"""Model loading + the two core scorers.

Two backends are supported, selected automatically from the model name or via
the ``backend`` argument to :func:`load_model`.

**HuggingFace backend** (default for Colab / GPU)
    Pass any HuggingFace Hub model ID (contains a ``/``), e.g.
    ``"Qwen/Qwen2.5-3B-Instruct"``.  Uses ``transformers`` + ``torch``.
    Supports *exact* teacher-forced log-prob scoring.

**Ollama backend** (for local machines)
    Pass any Ollama model tag (no ``/``), e.g. ``"llama3.2:3b"`` or
    ``"qwen2.5:3b"``.  Requires a running ``ollama serve`` and
    ``pip install openai``.  Log-prob scoring uses the OpenAI-compatible
    endpoint with ``logprobs=True`` (Ollama >= 0.3).

Public API (unchanged from the original single-backend version):

    model, tok = load_model(name)
    text        = report_generate(model, tok, system, user)
    result      = sequence_logprob(model, tok, system, user_prefix, target, ...)

For the Ollama backend ``tok`` is ``None``; callers never need to inspect it.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# -- defaults ------------------------------------------------------------------
DEFAULT_HF_MODEL     = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL      = "http://localhost:11434/v1"


# ==============================================================================
# Ollama backend
# ==============================================================================

class OllamaBackend:
    """Wrapper around Ollama's OpenAI-compatible endpoint.

    Parameters
    ----------
    model_name : str
        Ollama model tag, e.g. "llama3.2:3b" or "qwen2.5:3b".
    base_url : str
        Base URL of the Ollama server (default http://localhost:11434/v1).
    """

    def __init__(self, model_name: str, base_url: str = OLLAMA_BASE_URL) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for the Ollama backend.\n"
                "Install it with:  pip install openai"
            ) from exc
        self.model_name = model_name
        self._client = OpenAI(base_url=base_url, api_key="ollama")

    # -- generation ------------------------------------------------------------

    def generate(self, system: str, user: str, max_new_tokens: int = 256) -> str:
        """Free generation (greedy, temperature=0) -> assistant text."""
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=_build_messages(system, user),
            max_tokens=max_new_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    # -- log-prob scoring ------------------------------------------------------

    def sequence_logprob(self, system: str, user_prefix: str,
                         target_text: str, prefilled_assistant: str = "") -> dict:
        """Approximate teacher-forced log-prob via Ollama's logprobs API.

        The trick: pass ``prefilled_assistant`` as the last assistant message so
        Ollama continues exactly from that prefix, then collect log-probs for the
        tokens that cover ``target_text``.

        Requires Ollama >= 0.3.  Returns NaN fields on failure.
        """
        messages = _build_messages(system, user_prefix)
        if prefilled_assistant:
            messages.append({"role": "assistant", "content": prefilled_assistant})

        max_tokens = max(len(target_text.split()) * 4 + 10, 30)

        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                logprobs=True,
                top_logprobs=20,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[OllamaBackend] logprobs request failed ({exc}). "
                "Returning NaN for log-prob fields."
            )
            return _nan_logprob()

        lp_obj = resp.choices[0].logprobs
        token_data = getattr(lp_obj, "content", None) if lp_obj else None
        if not token_data:
            warnings.warn(
                "[OllamaBackend] No logprob data in response. "
                "Make sure you are running Ollama >= 0.3."
            )
            return _nan_logprob()

        return _extract_target_logprobs(token_data, target_text)


# ==============================================================================
# Public factory
# ==============================================================================

def load_model(
    name: Optional[str] = None,
    backend: str = "auto",
    load_in_4bit: bool = False,
    ollama_base_url: str = OLLAMA_BASE_URL,
):
    """Load a model and return (model, tok).

    Backend selection
    -----------------
    backend="auto" (default):
        name contains "/"  ->  HuggingFace  e.g. "Qwen/Qwen2.5-3B-Instruct"
        otherwise          ->  Ollama        e.g. "llama3.2:3b"
    backend="hf"      -> force HuggingFace
    backend="ollama"  -> force Ollama

    Returns
    -------
    (model, tok)
        HuggingFace: (AutoModelForCausalLM, AutoTokenizer) as before.
        Ollama:      (OllamaBackend, None)  -- tok is always None; treat as opaque.
    """
    resolved = _resolve_backend(name, backend)

    if resolved == "ollama":
        model_name = name or DEFAULT_OLLAMA_MODEL
        return OllamaBackend(model_name, base_url=ollama_base_url), None

    model_name = name or DEFAULT_HF_MODEL
    return _load_hf(model_name, load_in_4bit)


# ==============================================================================
# Module-level scorer functions (dispatch on backend type)
# ==============================================================================

def report_generate(model, tok, system: str, user: str,
                    max_new_tokens: int = 256) -> str:
    """Free generation -> decoded assistant text.  Binary report measure."""
    if isinstance(model, OllamaBackend):
        return model.generate(system, user, max_new_tokens)
    return _report_generate_hf(model, tok, system, user, max_new_tokens)


def sequence_logprob(model, tok, system: str, user_prefix: str,
                     target_text: str, prefilled_assistant: str = "") -> dict:
    """Joint log-prob of target_text continuing the prompt.

    Graded "unconscious strength" measure.  Dict keys:
        total_logprob, mean_logprob, joint_prob, n_tokens, per_token_logprob.
    For Ollama values may be NaN if logprobs are unavailable.
    """
    if isinstance(model, OllamaBackend):
        return model.sequence_logprob(system, user_prefix, target_text,
                                      prefilled_assistant)
    return _sequence_logprob_hf(model, tok, system, user_prefix, target_text,
                                prefilled_assistant)


# ==============================================================================
# HuggingFace implementation (unchanged logic from original)
# ==============================================================================

def _load_hf(name: str, load_in_4bit: bool = False):
    tok = AutoTokenizer.from_pretrained(name)
    kwargs: dict = dict(device_map="auto")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig  # type: ignore
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tok


def _build_prompt_ids(tok, system: str, user: str,
                      add_generation_prompt: bool = True):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    out = tok.apply_chat_template(
        msgs, add_generation_prompt=add_generation_prompt,
        return_tensors="pt"
    )
    # Newer transformers return a BatchEncoding (dict-like) instead of a
    # raw tensor; normalise to the input_ids tensor either way.
    if hasattr(out, "input_ids"):
        out = out.input_ids
    return out


# backward-compat alias (was public in the original)
build_prompt_ids = _build_prompt_ids


@torch.no_grad()
def _report_generate_hf(model, tok, system: str, user: str,
                         max_new_tokens: int = 256) -> str:
    ids = _build_prompt_ids(tok, system, user).to(model.device)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tok.eos_token_id,
    )
    gen = out[0, ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def _sequence_logprob_hf(model, tok, system: str, user_prefix: str,
                          target_text: str,
                          prefilled_assistant: str = "") -> dict:
    ids_prompt = _build_prompt_ids(tok, system, user_prefix,
                                   add_generation_prompt=True)
    if prefilled_assistant:
        pre = tok(prefilled_assistant, add_special_tokens=False,
                  return_tensors="pt").input_ids
        ids_prompt = torch.cat([ids_prompt, pre], dim=1)

    tgt  = tok(target_text, add_special_tokens=False,
               return_tensors="pt").input_ids
    full = torch.cat([ids_prompt, tgt], dim=1).to(model.device)

    logits   = model(full).logits          # [1, T, V]
    n_prompt = ids_prompt.shape[1]
    n_tgt    = tgt.shape[1]

    pred_logits = logits[0, n_prompt - 1: n_prompt + n_tgt - 1, :]
    logprobs    = F.log_softmax(pred_logits.float(), dim=-1)
    target_ids  = full[0, n_prompt: n_prompt + n_tgt]
    per_tok     = logprobs[torch.arange(n_tgt), target_ids]

    total = per_tok.sum().item()
    return {
        "total_logprob":     total,
        "mean_logprob":      total / max(n_tgt, 1),
        "joint_prob":        float(torch.exp(torch.tensor(total))),
        "n_tokens":          int(n_tgt),
        "per_token_logprob": per_tok.tolist(),
    }


# ==============================================================================
# Shared helpers
# ==============================================================================

def _build_messages(system: str, user: str) -> list:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs


def _nan_logprob() -> dict:
    return {
        "total_logprob":     float("nan"),
        "mean_logprob":      float("nan"),
        "joint_prob":        float("nan"),
        "n_tokens":          0,
        "per_token_logprob": [],
    }


def _extract_target_logprobs(token_data_list, target_text: str) -> dict:
    """Walk generated tokens and collect those covering target_text.

    Uses character-level alignment: track how much of target_text has been
    consumed as tokens are processed.  If the generated token matches the
    expected prefix, use its log-prob directly.  On mismatch, look the expected
    token up in top_logprobs; if absent, use a conservative lower bound.
    """
    per_tok: list = []
    remaining = target_text

    for td in token_data_list:
        if not remaining:
            break

        tok_str = td.token
        tok_lp  = td.logprob

        if remaining.startswith(tok_str):
            per_tok.append(tok_lp)
            remaining = remaining[len(tok_str):]

        elif tok_str.startswith(remaining):
            per_tok.append(tok_lp)
            remaining = ""

        else:
            top_list = getattr(td, "top_logprobs", None) or []
            found_lp = None
            consumed = 0
            for top in top_list:
                if remaining.startswith(top.token):
                    found_lp = top.logprob
                    consumed = len(top.token)
                    break
                if top.token.startswith(remaining):
                    found_lp = top.logprob
                    consumed = len(remaining)
                    break

            if found_lp is not None:
                per_tok.append(found_lp)
                remaining = remaining[consumed:]
            else:
                top_lps = [t.logprob for t in top_list]
                per_tok.append((min(top_lps) - 2.0) if top_lps else -25.0)
                remaining = remaining[len(tok_str):] if len(tok_str) <= len(remaining) else ""

    if not per_tok:
        return _nan_logprob()

    total = sum(per_tok)
    return {
        "total_logprob":     total,
        "mean_logprob":      total / len(per_tok),
        "joint_prob":        float(math.exp(max(total, -700))),
        "n_tokens":          len(per_tok),
        "per_token_logprob": per_tok,
    }


def _resolve_backend(name: Optional[str], backend: str) -> str:
    if backend not in ("auto", "hf", "ollama"):
        raise ValueError(
            f"backend must be 'auto', 'hf', or 'ollama'; got {backend!r}"
        )
    if backend != "auto":
        return backend
    return "hf" if (name and "/" in name) else "ollama"
