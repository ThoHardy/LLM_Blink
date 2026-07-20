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

# NOTE: torch / transformers are imported lazily inside the HuggingFace code
# paths only.  The Ollama backend must work without them installed (see README).

# -- defaults ------------------------------------------------------------------
DEFAULT_HF_MODEL     = "Qwen/Qwen2.5-3B-Instruct"
# Generation budget for the report pass. 256 was too small: in the cot regime
# the model often re-enumerates all 15 packets inside <Thinking> (~400+ tokens)
# and got cut off before reaching the T2 slot (45% of cot trials in the first
# 0.5B pilot). 1024 leaves ample headroom; a stop string caps the actual cost.
DEFAULT_MAX_NEW_TOKENS = 1024
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

    def __init__(self, model_name: str, base_url: str = OLLAMA_BASE_URL,
                 approx_logprobs: bool = False) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for the Ollama backend.\n"
                "Install it with:  pip install openai"
            ) from exc
        self.model_name = model_name
        self.approx_logprobs = approx_logprobs
        self._client = OpenAI(base_url=base_url, api_key="ollama")

    # -- generation ------------------------------------------------------------

    def generate(self, system: str, user: str,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 temperature: float = 0.0,
                 stop_at: Optional[str] = None,
                 return_truncated: bool = False):
        """Free generation (temperature=0 -> greedy by default) -> assistant text.

        ``stop_at``: optional stop string (generation halts there; the OpenAI-
        compatible API strips it from the returned text).
        ``return_truncated=True`` -> returns (text, truncated) where
        ``truncated`` means generation hit the ``max_new_tokens`` budget
        (finish_reason == "length") instead of finishing naturally.
        """
        kwargs: dict = dict(
            model=self.model_name,
            messages=_build_messages(system, user),
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        if stop_at:
            kwargs["stop"] = [stop_at]
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        if not return_truncated:
            return text
        truncated = getattr(resp.choices[0], "finish_reason", None) == "length"
        return text, truncated

    # -- log-prob scoring ------------------------------------------------------

    def sequence_logprob(self, system: str, user_prefix: str,
                         target_text: str, prefilled_assistant: str = "") -> dict:
        """Graded T2 log-prob — NOT AVAILABLE on the Ollama backend (returns NaN).

        Ollama's API cannot teacher-force an arbitrary continuation: the old
        implementation (kept below, opt-in via ``approx_logprobs=True``) sent
        the prefix as a trailing assistant message, let Ollama GENERATE ~30
        tokens greedily, then char-aligned the generated tokens to
        ``target_text`` — substituting top-20 lookups on mismatch and a
        ``min(top20)-2`` / flat -25.0 penalty when absent. That is not a joint
        log-prob: it largely re-codes the binary report measure, and mismatch
        trials collapse onto a ~-25 penalty floor (found 2026-07-20; it
        produced the spurious "inverted load effect" in the gemma3:4b run).
        Whether Ollama's OpenAI-compat endpoint even continues a trailing
        assistant message (vs. closing the turn) is unverified.

        Use the HuggingFace backend for graded read-outs, or re-score a saved
        Ollama CSV exactly with ``rescore_graded.py`` (same model via HF).
        """
        if not self.approx_logprobs:
            warnings.warn(
                "[OllamaBackend] Graded log-prob scoring is disabled on the "
                "Ollama backend (it cannot teacher-force; the old approximate "
                "path re-codes the report measure and has a -25 penalty "
                "floor). Returning NaN. Use a HuggingFace model for graded "
                "read-outs, re-score the saved CSV with rescore_graded.py, or "
                "opt back in explicitly with approx_logprobs=True."
            )
            return _nan_logprob()

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
    ollama_approx_logprobs: bool = False,
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
        return OllamaBackend(model_name, base_url=ollama_base_url,
                             approx_logprobs=ollama_approx_logprobs), None

    model_name = name or DEFAULT_HF_MODEL
    return _load_hf(model_name, load_in_4bit)


# ==============================================================================
# Module-level scorer functions (dispatch on backend type)
# ==============================================================================

def report_generate(model, tok, system: str, user: str,
                    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                    temperature: float = 0.0,
                    stop_at: Optional[str] = None,
                    return_truncated: bool = False):
    """Free generation -> decoded assistant text.  Binary report measure.

    temperature=0.0 (default) decodes greedily; any positive value switches to
    sampling at that temperature (both backends). Log-prob scoring is unaffected.

    ``stop_at``: optional stop string — generation halts once it appears
    (e.g. the closing template tag), so a large ``max_new_tokens`` budget only
    costs compute on trials that actually need it. HF keeps the stop string in
    the returned text; the Ollama/OpenAI API strips it.
    ``return_truncated=True`` -> returns ``(text, truncated)``;
    ``truncated`` is True when generation ran into the ``max_new_tokens``
    budget instead of finishing naturally (EOS or stop string) — i.e. the
    output tail was cut off and downstream parsing of it cannot be trusted.
    """
    if isinstance(model, OllamaBackend):
        return model.generate(system, user, max_new_tokens,
                              temperature=temperature, stop_at=stop_at,
                              return_truncated=return_truncated)
    return _report_generate_hf(model, tok, system, user, max_new_tokens,
                               temperature=temperature, stop_at=stop_at,
                               return_truncated=return_truncated)


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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        return_tensors="pt", return_dict=True,
    )
    # apply_chat_template may return a raw tensor, a BatchEncoding, or a plain
    # dict depending on the transformers version (e.g. v5 returns a dict-like
    # object whose .input_ids attribute is not always exposed). Normalise to
    # the input_ids tensor in every case.
    if isinstance(out, dict):           # dict or BatchEncoding (dict subclass)
        out = out["input_ids"]
    elif hasattr(out, "input_ids"):
        out = out.input_ids
    return out


# backward-compat alias (was public in the original)
build_prompt_ids = _build_prompt_ids


def _report_generate_hf(model, tok, system: str, user: str,
                         max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                         temperature: float = 0.0,
                         stop_at: Optional[str] = None,
                         return_truncated: bool = False):
    import torch

    gen_kwargs: dict = dict(max_new_tokens=max_new_tokens,
                            pad_token_id=tok.eos_token_id)
    if stop_at:
        # transformers >= 4.41; halts generation once the string is emitted
        # (the string itself stays in the decoded output).
        gen_kwargs.update(stop_strings=[stop_at], tokenizer=tok)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=float(temperature))
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        ids = _build_prompt_ids(tok, system, user).to(model.device)
        out = model.generate(ids, **gen_kwargs)
        gen = out[0, ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        if not return_truncated:
            return text
        # Truncated = the full budget was spent AND generation did not end
        # naturally (no EOS as last token, stop string never produced).
        n_gen = int(gen.shape[0])
        ended_eos = n_gen > 0 and int(gen[-1].item()) == tok.eos_token_id
        ended_stop = bool(stop_at) and (stop_at in text)
        truncated = (n_gen >= max_new_tokens) and not ended_eos and not ended_stop
        return text, truncated


def _sequence_logprob_hf(model, tok, system: str, user_prefix: str,
                          target_text: str,
                          prefilled_assistant: str = "") -> dict:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        return _sequence_logprob_hf_impl(
            torch, F, model, tok, system, user_prefix,
            target_text, prefilled_assistant,
        )


def _sequence_logprob_hf_impl(torch, F, model, tok, system: str,
                              user_prefix: str, target_text: str,
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
