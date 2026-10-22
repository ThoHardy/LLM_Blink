"""Read-out layer (design H): parse the free report, locate scoring slots.

Handles the task-stream design's free report — one line per task in the form
``- Task <NAME>: <result>`` — and provides the teacher-forced scoring prefix
for the passphrase task. Legacy-template parsing ("Target 2 Result:") stays
in experiment.py.

Parsing is deliberately tolerant (missing dash, any case, stray quotes,
bullets/numbering, quoted list items inside code fences): formatting slips
must not masquerade as detection failures. 2026-07-21 hardening: Qwen0.5B in
the direct regime answered inside a ```python code fence as quoted list items
(``'Task URCHIN: RED',``) — 17/20 direct+trivial trials parsed as 0 tasks
reported. The name-cued probe (two-stage report -> full access taxonomy) is
deliberately NOT implemented yet (Thomas, 2026-07-20): for now a task is
either spontaneously reported or not.
"""
from __future__ import annotations
import re


# One report line. Tolerated prefixes before "Task": an optional bullet
# (-, *, •) or numbering (1. / 1)), then an optional opening quote (straight
# or curly, single or double, or backtick). The answer capture is cleaned by
# _clean_response afterwards (closing quote + trailing comma of list items).
_TASK_LINE_RE = re.compile(
    r"^\s*(?:(?:[-*•]|\d+[.)])\s*)?[\"'‘’“”`]?"
    r"\s*Task\s+([A-Za-z0-9]+)\s*:\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE)
_FENCE_RE = re.compile(r"^\s*```.*$", re.MULTILINE)  # code-fence marker lines
_FINAL_OPEN = "<Final_Answers>"


def normalize_answer(text):
    """Uppercase, strip quotes/backticks and trailing punctuation, collapse
    whitespace. Applied to BOTH sides of every correctness comparison."""
    if text is None:
        return None
    t = str(text).strip().strip('"\N{APOSTROPHE}`').strip()
    t = re.sub(r"[.!?]+$", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t.upper()


def _clean_response(ans: str) -> str:
    """Strip list-item wrapping from a captured answer: trailing comma or
    semicolon (python-list / enumeration style) and symmetric quotes left by
    the tolerant line regex. Inner content untouched."""
    ans = ans.strip()
    ans = re.sub(r"[,;]\s*$", "", ans).strip()
    ans = re.sub(r"[\"'‘’“”`]+\s*$", "", ans)
    ans = re.sub(r"^[\"'‘’“”`]+", "", ans)
    return ans.strip()


def _answers_region(generated: str):
    """(region, offset) to parse for task lines: everything after
    <Final_Answers> when present — the CoT may legitimately restate task
    lines while reasoning — else the whole text."""
    i = generated.find(_FINAL_OPEN)
    if i == -1:
        return generated, 0
    off = i + len(_FINAL_OPEN)
    return generated[off:], off


_THINK_OPEN, _THINK_CLOSE = "<Thinking>", "</Thinking>"
_PACKET_REF_RE = re.compile(r"Packet\s*0?(\d{1,2})", re.IGNORECASE)


def _thinking_region(generated: str) -> str:
    """The model's <Thinking> block (empty string if absent). If the closing
    tag is missing (forced close / truncation), everything from <Thinking> up
    to <Final_Answers> (or the end) counts."""
    i = generated.find(_THINK_OPEN)
    if i == -1:
        return ""
    start = i + len(_THINK_OPEN)
    j = generated.find(_THINK_CLOSE, start)
    if j == -1:
        j = generated.find(_FINAL_OPEN, start)
    return generated[start:j] if j != -1 else generated[start:]


def access_cues(trial):
    """Content cues for the access probe: the target's own cues when set (e.g.
    the expression of a trivial_math target, whose bare integer answer would
    match any number in a math CoT), else the phrase itself."""
    return getattr(trial, "t2_access_cues", None) or trial.t2_phrase


def t2_in_cot(generated: str, t2_phrase,
              t2_task_name: str | None = None) -> bool:
    """Binary access probe (#14 §3.2): is the passphrase task mentioned or
    processed anywhere inside the model's own <Thinking> block, by name or
    content? Generalises ``t2_echoed_in_cot`` (which was phrase-only).
    ``t2_phrase`` may be a string or a tuple of alternative content cues.

    Accepts either a full trajectory (the <Thinking> span is extracted) or a
    bare CoT continuation (no tags — the whole string is searched). Used both
    on the realized CoT and on each pre_cot-forked sample.
    """
    if not generated:
        return False
    think = _thinking_region(generated)
    span = think if think else generated
    up = span.upper()
    cues = (t2_phrase,) if isinstance(t2_phrase, str) else tuple(t2_phrase or ())
    if any(c and str(c).upper() in up for c in cues):
        return True
    if t2_task_name:
        if re.search(r"\b" + re.escape(str(t2_task_name).upper()) + r"\b", up):
            return True
    return False


def cot_enumeration_stats(generated: str, task_packets) -> dict:
    """Compliance read-out for the anti-enumeration instruction (idea I1).

    Counts DISTINCT packet indices referenced inside <Thinking> ("Packet 07",
    "packet 3", ...). Mentioning task packets is legitimate (the worked
    example does it); enumeration NON-compliance = referencing filler packets.

    Returns: n_packets_in_cot (distinct indices referenced),
    n_filler_packets_in_cot (those not in task_packets),
    enumerated_fillers_in_cot (bool, >=2 filler refs — one stray mention is
    not an enumeration). Purely textual; usable retroactively via
    analyze.backfill_readout_columns on any saved CSV.
    """
    think = _thinking_region(generated)
    idxs = {int(m.group(1)) for m in _PACKET_REF_RE.finditer(think)}
    task_set = set(int(p) for p in task_packets)
    fillers = idxs - task_set
    return {"n_packets_in_cot": len(idxs),
            "n_filler_packets_in_cot": len(fillers),
            "enumerated_fillers_in_cot": len(fillers) >= 2}


def parse_task_report_names(generated: str, valid_names) -> dict:
    """Core parser, trial-free (usable on saved CSVs via the tasks JSON).

    ``valid_names``: iterable of task names for this trial. Returns
    ``{"reported": {NAME: raw_answer, ...}, "hallucinated": [...]}`` where
    NAME is uppercased and only the FIRST line per task name counts.
    Hallucinated = well-formed task lines whose name matches no task in the
    trial (fabricated tasks — a detection-side error worth logging).
    """
    region, _ = _answers_region(generated)
    region = _FENCE_RE.sub("", region)
    valid = {str(n).upper() for n in valid_names}
    reported, hallucinated = {}, []
    for m in _TASK_LINE_RE.finditer(region):
        name = m.group(1).upper()
        if name in valid:
            reported.setdefault(name, _clean_response(m.group(2)))
        else:
            hallucinated.append(name)
    return {"reported": reported, "hallucinated": hallucinated}


def parse_task_report(generated: str, trial) -> dict:
    """Parse the free report of a Trial (see parse_task_report_names)."""
    return parse_task_report_names(
        generated, (t["name"] for t in trial.tasks))


def answer_contains(response, answer) -> bool:
    """Lenient correctness: the normalized answer appears as a whole-word
    substring of the normalized response. Catches instruction-echo responses
    like 'copy-paste these three words: "SIERRA ROMEO LIMA".' which contain
    the correct passphrase verbatim — by the project's operational definition
    ("appears in the tokenized output") these count as access (2026-07-21).
    Word-bounded to avoid e.g. NO matching in NOT."""
    if response is None or answer is None:
        return False
    r, a = normalize_answer(response), normalize_answer(answer)
    if not a:
        return False
    return re.search(r"\b" + re.escape(a) + r"\b", r) is not None


def task_rows(trial_or_tasks, reported: dict) -> list:
    """Per-task result dicts (JSON-ready) for the ``tasks`` column.

    ``trial_or_tasks``: a Trial, or directly its task-dict list (so old CSVs
    can be re-parsed from the stored ``tasks`` JSON without rebuilding
    stimuli). Besides exact ``correct``, each row carries (2026-07-21):
    - ``correct_lenient``: answer_contains(response, answer) — exact-match
      undercounts access when the model wraps the right answer in echo text.
    - ``answer_migrated``: the (wrong) response exactly equals ANOTHER task's
      normalized answer — a binding error (illusory-conjunction analog):
      27-40% of reported lines in the 0.5B HAB pilot.
    """
    tasks = getattr(trial_or_tasks, "tasks", trial_or_tasks)
    norm_answers = {t["name"]: normalize_answer(t["answer"]) for t in tasks}
    rows = []
    for t in tasks:
        resp = reported.get(t["name"].upper())
        nresp = normalize_answer(resp) if resp is not None else None
        exact = (nresp == norm_answers[t["name"]]) if resp is not None else False
        others = [a for n, a in norm_answers.items() if n != t["name"]]
        rows.append(dict(
            name=t["name"], kind=t["kind"], packet=t["packet"], rank=t["rank"],
            answer=t["answer"],
            reported=resp is not None,
            response=resp,
            correct=exact,
            correct_lenient=answer_contains(resp, t["answer"]),
            answer_migrated=(not exact) and nresp is not None
                            and nresp in others,
        ))
    return rows


def _mention_pattern(task) -> re.Pattern:
    """A task is 'mentioned' via 'Task <NAME>', its packet number, or (for
    non-numeric names) its bare name. Bare numeric names ('ordered' naming)
    would match any digit in a math CoT, so they need the 'Task' prefix."""
    name = re.escape(str(task["name"]).upper())
    alts = [rf"\bTASK\s+{name}\b", rf"\bPACKET\s*0?{int(task['packet'])}\b"]
    if not str(task["name"]).isdigit():
        alts.append(rf"\b{name}\b")
    return re.compile("|".join(alts))


def task_processing_order(generated: str, trial, regime: str = "cot") -> list:
    """Uppercased task names in the order the model PROCESSED them.

    cot: order of first mention inside <Thinking> (tasks never mentioned are
    absent). direct, or a cot sample without a <Thinking> block: order of the
    task lines in <Final_Answers>."""
    think = _thinking_region(generated) if regime == "cot" else ""
    if think:
        up = think.upper()
        pos = {}
        for t in trial.tasks:
            m = _mention_pattern(t).search(up)
            if m:
                pos[str(t["name"]).upper()] = m.start()
        return sorted(pos, key=pos.get)
    return list(parse_task_report(generated, trial)["reported"].keys())


def instructed_order_keys(trial, compute_order: str = "none") -> dict:
    """name -> sort key of the instructed processing order. 'none' uses stream
    order (spontaneous compliance); 'load_first' ties all non-copy tasks."""
    if compute_order == "reverse":
        return {str(t["name"]).upper(): -t["rank"] for t in trial.tasks}
    if compute_order == "load_first":
        return {str(t["name"]).upper(): int(t["kind"] == "passphrase")
                for t in trial.tasks}
    return {str(t["name"]).upper(): t["rank"] for t in trial.tasks}


def compute_order_respected(generated: str, trial, compute_order: str = "none",
                            regime: str = "cot"):
    """Manipulation check for the compute-order axis: Kendall tau-b between
    the realised processing order (task_processing_order) and the instructed
    one. +1 = fully compliant, -1 = fully reversed, None if < 2 tasks seen or
    the instructed keys are all tied among the tasks seen. For 'load_first'
    (loads mutually unordered) it is 2*f - 1, f = fraction of the other
    processed tasks handled before the target (None if the target is absent)."""
    from scipy.stats import kendalltau
    keys = instructed_order_keys(trial, compute_order)
    seq = [keys[n] for n in task_processing_order(generated, trial, regime)
           if n in keys]
    if compute_order == "load_first":
        if 1 not in seq or len(seq) < 2:
            return None
        i = seq.index(1)
        return 2 * (i / (len(seq) - 1)) - 1
    if len(seq) < 2 or len(set(seq)) < 2:
        return None
    tau, _ = kendalltau(list(range(len(seq))), seq)
    return None if tau != tau else round(float(tau), 6)


def passphrase_processing_rank(generated: str, trial, regime: str = "cot"):
    """1-based rank of the target task in the realised processing order, or
    None when it was never processed (the blink itself)."""
    order = task_processing_order(generated, trial, regime)
    pp = str(trial.t2_task_name).upper()
    return order.index(pp) + 1 if pp in order else None


def t2_scoring_prefix_tasks(generated: str, task_name: str):
    """Split the generated text at the passphrase task's answer slot.

    Searches the <Final_Answers> region (NOT the CoT, which may re-enumerate
    packets and echo task names) for the model's own ``- Task <NAME>:`` line
    (tolerant to bullets/quotes like the report parser); the prefix ends
    right after the marker (+ optional opening quote), i.e. exactly where the
    model's own passphrase tokens would begin. If the model never emitted the
    slot, fall back to appending a fresh slot line to the full text and flag
    the trial (slot_missing=True) — same convention as the legacy
    ``_t2_scoring_prefix``.
    """
    region, off = _answers_region(generated)
    m = re.search(
        r"(?:(?:[-*•]|\d+[.)])\s*)?[\"'‘’“”`]?"
        r"\s*Task\s+" + re.escape(task_name) +
        r"\s*:\s*[\"'‘’“”`]?",
        region, re.IGNORECASE)
    if m:
        return generated[: off + m.end()], False
    return generated.rstrip() + f"\n- Task {task_name}: ", True
