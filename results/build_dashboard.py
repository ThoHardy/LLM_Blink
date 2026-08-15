"""Assemble the AB smoke-test dashboard HTML with embedded figures + leaderboard.
Writes to the path given as argv[1]. Reloads data each run so it can be redeployed
as more models finish.
"""
import base64
import glob
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
UPDATED = sys.argv[2] if len(sys.argv) > 2 else ""

BOOL = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}

# fixed display order, smallest -> largest
ORDER = ["gemma3:270m", "gemma3:1b", "gemma2:2b", "gemma3:4b",
         "gemma2:9b", "gemma3:12b", "gemma2:27b", "gemma3:27b"]
PARAMS = {"gemma3:270m": "0.27B", "gemma3:1b": "1B", "gemma2:2b": "2B",
          "gemma3:4b": "4B", "gemma2:9b": "9B", "gemma3:12b": "12B",
          "gemma2:27b": "27B", "gemma3:27b": "27B"}


def slug(name):
    return "ab_" + name.replace(":", "_") + "_t0_s10"


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def gated(df):
    for c in ["report_correct", "t1_correct", "output_truncated",
              "t2_slot_missing", "thinking_is_placeholder"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].map(BOOL)
    g = (df.output_truncated.fillna(0) == 0) & (df.t2_slot_missing.fillna(0) == 0)
    g &= ~((df.regime == "cot") & (df.thinking_is_placeholder.fillna(0) == 1))
    return df[g]


# ---- gather per-model status ----
models = []
for name in ORDER:
    csv = os.path.join(HERE, slug(name) + ".csv")
    if not os.path.exists(csv):
        models.append(dict(name=name, params=PARAMS[name], done=False))
        continue
    df = pd.read_csv(csv)
    dfok = gated(df.copy())
    ret = len(dfok) / len(df) if len(df) else 0
    # ceiling test on binary report (gated)
    rep = dfok.groupby("t1_load")["report_correct"].mean()
    rep_none = rep.get("none", float("nan"))
    rep_sem = rep.get("semantic_4", float("nan"))
    t1 = dfok[dfok.t1_load == "semantic_4"]["t1_correct"].mean()
    ph = df[df.regime == "cot"]["thinking_is_placeholder"].fillna(0).mean()
    fig = os.path.join(HERE, "fig_" + slug(name) + ".png")
    models.append(dict(name=name, params=PARAMS[name], done=True,
                       n=len(df), ret=ret, rep_none=rep_none, rep_sem=rep_sem,
                       t1=t1, placeholder=ph,
                       fig=b64(fig) if os.path.exists(fig) else None,
                       degenerate=ret < 0.1))

# ---- leaderboard ----
lb_path = os.path.join(HERE, "leaderboard_so_far.csv")
lb = pd.read_csv(lb_path) if os.path.exists(lb_path) else pd.DataFrame()

done_models = [m for m in models if m.get("done")]
n_done = len(done_models)


def ceiling_flag(m):
    if not m.get("done"):
        return ""
    if m["degenerate"]:
        return '<span class="pill pill-bad">degenerate</span>'
    if (m["rep_none"] >= 0.95) and (m["rep_sem"] >= 0.95):
        return '<span class="pill pill-warn">report at ceiling</span>'
    if m["rep_none"] < 0.9 or m["rep_sem"] < 0.9:
        return '<span class="pill pill-good">has headroom</span>'
    return '<span class="pill pill-warn">near ceiling</span>'


def fmt(x, p=2):
    try:
        if pd.isna(x):
            return "—"
        return f"{x:.{p}f}"
    except Exception:
        return "—"


# ---- build HTML ----
rows_status = ""
for m in models:
    if not m.get("done"):
        rows_status += (f'<tr class="pending"><td>{m["name"]}</td><td class="num">{m["params"]}</td>'
                        f'<td colspan="5" class="pending-cell">running / queued…</td></tr>')
        continue
    rows_status += (
        f'<tr><td>{m["name"]}</td><td class="num">{m["params"]}</td>'
        f'<td class="num">{m["ret"]*100:.0f}%</td>'
        f'<td class="num">{fmt(m["rep_none"])}</td>'
        f'<td class="num">{fmt(m["rep_sem"])}</td>'
        f'<td class="num">{fmt(m["t1"])}</td>'
        f'<td>{ceiling_flag(m)}</td></tr>')

figs_html = ""
for m in done_models:
    if not m.get("fig"):
        continue
    tag = ceiling_flag(m)
    figs_html += (
        f'<figure class="figcard"><figcaption><span class="fig-name">{m["name"]}</span>'
        f'<span class="fig-meta">{m["params"]} · gated n={int(m["ret"]*m["n"])}/{m["n"]} {tag}</span></figcaption>'
        f'<img alt="AB curves for {m["name"]}" src="data:image/png;base64,{m["fig"]}"></figure>')

lb_rows = ""
if not lb.empty:
    lb = lb[lb.model.isin([m["name"] for m in done_models])]
    lb = lb.sort_values(["regime", "BlinkScore_logp"], ascending=[True, False])
    for _, r in lb.iterrows():
        blink = r["BlinkScore_logp"]
        cls = "num pos" if blink > 0.3 else ("num neg" if blink < -0.3 else "num")
        lb_rows += (
            f'<tr><td>{r["model"]}</td><td>{r["regime"]}</td>'
            f'<td class="num">{fmt(r["LoadCost_report"])}</td>'
            f'<td class="num">{fmt(r["BlinkScore_report"])}</td>'
            f'<td class="num">{fmt(r["LoadCost_logp"])}</td>'
            f'<td class="{cls}">{fmt(r["BlinkScore_logp"])}</td></tr>')

HTML = f"""<title>LLM Attentional Blink — smoke test</title>
<style>
:root {{
  --bg:#f7f7f9; --panel:#ffffff; --ink:#1a1c22; --muted:#606472;
  --line:#e5e6ec; --accent:#4f46e5; --accent-soft:#ecebfb;
  --none:#1f77b4; --load:#e8791a;
  --good:#0f8a5f; --good-bg:#e2f4ec; --warn:#b7791f; --warn-bg:#fdf3e0;
  --bad:#c23a3a; --bad-bg:#fbe6e6;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --bg:#0f1014; --panel:#181a20; --ink:#e9eaf0; --muted:#9aa0b0;
    --line:#2a2d37; --accent:#8b84ff; --accent-soft:#23233a;
    --none:#5aa9dd; --load:#f0994a;
    --good:#4ecb95; --good-bg:#123527; --warn:#e0b45f; --warn-bg:#332a15;
    --bad:#e77; --bad-bg:#3a1e1e; }}
}}
:root[data-theme="dark"] {{ --bg:#0f1014; --panel:#181a20; --ink:#e9eaf0; --muted:#9aa0b0;
  --line:#2a2d37; --accent:#8b84ff; --accent-soft:#23233a; --none:#5aa9dd; --load:#f0994a;
  --good:#4ecb95; --good-bg:#123527; --warn:#e0b45f; --warn-bg:#332a15; --bad:#e77; --bad-bg:#3a1e1e; }}
:root[data-theme="light"] {{ --bg:#f7f7f9; --panel:#ffffff; --ink:#1a1c22; --muted:#606472;
  --line:#e5e6ec; --accent:#4f46e5; --accent-soft:#ecebfb; --none:#1f77b4; --load:#e8791a;
  --good:#0f8a5f; --good-bg:#e2f4ec; --warn:#b7791f; --warn-bg:#fdf3e0; --bad:#c23a3a; --bad-bg:#fbe6e6; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
  line-height:1.55; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:48px 24px 80px; }}
header .eyebrow {{ font-family:var(--mono); font-size:12px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--accent); margin:0 0 8px; }}
h1 {{ font-size:2.1rem; line-height:1.1; margin:0 0 10px; letter-spacing:-.02em; text-wrap:balance; }}
.lede {{ color:var(--muted); max-width:64ch; margin:0; font-size:1.02rem; }}
.status-line {{ font-family:var(--mono); font-size:12.5px; color:var(--muted); margin-top:14px; }}
.status-line b {{ color:var(--ink); }}
section {{ margin-top:40px; }}
h2 {{ font-size:1.25rem; margin:0 0 4px; letter-spacing:-.01em; }}
.sub {{ color:var(--muted); font-size:.92rem; margin:0 0 18px; }}
.callout {{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--warn);
  border-radius:10px; padding:20px 22px; }}
.callout h2 {{ margin-bottom:8px; }}
.callout p {{ margin:8px 0 0; color:var(--ink); }}
.callout .muted {{ color:var(--muted); font-size:.92rem; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
.tablecard {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  overflow:hidden; }}
.scroll {{ overflow-x:auto; }}
th,td {{ text-align:left; padding:10px 14px; border-bottom:1px solid var(--line); white-space:nowrap; }}
thead th {{ font-size:11.5px; letter-spacing:.05em; text-transform:uppercase; color:var(--muted);
  font-weight:600; background:var(--accent-soft); }}
tbody tr:last-child td {{ border-bottom:none; }}
.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.pos {{ color:var(--good); font-weight:600; }}
.neg {{ color:var(--bad); font-weight:600; }}
tr.pending td {{ color:var(--muted); }}
.pending-cell {{ font-family:var(--mono); font-size:12px; }}
.pill {{ display:inline-block; font-size:11px; font-family:var(--mono); padding:2px 8px;
  border-radius:999px; letter-spacing:.02em; }}
.pill-good {{ background:var(--good-bg); color:var(--good); }}
.pill-warn {{ background:var(--warn-bg); color:var(--warn); }}
.pill-bad {{ background:var(--bad-bg); color:var(--bad); }}
.figgrid {{ display:grid; grid-template-columns:1fr; gap:22px; }}
@media (min-width:760px) {{ .figgrid {{ grid-template-columns:1fr 1fr; }} }}
.figcard {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:14px; }}
.figcard img {{ width:100%; height:auto; border-radius:6px; display:block; margin-top:10px;
  background:#fff; }}
figcaption {{ display:flex; flex-direction:column; gap:2px; }}
.fig-name {{ font-family:var(--mono); font-weight:600; font-size:14px; }}
.fig-meta {{ color:var(--muted); font-size:12px; }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; font-size:13px; color:var(--muted); margin-top:6px; }}
.swatch {{ display:inline-block; width:11px; height:11px; border-radius:3px; vertical-align:middle;
  margin-right:6px; }}
.foot {{ margin-top:48px; color:var(--muted); font-size:12.5px; font-family:var(--mono);
  border-top:1px solid var(--line); padding-top:16px; }}
code {{ font-family:var(--mono); font-size:.9em; background:var(--accent-soft); padding:1px 5px;
  border-radius:4px; }}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Issue #9 · Gemma sweep · n_seeds = 10 (smoke)</p>
  <h1>Attentional&#8209;Blink smoke test</h1>
  <p class="lede">Does a lag-selective T2 deficit appear after a demanding T1? First pass across
  the Gemma fleet at 10 seeds/cell — a directional read to decide whether the effect is worth
  a large run, not a measurement.</p>
  <p class="status-line"><b>{n_done}/8</b> models done · greedy (T=0) · gated on
  <code>output_truncated</code>, <code>t2_slot_missing</code>, cot <code>placeholder</code> ·
  M4&nbsp;Max, 8-way parallel</p>
</header>

<section>
  <div class="callout">
    <h2>Headline: the binary report saturates at ceiling</h2>
    <p>From 2B up, the model reports the T2 passphrase almost perfectly under <em>both</em> loads
    (P≈1.0), so the "conscious" binary measure has no room to show a blink — <code>LoadCost</code>
    and <code>BlinkScore</code> on report collapse to ~0 mechanically. The 3-word NATO passphrase is
    too easy to report.</p>
    <p class="muted">Only the graded log-prob measure keeps dynamic range. Actionable next step for
    the real run: titrate T2 toward ~50% report (longer / masked / more confusable passphrase), or
    lead with the graded measure. n=10 is very noisy — treat every curve as directional.</p>
  </div>
</section>

<section>
  <h2>Per-model status &amp; quality</h2>
  <p class="sub">Gated retention, binary report by load (ceiling check), and T1 accuracy on the
  hard load (was the cost actually paid?).</p>
  <div class="tablecard scroll">
  <table>
    <thead><tr><th>Model</th><th class="num">Params</th><th class="num">Gated</th>
      <th class="num">report·none</th><th class="num">report·sem4</th>
      <th class="num">T1 acc·sem4</th><th>Signal</th></tr></thead>
    <tbody>{rows_status}</tbody>
  </table>
  </div>
</section>

<section>
  <h2>AB curves (gated)</h2>
  <p class="sub">Top row of each card: P(T2 reported), binary. Bottom row: mean log&#8202;p(T2), graded.
  Columns: cot | direct. Error bars = SEM over 10 seeds.</p>
  <div class="legend">
    <span><span class="swatch" style="background:var(--none)"></span>T1 = none (positional baseline)</span>
    <span><span class="swatch" style="background:var(--load)"></span>T1 = semantic_4 (hard load)</span>
  </div>
  <div class="figgrid" style="margin-top:16px">{figs_html}</div>
</section>

<section>
  <h2>Blink leaderboard <span class="sub" style="display:inline">(preliminary)</span></h2>
  <p class="sub"><code>Δ(lag)=P(none)−P(sem4)</code> per lag. <code>LoadCost</code>=mean Δ over all lags;
  <code>BlinkScore</code>=mean Δ over lags 2/4/6 minus mean over 0/10 (positive = lag-selective =
  blink-shaped). Report columns are ~0 at ceiling — read the log-prob columns for now.</p>
  <div class="tablecard scroll">
  <table>
    <thead><tr><th>Model</th><th>Regime</th>
      <th class="num">LoadCost·rep</th><th class="num">Blink·rep</th>
      <th class="num">LoadCost·logp</th><th class="num">Blink·logp</th></tr></thead>
    <tbody>{lb_rows}</tbody>
  </table>
  </div>
</section>

<p class="foot">Live — redeployed as 12B / 27B land ({n_done}/8 in). {UPDATED}
Data: <code>results/ab_*_t0_s10.csv</code> · issue #9, ThoHardy/LLM_Blink.</p>
</div>
"""

with open(OUT, "w") as f:
    f.write(HTML)
print(f"wrote {OUT} ({len(HTML)//1024} KB, {n_done} models)")
