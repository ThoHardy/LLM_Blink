"""Assemble the CoT-induced blindness dashboard (HTML) for the artifact.

Reads cib_master_table.csv + the three fig_cib_*.png and emits
results/cib_dashboard.html (body content only; the Artifact host wraps it).
Theme-aware, self-contained (images inlined as data URIs).
"""
from __future__ import annotations
import base64
import os
import html
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
LOADS = ["trivial", "semantic_1", "semantic_2", "semantic_3", "semantic_4"]


def datauri(png):
    b = base64.b64encode(open(os.path.join(_HERE, png), "rb").read()).decode()
    return f"data:image/png;base64,{b}"


def leaderboard_rows(mt, load):
    sub = mt[(mt.k == 5) & (mt.load == load)].sort_values("cib", ascending=False)
    out = []
    for _, r in sub.iterrows():
        ceiling = (r.direct > 0.98 and r.cot > 0.98)
        cot_helps_t1 = r.t1_gain_cot > 0.02
        if ceiling:
            flag, cls = "ceiling", "muted"
        elif not cot_helps_t1 and load != "trivial":
            flag, cls = "CoT⊥load", "warn"     # control-1 fail: CoT doesn't help the load
        elif r.cib > 0.15:
            flag, cls = "clean", "good"
        else:
            flag, cls = "weak", ""
        out.append((r.model, int(r.params_b), r.direct, r.cot, r.cib,
                    r.t1_gain_cot, flag, cls))
    return out


def table_html(rows, load):
    body = ""
    for m, p, d, c, cib, t1g, flag, cls in rows:
        cibcol = "pos" if cib > 0.05 else ("neg" if cib < -0.05 else "")
        body += (f"<tr class='{cls}'><td class='mono'>{html.escape(m)}</td>"
                 f"<td class='num'>{p}</td><td class='num'>{d:.2f}</td>"
                 f"<td class='num'>{c:.2f}</td>"
                 f"<td class='num cib {cibcol}'>{cib:+.2f}</td>"
                 f"<td class='num'>{t1g:+.2f}</td>"
                 f"<td class='flag'>{flag}</td></tr>")
    return (f"<table><thead><tr><th>model</th><th>B</th><th>Direct</th>"
            f"<th>CoT</th><th>CIB</th><th>t1 gain<br>(CoT−Direct)</th><th></th>"
            f"</tr></thead><tbody>{body}</tbody></table>")


def main():
    mt = pd.read_csv(os.path.join(_HERE, "cib_master_table.csv"))
    n_models = mt["model"].nunique()

    tabs = "".join(
        f"<button class='tab' data-load='{l}'>{l}</button>" for l in LOADS)
    panels = ""
    for i, l in enumerate(LOADS):
        panels += (f"<div class='panel' data-load='{l}' "
                   f"style='display:{'block' if i==0 else 'none'}'>"
                   f"{table_html(leaderboard_rows(mt, l), l)}</div>")

    fig_paradox = datauri("fig_cib_paradox.png")
    fig_load = datauri("fig_cib_by_load.png")
    fig_cap = datauri("fig_cib_vs_capacity.png")

    HTML = f"""<title>CoT-induced blindness — scale sweep</title>
<style>
  :root {{
    --bg:#faf9f7; --card:#ffffff; --ink:#0b0b0b; --muted:#52514e; --line:#e6e6e3;
    --good:#0a7d43; --warn:#b5540a; --pos:#2a78d6; --neg:#c0392b; --accent:#2a78d6;
  }}
  @media (prefers-color-scheme:dark){{ :root:not([data-theme=light]){{
    --bg:#141413; --card:#1e1e1c; --ink:#f5f4ef; --muted:#a8a79e; --line:#33322f;
    --good:#5bce8f; --warn:#e0913f; --pos:#6da7ec; --neg:#e57367; --accent:#6da7ec;
  }}}}
  :root[data-theme=dark]{{
    --bg:#141413; --card:#1e1e1c; --ink:#f5f4ef; --muted:#a8a79e; --line:#33322f;
    --good:#5bce8f; --warn:#e0913f; --pos:#6da7ec; --neg:#e57367; --accent:#6da7ec;
  }}
  *{{box-sizing:border-box}}
  body,.wrap{{margin:0}}
  .wrap{{max-width:1080px;margin:0 auto;padding:32px 22px 80px;
    font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    color:var(--ink);background:var(--bg)}}
  h1{{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}}
  h2{{font-size:19px;margin:38px 0 12px;letter-spacing:-.01em}}
  .sub{{color:var(--muted);font-size:14px;margin-bottom:6px}}
  .status{{display:inline-block;background:var(--card);border:1px solid var(--line);
    border-radius:8px;padding:6px 12px;font-size:13px;color:var(--muted);margin-top:8px}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;
    padding:18px;margin:14px 0}}
  .lead{{font-size:16px}}
  .lead b{{color:var(--accent)}}
  ul{{margin:8px 0;padding-left:20px}} li{{margin:5px 0}}
  img{{max-width:100%;height:auto;display:block;border-radius:8px;background:#fff;padding:6px}}
  figcaption{{color:var(--muted);font-size:13px;margin-top:8px}}
  .tabs{{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 12px}}
  .tab{{background:var(--card);border:1px solid var(--line);color:var(--muted);
    border-radius:20px;padding:5px 14px;font-size:13px;cursor:pointer;font-family:inherit}}
  .tab.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
  .tablewrap{{overflow-x:auto}}
  table{{border-collapse:collapse;width:100%;font-size:13.5px}}
  th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid var(--line)}}
  th{{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}}
  td.num,th{{}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
  .mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}}
  .cib{{font-weight:700}} .cib.pos{{color:var(--pos)}} .cib.neg{{color:var(--neg)}}
  tr.muted td{{color:var(--muted);opacity:.6}}
  tr.good td.flag{{color:var(--good);font-weight:600}}
  tr.warn td.flag{{color:var(--warn);font-weight:600}}
  .flag{{font-size:11.5px}}
  .legend{{font-size:12.5px;color:var(--muted);margin-top:8px}}
  .legend b{{color:var(--ink)}}
  .quote{{border-left:3px solid var(--accent);padding:6px 0 6px 14px;margin:10px 0;
    color:var(--muted);font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:pre-wrap}}
  code{{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:12.5px}}
</style>
<div class="wrap">
  <h1>CoT-induced blindness — scale sweep</h1>
  <div class="sub">Issue #10 · simplified design: <b>k=5, budget=None (unlimited CoT), passphrase-last, per-load</b> · Ollama · temp 0 · n_seeds=100</div>
  <div class="status">🌙 {n_models} models done overnight (0.5B→12B); sweep continuing up the ladder to 72B · updated live</div>

  <div class="card lead">
    <b>Headline.</b> At <b>unlimited CoT budget</b>, CoT-induced blindness is <b>real but modest</b>, and it lives in <b>small-to-mid models on semantic loads</b>. Removing the token-budget cap removed most of the effect the pilots saw — <b>the budget was doing the heavy lifting</b>. Above ~9B both regimes sit at ceiling (nothing to measure with these loads). The clean paradox still shows: <b>CoT is neutral/helpful at k=1 but blinds the model to the passphrase at k=5</b>.
  </div>

  <h2>The paradox — k=1 vs k=5</h2>
  <figure class="card">
    <img src="{fig_paradox}" alt="k=1 vs k=5 slopegraph">
    <figcaption>Each line is one model, Direct→CoT. <b>Left (k=1):</b> a single passphrase task — CoT is flat/up (no blindness). <b>Right (k=5):</b> 4 load tasks + passphrase — CoT slopes down = blindness. Semantic loads pooled. Models at ceiling in both regimes are greyed. <code>llama3.2:3b</code> is the exception (CoT helps it even at k=5).</figcaption>
  </figure>

  <h2>Per-load leaderboards</h2>
  <div class="sub">CoT-induced blindness <b>CIB = Direct − CoT</b> passphrase detection (report_contains, gated on not-truncated), k=5, sorted by CIB. <span class="flag" style="color:var(--good)">clean</span> = strong CIB & CoT helps the load task · <span class="flag" style="color:var(--warn)">CoT⊥load</span> = control-1 fail (CoT does <i>not</i> raise load accuracy, so blindness may be general CoT degradation) · <span style="color:var(--muted)">ceiling</span> = Direct≈CoT≈1.0.</div>
  <div class="tabs">{tabs}</div>
  <div class="card tablewrap">{panels}</div>
  <div class="legend"><b>Cleanest single case:</b> <code>qwen2.5:3b</code> — CoT <b>raises</b> load-task accuracy (+0.05–0.11) yet <b>drops</b> passphrase detection by 0.19–0.32. Capability up, awareness of the un-reasoned item down. That is the effect in its purest form.</div>

  <h2>CoT-induced blindness vs Direct capacity (semantic_2)</h2>
  <figure class="card">
    <img src="{fig_cap}" alt="CIB vs capacity">
    <figcaption>Blindness is not a simple function of raw ability: models with similar Direct capacity show very different CIB. The strongest blinders (gemma2:2b, mistral:7b, qwen2.5:3b) sit at mid-high Direct capacity; ceiling models cluster at CIB≈0, top-right.</figcaption>
  </figure>

  <h2>Mechanism (individual trials, k=5 CoT misses)</h2>
  <div class="card">
    <p>Reading the raw <code>&lt;Thinking&gt;</code> of missed-passphrase trials: the model works through the <b>load (reasoning) tasks</b> one by one and simply <b>never processes the passphrase task</b> — it reports the 4 tasks it reasoned about and drops the passphrase. It is <b>not</b> filler-enumeration (<code>enumerated_fillers=False</code>) — the anti-enumeration instruction holds. The passphrase needs no reasoning (just copy 3 words), so it falls outside the chain of thought and gets crowded out. Direct read-out catches it far more often.</p>
    <div class="quote">qwen2.5:3b · passphrase task GONDOLA (last) · reported 4/5, passphrase absent:
&lt;Thinking&gt; Packet 04 Task ICEBERG: … VALID.  Packet 05 Task OBELISK: … VALID.
 Packet 09 Task METRONOME: … VALID.  Packet 10 Task PYRAMID: … &lt;/Thinking&gt;
&lt;Final_Answers&gt; ICEBERG / OBELISK / METRONOME / PYRAMID  ← GONDOLA never appears</div>
  </div>

  <h2>Per-load slopegraphs</h2>
  <figure class="card">
    <img src="{fig_load}" alt="per-load slopegraphs">
    <figcaption>Direct→CoT per model, one panel per load. Trivial shows the least blindness; the semantic loads show consistent down-slopes for the interpretable (non-ceiling) models.</figcaption>
  </figure>

  <h2>Honest caveats</h2>
  <div class="card">
    <ul>
      <li><b>The effect is much weaker than the budget-capped pilots.</b> At budget=None the pilots' mechanism (serial scan runs out of tokens before late items) is gone; what remains is a smaller "un-reasoned item gets crowded out" effect.</li>
      <li><b>Scale vs ceiling are confounded.</b> ≥9B models solve everything under unlimited CoT (Direct≈CoT≈1.0), so "big models don't blink" is inseparable from "the load is too easy for them" here. Harder loads (semantic_5/math) are needed to test scale cleanly at the top.</li>
      <li><b>Control-1 flags.</b> gemma2:2b & mistral:7b show the largest CIB but CoT does <i>not</i> raise their load accuracy → their blindness may be partly general CoT degradation, not selective. Trust the <span style="color:var(--good)">clean</span> rows first.</li>
      <li><b>Graded measure is report-only</b> (Ollama can't teacher-force); re-score on HF for t2 log-probs.</li>
      <li><b>qwen2.5:0.5b</b> is floor-compressed (Direct capacity low) — its large CIB is unreliable.</li>
    </ul>
  </div>
  <div class="sub" style="margin-top:24px">Design faithful to the simplified spec (k=5, budget=None, passphrase_last, per-load, n_seeds=100). Term "CoT-induced blindness" per request. Pipeline: <code>overnight_cib.py</code> → <code>analyze_cib.py</code> → <code>make_cib_figures.py</code>.</div>
</div>
<script>
  document.querySelectorAll('.tab').forEach((t,i)=>{{
    if(i===0)t.classList.add('active');
    t.onclick=()=>{{
      document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
      t.classList.add('active');
      const l=t.dataset.load;
      document.querySelectorAll('.panel').forEach(p=>{{
        p.style.display = p.dataset.load===l ? 'block':'none';
      }});
    }};
  }});
</script>"""
    out = os.path.join(_HERE, "cib_dashboard.html")
    open(out, "w").write(HTML)
    print("wrote", out, len(HTML), "bytes")


if __name__ == "__main__":
    main()
