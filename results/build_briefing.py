"""Build the overnight briefing artifact (for Thomas). Narrative + embedded figures.
Usage: python build_briefing.py <out_html> "<updated stamp>"
"""
import base64
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
UPDATED = sys.argv[2] if len(sys.argv) > 2 else ""


def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


fig_mech = b64(os.path.join(HERE, "fig_4b_mechanism.png"))
fig_june = b64(os.path.join(HERE, "first_test_recovered",
                            "2026-06-05_gemma3-4b_AB-curves_1000trials.png"))

HTML = f"""<title>Attentional Blink — overnight briefing</title>
<style>
:root {{
  --bg:#f6f7f9; --panel:#fff; --ink:#191c22; --muted:#5c6270; --line:#e4e6ec;
  --accent:#3f6cd8; --accent-soft:#e9eefb; --good:#0f8a5f; --good-bg:#e2f4ec;
  --warn:#b7791f; --warn-bg:#fdf3e0; --bad:#c23a3a; --bad-bg:#fbe6e6;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
}}
@media (prefers-color-scheme:dark) {{ :root {{
  --bg:#0f1014; --panel:#181a20; --ink:#e9eaf0; --muted:#9aa0b0; --line:#2a2d37;
  --accent:#7fa2f0; --accent-soft:#1e2536; --good:#4ecb95; --good-bg:#123527;
  --warn:#e0b45f; --warn-bg:#332a15; --bad:#e77; --bad-bg:#3a1e1e; }} }}
:root[data-theme="dark"] {{ --bg:#0f1014; --panel:#181a20; --ink:#e9eaf0; --muted:#9aa0b0;
  --line:#2a2d37; --accent:#7fa2f0; --accent-soft:#1e2536; --good:#4ecb95; --good-bg:#123527;
  --warn:#e0b45f; --warn-bg:#332a15; --bad:#e77; --bad-bg:#3a1e1e; }}
:root[data-theme="light"] {{ --bg:#f6f7f9; --panel:#fff; --ink:#191c22; --muted:#5c6270;
  --line:#e4e6ec; --accent:#3f6cd8; --accent-soft:#e9eefb; --good:#0f8a5f; --good-bg:#e2f4ec;
  --warn:#b7791f; --warn-bg:#fdf3e0; --bad:#c23a3a; --bad-bg:#fbe6e6; }}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6}}
.wrap{{max-width:820px;margin:0 auto;padding:44px 22px 90px}}
.banner{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-family:var(--mono);
  font-size:12px;background:var(--accent-soft);color:var(--accent);padding:9px 14px;
  border-radius:8px;margin-bottom:26px}}
.dot{{width:8px;height:8px;border-radius:50%;background:var(--good);display:inline-block;
  box-shadow:0 0 0 0 var(--good);animation:pulse 2s infinite}}
@keyframes pulse{{0%{{box-shadow:0 0 0 0 rgba(78,203,149,.5)}}70%{{box-shadow:0 0 0 7px rgba(78,203,149,0)}}100%{{box-shadow:0 0 0 0 rgba(78,203,149,0)}}}}
@media (prefers-reduced-motion:reduce){{.dot{{animation:none}}}}
h1{{font-size:2rem;line-height:1.12;letter-spacing:-.02em;margin:0 0 8px;text-wrap:balance}}
.lede{{color:var(--muted);font-size:1.05rem;margin:0}}
.for{{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:10px}}
section{{margin-top:38px}}
h2{{font-size:1.3rem;letter-spacing:-.01em;margin:0 0 6px;text-wrap:balance}}
h2 .n{{color:var(--accent);font-family:var(--mono);font-size:.8em;margin-right:8px}}
p{{margin:10px 0}}
.tldr{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--good);
  border-radius:10px;padding:20px 22px}}
.tldr strong{{color:var(--good)}}
.callout{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:10px;padding:16px 20px;margin:16px 0}}
figure{{margin:18px 0 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}}
figure img{{width:100%;height:auto;border-radius:6px;display:block;background:#fff}}
figcaption{{color:var(--muted);font-size:12.5px;margin-top:8px;font-family:var(--mono)}}
.scroll{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:12px}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap}}
thead th{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  background:var(--accent-soft)}}
.mono,td.num{{font-family:var(--mono);font-variant-numeric:tabular-nums}}
.pill{{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:999px}}
.p-good{{background:var(--good-bg);color:var(--good)}}
.p-warn{{background:var(--warn-bg);color:var(--warn)}}
.p-bad{{background:var(--bad-bg);color:var(--bad)}}
ul{{padding-left:20px}} li{{margin:5px 0}}
code{{font-family:var(--mono);font-size:.88em;background:var(--accent-soft);padding:1px 5px;border-radius:4px}}
.foot{{margin-top:46px;border-top:1px solid var(--line);padding-top:16px;color:var(--muted);
  font-size:12px;font-family:var(--mono)}}
.steps{{list-style:none;padding:0;counter-reset:s}}
.steps li{{position:relative;padding:4px 0 14px 34px;border-left:2px solid var(--line);margin-left:8px}}
.steps li::before{{counter-increment:s;content:counter(s);position:absolute;left:-13px;top:2px;
  width:24px;height:24px;border-radius:50%;background:var(--accent);color:#fff;font-family:var(--mono);
  font-size:12px;display:grid;place-items:center}}
.steps li:last-child{{border-left-color:transparent}}
</style>

<div class="wrap">
<div class="banner"><span class="dot"></span> AUTONOMOUS OVERNIGHT RUN · updating live · {UPDATED}</div>

<h1>The June "blink" looks like a truncation artifact</h1>
<p class="lede">An overnight re-investigation of the attentional-blink experiment on
<code>gemma3:4b</code> — the model from your first test on June 5.</p>
<p class="for">For Thomas · Ulysse is asleep (back in ~8h) · Claude is running experiments
autonomously through the night and posting results here as they land.</p>

<section>
  <div class="tldr">
    <p style="margin-top:0"><strong>TL;DR.</strong> Your June 5 gemma3:4b run showed a
    beautiful CoT-specific, lag-dependent T2 deficit — the "blink". Tonight, with the
    current code, it's gone (T2 report sits at ceiling). Digging in: the June effect is
    best explained as a <strong>generation-length / truncation artifact</strong>, not a
    genuine attentional bottleneck. The measure that can't be fooled by truncation — the
    graded T2 log-prob at a large token budget — shows <strong>no blink</strong> (if
    anything, load slightly <em>raises</em> T2 log-prob). Per Ulysse's rule, I'm <em>not</em>
    scaling to other models until a real effect reproduces on 4b.</p>
  </div>
</section>

<section>
  <h2><span class="n">01</span>What we were chasing</h2>
  <p>Your June 5 figure (recovered from WhatsApp — the originals were lost from
  <code>/tmp</code>). In <strong>COT</strong>, T2 report is suppressed under load and
  recovers as T2 moves away from T1; in <strong>DIRECT</strong> everything is at ceiling.
  That dissociation is exactly what an attentional blink would look like.</p>
  <figure>
    <img alt="June 5 AB curves" src="data:image/png;base64,{fig_june}">
    <figcaption>gemma3:4b, June 5, 1000 trials. Top-left = "the AB curve" (COT): loads
    dip at short lag, recover. Top-right = DIRECT: flat at ceiling.</figcaption>
  </figure>
</section>

<section>
  <h2><span class="n">02</span>The mechanism — one figure</h2>
  <p>Same model, current code, sweeping the token budget and reading both measures:</p>
  <figure>
    <img alt="gemma3:4b mechanism figure" src="data:image/png;base64,{fig_mech}">
    <figcaption>Top row: binary report in COT. At 1024 tokens it's at ceiling (no
    deficit); at 256 tokens it collapses for <em>every</em> load because the CoT is
    truncated before the T2 slot (truncation ≈ 100%). Bottom row: the truncation-immune
    graded log-prob at 1024 — no lag dip, and load sits <em>above</em> none (inverted),
    in both COT and DIRECT.</figcaption>
  </figure>
  <div class="callout">
    <strong>Why truncation mimics a blink.</strong> In COT under load the model writes
    more reasoning (it works through T1) → longer output → at a short token budget it hits
    the wall before emitting T2 → "report failure". DIRECT writes no reasoning → never
    truncates → ceiling. Harder load = longer reasoning = more truncation. That reproduces
    the "CoT-only, load-graded" shape with <em>no attentional mechanism</em>. The old
    budget was 256 (README notes it cut ~45% of CoT trials); it's now 1024, and the
    deficit vanished.
  </div>
</section>

<section>
  <h2><span class="n">03</span>Hypotheses tested tonight (gemma3:4b)</h2>
  <div class="scroll"><table>
    <thead><tr><th>Hypothesis</th><th>Test</th><th>Result</th></tr></thead>
    <tbody>
      <tr><td>Worked example masks it</td><td><code>--no-example</code></td>
        <td><span class="pill p-bad">rejected</span> report still ceiling; T1 acc drops 0.85→0.29 but T2 unaffected</td></tr>
      <tr><td>June blink = truncation</td><td><code>max_new_tokens=256</code></td>
        <td><span class="pill p-good">supported</span> COT report collapses for all loads; truncation ≈100%</td></tr>
      <tr><td>Real effect in graded log-prob</td><td>1024, truncation-immune</td>
        <td><span class="pill p-bad">no blink</span> no dip; load raises T2 log-prob (inverted)</td></tr>
      <tr><td>Titrate T2 off ceiling</td><td><code>t2_words=6</code></td>
        <td><span class="pill p-bad">no blink</span> report off ceiling but none is as low as load; graded log-prob still inverted</td></tr>
      <tr><td>Load paid? (condition on T1 correct)</td><td>t1_correct==True subset</td>
        <td><span class="pill p-bad">no blink</span> same inverted picture on trials where the load was genuinely solved</td></tr>
    </tbody>
  </table></div>
  <p style="color:var(--muted);font-size:.95rem">All four hypotheses tested → the null is
  consistent across binary report, graded log-prob, titration, and load-paid conditioning.
  A powered <code>n=50</code> run (1800 trials) <b>confirms it</b> with tight error bars —
  the figure above uses that data.</p>
</section>

<section>
  <h2><span class="n">04</span>What happens next tonight</h2>
  <ul class="steps">
    <li><b>Done:</b> the powered <code>n=50</code> confirmation — the null holds with tight CIs (figure above).</li>
    <li><b>Optional next:</b> check out the June commit in an isolated worktree and reproduce the June curve directly, then show the graded measure is flat even there — belt-and-braces on the truncation story.</li>
    <li>The June effect did <b>not</b> reproduce as a genuine effect, so per Ulysse's rule I am <b>not</b> scaling to other Gemma sizes.</li>
    <li><b>Bottom line for the morning:</b> on gemma3:4b this is a <b>null</b> — a legitimate outcome the issue explicitly allows (transformers attend to the whole context in parallel, so a serial blink isn't guaranteed). Open question to discuss: is there a task design that would create a genuine serial bottleneck, or do we record the null?</li>
  </ul>
  <p style="color:var(--muted);font-size:.95rem">This page redeploys to the same link as
  each result lands. Full log: <code>results/NIGHT_LOG.md</code>; recovered June figures:
  <code>results/first_test_recovered/</code>.</p>
</section>

<p class="foot">gemma3:4b · greedy (T=0) · n=10 seeds/cell · M4 Max, 8-way parallel · {UPDATED}<br>
Autonomous overnight run for issue #9 (ThoHardy/LLM_Blink). Numbers are n=10 — directional; a clean run follows if an effect survives.</p>
</div>
"""
with open(OUT, "w") as f:
    f.write(HTML)
print(f"wrote {OUT} ({len(HTML)//1024} KB)")
