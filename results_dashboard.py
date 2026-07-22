"""Flask dashboard for the A×B×H pilot results.

Run from LLM_Blink/ on Windows:
    python results_dashboard.py            # http://127.0.0.1:5000
    python results_dashboard.py --port 8080

Discovers every ab_results_*_hab_*.csv in the script's folder, passes each
through backfill_readout_columns (mandatory re-parse), and serves interactive
charts: per-rank detection curves by budget (the serial-scan cutoff), passphrase
detection by budget x load, report coverage, rank-arm contrast, quality flags.
Report analyses gate on output_truncated only (2026-07-21 rule); slot_missing
is shown as a quality metric, not used as a gate.
"""
import argparse, glob, json, os, re, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from LLM_Blink import backfill_readout_columns

HERE = Path(__file__).resolve().parent
BOOL_COLS = ["output_truncated", "t2_slot_missing", "cot_forced_closed",
             "report_correct", "report_contains", "phrase_anywhere"]


def _to_bool(v):
    return str(v).strip().lower() in ("true", "1", "1.0")


def load_all():
    frames = []
    for f in sorted(glob.glob(str(HERE / "ab_results_*_hab_*.csv"))):
        name = os.path.basename(f)
        m = re.match(r"ab_results_(.+?)_hab_(cot|direct)(_passphraselast)?\.csv", name)
        if not m:
            continue
        df = backfill_readout_columns(pd.read_csv(f))
        df["model"] = m.group(1)
        # arm from the passphrase_last COLUMN when present (2026-07-22: one
        # CSV may mix both arms); filename suffix as fallback for old files.
        if "passphrase_last" in df.columns and df["passphrase_last"].notna().any():
            df["arm"] = df["passphrase_last"].map(
                lambda v: "last" if _to_bool(v) else "random")
        else:
            df["arm"] = "last" if m.group(3) else "random"
        df["file"] = name
        frames.append(df)
    if not frames:
        raise SystemExit("No ab_results_*_hab_*.csv found next to this script.")
    big = pd.concat(frames, ignore_index=True)
    for c in BOOL_COLS:
        if c in big.columns:
            big[c] = big[c].map(_to_bool)
    big["budget"] = big["finite_budget"].fillna(-1).astype(int)  # -1 = direct
    return big


def _blabel(b):
    return "direct" if b < 0 else f"b{b}"


def compute_payload(big):
    dfr = big[~big.output_truncated]
    payload = {"quality": [], "panels": []}

    for fn, g in big.groupby("file"):
        row = {"file": fn, "n": len(g),
               "trunc": round(g.output_truncated.mean(), 2),
               "slot_missing": round(g.t2_slot_missing.mean(), 2)}
        if (g.regime == "cot").any():
            row["forced_close"] = round(g[g.regime == "cot"].cot_forced_closed.mean(), 2)
        payload["quality"].append(row)

    for (model, arm), g in dfr.groupby(["model", "arm"]):
        panel = {"model": model, "arm": arm}
        kmax = int(g.n_tasks.max())
        gk = g[g.n_tasks == kmax]
        ranks = {}
        for b, gb in gk.groupby("budget"):
            det = [[] for _ in range(kmax)]
            for _, r in gb.iterrows():
                try:
                    tasks = json.loads(r.tasks)
                except Exception:
                    continue
                for i, t in enumerate(tasks[:kmax]):
                    det[i].append(bool(t.get("reported", t.get("detected", False))))
            ranks[_blabel(b)] = [round(sum(d) / len(d), 2) if d else None for d in det]
        panel["rank_curves"] = {"kmax": kmax, "series": ranks}

        gk1 = g[g.n_tasks > 1]
        det = gk1.groupby(["budget", "t1_load"]).report_contains.mean().round(2)
        panel["detection"] = {_blabel(b): {ld: v for (bb, ld), v in det.items() if bb == b}
                              for b in sorted(gk1.budget.unique())}

        cov = (gk1.n_tasks_reported / gk1.n_tasks).groupby(gk1.budget).mean().round(2)
        panel["coverage"] = {_blabel(b): v for b, v in cov.items()}
        payload["panels"].append(panel)

    arm_rows = []
    for model, g in dfr[dfr.n_tasks > 1].groupby("model"):
        if g.arm.nunique() < 2:
            continue
        t = g.groupby(["budget", "arm"]).report_contains.mean().round(2)
        arm_rows.append({"model": model,
                         "budgets": [_blabel(b) for b in sorted(g.budget.unique())],
                         "random": [t.get((b, "random"), None) for b in sorted(g.budget.unique())],
                         "last": [t.get((b, "last"), None) for b in sorted(g.budget.unique())]})
    payload["arm_contrast"] = arm_rows
    return payload


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>LLM Blink — A&times;B&times;H results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:2.5rem} h3{font-size:.95rem;color:#555}
table{border-collapse:collapse;font-size:.85rem} td,th{border:1px solid #ddd;padding:4px 10px;text-align:right}
th{background:#f5f5f4} td:first-child,th:first-child{text-align:left}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px} .chartbox{position:relative;height:260px}
.note{color:#777;font-size:.8rem}
</style></head><body>
<h1>LLM attentional blink — A&times;B&times;H pilot dashboard</h1>
<p class="note">Report gate = output_truncated only. slot_missing = "not spontaneously reported"
(H detection outcome), shown as a metric, never a gate. All CSVs re-parsed via
backfill_readout_columns at page load.</p>
<h2>Quality flags</h2><div id="quality"></div>
<h2>Per-rank detection (largest k, cot budgets vs direct)</h2>
<p class="note">The serial-scan cutoff: primacy under capped CoT, flat under direct.</p>
<div class="grid" id="ranks"></div>
<h2>Passphrase detection by budget &times; load (k&gt;1)</h2><div class="grid" id="det"></div>
<h2>Report coverage (tasks reported / k, k&gt;1)</h2><div class="grid" id="cov"></div>
<h2>Passphrase-position arm: random vs last (k&gt;1, loads pooled)</h2><div class="grid" id="arm"></div>
<script>
const BLUES=["#b5d4f4","#378add","#0c447c","#042c53"], GRAY="#898781";
const PAIR=["#2a78d6","#eb6834","#1baf7a"];
fetch("/api/data").then(r=>r.json()).then(D=>{
  const q=document.getElementById("quality");
  let h="<table><tr><th>file</th><th>n</th><th>trunc</th><th>slot_missing</th><th>forced_close</th></tr>";
  D.quality.forEach(r=>{h+=`<tr><td>${r.file}</td><td>${r.n}</td><td>${r.trunc}</td><td>${r.slot_missing}</td><td>${r.forced_close??"—"}</td></tr>`});
  q.innerHTML=h+"</table>";
  function box(parent,title){const d=document.createElement("div");
    d.innerHTML=`<h3>${title}</h3><div class="chartbox"><canvas></canvas></div>`;
    document.getElementById(parent).appendChild(d);return d.querySelector("canvas");}
  const yopt={min:0,max:1};
  D.panels.forEach(p=>{
    const rc=p.rank_curves, labels=[...Array(rc.kmax).keys()].map(i=>"rank "+(i+1));
    const keys=Object.keys(rc.series).sort((a,b)=>(a==="direct")-(b==="direct")||parseInt(a.slice(1)||0)-parseInt(b.slice(1)||0));
    let bi=0;
    const ds=keys.map(k=>({label:k,data:rc.series[k],borderWidth:2,tension:.15,
      borderColor:k==="direct"?GRAY:BLUES[bi],backgroundColor:k==="direct"?GRAY:BLUES[bi++],
      borderDash:k==="direct"?[6,4]:[]}));
    new Chart(box("ranks",`${p.model} — ${p.arm} arm (k=${rc.kmax})`),
      {type:"line",data:{labels,datasets:ds},options:{responsive:true,maintainAspectRatio:false,scales:{y:yopt}}});
    const dl=Object.keys(p.detection), loads=[...new Set(dl.flatMap(b=>Object.keys(p.detection[b])))].sort();
    const dds=loads.map((ld,i)=>({label:ld,data:dl.map(b=>p.detection[b][ld]??null),backgroundColor:PAIR[i%3],borderRadius:4}));
    new Chart(box("det",`${p.model} — ${p.arm} arm`),
      {type:"bar",data:{labels:dl,datasets:dds},options:{responsive:true,maintainAspectRatio:false,scales:{y:yopt}}});
    const cl=Object.keys(p.coverage);
    new Chart(box("cov",`${p.model} — ${p.arm} arm`),
      {type:"bar",data:{labels:cl,datasets:[{label:"coverage",data:cl.map(b=>p.coverage[b]),backgroundColor:"#2a78d6",borderRadius:4}]},
       options:{responsive:true,maintainAspectRatio:false,scales:{y:yopt},plugins:{legend:{display:false}}}});
  });
  D.arm_contrast.forEach(a=>{
    new Chart(box("arm",a.model),{type:"bar",data:{labels:a.budgets,datasets:[
      {label:"random",data:a.random,backgroundColor:PAIR[0],borderRadius:4},
      {label:"last",data:a.last,backgroundColor:PAIR[1],borderRadius:4}]},
      options:{responsive:true,maintainAspectRatio:false,scales:{y:yopt}}});
  });
});
</script></body></html>"""


def create_app():
    from flask import Flask, jsonify
    app = Flask(__name__)
    payload = compute_payload(load_all())

    @app.route("/")
    def index():
        return PAGE

    @app.route("/api/data")
    def data():
        return jsonify(payload)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    create_app().run(port=args.port, debug=False)
