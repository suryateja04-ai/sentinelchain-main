"""Patch a built portal bundle from ten agents to three.

usage: python tools/three_agents_patch.py <site folder>

Works on minified builds by finding each piece by pattern rather than by minified name.
Every required replacement must match exactly, otherwise nothing is written.
"""
import hashlib
import json
import os
import re
import sys

site = sys.argv[1]
assets = os.path.join(site, "assets")
js = [f for f in os.listdir(assets) if f.startswith("index-") and f.endswith(".js")]
css = [f for f in os.listdir(assets) if f.startswith("index-") and f.endswith(".css")]
assert len(js) == 1 and len(css) == 1, (js, css)
s = open(os.path.join(assets, js[0]), encoding="utf-8").read()

GROUPS = [
    ("data-agent", "Data Gathering & Preprocessing", ["data-loader", "validation", "preprocessing", "temporal"]),
    ("prediction", "Sepsis Prediction", ["prediction", "alert"]),
    ("explain-agent", "Explainability & Clinical Guidance", ["explanation", "safety", "report"]),
]
DEFS = ('[{id:"data-agent",name:"Data Gathering & Preprocessing",code:"AGENT 1 · DATA",lane:"ingest",'
        'summary:"Gathers the latest 16-hour window, validates and cleans it, applies the frozen imputer and scaler, '
        'and measures recent trends.",input:"Patient ID, window end hour",output:"Validated, standardised 16 × 39 '
        'window with quality report and trends",baseLatency:24,cpu:12,memory:400},'
        '{id:"prediction",name:"Sepsis Prediction",code:"AGENT 2 · PREDICTION",lane:"analyse",summary:"Runs eight '
        'trained models, isotonic calibration and the weighted ensemble, decides the alert at threshold 0.047 and '
        'reports precision and sensitivity at that threshold.",input:"Standardised window",output:"Calibrated '
        'probability, per-model scores, alert decision",baseLatency:120,cpu:42,memory:1508},'
        '{id:"explain-agent",name:"Explainability & Clinical Guidance",code:"AGENT 3 · GUIDANCE",lane:"govern",'
        'summary:"Occlusion attribution, observational range flags, decision-support wording (never prescribes) and '
        'the clinician report.",input:"Window, prediction, latest values",output:"Signal contributions, flags, '
        'guidance and report",baseLatency:74,cpu:27,memory:402}]')


def merge(stages):
    out = []
    for gid, name, ids in GROUPS:
        part = [next(x for x in stages if x["id"] == i) for i in ids]
        bad = [x for x in part if x["status"] != "completed"]
        out.append({"id": gid, "name": name, "status": bad[0]["status"] if bad else "completed",
                    "ms": round(sum(float(x["ms"]) for x in part), 2), "input": part[0]["input"],
                    "output": " · ".join(x["output"] for x in part),
                    "errors": [e for x in part for e in x.get("errors", [])]})
    return out


def sub(old, new, count=1):
    """Plain replacement; count=None means optional (0 or more)."""
    global s
    n = s.count(old)
    if count is not None and n != count:
        raise SystemExit(f"expected {count} match(es), found {n}: {old[:80]!r}")
    s = s.replace(old, new)


def rsub(pattern, repl, count=1):
    global s
    s, n = re.subn(pattern, repl, s)
    if n != count:
        raise SystemExit(f"expected {count} match(es), found {n}: {pattern}")


# 1. the ten agent definitions -> three (array name differs per build)
m = re.search(r'(\w+)=\[\{id:"data-loader"', s)
assert m, "agent definitions not found"
a = m.start() + len(m.group(1)) + 1
b = s.index("}];", a) + 2
s = s[:a] + DEFS + s[b:]
AGENTS = m.group(1)

# 2. recorded runs: merge each run's ten stages into three
m = re.search(r"\w+=JSON\.parse\('(?=\[\{\"trace_id\")", s)
assert m, "recorded runs not found"
a = m.end()
b = s.index("')", a)
runs = json.loads(s[a:b])
for r in runs:
    assert len(r["stages"]) == 10, r["trace_id"]
    r["stages"] = merge(r["stages"])
body = json.dumps(runs, ensure_ascii=False, separators=(",", ":"))
assert "'" not in body and "\\" not in body
s = s[:a] + body + s[b:]

# 3. live service answers still carry ten stages: merge them on arrival
assert "__mergeAgents3" not in s
merge_js = ('const __mergeAgents3=r=>{if(!r||!Array.isArray(r.stages)||r.stages.length<=3)return r;const g='
            + json.dumps(GROUPS) +
            ';return{...r,stages:g.map(([id,name,ids])=>{const p=ids.map(x=>r.stages.find(t=>t.id===x)).filter(Boolean),'
            'b=p.find(t=>t.status!=="completed");return{id,name,status:b?b.status:"completed",'
            'ms:+p.reduce((a,t)=>a+(+t.ms||0),0).toFixed(2),input:p[0]?p[0].input:"",'
            'output:p.map(t=>t.output).join(" · "),errors:p.flatMap(t=>t.errors||[])}})}};')
rsub(r'async function (\w+)\(e,\{live:t\}\)\{', lambda m: merge_js + m.group(0))
rsub(r'return t\?(\{\.\.\.await \w+\("/api/ml/analyze",\{method:"POST",body:\{sample_id:e\}\}\),'
     r'label:n\.label,category:n\.category,source:"live"\}):',
     lambda m: "return t?__mergeAgents3(" + m.group(1) + "):")

# 4. run animation: agent 1, then prediction, then agent 3 (was 4 + 1 + 5 by position)
rsub(r'for\(let (\w+)=0;\1<4;\1\+=1\)\{if\((\w+)\.current\)return;',
     lambda m: m.group(0).replace("<4;", "<1;"))
rsub(r'prediction:(\w+)\.stages\[4\]\.status',
     lambda m: f'prediction:({m.group(1)}.stages.find(t=>t.id==="prediction")||{{status:"completed"}}).status')
rsub(r'for\(let (\w+)=5;\1<(\w+)\.length;\1\+=1\)', lambda m: m.group(0).replace("=5;", "=2;"))
rsub(r'(\w+)===5\?620:360', lambda m: f"{m.group(1)}===2?620:360")
rsub(r'\[(\w+)\[(\w+)\]\.id\]:(\w+)\.stages\[\2\]\.status',
     lambda m: f'[{m.group(1)}[{m.group(2)}].id]:({m.group(3)}.stages.find(t=>t.id==={m.group(1)}[{m.group(2)}].id)'
               f'||{{status:"completed"}}).status')

# 5. ward event log keyed by agent
a = s.index('i={"data-loader":["Loaded observation window"')
b = s.index("]};", a) + 3
s = s[:a] + ('i={"data-agent":["Prepared observation window","Patient ICU-2047, 16 h","Standardised 16×39 tensor · 1 '
             'validation warning (temporal gap) · RR +6/min over 4 h"],prediction:["Generated risk score and alert",'
             '"Standardised tensor","Risk 72% · routed to clinician queue"],"explain-agent":["Explained risk and added '
             'guidance","Prediction + window","RR, Lactate, HR ↑ · Clinical review recommended · Report RPT-2047"]};') + s[b:]
rsub(r'id:(\w)===4\?"SC-2047-9F2A":void 0', lambda m: f'id:{m.group(1)}===1?"SC-2047-9F2A":void 0')
rsub(r'status:(\w)===1\?"warning":"success",at:t-240\*1e3\+\1\*18',
     lambda m: m.group(0).replace("===1?", "===0?"))

# 6. event labels that used the old agent names
sub('agent:"Validation"', 'agent:"Data Gathering & Preprocessing"', 2)
sub('agent:"Prediction"', 'agent:"Sepsis Prediction"', 3)
sub('agent:"Alert"', 'agent:"Sepsis Prediction"', 3)
sub('agent:"Audit"', 'agent:"Orchestrator"', 1)
sub('agent:"Report"', 'agent:"Explainability & Clinical Guidance"', 2)

# 7. texts (some exist only in one build)
sub("A ten-agent pipeline turns", "A three-agent pipeline turns")
sub("Runtime health and resource use of the ten orchestrated agents.",
    "Runtime health and resource use of the three orchestrated agents.")
sub("Ten agents turn a 16-hour observation window", "Three agents turn a 16-hour observation window")
sub("then ten agents validate, explain, safety-check, report and audit it.",
    "then three agents prepare the data, predict the risk, and explain it with clinical guidance.", None)
sub('children:"Ten agents"', 'children:"Three agents"', None)
sub("10/10 agents completed", "3/3 agents completed", 2)
sub('title:"Explanation agent"', 'title:"Agent 3 · Explanation"')
sub('title:"Clinical safety agent"', 'title:"Agent 3 · Clinical guidance"')
sub('title:"Temporal analysis agent"', 'title:"Agent 1 · Temporal trends"')
sub('"Validation agent findings"', '"Agent 1 · Validation findings"')
sub('"Data validation & temporal analysis"', '"Agent 1 · Data gathering & preprocessing"')
sub('"Eight-model calibrated ensemble"', '"Agent 2 · Sepsis prediction (eight-model calibrated ensemble)"')
sub('"Explanation, safety review & audit trail"', '"Agent 3 · Explainability & clinical guidance"')

left = [w for w in ("Ten agents", "ten agents", "10/10", "ten-agent", "ten orchestrated", 'id:"data-loader"',
                    "stages[4]") if w in s]
assert not left, left

# 8. one row of three agents, lane labels in thirds, no U-turn to a second row
c = open(os.path.join(assets, css[0]), encoding="utf-8").read()
c += (".orch-row{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}"
      ".orch-lanes{grid-template-columns:1fr 1fr 1fr}"
      ".orch-turn,.orch-row--reverse{display:none}")

# write under new content hashes so browsers do not keep the old files
names = {}
for old, text, ext in ((js[0], s, "js"), (css[0], c, "css")):
    new = f"index-{hashlib.sha256(text.encode()).hexdigest()[:8]}.{ext}"
    with open(os.path.join(assets, new), "w", encoding="utf-8") as f:
        f.write(text)
    os.remove(os.path.join(assets, old))
    names[old] = new
for page in ("index.html", "404.html"):
    p = os.path.join(site, page)
    t = open(p, encoding="utf-8").read()
    for old, new in names.items():
        assert old in t, (page, old)
        t = t.replace(old, new)
    open(p, "w", encoding="utf-8").write(t)
print("agents array:", AGENTS, "|", names, "| runs:",
      [(r["trace_id"], [x["ms"] for x in r["stages"]], r["total_ms"]) for r in runs])
