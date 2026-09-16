"""Score raw predictions (from a remote decoder) with local RDKit."""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import recovery_metrics as RM

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--out_dir", default="results_recovery")
ap.add_argument("--tag", required=True)
ap.add_argument("--mcs_n", type=int, default=400)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

rows = [json.loads(l) for l in open(a.preds) if l.strip()]
recs = []
for i, r in enumerate(rows):
    sc = RM.score(r["ground_truth"], r["pred"], with_mcs=(i < a.mcs_n))
    recs.append({**{k: r.get(k) for k in ("ground_truth", "corrupted", "category", "pred")}, **sc})
agg = RM.aggregate(recs)
agg["method"] = a.tag
cats = {}
for c in sorted({r["category"] for r in recs if r["category"]}):
    sub = [r for r in recs if r["category"] == c]
    x = RM.aggregate(sub)
    cats[c] = {"n": x["n"], "validity": x["validity"],
               "exact_recovery": x["exact_recovery"],
               "tanimoto_given_valid": x["tanimoto_given_valid"]}
agg["per_category"] = cats
with open(os.path.join(a.out_dir, f"{a.tag}.jsonl"), "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
with open(os.path.join(a.out_dir, f"{a.tag}.summary.json"), "w") as f:
    json.dump(agg, f, indent=2)
print(json.dumps({k: v for k, v in agg.items() if k != "per_category"}, indent=2))
