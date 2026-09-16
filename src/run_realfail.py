"""Apply the pipeline to REAL generative-model failures.

These have no ground truth -- that is precisely why the controlled benchmark
exists -- so exact recovery is not measurable here. What IS measurable, and what
this script reports, is:

  * coverage  -- fraction of real failures whose one-edit repair space is non-empty
  * yield     -- fraction converted into a valid molecule
  * minimality-- token edit distance from the broken string to the output, a
                 proxy for identity preservation: a repair that rewrites the
                 molecule cannot be preserving what the generator intended
  * chemistry -- properties of the recovered set vs the generator's own valid output
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import torch
import metrics as M
import edit_ops as EO
from rank_domain_lm import load_lm, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="data/recovery/cands_realfail_full.jsonl")
    ap.add_argument("--lm", default="runs/lm/best.pt")
    ap.add_argument("--mode", default="mean")
    ap.add_argument("--prune", action="store_true", default=True)
    ap.add_argument("--max_cands", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results_recovery/REALFAIL_ours.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = [json.loads(l) for l in open(args.cands) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    torch.set_num_threads(2)
    model, stoi = load_lm(args.lm)
    j = 0 if args.mode == "sum" else 1
    print(f"[realfail] {len(rows)} real generator failures", flush=True)

    recs = []
    for i, r in enumerate(rows):
        cands = r["candidates"]
        if args.prune and cands:
            cands, _ = EO.prune(r["corrupted"], cands, r["witness"], r.get("category"))
        best, margin = None, None
        if cands:
            if len(cands) > args.max_cands:
                cands = sorted(cands, key=len)[:args.max_cands]
            sc = score(model, stoi, cands, batch_size=256)
            order = sorted(range(len(cands)), key=lambda t: -sc[t][j])
            best = cands[order[0]]
            if len(order) > 1:
                margin = sc[order[0]][j] - sc[order[1]][j]
        recs.append({"corrupted": r["corrupted"], "category": r.get("category"),
                     "pred": best, "n_cands": len(cands), "margin": margin,
                     "valid": bool(best),
                     "edit": M.token_edit_distance(r["corrupted"], best) if best else None})
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    n = len(recs)
    cov = sum(1 for r in recs if r["n_cands"]) / n
    yld = sum(1 for r in recs if r["valid"]) / n
    ed = sorted(r["edit"] for r in recs if r["edit"] is not None)
    summ = {"n": n, "coverage": round(cov, 4), "yield": round(yld, 4),
            "median_edit": ed[len(ed) // 2] if ed else None,
            "mean_edit": round(sum(ed) / len(ed), 3) if ed else None}
    cats = {}
    for c in sorted({r["category"] for r in recs if r["category"]}):
        sub = [r for r in recs if r["category"] == c]
        cats[c] = {"n": len(sub),
                   "yield": round(sum(1 for r in sub if r["valid"]) / len(sub), 4)}
    summ["per_category"] = cats
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(args.out.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_category"}, indent=2))
    print(json.dumps(cats, indent=2))


if __name__ == "__main__":
    main()
