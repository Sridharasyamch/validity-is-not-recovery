"""Rank enumerated repairs with the SMILES Transformer LM (Ranker B).

Same CLI and same scoring contract as `rank_domain_lm.py`, so the two rankers
are interchangeable in the evaluation harness and differ only in architecture
(the character vocabulary is shared verbatim).
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import recovery_metrics as RM
import edit_ops as EO
from transformer_lm import build, score


def load_tlm(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck["cfg"], device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["vocab"]["stoi"], ck["cfg"]["max_len"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--lm", default="runs/tlm/best.pt")
    ap.add_argument("--mode", default="mean", choices=["sum", "mean"])
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--max_cands", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mcs_n", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", default="results_recovery")
    ap.add_argument("--tag", default="enumrank_tlm")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.cands) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    torch.set_num_threads(2)
    model, stoi, max_len = load_tlm(args.lm)
    print(f"[tlm-rank] {len(rows)} items", flush=True)

    j = 0 if args.mode == "sum" else 1
    recs = []
    for i, r in enumerate(rows):
        cands = r["candidates"]
        if args.prune and cands:
            cands, _ = EO.prune(r["corrupted"], cands, r["witness"], r.get("category"))
        best, margin = None, None
        if cands:
            if len(cands) > args.max_cands:
                cands = sorted(cands, key=len)[:args.max_cands]
            sc = score(model, stoi, cands, batch_size=args.batch_size,
                       max_len=max_len)
            order = sorted(range(len(cands)), key=lambda t: -sc[t][j])
            best = cands[order[0]]
            if len(order) > 1:
                margin = sc[order[0]][j] - sc[order[1]][j]
        s = RM.score(r["ground_truth"], best, with_mcs=(i < args.mcs_n))
        recs.append({"ground_truth": r["ground_truth"], "corrupted": r["corrupted"],
                     "category": r.get("category"), "pred": best,
                     "n_cands": len(cands), "margin": margin,
                     "truth_in_candidates": r.get("truth_in_candidates"), **s})
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    agg = RM.aggregate(recs)
    agg.update({"ranker": "smiles_transformer_lm", "mode": args.mode,
                "prune": args.prune})
    cats = {}
    for c in sorted({r["category"] for r in recs if r["category"]}):
        sub = [r for r in recs if r["category"] == c]
        a = RM.aggregate(sub)
        cats[c] = {"n": a["n"], "validity": a["validity"],
                   "exact_recovery": a["exact_recovery"],
                   "tanimoto_given_valid": a["tanimoto_given_valid"]}
    agg["per_category"] = cats
    with open(os.path.join(args.out_dir, f"{args.tag}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, f"{args.tag}.summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps({k: v for k, v in agg.items() if k != "per_category"}, indent=2))


if __name__ == "__main__":
    main()
