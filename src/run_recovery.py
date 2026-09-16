"""Run recovery methods on the ground-truth corruption benchmark.

Measures VALIDITY and IDENTITY (exact recovery / Tanimoto / scaffold / MCS)
side by side, which is the whole point: a method can be 100% valid and still
recover the wrong molecule.
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import symbolic as S
import recovery_metrics as RM

SMISELF_PATH = os.environ.get("SMISELF_PATH", "third_party/SmiSelf")


def m_identity(smi):
    return smi


def m_rule(smi):
    return S.rule_repair(smi)


def _get_smiself():
    if SMISELF_PATH not in sys.path:
        sys.path.insert(0, SMISELF_PATH)
    import smiself
    return smiself


def m_smiself(smi):
    """SmiSelf (EMNLP 2025): invalid SMILES -> SELFIES (grammar rules) -> valid SMILES."""
    sf = _get_smiself()
    try:
        return sf.decoder(sf.encoder(smi))
    except Exception:
        return None


def m_selfies_vanilla(smi):
    """Stock SELFIES round-trip (encoder requires valid input) -- shows why a
    tolerant encoder like SmiSelf's is needed at all."""
    import selfies as sf
    try:
        return sf.decoder(sf.encoder(smi))
    except Exception:
        return None


METHODS = {"identity": m_identity, "rule": m_rule,
           "smiself": m_smiself, "selfies_vanilla": m_selfies_vanilla}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--out_dir", default="results_recovery")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mcs_n", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.benchmark) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    fn = METHODS[args.method]

    recs = []
    for i, r in enumerate(rows):
        pred = fn(r["corrupted"])
        # MCS is expensive; compute on a fixed prefix subsample only
        sc = RM.score(r["ground_truth"], pred, with_mcs=(i < args.mcs_n))
        recs.append({"ground_truth": r["ground_truth"], "corrupted": r["corrupted"],
                     "category": r["observed_category"], "pred": pred, **sc})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    agg = RM.aggregate(recs)
    agg["method"] = args.method
    # per-category breakdown
    cats = {}
    for c in sorted(set(r["category"] for r in recs)):
        sub = [r for r in recs if r["category"] == c]
        a = RM.aggregate(sub)
        cats[c] = {"n": a["n"], "validity": a["validity"],
                   "exact_recovery": a["exact_recovery"],
                   "tanimoto_given_valid": a["tanimoto_given_valid"]}
    agg["per_category"] = cats

    with open(os.path.join(args.out_dir, f"{args.method}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, f"{args.method}.summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps({k: v for k, v in agg.items() if k != "per_category"}, indent=2))


if __name__ == "__main__":
    main()
