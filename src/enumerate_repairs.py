"""Symbolic enumeration of the one-token-edit repair space.

For a corrupted SMILES this returns every DISTINCT valid molecule reachable by a
single token edit (delete / substitute / insert). The set is:

  * sound     -- every member parses and sanitises under RDKit, so any choice
                 from it is valid by construction;
  * complete  -- for single-edit corruptions it provably contains the ground
                 truth (empirically verified: `ceiling_one_edit` in ambiguity.json).

Enumeration is the expensive half, so results are cached to disk and reused by
every method that ranks over the candidate set.
"""
from __future__ import annotations
import argparse, json, os, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import symbolic as S
from ambiguity import build_vocab, one_edit_neighbours

_VOCAB = None


def _init(vocab):
    global _VOCAB
    _VOCAB = vocab
    RDLogger.DisableLog("rdApp.*")


def _enumerate(smi):
    """Distinct canonical molecules one token edit from `smi`, with a witness."""
    best = {}
    for cand in one_edit_neighbours(smi, _VOCAB):
        m = Chem.MolFromSmiles(cand)
        if m is None:
            continue
        try:
            c = Chem.MolToSmiles(m)
        except Exception:
            continue
        # keep the shortest witness string for each distinct molecule
        if c not in best or len(cand) < len(best[c]):
            best[c] = cand
    return best


def _work(args):
    i, smi = args
    try:
        return i, _enumerate(smi)
    except Exception:
        return i, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/recovery/recovery_test.jsonl")
    ap.add_argument("--out", default="data/recovery/cands_test.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--field", default="corrupted")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = [json.loads(l) for l in open(args.benchmark) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    vocab = build_vocab(rows if "ground_truth" in rows[0] else rows)
    print(f"[enum] n={len(rows)} vocab={len(vocab)} workers={args.workers}", flush=True)

    out = [None] * len(rows)
    with Pool(args.workers, initializer=_init, initargs=(vocab,)) as p:
        for n, (i, best) in enumerate(
                p.imap_unordered(_work, [(i, r[args.field]) for i, r in enumerate(rows)],
                                 chunksize=8), 1):
            out[i] = best
            if n % 250 == 0:
                print(f"  {n}/{len(rows)}", flush=True)

    n_empty = sum(1 for b in out if not b)
    sizes = sorted(len(b) for b in out)
    with open(args.out, "w") as f:
        for r, best in zip(rows, out):
            rec = {"corrupted": r[args.field],
                   "candidates": sorted(best.keys()),
                   "witness": best}
            if "ground_truth" in r:
                rec["ground_truth"] = r["ground_truth"]
                rec["truth_in_candidates"] = r["ground_truth"] in best
            if "observed_category" in r:
                rec["category"] = r["observed_category"]
            f.write(json.dumps(rec) + "\n")

    summ = {"n": len(rows), "empty_candidate_sets": n_empty,
            "median_candidates": sizes[len(sizes) // 2],
            "mean_candidates": round(sum(sizes) / len(sizes), 1)}
    if "ground_truth" in rows[0]:
        summ["truth_in_candidates"] = round(
            sum(1 for r, b in zip(rows, out) if r["ground_truth"] in b) / len(rows), 4)
    print(json.dumps(summ, indent=2))
    json.dump(summ, open(args.out.replace(".jsonl", ".summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
