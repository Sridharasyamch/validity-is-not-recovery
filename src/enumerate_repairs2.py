"""Bounded two-edit repair search: how much scope does a second edit buy, and
at what cost?

A single-token enumerator collapses on multi-error strings (coverage 99.7% at
one edit, 0.5% at two, 0.0% at three). A complete two-edit enumeration is
quadratic and infeasible, so this performs a bounded search:

  level 1  enumerate all one-edit neighbours; the valid ones are candidates,
           the invalid ones are ranked by a purely SYNTACTIC imbalance score
           (unclosed branches, odd ring-closure digits, bracket mismatch) that
           uses no ground truth;
  level 2  keep the best `beam` invalid neighbours and enumerate their one-edit
           neighbours, collecting every valid molecule found.

Each item gets a wall-clock budget, and parse counts and elapsed time are
recorded, so recovery can be reported against compute rather than for free.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import symbolic as S
from ambiguity import build_vocab, one_edit_neighbours

_V = None
_BEAM = 60
_BUDGET = 4.0


def _init(vocab, beam, budget):
    global _V, _BEAM, _BUDGET
    _V, _BEAM, _BUDGET = vocab, beam, budget
    RDLogger.DisableLog("rdApp.*")


def imbalance(smi):
    """Observable syntactic defect count -- no ground truth involved."""
    try:
        toks = S.tokenize(smi)
    except Exception:
        return 999
    depth = bad = 0
    for t in toks:
        if t == "(":
            depth += 1
        elif t == ")":
            depth -= 1
            if depth < 0:
                bad += 1
                depth = 0
    bad += depth
    rings = Counter(t for t in toks if t.isdigit() or t.startswith("%"))
    bad += sum(1 for v in rings.values() if v % 2)
    bad += abs(smi.count("[") - smi.count("]"))
    return bad


def _search(args):
    i, smi = args
    t0 = time.time()
    parses = 0
    found = {}
    budget_hit = False
    try:
        lvl1 = one_edit_neighbours(smi, _V)
        invalid = []
        for cand in lvl1:
            m = Chem.MolFromSmiles(cand)
            parses += 1
            if m is None:
                invalid.append(cand)
                continue
            try:
                c = Chem.MolToSmiles(m)
            except Exception:
                continue
            if c not in found or len(cand) < len(found[c]):
                found[c] = cand
        invalid.sort(key=imbalance)
        for mid in invalid[:_BEAM]:
            if time.time() - t0 > _BUDGET:
                budget_hit = True
                break
            for cand in one_edit_neighbours(mid, _V):
                m = Chem.MolFromSmiles(cand)
                parses += 1
                if m is None:
                    continue
                try:
                    c = Chem.MolToSmiles(m)
                except Exception:
                    continue
                if c not in found or len(cand) < len(found[c]):
                    found[c] = cand
            if time.time() - t0 > _BUDGET:
                budget_hit = True
                break
    except Exception:
        pass
    return i, found, parses, round(time.time() - t0, 3), budget_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--beam", type=int, default=60)
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--field", default="corrupted")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = [json.loads(l) for l in open(args.benchmark) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    vocab = build_vocab(rows)
    print(f"[enum2] n={len(rows)} vocab={len(vocab)} beam={args.beam} "
          f"budget={args.budget}s workers={args.workers}", flush=True)

    out = [None] * len(rows)
    with Pool(args.workers, initializer=_init,
              initargs=(vocab, args.beam, args.budget)) as p:
        for n, (i, found, parses, el, hit) in enumerate(
                p.imap_unordered(_search,
                                 [(i, r[args.field]) for i, r in enumerate(rows)],
                                 chunksize=4), 1):
            out[i] = (found, parses, el, hit)
            if n % 50 == 0:
                print(f"  {n}/{len(rows)}", flush=True)

    with open(args.out, "w") as f:
        for r, (found, parses, el, hit) in zip(rows, out):
            rec = {"corrupted": r[args.field], "candidates": sorted(found),
                   "witness": found, "n_parses": parses, "seconds": el,
                   "budget_hit": hit}
            if "ground_truth" in r:
                rec["ground_truth"] = r["ground_truth"]
                rec["truth_in_candidates"] = r["ground_truth"] in found
            if "observed_category" in r:
                rec["category"] = r["observed_category"]
            f.write(json.dumps(rec) + "\n")

    sizes = sorted(len(b[0]) for b in out)
    n = len(rows)
    summ = {"n": n, "beam": args.beam, "budget_s": args.budget,
            "empty_candidate_sets": sum(1 for b in out if not b[0]),
            "median_candidates": sizes[n // 2],
            "mean_parses_per_item": round(sum(b[1] for b in out) / n, 1),
            "mean_seconds_per_item": round(sum(b[2] for b in out) / n, 3),
            "budget_hit_frac": round(sum(1 for b in out if b[3]) / n, 4)}
    if "ground_truth" in rows[0]:
        summ["truth_in_candidates"] = round(
            sum(1 for r, b in zip(rows, out) if r["ground_truth"] in b[0]) / n, 4)
    json.dump(summ, open(args.out.replace(".jsonl", ".summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
