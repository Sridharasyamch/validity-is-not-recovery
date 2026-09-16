"""Multi-edit corruption benchmark: how does recovery degrade with edit distance?

The main benchmark applies a single token edit, which matches the modal real
generator failure but is also the assumption our enumerator is built on. This
builds matched benchmarks at k = 1, 2, 3 simultaneous edits so the degradation
can be measured rather than assumed.

Molecules are drawn from the same held-out pool but explicitly EXCLUDE every
molecule used in the main benchmark, so this is a separate probe rather than a
re-scoring of the same items. We record the true token edit distance between
ground truth and corrupted string, because independent edits can interact (two
edits can partially cancel, making a k=2 item reachable in one edit); reporting
the realised distance keeps the degradation curve honest.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import symbolic as S
import metrics as M
from corrupt import REAL_MIX, OPS

SEMANTIC = {"semantic_valence", "semantic_kekulize"}


def apply_k(smi, k, rng, max_try=8):
    """Apply k single-token operators. Semantic operators need a parseable
    string, so they are applied first, while the molecule still parses."""
    cats, weights = zip(*REAL_MIX.items())
    for _ in range(max_try):
        picks = rng.choices(cats, weights=weights, k=k)
        # semantic operators first, while the string is still valid
        picks = sorted(picks, key=lambda c: 0 if c in SEMANTIC else 1)
        cur, applied = smi, []
        ok = True
        for c in picks:
            if c in SEMANTIC and Chem.MolFromSmiles(cur) is None:
                # cannot apply a semantic operator to an unparseable string;
                # fall back to a syntactic one
                c = rng.choice([x for x in cats if x not in SEMANTIC])
            nxt = OPS[c](cur, rng)
            if not nxt or nxt == cur:
                ok = False
                break
            cur, _ = nxt, applied.append(c)
        if not ok:
            continue
        d = S.diagnose(cur)
        if d.valid:
            continue
        return cur, applied, d.category
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/guacamol_test.smiles")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=909)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--exclude", nargs="*",
                    default=["data/recovery/recovery_test.jsonl",
                             "data/recovery/recovery_dev.jsonl"])
    ap.add_argument("--out_dir", default="data/recovery_multi")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed + 1000 * args.k)

    used = set()
    for p in args.exclude:
        if os.path.exists(p):
            for l in open(p):
                if l.strip():
                    used.add(json.loads(l)["ground_truth"])
    print(f"[multi] excluding {len(used)} molecules already in the main benchmark")

    pool = []
    with open(args.source) as f:
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            if s and len(s) <= args.max_len:
                pool.append(s)
    rng.shuffle(pool)

    items, seen, i = [], set(), 0
    while len(items) < args.n and i < len(pool):
        smi = pool[i]; i += 1
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        canon = Chem.MolToSmiles(m)
        if canon in used or canon in seen:
            continue
        cur, applied, cat = apply_k(smi, args.k, rng)
        if not cur:
            continue
        seen.add(canon)
        items.append({"ground_truth": canon, "source_smiles": smi,
                      "corrupted": cur, "k_requested": args.k,
                      "ops_applied": applied, "observed_category": cat,
                      "true_token_edit": M.token_edit_distance(canon, cur)})
        if len(items) % 200 == 0:
            print(f"  built {len(items)}/{args.n} (scanned {i})", flush=True)

    out = os.path.join(args.out_dir, f"k{args.k}.jsonl")
    with open(out, "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")
    ed = sorted(r["true_token_edit"] for r in items)
    stats = {"k": args.k, "n": len(items),
             "median_true_token_edit": ed[len(ed) // 2] if ed else None,
             "mean_true_token_edit": round(sum(ed) / len(ed), 2) if ed else None,
             "observed_categories": dict(Counter(r["observed_category"] for r in items))}
    json.dump(stats, open(out.replace(".jsonl", ".stats.json"), "w"), indent=2)
    print(f"[multi] wrote {out}: {len(items)} items")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
