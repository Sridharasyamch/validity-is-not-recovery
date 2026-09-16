"""How ambiguous is SMILES repair? An exact chance baseline and recovery ceiling.

Every corruption in the benchmark is a single-token edit, so the ground truth is
by construction reachable within one token edit of the corrupted string. That
lets us enumerate the ENTIRE space of one-edit repairs and ask:

  * ceiling      -- is the truth actually recoverable at one edit? (sanity: ~1.0)
  * |C|          -- how many DISTINCT valid molecules sit one edit away?
  * chance       -- E[1/|C|], the exact expected exact-recovery of a repairer
                    that picks uniformly among valid one-edit repairs.

`chance` is the honest null hypothesis for this benchmark. A method that scores
at or below it has demonstrated no ability to recover identity, however high its
validity rate.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import symbolic as S

# Token alphabet for insertions/substitutions, mined from the source corpus.
DEFAULT_VOCAB = list("BCNOSPFIbcnosp()[]=#-+/\\@123456789%") + [
    "Cl", "Br", "[nH]", "[C@H]", "[C@@H]", "[N+]", "[O-]", "0",
]


def one_edit_neighbours(smi, vocab):
    """All strings within one TOKEN edit (delete / substitute / insert)."""
    toks = S.tokenize(smi)
    out = set()
    for i in range(len(toks)):
        out.add("".join(toks[:i] + toks[i + 1:]))                 # delete
        for v in vocab:
            if v != toks[i]:
                out.add("".join(toks[:i] + [v] + toks[i + 1:]))   # substitute
    for i in range(len(toks) + 1):
        for v in vocab:
            out.add("".join(toks[:i] + [v] + toks[i:]))           # insert
    out.discard(smi)
    return out


def valid_neighbour_molecules(smi, vocab):
    """Canonical SMILES of every valid molecule one token edit away."""
    mols = set()
    for cand in one_edit_neighbours(smi, vocab):
        m = Chem.MolFromSmiles(cand)
        if m is not None:
            try:
                mols.add(Chem.MolToSmiles(m))
            except Exception:
                pass
    return mols


def build_vocab(rows, extra_top=40):
    """Token alphabet = default set plus the most frequent tokens in the truths."""
    c = Counter()
    for r in rows:
        s = r.get("ground_truth") or r.get("corrupted") or r.get("smiles")
        if not s:
            continue
        try:
            c.update(S.tokenize(s))
        except Exception:
            pass
    vocab = list(dict.fromkeys(DEFAULT_VOCAB + [t for t, _ in c.most_common(extra_top)]))
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/recovery/recovery_test.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="results_recovery/ambiguity.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = [json.loads(l) for l in open(args.benchmark) if l.strip()][:args.n]
    vocab = build_vocab(rows)
    print(f"[ambiguity] n={len(rows)} vocab={len(vocab)} tokens")

    recs = []
    for i, r in enumerate(rows):
        truth = r["ground_truth"]
        mols = valid_neighbour_molecules(r["corrupted"], vocab)
        recs.append({"category": r["observed_category"],
                     "n_valid_one_edit": len(mols),
                     "truth_reachable": truth in mols})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    n = len(recs)
    reach = [r for r in recs if r["truth_reachable"]]
    # exact expected accuracy of uniform choice among valid one-edit repairs
    chance = sum(1.0 / r["n_valid_one_edit"] for r in reach if r["n_valid_one_edit"]) / n
    sizes = sorted(r["n_valid_one_edit"] for r in recs)
    summary = {
        "n": n,
        "ceiling_one_edit": round(len(reach) / n, 4),
        "chance_uniform_one_edit": round(chance, 4),
        "median_valid_one_edit": sizes[n // 2],
        "mean_valid_one_edit": round(sum(sizes) / n, 1),
        "vocab_size": len(vocab),
    }
    per_cat = {}
    for c in sorted(set(r["category"] for r in recs)):
        sub = [r for r in recs if r["category"] == c]
        sr = [r for r in sub if r["truth_reachable"]]
        ss = sorted(r["n_valid_one_edit"] for r in sub)
        per_cat[c] = {
            "n": len(sub),
            "ceiling": round(len(sr) / len(sub), 4),
            "chance": round(sum(1.0 / r["n_valid_one_edit"] for r in sr
                                if r["n_valid_one_edit"]) / len(sub), 4),
            "median_valid_one_edit": ss[len(ss) // 2],
        }
    summary["per_category"] = per_cat
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
