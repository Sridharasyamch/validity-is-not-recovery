"""Ground-truth corruption benchmark (leakage-controlled).

Takes VALID molecules from a held-out pool (never seen by our generator), applies
a corruption operator, and verifies the result is genuinely invalid. Because we
start from a known molecule we know the ground truth, which lets us measure
RECOVERY (did we get the intended molecule back?) rather than only VALIDITY.

Operator mix is calibrated to the error-category distribution measured from REAL
generator failures, so the synthetic benchmark mirrors reality.
"""
from __future__ import annotations

import argparse, json, random, re, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import symbolic as S

# Target mix measured from the real generator-failure pool (pooled T=1.0/1.2/1.5)
REAL_MIX = {
    "semantic_kekulize": 0.452,
    "syntax_ring":       0.313,
    "semantic_valence":  0.113,
    "syntax_other":      0.054,
    "syntax_parens":     0.037,
    "syntax_brackets":   0.031,
}


def _atom_token_idx(tokens):
    """Indices of atom tokens in a token list."""
    return [i for i, t in enumerate(tokens)
            if t.startswith("[") or (t[0].isalpha() and t not in ("Cl", "Br") or t in ("Cl", "Br"))]


def corrupt_ring(smi, rng):
    """Delete one occurrence of a ring-closure digit -> unclosed ring."""
    toks = S.tokenize(smi)
    ring_pos = [i for i, t in enumerate(toks) if t.isdigit() or t.startswith("%")]
    if not ring_pos:
        return None
    i = rng.choice(ring_pos)
    return "".join(toks[:i] + toks[i + 1:])


def corrupt_parens(smi, rng):
    """Delete a parenthesis -> unbalanced branches."""
    pos = [i for i, c in enumerate(smi) if c in "()"]
    if not pos:
        return None
    i = rng.choice(pos)
    return smi[:i] + smi[i + 1:]


def corrupt_brackets(smi, rng):
    """Delete a closing bracket -> unterminated bracket atom."""
    pos = [i for i, c in enumerate(smi) if c == "]"]
    if not pos:
        return None
    i = rng.choice(pos)
    return smi[:i] + smi[i + 1:]


def corrupt_valence(smi, rng):
    """Replace a high-degree carbon with a monovalent atom -> valence violation."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    toks = S.tokenize(smi)
    # organic-subset carbons written as plain 'C' are the safe substitution sites
    cands = [i for i, t in enumerate(toks) if t == "C"]
    rng.shuffle(cands)
    for i in cands:
        for repl in ("F", "Cl"):
            cand = "".join(toks[:i] + [repl] + toks[i + 1:])
            if Chem.MolFromSmiles(cand) is None:
                return cand
    return None


def corrupt_kekulize(smi, rng):
    """Break aromaticity so the ring cannot be kekulized.

    Two reliable inducers: (a) swap an aromatic heteroatom in a 5-ring for 'c'
    (e.g. furan -> all-carbon 5-ring), (b) delete one aromatic atom from a
    6-ring, leaving an odd all-aromatic ring."""
    toks = S.tokenize(smi)
    het = [i for i, t in enumerate(toks) if t in ("o", "s", "n")]
    rng.shuffle(het)
    for i in het:
        cand = "".join(toks[:i] + ["c"] + toks[i + 1:])
        d = S.diagnose(cand)
        if not d.valid and d.category == "semantic_kekulize":
            return cand
    arom_c = [i for i, t in enumerate(toks) if t == "c"]
    rng.shuffle(arom_c)
    for i in arom_c:
        cand = "".join(toks[:i] + toks[i + 1:])
        d = S.diagnose(cand)
        if not d.valid and d.category == "semantic_kekulize":
            return cand
    return None


def corrupt_other(smi, rng):
    """Duplicate a bond symbol / inject a stray token -> generic parse error."""
    toks = S.tokenize(smi)
    bond_pos = [i for i, t in enumerate(toks) if t in ("=", "#", "-")]
    if bond_pos:
        i = rng.choice(bond_pos)
        cand = "".join(toks[:i] + [toks[i] * 2] + toks[i + 1:])
        if Chem.MolFromSmiles(cand) is None:
            return cand
    # fallback: lowercase an aliphatic atom mid-chain
    cands = [i for i, t in enumerate(toks) if t in ("N", "O", "S")]
    rng.shuffle(cands)
    for i in cands:
        cand = "".join(toks[:i] + [toks[i].lower()] + toks[i + 1:])
        if Chem.MolFromSmiles(cand) is None:
            return cand
    return None


OPS = {
    "semantic_kekulize": corrupt_kekulize,
    "syntax_ring": corrupt_ring,
    "semantic_valence": corrupt_valence,
    "syntax_other": corrupt_other,
    "syntax_parens": corrupt_parens,
    "syntax_brackets": corrupt_brackets,
}


def make_item(smi, target_cat, rng, max_try=6):
    """Corrupt `smi` into an invalid string of (ideally) category `target_cat`."""
    for _ in range(max_try):
        cand = OPS[target_cat](smi, rng)
        if not cand or cand == smi:
            continue
        d = S.diagnose(cand)
        if d.valid:
            continue
        return {"corrupted": cand, "observed_category": d.category,
                "target_category": target_cat,
                "category_match": d.category == target_cat}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/guacamol_test.smiles",
                    help="held-out molecule pool (NOT seen by the generator)")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--dev_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", default="data/recovery")
    ap.add_argument("--max_len", type=int, default=120)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    pool = []
    with open(args.source) as f:
        for line in f:
            s = line.strip().split()[0] if line.strip() else ""
            if s and len(s) <= args.max_len:
                pool.append(s)
    rng.shuffle(pool)
    print(f"[corrupt] held-out source pool: {len(pool)}")

    # category draw order matching the real failure mix
    cats, weights = zip(*REAL_MIX.items())
    items, seen_src, i = [], set(), 0
    while len(items) < args.n and i < len(pool):
        smi = pool[i]; i += 1
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        canon = Chem.MolToSmiles(m)
        if canon in seen_src:
            continue
        cat = rng.choices(cats, weights=weights, k=1)[0]
        made = make_item(smi, cat, rng)
        if not made:
            continue
        seen_src.add(canon)
        items.append({"ground_truth": canon, "source_smiles": smi, **made})
        if len(items) % 500 == 0:
            print(f"  built {len(items)}/{args.n} (scanned {i})", flush=True)

    rng.shuffle(items)
    n_dev = int(len(items) * args.dev_frac)
    dev, test = items[:n_dev], items[n_dev:]
    for name, part in (("dev", dev), ("test", test)):
        p = os.path.join(args.out_dir, f"recovery_{name}.jsonl")
        with open(p, "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        print(f"[corrupt] wrote {p}: {len(part)}")

    obs = Counter(r["observed_category"] for r in items)
    match = sum(r["category_match"] for r in items)
    stats = {"n_total": len(items), "n_dev": len(dev), "n_test": len(test),
             "source": args.source, "seed": args.seed,
             "observed_category_distribution": dict(obs),
             "target_mix": REAL_MIX,
             "operator_hit_intended_category_pct": round(100 * match / max(len(items), 1), 1)}
    json.dump(stats, open(os.path.join(args.out_dir, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
