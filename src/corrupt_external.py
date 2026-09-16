"""EXTERNAL benchmark: another group's corruption operators, another molecule source.

Our benchmark uses corruption operators we wrote, on GuacaMol molecules. A fair
objection is that the method may be tuned to that particular corruption process.
This builds an independent benchmark using the corruption code released with
UnCorrupt SMILES (Schoenmaker et al., J. Cheminformatics 2023) applied to their
molecule source (PAPYRUS, a bioactivity corpus), at one error per molecule.

Nothing here is ours: the operators, the fragment library and the molecules all
come from their repository. The ranker has never seen PAPYRUS molecules, so this
is simultaneously an out-of-distribution test for the chemical prior.

Note on ground truth: their `introduce_error` returns (invalid, corfrag), and for
aromaticity/valence operators it GRAFTS a fragment onto the molecule -- in which
case the intended molecule is `corfrag` (original.Fragment), not the input. We
therefore score against corfrag, exactly as their own pipeline does.
"""
from __future__ import annotations
import argparse, csv, json, os, random, sys

sys.path.insert(0, os.path.dirname(__file__))
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import symbolic as S
import metrics as M

UNCORRUPT = os.environ.get("UNCORRUPT_PATH", "third_party/uncorrupt")


def sniff_smiles_column(path, probe=40):
    """Find which column of a CSV holds SMILES, by trying to parse them."""
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        hits = [0] * len(header)
        rows = 0
        for row in rdr:
            rows += 1
            for i, v in enumerate(row[:len(header)]):
                v = (v or "").strip()
                if 5 < len(v) < 200 and Chem.MolFromSmiles(v) is not None:
                    hits[i] += 1
            if rows >= probe:
                break
    best = max(range(len(hits)), key=lambda i: hits[i])
    return header, best, hits[best], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--scan_limit", type=int, default=60000)
    ap.add_argument("--out", default="data/recovery_external/papyrus_ext.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if UNCORRUPT not in sys.path:
        sys.path.insert(0, UNCORRUPT)
    from src.invalidSMILES import introduce_error

    frag_path = f"{UNCORRUPT}/RawData/gbd_8.csv"
    mol_path = f"{UNCORRUPT}/RawData/PAPYRUS.csv"

    frags = []
    with open(frag_path, newline="") as f:
        rdr = csv.reader(f)
        next(rdr, None)
        for row in rdr:
            if row and row[0].strip():
                frags.append(row[0].strip())
    print(f"[ext] fragment library: {len(frags)}")

    header, col, hits, probed = sniff_smiles_column(mol_path)
    print(f"[ext] PAPYRUS columns: {header}")
    print(f"[ext] using column {col} ({header[col]!r}) -- {hits}/{probed} parsed")

    rng = random.Random(args.seed)
    random.seed(args.seed)          # their operators use the global `random`
    vocab = set()

    items, seen, scanned, attempts = [], set(), 0, 0
    with open(mol_path, newline="") as f:
        rdr = csv.reader(f)
        next(rdr, None)
        for row in rdr:
            if len(items) >= args.n or scanned >= args.scan_limit:
                break
            scanned += 1
            if col >= len(row):
                continue
            smi = (row[col] or "").strip()
            if not smi or len(smi) > args.max_len:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            canon = Chem.MolToSmiles(m)
            if canon in seen:
                continue
            attempts += 1
            try:
                bad, corfrag = introduce_error(
                    smi, rng.choice(frags), vocab,
                    invalid_type="all", num_errors=1)
            except Exception:
                continue
            if not bad:
                continue
            d = S.diagnose(bad)
            if d.valid:
                continue
            tm = Chem.MolFromSmiles(corfrag or "")
            if tm is None:
                continue
            truth = Chem.MolToSmiles(tm)
            seen.add(canon)
            items.append({"ground_truth": truth, "source_smiles": smi,
                          "corrupted": bad, "observed_category": d.category,
                          "fragment_grafted": truth != canon,
                          "true_token_edit": M.token_edit_distance(truth, bad)})
            if len(items) % 200 == 0:
                print(f"  built {len(items)}/{args.n} (scanned {scanned})", flush=True)

    with open(args.out, "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    ed = sorted(r["true_token_edit"] for r in items)
    stats = {"n": len(items), "scanned": scanned, "attempts": attempts,
             "source": "PAPYRUS (UnCorrupt repo)",
             "operators": "UnCorrupt introduce_error, invalid_type=all, num_errors=1",
             "fragment_grafted_frac": round(
                 sum(r["fragment_grafted"] for r in items) / max(len(items), 1), 4),
             "median_true_token_edit": ed[len(ed) // 2] if ed else None,
             "mean_true_token_edit": round(sum(ed) / len(ed), 2) if ed else None,
             "observed_categories": dict(Counter(r["observed_category"] for r in items))}
    json.dump(stats, open(args.out.replace(".jsonl", ".stats.json"), "w"), indent=2)
    print(f"[ext] wrote {args.out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
