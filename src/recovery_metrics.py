"""Identity-aware RECOVERY metrics (vs. validity-only).

The point of the ground-truth benchmark is that we know the molecule the broken
string was supposed to encode, so we can ask whether a method recovered THAT
molecule, not merely whether it emitted A valid one.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

_MORGAN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


def canon(smi):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def tanimoto(a, b):
    ma, mb = Chem.MolFromSmiles(a or ""), Chem.MolFromSmiles(b or "")
    if ma is None or mb is None:
        return None
    return DataStructs.TanimotoSimilarity(_MORGAN.GetFingerprint(ma),
                                          _MORGAN.GetFingerprint(mb))


def scaffold_match(a, b):
    try:
        ma, mb = Chem.MolFromSmiles(a or ""), Chem.MolFromSmiles(b or "")
        if ma is None or mb is None:
            return None
        return (MurckoScaffold.MurckoScaffoldSmiles(mol=ma)
                == MurckoScaffold.MurckoScaffoldSmiles(mol=mb))
    except Exception:
        return None


def mcs_frac(a, b, timeout=2):
    """Size of the maximum common substructure as a fraction of the truth's atoms."""
    ma, mb = Chem.MolFromSmiles(a or ""), Chem.MolFromSmiles(b or "")
    if ma is None or mb is None or ma.GetNumAtoms() == 0:
        return None
    try:
        r = rdFMCS.FindMCS([ma, mb], timeout=timeout,
                           ringMatchesRingOnly=True, completeRingsOnly=False)
        if r.canceled and r.numAtoms == 0:
            return 0.0
        return r.numAtoms / ma.GetNumAtoms()
    except Exception:
        return None


def score(truth, pred, with_mcs=True):
    """Full recovery scorecard for one item. `truth` is the ground-truth SMILES."""
    ct, cp = canon(truth), canon(pred)
    valid = cp is not None
    return {
        "valid": valid,
        "exact_recovery": bool(valid and cp == ct),
        "tanimoto": tanimoto(ct, cp) if valid else None,
        "scaffold_match": scaffold_match(ct, cp) if valid else None,
        "mcs_frac": (mcs_frac(ct, cp) if (valid and with_mcs) else None),
    }


def aggregate(rows):
    n = len(rows)
    if n == 0:
        return {}
    val = [r for r in rows if r["valid"]]

    def mean(key, subset):
        xs = [r[key] for r in subset if r.get(key) is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "n": n,
        "validity": round(len(val) / n, 4),
        "exact_recovery": round(sum(r["exact_recovery"] for r in rows) / n, 4),
        # identity metrics conditioned on producing something valid
        "tanimoto_given_valid": mean("tanimoto", val),
        "scaffold_match_given_valid": (
            round(sum(1 for r in val if r.get("scaffold_match")) / len(val), 4) if val else None),
        "mcs_frac_given_valid": mean("mcs_frac", val),
    }
