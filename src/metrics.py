"""Evaluation metrics for SMILES repair.

Validity, canonicalization, similarity (Tanimoto), faithfulness (edit distance
between the broken string and its repair), scaffold preservation, and chemical
quality (QED, descriptors, diversity, novelty)."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

from symbolic import tokenize  # reuse the SMILES tokenizer

RDLogger.DisableLog("rdApp.*")  # metrics code should be quiet

_MORGAN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


def canonical(smi: str) -> Optional[str]:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi.strip())
    return Chem.MolToSmiles(m) if m is not None else None


def is_valid(smi: str) -> bool:
    return canonical(smi) is not None


@lru_cache(maxsize=200_000)
def _fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    return _MORGAN.GetFingerprint(m) if m is not None else None


def tanimoto(a: str, b: str) -> Optional[float]:
    """Morgan(r=2,2048) Tanimoto between two SMILES; None if either invalid."""
    fa, fb = _fp(a or ""), _fp(b or "")
    if fa is None or fb is None:
        return None
    return DataStructs.TanimotoSimilarity(fa, fb)


def edit_distance(a: str, b: str) -> int:
    """Character-level Levenshtein distance."""
    a, b = a or "", b or ""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def token_edit_distance(a: str, b: str) -> int:
    """SMILES-token-level Levenshtein distance."""
    ta, tb = tokenize(a or ""), tokenize(b or "")
    if ta == tb:
        return 0
    if not ta:
        return len(tb)
    if not tb:
        return len(ta)
    prev = list(range(len(tb) + 1))
    for i, ca in enumerate(ta, 1):
        cur = [i]
        for j, cb in enumerate(tb, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def scaffold(smi: str) -> Optional[str]:
    m = Chem.MolFromSmiles(smi or "")
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None


def qed(smi: str) -> Optional[float]:
    m = Chem.MolFromSmiles(smi or "")
    if m is None:
        return None
    try:
        return QED.qed(m)
    except Exception:
        return None


def descriptors(smi: str) -> Optional[dict]:
    m = Chem.MolFromSmiles(smi or "")
    if m is None:
        return None
    return {
        "mw": Descriptors.MolWt(m),
        "logp": Descriptors.MolLogP(m),
        "hbd": Descriptors.NumHDonors(m),
        "hba": Descriptors.NumHAcceptors(m),
        "tpsa": Descriptors.TPSA(m),
        "rings": Descriptors.RingCount(m),
        "heavy": m.GetNumHeavyAtoms(),
    }


def internal_diversity(smiles: Iterable[str], sample: int = 2000, seed: int = 0) -> Optional[float]:
    """Mean pairwise (1 - Tanimoto) over valid molecules (subsampled)."""
    fps = [_fp(s) for s in smiles]
    fps = [f for f in fps if f is not None]
    if len(fps) < 2:
        return None
    rng = np.random.default_rng(seed)
    if len(fps) > sample:
        idx = rng.choice(len(fps), sample, replace=False)
        fps = [fps[i] for i in idx]
    sims = []
    for i in range(len(fps)):
        s = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        sims.extend(s)
    return float(1.0 - np.mean(sims)) if sims else None


def novelty(smiles: Iterable[str], reference: set[str]) -> Optional[float]:
    """Fraction of (valid, canonicalized) molecules not present in reference."""
    cans = [canonical(s) for s in smiles]
    cans = [c for c in cans if c is not None]
    if not cans:
        return None
    return float(np.mean([c not in reference for c in cans]))


def uniqueness(smiles: Iterable[str]) -> Optional[float]:
    cans = [canonical(s) for s in smiles]
    cans = [c for c in cans if c is not None]
    if not cans:
        return None
    return len(set(cans)) / len(cans)
