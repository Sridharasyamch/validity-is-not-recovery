"""Recover which token edit produced each enumerated candidate, and use the
symbolic diagnosis to prune the candidate set.

The enumerator stores, for every distinct reachable molecule, the witness string
that produced it. Diffing the witness against the corrupted string recovers the
edit (operation + token), which lets the RDKit diagnosis constrain the search:
an unclosed ring should be repaired by a ring-closure digit, an unbalanced branch
by a parenthesis, and so on. This is the symbolic half doing real work -- it
shrinks the hypothesis space before the neural ranker ever sees it.
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import symbolic as S

RING = set("0123456789")


def token_class(tok):
    if not tok:
        return "none"
    if tok in ("(", ")"):
        return "paren"
    if tok.startswith("[") or tok == "]":
        return "bracket"
    if tok in RING or tok.startswith("%"):
        return "ring"
    if tok in ("=", "#", "-", ":", "~", "/", "\\", "+", "@", "."):
        return "bond"
    if tok[0].islower():
        return "aromatic_atom"
    return "atom"


def diff_edit(src, dst):
    """(op, token) turning token(src) into token(dst) by one edit; None if not 1 edit."""
    try:
        a, b = S.tokenize(src), S.tokenize(dst)
    except Exception:
        return None
    if abs(len(a) - len(b)) > 1:
        return None
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    if len(a) == len(b):
        if i == len(a):
            return ("none", "")
        if a[i + 1:] != b[i + 1:]:
            return None
        return ("substitute", b[i])
    if len(b) == len(a) + 1:                 # insertion into the corrupted string
        if a[i:] != b[i + 1:]:
            return None
        return ("insert", b[i])
    if a[i:] != b[i - 0:][:0] + b[i:]:       # deletion from the corrupted string
        pass
    if a[i + 1:] != b[i:]:
        return None
    return ("delete", a[i])


# which token classes can plausibly repair each diagnosed error category
ALLOWED = {
    "syntax_ring":       {"ring"},
    "syntax_parens":     {"paren"},
    "syntax_brackets":   {"bracket", "atom"},
    "semantic_valence":  {"atom", "aromatic_atom", "bond", "bracket"},
    "semantic_kekulize": {"aromatic_atom", "atom", "bracket"},
    "syntax_other":      None,               # no constraint
}


def prune(corrupted, candidates, witness, category=None):
    """Keep only candidates whose edit is consistent with the diagnosed error."""
    if category is None:
        category = S.diagnose(corrupted).category
    allowed = ALLOWED.get(category)
    if allowed is None:
        return list(candidates), category
    keep = []
    for c in candidates:
        w = witness.get(c)
        if not w:
            continue
        e = diff_edit(corrupted, w)
        if e is None:
            continue
        if token_class(e[1]) in allowed:
            keep.append(c)
    return (keep if keep else list(candidates)), category
