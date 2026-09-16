"""Symbolic layer: deterministic SMILES diagnosis + rule-based repair.

This is the "symbolic" half of the neuro-symbolic loop. It does two things:

1. ``diagnose(smi)`` -- a *reproducible* two-stage RDKit check that classifies a
   SMILES string as valid / syntactically broken / semantically broken, and
   returns a structured, human-readable diagnostic. The diagnostic is what we
   feed to the LLM as feedback (the neuro-symbolic signal).

2. ``rule_repair(smi)`` -- a best-effort deterministic repair that fixes the
   easy, purely-syntactic failure modes (paren/bracket balance, unmatched ring
   bonds). This is the non-neural baseline the LLM must beat.

Design note: we do NOT rely solely on scraping RDKit's stderr log (fragile across
versions). We tokenize the SMILES ourselves and run structural checks, then use
RDKit's ``DetectChemistryProblems`` for semantic problems and the parse log only
as a human-readable detail string.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from contextlib import redirect_stderr
from dataclasses import dataclass, asdict
from typing import Optional

from rdkit import Chem, rdBase

# Route RDKit's C++ logs to Python stderr so we can capture the parse message.
rdBase.LogToPythonStderr()

# Standard SMILES tokenizer (Schwaller et al., 2018 / Molecular Transformer).
_SMI_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)

# Error taxonomy (string constants used everywhere downstream).
VALID = "valid"
SYN_PARENS = "syntax_parens"
SYN_BRACKETS = "syntax_brackets"
SYN_RING = "syntax_ring"
SYN_OTHER = "syntax_other"
SEM_VALENCE = "semantic_valence"
SEM_KEKULIZE = "semantic_kekulize"
SEM_OTHER = "semantic_other"

SYNTACTIC = {SYN_PARENS, SYN_BRACKETS, SYN_RING, SYN_OTHER}
SEMANTIC = {SEM_VALENCE, SEM_KEKULIZE, SEM_OTHER}


def tokenize(smi: str) -> list[str]:
    """Tokenize a SMILES string into chemically-meaningful tokens."""
    return _SMI_REGEX.findall(smi)


def _ring_digit_counts(tokens: list[str]) -> Counter:
    """Count ring-bond labels (bare digits and %NN) in the token stream.

    Tokens inside [...] are captured as a single bracket token by the regex, so
    isotope/charge/H-count digits never leak in here -- a bare digit token is
    always a ring-closure label."""
    counts: Counter = Counter()
    for t in tokens:
        if t.isdigit():
            counts[t] += 1
        elif t.startswith("%"):
            counts[t] += 1
    return counts


@dataclass
class Diagnosis:
    smiles: str
    valid: bool
    category: str          # one of the taxonomy constants
    is_syntactic: bool
    message: str           # short human-readable diagnostic (fed to the LLM)
    rdkit_log: str         # raw RDKit parse log (detail / provenance)

    def as_dict(self) -> dict:
        return asdict(self)

    def feedback(self) -> str:
        """The natural-language feedback string handed to the repair LLM."""
        return self.message


def _parse_no_sanitize(smi: str):
    """Parse without sanitization; capture the RDKit log. Returns (mol, log)."""
    buf = io.StringIO()
    with redirect_stderr(buf):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
    return mol, buf.getvalue().strip()


_TS = re.compile(r"^\[\d\d:\d\d:\d\d\]\s*")


def _clean_log(log: str) -> str:
    """Turn RDKit's multi-line stderr into one concise, specific diagnostic line.

    Keeps the informative content ('Explicit valence for atom # 14 ...',
    'Unkekulized atoms: 24 26 27', 'extra open parentheses ... position 17') and
    drops timestamps and the ASCII caret/echo lines."""
    keep = []
    for ln in log.splitlines():
        ln = _TS.sub("", ln).strip()
        if not ln or set(ln) <= set("~^ "):          # caret pointer lines
            continue
        low = ln.lower()
        if low.startswith("smiles parse error: failed parsing"):
            continue                                  # redundant echo of the string
        if low.startswith("smiles parse error: check for mistakes"):
            keep.append(ln.split(":", 1)[-1].strip()) # keep "around position N"
            continue
        keep.append(ln)
    # de-duplicate while preserving order
    seen, out = set(), []
    for k in keep:
        if k not in seen:
            seen.add(k); out.append(k)
    return "  ".join(out)[:300]


def diagnose(smi: str) -> Diagnosis:
    """Crash-safe two-stage diagnosis: sanitized parse -> structural / sanitize-flags.

    NOTE: we deliberately avoid ``DetectChemistryProblems`` and any operation on
    an unsanitized mol that can SIGSEGV on adversarial generator output. Validity
    is decided by the standard sanitizing parser (returns None, never raises),
    and the semantic error type is obtained from ``SanitizeMol(catchErrors=True)``
    which returns a flag instead of raising or crashing."""
    smi = (smi or "").strip()
    tokens = tokenize(smi)

    # ---- Stage 0: cheap deterministic structural checks (pure Python, safe) ----
    n_open, n_close = smi.count("("), smi.count(")")
    n_lb, n_rb = smi.count("["), smi.count("]")
    ring_counts = _ring_digit_counts(tokens)
    odd_rings = [lbl for lbl, c in ring_counts.items() if c % 2 == 1]

    # ---- Stage 1: SAFE validity check; capture RDKit's *specific* error log ----
    buf = io.StringIO()
    with redirect_stderr(buf):
        m = Chem.MolFromSmiles(smi)          # sanitizing parser; never raises/crashes (local)
    log = _clean_log(buf.getvalue())          # e.g. "Explicit valence for atom # 14 C, 5, ..."
    if m is not None:
        return Diagnosis(smi, True, VALID, False, "Valid SMILES.", "")

    # ---- Stage 2: invalid. Parse-only to split syntactic vs semantic ----
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        if n_open != n_close:
            cat, hint = SYN_PARENS, (f"Unbalanced parentheses: {n_open} '(' vs {n_close} ')'.")
        elif n_lb != n_rb:
            cat, hint = SYN_BRACKETS, (f"Unbalanced square brackets: {n_lb} '[' vs {n_rb} ']'.")
        elif odd_rings:
            cat, hint = SYN_RING, (f"Unclosed ring: ring label(s) {sorted(odd_rings)} appear an "
                                   "odd number of times (each must occur exactly twice).")
        else:
            cat, hint = SYN_OTHER, ("Invalid SMILES syntax (an invalid token/atom symbol or a "
                                    "misplaced bond).")
        msg = f"{log}  {hint}".strip() if log else hint
        return Diagnosis(smi, False, cat, True, msg, log)

    # parseable skeleton but failed sanitization -> semantic error
    try:
        flags = Chem.SanitizeMol(mol, catchErrors=True)
    except Exception as e:
        return Diagnosis(smi, False, SEM_OTHER, False, (log or f"Sanitization error: {e}"), log)
    if flags & Chem.SanitizeFlags.SANITIZE_KEKULIZE:
        cat, hint = SEM_KEKULIZE, ("Aromaticity error: the listed aromatic atoms cannot be "
                                   "kekulized; fix the aromatic ring (e.g. an aromatic atom that "
                                   "should be aliphatic, or vice versa).")
    elif flags & Chem.SanitizeFlags.SANITIZE_PROPERTIES:
        cat, hint = SEM_VALENCE, ("Valence error: the named atom has more bonds than its element "
                                  "permits; remove an extra bond or adjust the atom.")
    else:
        cat, hint = SEM_OTHER, "Chemical sanitization failed."
    msg = f"{log}  {hint}".strip() if log else hint
    return Diagnosis(smi, False, cat, False, msg, log)


def is_valid(smi: str) -> bool:
    try:
        return Chem.MolFromSmiles((smi or "").strip()) is not None
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Rich feedback: translate RDKit's *atom-index* diagnosis into string-level
# pointers the LLM can actually act on (the key neuro-symbolic enhancement).
# --------------------------------------------------------------------------- #
_UNKEK_RE = re.compile(r"Unkekulized atoms:\s*([\d ]+)")
_VALENCE_RE = re.compile(r"atom #\s*(\d+)")
_ELEM_RE = re.compile(r"(Cl|Br|[A-Z][a-z]?|se|as|[bcnops])")


def _atom_spans(smi: str):
    """(char_pos, symbol) for each atom token, in SMILES parse order.
    The list index equals the RDKit atom index for a sanitize=False parse."""
    spans = []
    for m in _SMI_REGEX.finditer(smi):
        tok = m.group(1)
        if tok.startswith("["):
            inner = re.sub(r"^\d+", "", tok[1:-1])      # drop leading isotope
            em = _ELEM_RE.search(inner)
            spans.append((m.start(), em.group(1) if em else inner))
        elif tok[0].isalpha():
            spans.append((m.start(), tok))
    return spans


def rich_feedback(smi: str) -> str:
    """String-localized diagnosis for the LLM. Falls back to the base message."""
    d = diagnose(smi)
    if d.valid:
        return "Valid SMILES."
    text = d.message + "  " + (d.rdkit_log or "")
    spans = _atom_spans(smi)

    if d.category == SEM_KEKULIZE:
        m = _UNKEK_RE.search(text)
        if m:
            locs = []
            for i in (int(x) for x in m.group(1).split()):
                if 0 <= i < len(spans):
                    pos, sym = spans[i]
                    locs.append(f"'{sym}' at position {pos}")
            if locs:
                return ("Kekulization/aromaticity error: the aromatic atoms that cannot "
                        "be kekulized are " + ", ".join(locs) + " (position counts the "
                        "character index in the string). Fix the aromatic ring -- change "
                        "one of these aromatic (lowercase) atoms to its aliphatic (uppercase) "
                        "form, add a missing ring atom, or correct ring membership.")
    if d.category == SEM_VALENCE:
        m = _VALENCE_RE.search(text)
        if m:
            i = int(m.group(1))
            if 0 <= i < len(spans):
                pos, sym = spans[i]
                return (f"Valence error: atom '{sym}' at position {pos} (character index in "
                        "the string) has more bonds than its element permits. Remove one of "
                        "its bonds or change the atom so its valence is allowed.")
    return d.message


# --------------------------------------------------------------------------- #
# Deterministic rule-based repair (the non-neural baseline).
# --------------------------------------------------------------------------- #

def _balance_parens(smi: str) -> list[str]:
    """Return candidate repairs that balance parentheses."""
    cands = []
    n_open, n_close = smi.count("("), smi.count(")")
    if n_open > n_close:
        cands.append(smi + ")" * (n_open - n_close))          # close trailing
        # drop the last unmatched '(' (handles a stray trailing branch opener)
        s = smi
        for _ in range(n_open - n_close):
            idx = s.rfind("(")
            if idx >= 0:
                s = s[:idx] + s[idx + 1:]
        cands.append(s)
    elif n_close > n_open:
        cands.append("(" * (n_close - n_open) + smi)          # open leading
        # drop trailing extra ')'
        s = smi
        for _ in range(n_close - n_open):
            idx = s.rfind(")")
            if idx >= 0:
                s = s[:idx] + s[idx + 1:]
        cands.append(s)
    return cands


def _fix_rings(smi: str) -> list[str]:
    """Drop unmatched ring-closure labels (best effort)."""
    tokens = tokenize(smi)
    counts = _ring_digit_counts(tokens)
    odd = {lbl for lbl, c in counts.items() if c % 2 == 1}
    if not odd:
        return []
    out, seen = [], Counter()
    for t in tokens:
        if (t.isdigit() or t.startswith("%")) and t in odd:
            seen[t] += 1
            # drop the last (odd) occurrence
            if seen[t] == counts[t]:
                continue
        out.append(t)
    return ["".join(out)]


def rule_repair(smi: str, max_candidates: int = 8) -> Optional[str]:
    """Best-effort deterministic repair. Returns a valid SMILES or None.

    Handles the easy syntactic failures (paren balance, unmatched rings, simple
    bracket trimming). Semantic failures (valence, kekulization) are mostly out
    of reach for rules -- that gap is exactly what the LLM is meant to close."""
    smi = (smi or "").strip()
    if is_valid(smi):
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    candidates: list[str] = []
    candidates += _balance_parens(smi)
    candidates += _fix_rings(smi)
    # combine ring fix + paren balance
    for r in _fix_rings(smi):
        candidates += _balance_parens(r)
    # bracket trim: drop a stray unmatched bracket
    if smi.count("[") != smi.count("]"):
        candidates.append(smi.replace("[", "").replace("]", ""))

    seen = set()
    for c in candidates[: max_candidates * 3]:
        if c in seen:
            continue
        seen.add(c)
        if is_valid(c):
            return Chem.MolToSmiles(Chem.MolFromSmiles(c))
    return None


if __name__ == "__main__":
    # quick manual check
    for s in ["CCO", "c1ccccc1", "CC(C)(C)(C)C", "c1ccccc1(", "C1CCCCC", "CN(C)(C)(C)C"]:
        d = diagnose(s)
        print(f"{s:18} -> {d.category:16} valid={d.valid}  {d.message[:60]}")
