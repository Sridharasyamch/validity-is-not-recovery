"""Neuro-symbolic repair harness.

Three repair conditions, sharing one LLM:
  - "no_feedback"  : ablation. Tell the model the SMILES is invalid; ask for a fix.
  - "feedback"     : single-shot neuro-symbolic. Include the RDKit diagnosis.
  - "iterative"    : the full loop. Repair -> re-diagnose -> re-prompt with the
                     new error, up to K rounds. This is the proposed method.

The LLM is reached through an OpenAI-compatible endpoint, so the same code drives
a local vLLM server (open models on Thunder) or any hosted API.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

import symbolic as S

SYSTEM = (
    "You are an expert cheminformatician and a precise SMILES-repair engine. "
    "You are given an INVALID SMILES string produced by a generative model. "
    "Return a corrected, VALID SMILES that stays as close as possible to the "
    "original (make the smallest change that fixes it). "
    "Respond with ONLY the corrected SMILES string on a single line, no prose, "
    "no code fences, no explanation."
)

PROMPT_NO_FEEDBACK = (
    "The following SMILES string is invalid. Output a corrected, valid SMILES "
    "that is as close as possible to the original.\n\nINVALID SMILES: {smi}\n\n"
    "Corrected SMILES:"
)

PROMPT_FEEDBACK = (
    "The following SMILES string is invalid. A chemistry checker (RDKit) reports "
    "the specific problem below. Use it to make the smallest fix that yields a "
    "valid SMILES, staying as close as possible to the original.\n\n"
    "INVALID SMILES: {smi}\n"
    "CHECKER DIAGNOSIS: {diag}\n\n"
    "Corrected SMILES:"
)

# defensive extraction of a SMILES from a possibly-chatty model reply
_FENCE = re.compile(r"```[a-zA-Z]*")
_SMILES_CHARS = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$:/\\.%]+")


def extract_smiles(text: str) -> str:
    if not text:
        return ""
    t = _FENCE.sub("", text).strip()
    # drop common prefixes
    for line in reversed([l.strip() for l in t.splitlines() if l.strip()]):
        line = re.sub(r"(?i)^(corrected\s+smiles|smiles|answer)\s*[:=]\s*", "", line).strip()
        m = _SMILES_CHARS.findall(line)
        if m:
            # the longest contiguous SMILES-like token on the line
            return max(m, key=len)
    return t.split()[0] if t.split() else ""


@dataclass
class RepairResult:
    original: str
    repaired: Optional[str]
    valid: bool
    condition: str
    iterations: int
    trace: list = field(default_factory=list)   # list of (attempt, diagnosis)
    raw_replies: list = field(default_factory=list)
    latency_s: float = 0.0
    error: Optional[str] = None


class LLMRepairer:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 temperature: float = 0.0, max_tokens: int = 160, timeout: float = 60.0,
                 retries: int = 3, n_samples: int = 1):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.n_samples = max(1, n_samples)
        # pass@k needs diversity: use a sampling temperature when drawing >1 candidate
        self.temperature = temperature if (self.n_samples == 1) else max(temperature, 0.7)
        self.max_tokens = max_tokens
        self.retries = retries

    def _chat(self, prompt: str) -> list[str]:
        """Return up to n_samples candidate replies for one prompt."""
        last = None
        for attempt in range(self.retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=self.n_samples,
                )
                return [c.message.content or "" for c in r.choices]
            except Exception as e:  # network / server hiccup
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.retries} retries: {last}")

    def repair(self, smi: str, condition: str = "iterative", max_iters: int = 3) -> RepairResult:
        """Unified repair loop. Condition flags:
        - 'no_feedback'  : no diagnosis; 'feedback'/'iterative' : RDKit diagnosis
        - add '_rich'    : use string-localized feedback (symbolic.rich_feedback)
        - n_samples>1    : pass@k -- accept the first valid of k candidates / round
        Iteration is enabled for conditions starting with 'iterative'."""
        t0 = time.time()
        use_feedback = not condition.startswith("no_feedback")
        use_rich = "rich" in condition
        iterate = condition.startswith("iterative")
        K = max_iters if iterate else 1
        res = RepairResult(original=smi, repaired=None, valid=False,
                           condition=condition, iterations=0)
        try:
            cur = smi
            for it in range(1, K + 1):
                d = S.diagnose(cur)
                if d.valid:
                    res.repaired = cur
                    res.valid = True
                    res.iterations = it - 1
                    break
                if not use_feedback:
                    prompt = PROMPT_NO_FEEDBACK.format(smi=cur)
                    diag = "(none)"
                else:
                    diag = S.rich_feedback(cur) if use_rich else d.message
                    prompt = PROMPT_FEEDBACK.format(smi=cur, diag=diag)
                cands = self._chat(prompt)
                res.raw_replies.extend(cands)
                res.trace.append((cur, diag))
                res.iterations = it
                # pass@k: take the first candidate that validates, else the first
                picked = None
                for c in cands:
                    e = extract_smiles(c)
                    if S.is_valid(e):
                        picked = e
                        break
                cur = picked if picked is not None else extract_smiles(cands[0])
                if S.is_valid(cur):
                    res.repaired = cur
                    res.valid = True
                    break
            if res.repaired is None:
                res.repaired = cur
                res.valid = S.is_valid(cur)
        except Exception as e:
            res.error = str(e)
        res.latency_s = time.time() - t0
        return res


if __name__ == "__main__":
    # offline check of the parser (no server needed)
    tests = [
        "CCO",
        "```\nCC(=O)Oc1ccccc1C(=O)O\n```",
        "Corrected SMILES: c1ccccc1",
        "The answer is CN1C=NC2=C1C(=O)N(C(=O)N2C)C which is caffeine.",
        "Here you go:\nCCN(CC)CC",
    ]
    for t in tests:
        print(repr(t[:40]), "->", extract_smiles(t))
