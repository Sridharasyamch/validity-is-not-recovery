"""Statistical significance for repair-rate comparisons (runs on laptop).

- Wilson 95% confidence intervals on each condition's repair rate.
- McNemar paired tests between conditions evaluated on the SAME benchmark items.
"""
from __future__ import annotations

import json
import math
import os

from scipy.stats import binomtest


def load_valid(tag, results_dir="results"):
    """Return ordered list of (original, valid_bool) for a condition."""
    p = os.path.join(results_dir, f"{tag}.jsonl")
    out = []
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            out.append((r["original"], bool(r["valid"])))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, center - half, center + half)


def mcnemar(a, b):
    """Exact McNemar (binomial) on paired binary outcomes a,b (aligned)."""
    assert len(a) == len(b)
    b_only = sum(1 for (_, x), (_, y) in zip(a, b) if x and not y)   # a fixes, b doesn't
    c_only = sum(1 for (_, x), (_, y) in zip(a, b) if y and not x)   # b fixes, a doesn't
    n = b_only + c_only
    if n == 0:
        return b_only, c_only, 1.0
    p = binomtest(b_only, n, 0.5).pvalue
    return b_only, c_only, p


def main():
    rd = "results"
    conds = {
        "rule": "rule",
        "7B no-feedback": "Qwen2.5-7B-Instruct__no_feedback",
        "7B +symbolic": "Qwen2.5-7B-Instruct__feedback",
        "7B loop": "Qwen2.5-7B-Instruct__iterative",
        "14B no-feedback": "Qwen2.5-14B-Instruct__no_feedback",
        "14B +symbolic": "Qwen2.5-14B-Instruct__feedback",
        "14B loop": "Qwen2.5-14B-Instruct__iterative",
        "14B loop+rich": "Qwen2.5-14B-Instruct__iterative_rich",
        "14B loop+pass5": "Qwen2.5-14B-Instruct__iterative_rich_pass5",
        "cascade(best)": "cascade__14B_best",
    }
    data = {name: load_valid(tag, rd) for name, tag in conds.items()}

    print("=== Repair rate with Wilson 95% CI ===")
    for name, rows in data.items():
        k = sum(v for _, v in rows); n = len(rows)
        p, lo, hi = wilson(k, n)
        print(f"  {name:18} {p*100:5.1f}%  [{lo*100:4.1f}, {hi*100:4.1f}]   ({k}/{n})")

    print("\n=== McNemar paired tests (b=left-only fixes, c=right-only fixes) ===")
    pairs = [
        ("14B no-feedback", "14B +symbolic"),
        ("14B +symbolic", "14B loop"),
        ("14B loop", "14B loop+rich"),
        ("14B loop+rich", "14B loop+pass5"),
        ("14B loop+pass5", "cascade(best)"),
        ("rule", "cascade(best)"),
        ("7B loop", "14B loop"),
    ]
    for left, right in pairs:
        b, c, pval = mcnemar(data[left], data[right])
        sig = "***" if pval < 1e-3 else "**" if pval < 1e-2 else "*" if pval < 5e-2 else "ns"
        print(f"  {left:16} vs {right:16}  b={b:4} c={c:4}  p={pval:.2e}  {sig}")


if __name__ == "__main__":
    main()
