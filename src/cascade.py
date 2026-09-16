"""Cascade two methods into one deployed system: take the first method's repair
if it is valid, else fall back to the second. Writes a result file in the same
schema as evaluate.py so analyze.py/figures.py/stats.py pick it up.

Usage:
    python cascade.py --first rule --second Qwen2.5-14B-Instruct__iterative_rich \
        --tag cascade__14B_rich --results_dir results
"""
from __future__ import annotations

import argparse
import json
import os

from evaluate import summarize


def load(tag, rdir):
    return [json.loads(l) for l in open(os.path.join(rdir, f"{tag}.jsonl")) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default="rule")
    ap.add_argument("--second", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    a = load(args.first, args.results_dir)
    b = load(args.second, args.results_dir)
    bya = {r["original"]: r for r in a}
    out = []
    for rb in b:
        ra = bya.get(rb["original"])
        # prefer a valid repair from the first method (cheap/deterministic), else second
        if ra is not None and ra.get("valid"):
            rec = dict(ra)
        else:
            rec = dict(rb)
        rec["valid"] = bool((ra and ra.get("valid")) or rb.get("valid"))
        rec["condition"] = args.tag
        out.append(rec)

    with open(os.path.join(args.results_dir, f"{args.tag}.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    summary = summarize(out, args.tag)
    with open(os.path.join(args.results_dir, f"{args.tag}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"cascade {args.first} -> {args.second}: repair {summary['repair_rate']*100:.1f}% "
          f"(syn {(summary.get('syntactic_repair_rate') or 0)*100:.0f}%, "
          f"sem {(summary.get('semantic_repair_rate') or 0)*100:.0f}%)")


if __name__ == "__main__":
    main()
