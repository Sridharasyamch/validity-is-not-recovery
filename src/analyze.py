"""Aggregate results/*.summary.json into paper-ready tables (runs on laptop)."""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="paper")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    cat_rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.summary.json"))):
        s = json.load(open(path))
        model, _, cond = s["tag"].partition("__")
        rows.append({
            "model": model, "condition": cond or model,
            "repair_rate": s["repair_rate"],
            "syntactic": s.get("syntactic_repair_rate"),
            "semantic": s.get("semantic_repair_rate"),
            "tok_edit": s.get("mean_token_edit_fixed"),
            "qed": s.get("mean_qed_fixed"),
            "iters": s.get("mean_iterations"),
            "lat_s": s.get("mean_latency_s"),
            "n": s["n"],
        })
        for cat, v in s.get("per_category", {}).items():
            cat_rows.append({"tag": s["tag"], "category": cat,
                             "n": v["n"], "repair_rate": v["repair_rate"]})

    df = pd.DataFrame(rows).sort_values(["model", "condition"])
    cat = pd.DataFrame(cat_rows)

    df.to_csv(os.path.join(args.out_dir, "main_results.csv"), index=False)
    if not cat.empty:
        pivot = cat.pivot_table(index="category", columns="tag",
                                values="repair_rate", aggfunc="first")
        pivot.to_csv(os.path.join(args.out_dir, "per_category.csv"))

    print("=== MAIN RESULTS (repair rate by model x condition) ===")
    print(df.to_string(index=False))
    with open(os.path.join(args.out_dir, "main_results.md"), "w") as f:
        f.write(df.to_markdown(index=False))
    print(f"\nwrote {args.out_dir}/main_results.csv, main_results.md, per_category.csv")


if __name__ == "__main__":
    main()
