"""Generate paper figures from results/ (runs on laptop; small data)."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})

COND_ORDER = ["rule", "no_feedback", "feedback", "iterative",
              "iterative_rich", "iterative_rich_pass5"]
COND_LABEL = {"rule": "Rule-based", "no_feedback": "LLM (no feedback)",
              "feedback": "LLM + symbolic", "iterative": "Neuro-symbolic loop",
              "iterative_rich": "+ rich feedback", "iterative_rich_pass5": "+ pass@5"}


def load_summaries(rdir):
    out = {}
    for p in glob.glob(os.path.join(rdir, "*.summary.json")):
        s = json.load(open(p))
        out[s["tag"]] = s
    return out


def load_items(rdir, tag):
    p = os.path.join(rdir, f"{tag}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def fig_main(summ, outdir):
    """Grouped bar: repair rate by condition, per model (+ rule baseline)."""
    models = sorted({t.split("__")[0] for t in summ if "__" in t})
    conds = [c for c in COND_ORDER if c != "rule"]
    x = np.arange(len(models)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, c in enumerate(conds):
        vals = [summ.get(f"{m}__{c}", {}).get("repair_rate", 0) for m in models]
        ax.bar(x + (j - 1) * w, vals, w, label=COND_LABEL[c])
    if "rule" in summ:
        ax.axhline(summ["rule"]["repair_rate"], ls="--", color="gray",
                   label=f"Rule-based ({summ['rule']['repair_rate']:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=10)
    ax.set_ylabel("Repair rate (validity)"); ax.set_ylim(0, 1)
    ax.set_title("SMILES repair rate by method"); ax.legend(fontsize=9)
    fig.savefig(os.path.join(outdir, "fig_main_repair_rate.pdf"))
    plt.close(fig)


def fig_syntactic_semantic(summ, outdir):
    models = sorted({t.split("__")[0] for t in summ if "__" in t})
    conds = [c for c in COND_ORDER if c != "rule"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, title in zip(axes, ["syntactic_repair_rate", "semantic_repair_rate"],
                              ["Syntactic errors", "Semantic errors"]):
        x = np.arange(len(models)); w = 0.25
        for j, c in enumerate(conds):
            vals = [summ.get(f"{m}__{c}", {}).get(key) or 0 for m in models]
            ax.bar(x + (j - 1) * w, vals, w, label=COND_LABEL[c])
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=10)
        ax.set_title(title); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Repair rate"); axes[1].legend(fontsize=9)
    fig.suptitle("Repair rate by error type")
    fig.savefig(os.path.join(outdir, "fig_syntactic_semantic.pdf"))
    plt.close(fig)


def fig_iteration_curve(rdir, outdir):
    """Cumulative repair rate vs iteration for the iterative condition."""
    fig, ax = plt.subplots(figsize=(6, 4)); plotted = False
    for p in glob.glob(os.path.join(rdir, "*__iterative.jsonl")):
        tag = os.path.basename(p).replace(".jsonl", "")
        items = [json.loads(l) for l in open(p) if l.strip()]
        if not items:
            continue
        maxit = max((it.get("iterations", 0) for it in items), default=0)
        ks = list(range(1, max(maxit, 1) + 1))
        cum = [np.mean([it["valid"] and it.get("iterations", 99) <= k for it in items]) for k in ks]
        ax.plot(ks, cum, marker="o", label=tag.split("__")[0]); plotted = True
    if plotted:
        ax.set_xlabel("Iteration"); ax.set_ylabel("Cumulative repair rate")
        ax.set_title("Repair rate vs. neuro-symbolic loop iterations")
        ax.set_ylim(0, 1); ax.legend(fontsize=9)
        fig.savefig(os.path.join(outdir, "fig_iteration_curve.pdf"))
    plt.close(fig)


def fig_edit_distance(rdir, outdir):
    """Token-edit-distance distribution for successful repairs (faithfulness)."""
    fig, ax = plt.subplots(figsize=(6, 4)); plotted = False
    for c in ["no_feedback", "feedback", "iterative"]:
        edits = []
        for p in glob.glob(os.path.join(rdir, f"*__{c}.jsonl")):
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    if r.get("valid") and r.get("token_edit") is not None:
                        edits.append(r["token_edit"])
        if edits:
            ax.hist(edits, bins=range(0, 20), alpha=0.5, label=COND_LABEL.get(c, c), density=True)
            plotted = True
    if plotted:
        ax.set_xlabel("Token edit distance (input - repair)")
        ax.set_ylabel("Density"); ax.set_title("Repair faithfulness"); ax.legend(fontsize=9)
        fig.savefig(os.path.join(outdir, "fig_edit_distance.pdf"))
    plt.close(fig)


def fig_progression(summ, outdir):
    """The headline climb: each lever stacked, for the 14B model + cascade."""
    steps = [
        ("rule", "Rule-based"),
        ("Qwen2.5-14B-Instruct__no_feedback", "LLM\n(no feedback)"),
        ("Qwen2.5-14B-Instruct__feedback", "+ symbolic\nfeedback"),
        ("Qwen2.5-14B-Instruct__iterative", "+ iteration\n(loop)"),
        ("Qwen2.5-14B-Instruct__iterative_rich_pass5", "+ pass@5"),
        ("cascade__14B_best", "+ rule\ncascade"),
    ]
    xs, ys = [], []
    for tag, lab in steps:
        if tag in summ:
            xs.append(lab); ys.append(summ[tag]["repair_rate"] * 100)
    if not ys:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(range(len(ys)), ys, color="#3a7", edgecolor="black", linewidth=0.5)
    bars[0].set_color("#999")
    for i, v in enumerate(ys):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=9)
    ax.set_ylabel("Repair rate (%)"); ax.set_ylim(0, max(ys) * 1.18)
    ax.set_title("Raising the repair rate, lever by lever (14B model)")
    fig.savefig(os.path.join(outdir, "fig_progression.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="paper/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summ = load_summaries(args.results_dir)
    if not summ:
        print("no summaries found yet"); return
    fig_main(summ, args.out_dir)
    fig_progression(summ, args.out_dir)
    fig_syntactic_semantic(summ, args.out_dir)
    fig_iteration_curve(args.results_dir, args.out_dir)
    fig_edit_distance(args.results_dir, args.out_dir)
    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
