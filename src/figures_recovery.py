"""Figures for the 'Validity is not Recovery' paper.

Every number is read from the result files in results_recovery/ so the figures
cannot drift from the tables.
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "results_recovery"
OUT = "paper/figures_recovery"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    # Type 3 fonts are rejected by IEEE PDF eXpress; 42 embeds TrueType
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False,
})

C_OURS = "#1b6ca8"
C_SUP = "#e07b39"
C_LLM = "#7a9e3f"
C_REF = "#9aa0a6"
C_PRIOR = "#c0392b"


def rows(tag):
    return [json.loads(l) for l in open(f"{R}/{tag}.jsonl") if l.strip()]


def agg(tag):
    d = json.load(open(f"{R}/{tag}.summary.json"))
    return d


def fig_validity_recovery_plane():
    """THE headline figure: validity and recovery are close to independent."""
    pts = [
        ("Ours (enumerate+prune+rank)", "TEST_enumrank_domainlm_prune", C_OURS, "o", 70),
        ("Supervised seq2seq", "TEST_seq2seq_supervised", C_SUP, "s", 55),
        ("Enumerate + LLM rank", "TEST_enumrank_llm_prune", C_LLM, "^", 55),
        ("Enumerate + shortest", "TEST_enum_shortest", C_REF, "v", 45),
        ("Enumerate + uniform", "TEST_enum_uniform", C_REF, "D", 45),
        ("Free LLM loop", "TEST_looprich_k5_minedit_s0", C_REF, "P", 45),
        ("Rule-based", "rule", C_REF, "X", 45),
        ("SmiSelf", "smiself", C_PRIOR, "*", 140),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    # iso-lines of constant valid-outputs-per-correct-recovery, labelled in-axes
    xs = np.linspace(0.001, 1.0, 200)
    for ratio, lab in ((1, "1:1"), (3, "3:1"), (10, "10:1"), (75, "75:1")):
        ax.plot(xs, xs / ratio, color="0.88", lw=0.8, zorder=0)
        xa = 0.86
        ya = xa / ratio
        if ya < 0.86:
            ax.text(xa, ya + 0.012, lab, color="0.62", fontsize=6.5,
                    rotation=np.degrees(np.arctan2(1.0 / ratio, 1.0)) * 0.62,
                    ha="center", va="bottom")
    for lab, tag, c, m, s_ in pts:
        a = agg(tag)
        ax.scatter(a["validity"], a["exact_recovery"], c=c, marker=m, s=s_,
                   zorder=3, edgecolor="white", linewidth=0.6, label=lab)
    ax.set_xlabel("Validity  (emits a parseable molecule)")
    ax.set_ylabel("Exact recovery  (emits the intended molecule)")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.04, 0.95)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.32)

    # annotate the two systems whose ordering depends on which axis you read
    a_ss = agg("smiself")
    a_ll = agg("TEST_looprich_k5_minedit_s0")
    ax.annotate(f"SmiSelf\n{a_ss['validity']*100:.1f}% valid, "
                f"{a_ss['exact_recovery']*100:.1f}% recovered",
                xy=(a_ss["validity"], a_ss["exact_recovery"]),
                xytext=(a_ss["validity"] - 0.02, 0.20), fontsize=6.8,
                color="0.3", ha="center",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7))
    ax.annotate(f"Free LLM loop\n{a_ll['validity']*100:.1f}% valid, "
                f"{a_ll['exact_recovery']*100:.1f}% recovered",
                xy=(a_ll["validity"], a_ll["exact_recovery"]),
                xytext=(a_ll["validity"] - 0.03, 0.30), fontsize=6.8,
                color="0.3", ha="center",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7))
    fig.savefig(f"{OUT}/fig_validity_recovery_plane.pdf")
    plt.close(fig)
    print("  fig_validity_recovery_plane")


def fig_ranking_ladder():
    """Where the improvement actually comes from."""
    steps = [("Free LLM\nloop", "TEST_looprich_k5_minedit_s0", C_REF),
             ("Enumerate\n+ uniform", "TEST_enum_uniform", C_REF),
             ("+ shortest\ncanonical", "TEST_enum_shortest", C_REF),
             ("+ general LLM\nrank", "TEST_enumrank_llm_prune", C_LLM),
             ("+ domain\nTransformer", "TEST_enumrank_tlm_prune", C_SUP),
             ("+ domain\nchar-LSTM (ours)", "TEST_enumrank_domainlm_prune", C_OURS)]
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    xs = np.arange(len(steps))
    rec = [agg(t)["exact_recovery"] for _, t, _ in steps]
    val = [agg(t)["validity"] for _, t, _ in steps]
    ax.bar(xs, rec, color=[c for _, _, c in steps], width=0.62, zorder=2)
    ax.plot(xs, val, color="0.35", marker="o", ms=3.5, lw=1.0, ls="--",
            zorder=3, label="validity")
    for x, r in zip(xs, rec):
        ax.text(x, r + 0.018, f"{r*100:.1f}%", ha="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels([s for s, _, _ in steps], fontsize=7)
    ax.set_ylabel("exact recovery")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(f"{OUT}/fig_ranking_ladder.pdf")
    plt.close(fig)
    print("  fig_ranking_ladder")


def fig_candidate_space():
    """The repair space is small, and it almost always contains the answer."""
    cands = [json.loads(l) for l in open("data/recovery/cands_test.jsonl")]
    sizes = np.array([len(c["candidates"]) for c in cands])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.5))
    a1.hist(np.clip(sizes, 0, 300), bins=60, color=C_OURS, alpha=0.85)
    a1.axvline(np.median(sizes), color="0.25", ls="--", lw=1.0)
    a1.text(np.median(sizes) + 8, a1.get_ylim()[1] * 0.85,
            f"median {int(np.median(sizes))}", fontsize=7.5, color="0.25")
    a1.set_xlabel("valid molecules within one token edit")
    a1.set_ylabel("benchmark items")

    cats, cov, unif = [], [], []
    by = {}
    for c in cands:
        by.setdefault(c.get("category"), []).append(c)
    order = sorted(by, key=lambda k: -len(by[k]))
    for k in order:
        sub = by[k]
        cats.append(k.replace("semantic_", "sem:").replace("syntax_", "syn:"))
        cov.append(sum(1 for r in sub if r.get("truth_in_candidates")) / len(sub))
        unif.append(sum(1.0 / max(len(r["candidates"]), 1) for r in sub
                        if r.get("truth_in_candidates")) / len(sub))
    y = np.arange(len(cats))
    a2.barh(y, cov, color=C_OURS, alpha=0.85, label="truth in candidate set")
    a2.barh(y, unif, color=C_PRIOR, alpha=0.9, height=0.45,
            label="uniform choice")
    a2.set_yticks(y); a2.set_yticklabels(cats, fontsize=7)
    a2.invert_yaxis(); a2.set_xlim(0, 1.05)
    a2.set_xlabel("fraction")
    # the legend must sit outside the axes: its swatches are the same colours as
    # the bars, so inside it reads as bar and hides the one category (brackets)
    # whose coverage is not ~1
    a2.legend(frameon=False, fontsize=6.8, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=2, columnspacing=1.2,
              handletextpad=0.4)
    fig.savefig(f"{OUT}/fig_candidate_space.pdf")
    plt.close(fig)
    print("  fig_candidate_space")


def fig_per_category():
    systems = [("SmiSelf", "smiself", C_PRIOR),
               ("Free LLM loop", "TEST_looprich_k5_minedit_s0", C_REF),
               ("Supervised seq2seq", "TEST_seq2seq_supervised", C_SUP),
               ("Ours", "TEST_enumrank_domainlm_prune", C_OURS)]
    base = agg("TEST_enumrank_domainlm_prune")["per_category"]
    cats = sorted(base, key=lambda k: -base[k]["n"])
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    w = 0.2
    xs = np.arange(len(cats))
    for i, (lab, tag, c) in enumerate(systems):
        pc = agg(tag)["per_category"]
        vals = [pc.get(k, {}).get("exact_recovery", 0) or 0 for k in cats]
        ax.bar(xs + (i - 1.5) * w, vals, width=w, color=c, label=lab, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k.replace('semantic_','sem:').replace('syntax_','syn:')}\n"
                        f"n={base[k]['n']}" for k in cats], fontsize=7)
    ax.set_ylabel("exact recovery")
    ax.set_ylim(0, 1.18)
    # legend above the axes so it cannot sit on top of a bar
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), fontsize=7, columnspacing=1.1,
              handletextpad=0.4)
    fig.savefig(f"{OUT}/fig_per_category.pdf")
    plt.close(fig)
    print("  fig_per_category")


def fig_risk_coverage():
    rr = rows("TEST_enumrank_domainlm_prune")
    sc = sorted([(r["margin"], r["exact_recovery"]) for r in rr
                 if r.get("margin") is not None], key=lambda t: -t[0])
    n = len(sc)
    covs = np.linspace(0.02, 1.0, 60)
    prec = []
    for f in covs:
        k = max(1, int(n * f))
        prec.append(sum(1 for _, h in sc[:k] if h) / k)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(covs * 100, np.array(prec) * 100, color=C_OURS, lw=1.6)
    full = sum(1 for r in rr if r["exact_recovery"]) / len(rr)
    ax.axhline(full * 100, color="0.55", ls="--", lw=0.9)
    ax.text(4, full * 100 + 1.2, f"no abstention ({full*100:.1f}%)",
            fontsize=7, color="0.4")
    ax.set_xlabel("coverage (% of inputs answered)")
    ax.set_ylabel("precision (% exactly recovered)")
    ax.set_ylim(70, 102)
    fig.savefig(f"{OUT}/fig_risk_coverage.pdf")
    plt.close(fig)
    print("  fig_risk_coverage")


if __name__ == "__main__":
    which = sys.argv[1:] or ["plane", "ladder", "space", "cat", "risk"]
    if "plane" in which: fig_validity_recovery_plane()
    if "ladder" in which: fig_ranking_ladder()
    if "space" in which: fig_candidate_space()
    if "cat" in which: fig_per_category()
    if "risk" in which: fig_risk_coverage()
    print(f"figures -> {OUT}/")
