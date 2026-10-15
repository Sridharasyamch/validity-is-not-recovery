"""ACS Table-of-Contents graphic: the validity/recovery inversion in one panel.

Sized to the ACS limit (3.25 x 1.75 in) and drawn from the result files, so it
cannot disagree with the tables. Writes a vector PDF for the manuscript and a
300 dpi TIFF for the submission system.
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results_recovery"
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})

def agg(tag):
    return json.load(open(f"{R}/{tag}.summary.json"))

PTS = [
    ("EnumRank (this work)", "TEST_enumrank_domainlm_prune", "#1b6ca8", "o", 55, (-2, 11)),
    ("supervised",           "TEST_seq2seq_supervised",      "#e07b39", "s", 34, (0, 10)),
    ("free LLM loop",        "TEST_looprich_k5_minedit_s0",  "#9aa0a6", "P", 34, (0, 10)),
    ("SmiSelf",              "smiself",                      "#c0392b", "*", 90, (0, -11)),
]

fig, ax = plt.subplots(figsize=(3.25, 1.75))
for lab, tag, c, m, s, off in PTS:
    a = agg(tag)
    x, y = a["validity"] * 100, a["exact_recovery"] * 100
    ax.scatter(x, y, c=c, marker=m, s=s, zorder=3, edgecolor="white", linewidth=0.5)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=off,
                fontsize=6.5, color=c, ha="center", va="center", linespacing=0.95)

ax.set_xlim(34, 118); ax.set_ylim(-14, 100)
ax.set_xlabel("validity (%)", labelpad=1.5)
ax.set_ylabel("exact recovery (%)", labelpad=1.5)
ax.set_xticks([50, 75, 100]); ax.set_yticks([0, 40, 80])
ax.tick_params(length=2, pad=1.5)
ax.text(0.015, 0.96, "validity is not recovery", transform=ax.transAxes,
        fontsize=7.5, color="0.25", ha="left", va="top")
fig.subplots_adjust(left=0.135, right=0.995, top=0.97, bottom=0.21)

os.makedirs("paper_vnr/figures", exist_ok=True)
fig.savefig("paper_vnr/figures/toc_graphic.pdf", bbox_inches=None)
tif = "paper_vnr/figures/TOC_graphic.tif"
try:
    fig.savefig(tif, dpi=300, bbox_inches=None, pil_kwargs={"compression": "tiff_lzw"})
except Exception:
    tif = tif.replace(".tif", ".png"); fig.savefig(tif, dpi=300, bbox_inches=None)
if tif.endswith(".tif"):                      # ACS wants RGB, not RGBA
    from PIL import Image
    _im = Image.open(tif)
    if _im.mode != "RGB":
        _bg = Image.new("RGB", _im.size, (255, 255, 255))
        _bg.paste(_im, mask=_im.split()[-1])
        _bg.save(tif, dpi=(300, 300), compression="tiff_lzw")

w, h = fig.get_size_inches()
print(f"  toc_graphic.pdf + {os.path.basename(tif)}  {w:.2f} x {h:.2f} in (ACS max 3.25 x 1.75)")
