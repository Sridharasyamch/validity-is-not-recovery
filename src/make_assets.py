"""Generate extra paper figures (local, no GPU):
 1. molecule renderings of example repairs (RDKit)
 2. benchmark error-category distribution
 3. QED distribution: repaired molecules vs native valid generations
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, QED
RDLogger.DisableLog("rdApp.*")
plt.rcParams.update({"font.size": 11, "savefig.bbox": "tight", "savefig.dpi": 160})

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

# ---- 1. molecule renderings of repaired examples ----
EX = [
    ("c1cncc(F)C2CNC2c1", "Valence fix\n(F split to own branch)"),
    ("Oc1ccnnc1-n1ccnc1", "Kekulization fix\n(de-aromatized)"),
    ("CC1(C)CO1", "Ring fix\n(closed ring 1)"),
    ("CC(=NNC(=N)S)", "Parenthesis fix\n(added ')')"),
    ("CCN(CC)[CH]", "Bracket fix\n(closed '[CH]')"),
    ("COC1CCC(Oc2cnc3cc(Cl)ccc3c2)CC1O", "Recovered drug-like\nmolecule (QED 0.95)"),
]
mols, legs = [], []
for smi, lab in EX:
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        mols.append(m); legs.append(lab)
img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(260, 200), legends=legs)
img.save(os.path.join(OUT, "mol_examples.png"))
print("wrote mol_examples.png")

# ---- 2. benchmark category distribution ----
# IMPORTANT: use the SAME per-item data the results are computed on (the 4,009
# T=1.0 primary benchmark), NOT data/benchmark/stats.json which is the pooled
# T in {1.0,1.2,1.5} harvest (34,353 invalids). This keeps Fig. 2 consistent
# with Table I and every reported number.
from collections import Counter as _Counter
_rows = [json.loads(_l) for _l in open("results/rule.jsonl") if _l.strip()]
dist = dict(_Counter(r["category"] for r in _rows))
assert sum(dist.values()) == 4009, f"expected 4009 T=1.0 items, got {sum(dist.values())}"
label_map = {"semantic_kekulize": "Kekulization (sem)", "syntax_ring": "Unclosed ring (syn)",
             "semantic_valence": "Valence (sem)", "syntax_other": "Other (syn)",
             "syntax_parens": "Parentheses (syn)", "syntax_brackets": "Brackets (syn)"}
items = sorted(dist.items(), key=lambda x: -x[1])
labels = [label_map.get(k, k) for k, _ in items]
vals = [v for _, v in items]
colors = ["#d1495b" if "sem" in label_map.get(k, "") else "#3a7" for k, _ in items]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v + 30, i, f"{v} ({100*v/sum(vals):.0f}%)", va="center", fontsize=9)
ax.set_xlabel("Count"); ax.set_xlim(0, max(vals) * 1.25)
ax.set_title("Benchmark composition: real invalid SMILES by error type ($T{=}1.0$)")
fig.text(0.62, 0.30, "red = semantic\ngreen = syntactic", fontsize=9)
fig.savefig(os.path.join(OUT, "benchmark_dist.pdf")); plt.close(fig)
print("wrote benchmark_dist.pdf")

# ---- 3. QED distribution: repaired vs native valid generations ----
rep = []
for l in open("results/cascade__14B_best.jsonl"):
    r = json.loads(l)
    if r.get("valid") and r.get("qed") is not None:
        rep.append(r["qed"])
native = []
for s in (x.strip() for x in open("runs/gen_t1.0.smiles")):
    m = Chem.MolFromSmiles(s) if s else None
    if m is not None:
        try:
            native.append(QED.qed(m))
        except Exception:
            pass
    if len(native) >= 3000:
        break
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(native, bins=30, density=True, alpha=0.55, label=f"Native valid generations (n={len(native)})", color="#888")
ax.hist(rep, bins=30, density=True, alpha=0.6, label=f"Repaired molecules (n={len(rep)})", color="#3a7")
ax.set_xlabel("QED (drug-likeness)"); ax.set_ylabel("Density")
ax.set_title("Repaired molecules match native drug-likeness"); ax.legend(fontsize=9)
ax.axvline(np.mean(native), color="#555", ls="--", lw=1); ax.axvline(np.mean(rep), color="#274", ls="--", lw=1)
fig.savefig(os.path.join(OUT, "qed_dist.pdf")); plt.close(fig)
print(f"wrote qed_dist.pdf (native QED {np.mean(native):.3f}, repaired {np.mean(rep):.3f})")
