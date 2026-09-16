"""Extended cheminformatics evaluation of the repaired set vs references.
Reports: MW, LogP, SA score, QED, Lipinski Ro5 compliance, uniqueness, novelty,
internal diversity, scaffold diversity, and Frechet ChemNet Distance (FCD)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import RDConfig
RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa

def load_smiles_from_jsonl(path, key, valid_only=True):
    out = []
    for l in open(path):
        r = json.loads(l)
        s = r.get(key)
        if s and (not valid_only or r.get("valid")):
            out.append(s)
    return out

def clean(smiles, cap=None):
    ms = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            ms.append(m)
        if cap and len(ms) >= cap:
            break
    return ms

def ro5_violations(m):
    v = 0
    if Descriptors.MolWt(m) > 500: v += 1
    if Descriptors.MolLogP(m) > 5: v += 1
    if Descriptors.NumHDonors(m) > 5: v += 1
    if Descriptors.NumHAcceptors(m) > 10: v += 1
    return v

def scaffolds(mols):
    sc = set()
    for m in mols:
        try:
            sc.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            pass
    return sc

def internal_diversity(mols, sample=2000, seed=0):
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in mols]
    rng = np.random.default_rng(seed)
    if len(fps) > sample:
        idx = rng.choice(len(fps), sample, replace=False); fps = [fps[i] for i in idx]
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    return 1 - np.mean(sims)

def profile(mols, name, train_canon=None):
    mw = [Descriptors.MolWt(m) for m in mols]
    logp = [Descriptors.MolLogP(m) for m in mols]
    sa = [sascorer.calculateScore(m) for m in mols]
    qed = [QED.qed(m) for m in mols]
    ro5 = [ro5_violations(m) for m in mols]
    cans = [Chem.MolToSmiles(m) for m in mols]
    sc = scaffolds(mols)
    d = {
        "name": name, "n": len(mols),
        "MW_mean": round(np.mean(mw), 1), "MW_sd": round(np.std(mw), 1),
        "LogP_mean": round(np.mean(logp), 2), "LogP_sd": round(np.std(logp), 2),
        "SA_mean": round(np.mean(sa), 2), "SA_sd": round(np.std(sa), 2),
        "QED_mean": round(np.mean(qed), 3),
        "Lipinski_pass_pct": round(100 * np.mean([v <= 1 for v in ro5]), 1),
        "uniqueness": round(len(set(cans)) / len(cans), 3),
        "internal_diversity": round(float(internal_diversity(mols)), 3),
        "n_scaffolds": len(sc),
        "scaffold_diversity": round(len(sc) / len(mols), 3),
    }
    if train_canon is not None:
        d["novelty_vs_train"] = round(float(np.mean([c not in train_canon for c in cans])), 3)
    return d

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    repaired = clean(load_smiles_from_jsonl("results/cascade__14B_best.jsonl", "repaired"))
    native = clean([l.strip() for l in open("runs/gen_t1.0.smiles") if l.strip()][:20000], cap=3000)
    chembl = clean([l.strip() for l in open("data/guacamol_train.smiles")][:20000], cap=3000)
    train_canon = set(Chem.MolToSmiles(m) for m in clean(
        [l.strip() for l in open("data/guacamol_train.smiles")][:60000], cap=60000))
    print(f"repaired={len(repaired)} native={len(native)} chembl={len(chembl)}")

    profiles = [profile(repaired, "Repaired (cascade)", train_canon),
                profile(native, "Native valid gen.", train_canon),
                profile(chembl, "ChEMBL reference")]

    # FCD
    try:
        if not hasattr(np, "row_stack"):
            np.row_stack = np.vstack  # numpy>=2 compat for fcd_torch
        from fcd_torch import FCD
        fcd = FCD(device="cpu", n_jobs=4)
        rep_s = [Chem.MolToSmiles(m) for m in repaired]
        nat_s = [Chem.MolToSmiles(m) for m in native]
        chembl_s = [Chem.MolToSmiles(m) for m in chembl]
        fcd_rep_native = fcd(rep_s, nat_s)
        fcd_rep_chembl = fcd(rep_s, chembl_s)
        fcd_native_chembl = fcd(nat_s, chembl_s)
        fcd_out = {"FCD_repaired_vs_native": round(fcd_rep_native, 3),
                   "FCD_repaired_vs_chembl": round(fcd_rep_chembl, 3),
                   "FCD_native_vs_chembl": round(fcd_native_chembl, 3)}
    except Exception as e:
        fcd_out = {"FCD_error": str(e)}

    out = {"profiles": profiles, "fcd": fcd_out}
    json.dump(out, open("paper/chem_metrics.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
