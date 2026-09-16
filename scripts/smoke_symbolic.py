"""Smoke test for the symbolic layer + metrics on hand-built broken SMILES."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import symbolic as S
import metrics as M

CASES = [
    # (smiles, note)
    ("CCO",                              "valid: ethanol"),
    ("c1ccccc1",                         "valid: benzene"),
    ("CC(=O)Oc1ccccc1C(=O)O",            "valid: aspirin"),
    ("CC(C)(C)(C)C",                     "semantic: 5-valent carbon"),
    ("c1ccccc1C(=O)O(",                  "syntax: extra '('"),
    ("CC(C)C)CC",                        "syntax: extra ')'"),
    ("C1CCCCC",                          "syntax: unclosed ring 1"),
    ("c1ccncc1OO",                       "valid-ish: check"),
    ("[NH4",                             "syntax: unbalanced bracket"),
    ("c1ccc2ccccc2c1",                   "valid: naphthalene"),
    ("c1cccc1",                          "semantic: kekulize 5-ring aromatic"),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C",     "valid: caffeine"),
]

print(f"{'SMILES':32} {'category':17} {'valid':5} | rule_repair -> result")
print("-" * 100)
n_correct_valid = 0
for smi, note in CASES:
    d = S.diagnose(smi)
    rep = S.rule_repair(smi) if not d.valid else "(already valid)"
    rep_ok = "ok" if (rep and rep != "(already valid)" and M.is_valid(rep)) else (
        "-" if d.valid else "FAIL")
    print(f"{smi:32.32} {d.category:17} {str(d.valid):5} | {str(rep):28.28} [{rep_ok}]")
    print(f"    msg: {d.message[:88]}")

print("\n=== metrics sanity ===")
a, b = "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"  # aspirin vs itself
print("canonical(aspirin):", M.canonical(a))
print("tanimoto(self):", M.tanimoto(a, b))
print("tanimoto(aspirin, benzene):", round(M.tanimoto(a, "c1ccccc1"), 3))
print("edit_distance('CCO','CCN'):", M.edit_distance("CCO", "CCN"))
print("token_edit_distance('CC(=O)O','CC(=O)N'):", M.token_edit_distance("CC(=O)O", "CC(=O)N"))
print("scaffold(aspirin):", M.scaffold(a))
print("QED(aspirin):", round(M.qed(a), 3))
print("descriptors(aspirin):", M.descriptors(a))
print("\nSmoke test complete.")
