"""Cross-check every headline number in the manuscript against the result files.

After a figure/table mismatch earlier in this project, no number goes into the
paper unverified. This asserts the claims in main.tex against results_recovery/.
"""
from __future__ import annotations
import json, os, re, sys

R = "results_recovery"
TEX = "paper_vnr/main.tex"
fails, checks = [], 0


def summ(tag):
    return json.load(open(f"{R}/{tag}.summary.json"))


def check(label, claimed, actual, tol=0.05):
    global checks
    checks += 1
    if actual is None:
        fails.append(f"{label}: no actual value"); return
    if abs(claimed - actual) > tol:
        fails.append(f"{label}: paper says {claimed}, results say {actual:.4f}")


tex = open(TEX).read()

# main table rows: (label, tag, validity%, exact%, tanimoto, scaffold%)
rows = [
    ("ours", "TEST_enumrank_domainlm_prune", 100.0, 80.7, 0.937, 84.7),
    ("supervised", "TEST_seq2seq_supervised", 74.8, 59.4, 0.946, 84.1),
    ("llm rank", "TEST_enumrank_llm_prune", 100.0, 50.2, 0.805, 57.9),
    ("shortest", "TEST_enum_shortest", 100.0, 23.1, 0.683, 28.5),
    ("free loop", "TEST_looprich_k5_minedit_s0", 47.7, 13.8, 0.660, 34.9),
    ("uniform", "TEST_enum_uniform", 100.0, 13.6, 0.629, 19.7),
    ("smiself", "smiself", 86.6, 1.2, 0.356, 3.4),
]
for lab, tag, v, e, t, sc in rows:
    a = summ(tag)
    check(f"{lab} validity", v, a["validity"] * 100, 0.06)
    check(f"{lab} exact", e, a["exact_recovery"] * 100, 0.06)
    check(f"{lab} tanimoto", t, a["tanimoto_given_valid"], 0.0006)
    check(f"{lab} scaffold", sc, a["scaffold_match_given_valid"] * 100, 0.06)

# ablation ladder quoted in the text
for lab, tag, v, e in [("nofb", "TEST_nofb_k1", 23.4, 8.9),
                       ("fb", "TEST_fb_k1", 27.0, 11.7),
                       ("loop k1", "TEST_looprich_k1", 37.4, 12.1)]:
    a = summ(tag)
    check(f"{lab} validity", v, a["validity"] * 100, 0.06)
    check(f"{lab} exact", e, a["exact_recovery"] * 100, 0.06)

# candidate space
c = [json.loads(l) for l in open("data/recovery/cands_test.jsonl")]
n = len(c)
ceil = sum(1 for r in c if r.get("truth_in_candidates")) / n
sizes = sorted(len(r["candidates"]) for r in c)
unif = sum(1.0 / max(len(r["candidates"]), 1) for r in c
           if r.get("truth_in_candidates")) / n
check("ceiling", 99.79, ceil * 100, 0.006)
check("median cands", 50, sizes[n // 2], 0.5)
check("mean cands", 75.9, sum(sizes) / n, 0.06)
check("uniform expected", 12.7, unif * 100, 0.06)

# per-category claims in the text
pc = summ("TEST_enumrank_domainlm_prune")["per_category"]
for cat, val in [("semantic_kekulize", 67.5), ("syntax_ring", 93.1),
                 ("semantic_valence", 87.4)]:
    check(f"percat {cat}", val, pc[cat]["exact_recovery"] * 100, 0.06)

# abstention
rr = [json.loads(l) for l in open(f"{R}/TEST_enumrank_domainlm_prune.jsonl")]
sc = sorted([(r["margin"], r["exact_recovery"]) for r in rr
             if r.get("margin") is not None], key=lambda t: -t[0])
for frac, claimed in ((0.5, 96.4), (0.3, 99.2), (0.2, 99.7)):
    k = max(1, int(len(sc) * frac))
    check(f"abstain@{int(frac*100)}", claimed,
          sum(1 for _, h in sc[:k] if h) / k * 100, 0.06)

# contamination
loo = json.load(open(f"{R}/CONTAMINATION_LOO.json"))
worst = max(abs(v["delta_pp"]) for v in loo.values())
check("contamination max delta", 0.02, worst, 0.005)


# --- Experiment 6: real failures and generator distribution ---
rf = json.load(open(f"{R}/REALFAIL_ours.summary.json"))
check("realfail coverage", 91.5, rf["coverage"] * 100, 0.06)
check("realfail yield", 91.5, rf["yield"] * 100, 0.06)
ss = json.load(open(f"{R}/REALFAIL_smiself.summary.json"))
check("smiself realfail yield", 90.9, ss["yield"] * 100, 0.06)
ed = json.load(open(f"{R}/REALFAIL_edits.json"))
check("ours repair median", 1, ed["ours_witness_median"], 0.001)
check("ours repair mean", 1.00, ed["ours_witness_mean"], 0.006)
check("smiself repair median", 21, ed["smiself_median"], 0.001)
check("smiself repair mean", 21.47, ed["smiself_mean"], 0.006)
gd = summ("GENDIST_ours")
check("gendist exact", 78.9, gd["exact_recovery"] * 100, 0.06)
check("gendist validity", 99.9, gd["validity"] * 100, 0.06)
gr = json.load(open(f"{R}/GENDIST_reference.json"))
check("gendist ceiling", 99.7, gr["ceiling"] * 100, 0.06)
check("gendist median cands", 47, gr["median_cands"], 0.5)
check("gendist uniform", 12.5, gr["uniform_expected"] * 100, 0.06)


# --- claims the first pass did not cover (added after manual audit) ---
import torch as _torch
_st = json.load(open("data/recovery/stats.json"))
_tm = _st["target_mix"]
for _k, _v in [("semantic_kekulize", 45.2), ("syntax_ring", 31.3),
               ("semantic_valence", 11.3), ("syntax_other", 5.4),
               ("syntax_parens", 3.7), ("syntax_brackets", 3.1)]:
    check(f"target mix {_k}", _v, _tm[_k] * 100, 0.06)
_obs = _st["observed_category_distribution"]; _tot = sum(_obs.values())
for _k, _v in [("semantic_kekulize", 45.8), ("syntax_ring", 32.7),
               ("semantic_valence", 11.9), ("syntax_other", 5.6),
               ("syntax_parens", 3.6), ("syntax_brackets", 0.5)]:
    check(f"realised mix {_k}", _v, _obs[_k] / _tot * 100, 0.06)
check("category agreement", 99.6, _st["operator_hit_intended_category_pct"], 0.06)

_ck = _torch.load("runs/lm/best.pt", map_location="cpu", weights_only=False)
_np = sum(v.numel() for v in _ck["model"].values())
check("ranker params (M)", 12.2, _np / 1e6, 0.05)
assert _np / 14.7e9 * 100 < 0.1, "ranker param fraction must be under 0.1%"
checks += 1

# kekulize candidate medians, before and after pruning (the paper says 112 after)
sys.path.insert(0, "src")
import edit_ops as _EO
_kek = [r for r in c if r.get("category") == "semantic_kekulize"]
_before = sorted(len(r["candidates"]) for r in _kek)
_after = sorted(len(_EO.prune(r["corrupted"], r["candidates"], r["witness"],
                             r.get("category"))[0]) for r in _kek)
check("kekulize median before prune", 117, _before[len(_before) // 2], 0.5)
check("kekulize median after prune", 112, _after[len(_after) // 2], 0.5)

# per-category uniform reference quoted in the text
_pcu = {}
for r in c:
    a = _pcu.setdefault(r.get("category"), [0, 0.0])
    a[0] += 1
    if r.get("truth_in_candidates"):
        a[1] += 1.0 / max(len(r["candidates"]), 1)
for _k, _v in [("syntax_ring", 31.0), ("semantic_valence", 7.5),
               ("semantic_kekulize", 0.9)]:
    check(f"uniform {_k}", _v, _pcu[_k][1] / _pcu[_k][0] * 100, 0.06)

check("coverage residual", 0.21, 100 - ceil * 100, 0.006)
_brk = [r for r in c if r.get("category") == "syntax_brackets"]
check("bracket ceiling", 52.6,
      sum(1 for r in _brk if r.get("truth_in_candidates")) / len(_brk) * 100, 0.06)

import re as _re
_bib = open("paper_vnr/references.bib").read()
check("bib entries", 21, len(_re.findall(r"^@", _bib, _re.M)), 0.5)


# --- scope experiments: multi-edit degradation, 2-edit cost, external benchmark ---
for _k, _cov, _empty in [(1, 99.67, 0), (2, 0.50, 324), (3, 0.00, 536)]:
    _c = [json.loads(l) for l in open(f"data/recovery_multi/cands_k{_k}.jsonl")]
    _n = len(_c)
    check(f"k={_k} coverage", _cov,
          sum(1 for r in _c if r.get("truth_in_candidates")) / _n * 100, 0.006)
    check(f"k={_k} empty sets", _empty, sum(1 for r in _c if not r["candidates"]), 0.5)

_k1 = summ("MULTI_k1_ours")
check("k=1 replication validity", 100.0, _k1["validity"] * 100, 0.06)
check("k=1 replication exact", 80.0, _k1["exact_recovery"] * 100, 0.06)

_ex = summ("EXT_papyrus_ours")
check("external validity", 84.50, _ex["validity"] * 100, 0.06)
check("external exact", 60.42, _ex["exact_recovery"] * 100, 0.06)
check("external tanimoto", 0.888, _ex["tanimoto_given_valid"], 0.0006)

_exc = json.load(open(f"{R}/EXT_conditional.json"))
check("external answered", 84.50, _exc["answered_frac"] * 100, 0.06)
check("external recovery|answered", 71.50, _exc["recovery_given_answered"] * 100, 0.06)
check("external reachable", 69.08, _exc["reachable_frac"] * 100, 0.06)
check("external recovery|reachable", 87.45, _exc["recovery_given_reachable"] * 100, 0.06)
assert _exc["recovery_given_unreachable"] == 0.0, \
    "recovery on unreachable items must be exactly 0 (enumeration soundness)"
checks += 1

_exs = json.load(open("data/recovery_external/papyrus_ext.stats.json"))
check("external n", 1200, _exs["n"], 0.5)
check("external median true edit", 5, _exs["median_true_token_edit"], 0.5)
check("external fragment-graft frac", 11.08, _exs["fragment_grafted_frac"] * 100, 0.06)


# --- cross-generator 2x2, Transformer ranker rung, Gen B, external 2-edit ladder ---
_rung = summ("TEST_enumrank_tlm_prune")
check("transformer rung validity", 100.0, _rung["validity"] * 100, 0.06)
check("transformer rung exact", 75.43, _rung["exact_recovery"] * 100, 0.06)

for _tag, _lab, _v in [("GENDIST_ours", "genA x lstm", 78.85),
                       ("GENDIST_A_tlm", "genA x tlm", 73.10),
                       ("GENDIST_B_lstm", "genB x lstm", 64.95),
                       ("GENDIST_B_tlm", "genB x tlm", 73.75)]:
    _a = summ(_tag)
    check(f"2x2 {_lab}", _v, _a["exact_recovery"] * 100, 0.06)
    assert _a["validity"] >= 0.998, f"{_lab}: enumeration must keep validity ~1"
    checks += 1

# the interaction the paper claims: matched prior beats mismatched
_m = (summ("GENDIST_ours")["exact_recovery"] + summ("GENDIST_B_tlm")["exact_recovery"]) / 2
_x = (summ("GENDIST_A_tlm")["exact_recovery"] + summ("GENDIST_B_lstm")["exact_recovery"]) / 2
check("matched mean", 76.30, _m * 100, 0.06)
check("mismatched mean", 69.03, _x * 100, 0.06)
check("matched advantage (pp)", 7.27, (_m - _x) * 100, 0.06)
assert _m > _x, "matched-prior advantage must be positive"
checks += 1

_gb = json.load(open("data/recovery_gendist_B/cands_gendistB.summary.json"))
check("genB coverage", 99.80, _gb["truth_in_candidates"] * 100, 0.06)
check("genB median cands", 44, _gb["median_candidates"], 0.5)
_ga = json.load(open("data/recovery/cands_gendist.summary.json"))
check("genA coverage", 99.70, _ga["truth_in_candidates"] * 100, 0.06)

_e2 = json.load(open("data/recovery_external/cands2_uncovered.summary.json"))
check("external 2-edit reach", 5.0, _e2["truth_in_candidates"] * 100, 0.06)
check("external 2-edit parses", 122545.9, _e2["mean_parses_per_item"], 1.0)


# --- guard: the IEEE variant is GENERATED from main.tex and silently went stale
#     for six days because it was only ever recompiled, never regenerated. ---
import re as _re2, os as _os2
_main = open("paper_vnr/main.tex").read()
_ieee = open("paper_vnr/ieee_main.tex").read()
_secs_main = set(_re2.findall(r"\\subsection\{([^}]*)\}", _main))
_secs_ieee = set(_re2.findall(r"\\subsection\{([^}]*)\}", _ieee))
assert len(_secs_main) >= 9, f"section regex matched only {len(_secs_main)} -- guard would pass vacuously"
_missing = _secs_main - _secs_ieee
assert not _missing, ("ieee_main.tex is STALE -- regenerate it from main.tex. "
                      f"Missing sections: {sorted(_missing)}")
checks += 1
assert _os2.path.getmtime("paper_vnr/ieee_main.tex") >= _os2.path.getmtime("paper_vnr/main.tex") - 5, \
    "ieee_main.tex is older than main.tex -- regenerate it from main.tex"
checks += 1

# --- guard: em-dashes belong only in the two "not applicable" table rows ---
_prose = _re2.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", _main, flags=_re2.S)
assert "---" not in _prose, "em-dash found in main.tex prose (only table cells may use ---)"
checks += 1


# --- two-edit cost curve: these 12 numbers lived only in a volatile scratchpad
#     until they were rescued into results_recovery/twoedit/ ---
for _f, _cov, _par, _sec in [("cands2_k2_probe", 5.0, 28062.5, 4.424),
                             ("cands2_k2_mid", 30.0, 117818.9, 15.32),
                             ("cands2_k2_generous", 58.3, 437774.2, 60.316)]:
    _d = json.load(open(f"{R}/twoedit/{_f}.summary.json"))
    check(f"2edit {_f} coverage", _cov, _d["truth_in_candidates"] * 100, 0.06)
    check(f"2edit {_f} parses", _par, _d["mean_parses_per_item"], 1.0)
    check(f"2edit {_f} seconds", _sec, _d["mean_seconds_per_item"], 0.01)

# --- abstention denominator: the margin is undefined for single-candidate items,
#     so coverage percentages are over 3,881 not 4,200. The paper must say so. ---
_ab = [json.loads(l) for l in open(f"{R}/TEST_enumrank_domainlm_prune.jsonl")]
_withm = [r for r in _ab if r.get("margin") is not None]
check("abstention denominator", 3881, len(_withm), 0.5)
check("abstention as frac of full split", 46.2,
      int(len(_withm) * 0.5) / len(_ab) * 100, 0.1)
assert "3{,}881" in _main, "the paper must state the abstention denominator explicitly"
assert "half of all inputs" not in _main, "abstention denominator overclaim reintroduced"
checks += 2

# --- BMC caps research-article abstracts at 350 words ---
_abs = _re2.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", _main, _re2.S).group(1)
_abs = _re2.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", _abs)
_abs = _re2.sub(r"[{}\\]", " ", _abs)
_nw = len([x for x in _abs.split() if any(c.isalnum() for c in x)])
assert _nw <= 350, f"abstract is {_nw} words; BMC caps research-article abstracts at 350"
checks += 1


# --- generator validity by sampling temperature (cited in Background from our
#     own measurement, since [2,3] do not support the temperature claim) ---
_gt = json.load(open(f"{R}/generator_temperature.json"))
for _t, _v in [("1.0", 93.3), ("1.2", 85.3), ("1.5", 64.1)]:
    check(f"generator validity T={_t}", _v, _gt[_t]["validity"] * 100, 0.06)
assert "sampling temperature" not in _main or "93.3" in _main, \
    "the temperature claim must carry its own measurement, not a citation to [2,3]"
checks += 1


# --- the bibliography ships to the editor: no internal working notes in it ---
_bibtxt = open("paper_vnr/references.bib").read()
for _bad in ["Kerkhof", "TODO", "FIXME", "verified via", "Confirm it is still"]:
    assert _bad not in _bibtxt, f"internal note left in references.bib: {_bad!r}"
checks += 1


# --- novelty attribution: the constrained-selection formulation is transferred
#     from syntax repair, not invented here ---
assert "considine2025syntax" in _main, "Considine (arXiv:2507.11873) must be cited"
assert "That division is not ours" in _main, "the formulation's provenance must stay explicit"
checks += 2


# --- the corpus figure is the UNIQUE canonical count (1,272,851), not the number
#     of parsed lines (1,273,104, which includes 253 duplicates) ---
assert "1{,}272{,}851" in _main, "contamination corpus size must be the unique canonical count"
assert "1{,}273{,}104" not in _main, "1,273,104 is the parsed count, not the unique count"
# --- empty candidate sets are not evidence of multiple errors ---
assert "carry more than one simultaneous" not in _main, \
    "cannot attribute empty candidate sets to multiple errors without ground truth"
checks += 3


# --- claims must not outrun the evidence for REAL failures (no ground truth) ---
for _bad in ["carry multiple simultaneous", "admit a one-edit repair",
             "transfers undegraded", "matches the modal real failure",
             "All differences in Table"]:
    assert _bad not in _main, f"unsupported claim reintroduced: {_bad!r}"
checks += 5
# the decomposition must use the uniform baseline on the SAME pruned candidate set
assert "12.7\\%\\,$\\rightarrow$\\,80.7" not in _main, \
    "decomposition must compare 13.6% uniform to 80.7% on the same pruned set"
checks += 1


assert "barely\nbetter than published methods" not in _main and \
       "barely better than published methods" not in _main, \
    "uniform selection beats SmiSelf 11x; 'barely better' is wrong in both directions"
assert "The decomposition is clean" not in _main, "do not grade our own decomposition"
checks += 2

# every number the tex asserts must appear in this script
print(f"checked {checks} claims")
if fails:
    print("MISMATCHES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all paper numbers match the result files")
