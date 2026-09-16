"""Consolidated analysis of the recovery benchmark: tables, tests, risk-coverage.

Reads every per-item .jsonl in results_recovery/ and produces
  * the main results table (validity vs identity metrics, Wilson 95% CIs)
  * exact paired McNemar tests between named systems, Bonferroni-corrected
  * the validity-recovery gap (the ratio that motivates the paper)
  * risk-coverage curves for selective prediction
All items are aligned by ground truth so every comparison is genuinely paired.
"""
from __future__ import annotations
import argparse, json, math, os, sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(__file__))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(a_hits, b_hits):
    """Exact two-sided binomial McNemar on paired boolean outcomes."""
    b = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    c = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    from math import comb
    lo = min(b, c)
    p = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n) * 2
    return b, c, min(1.0, p)


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {r["ground_truth"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_recovery")
    ap.add_argument("--out", default="results_recovery/ANALYSIS.json")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="pairs 'A:B' to McNemar-test on exact recovery")
    args = ap.parse_args()

    systems = OrderedDict()
    for fn in sorted(os.listdir(args.dir)):
        if fn.endswith(".jsonl"):
            name = fn[:-6]
            try:
                systems[name] = load(os.path.join(args.dir, fn))
            except Exception as e:
                print(f"  skip {fn}: {e}")
    if not systems:
        print("no result files found"); return

    # common item set so every reported comparison is paired
    common = set.intersection(*(set(d) for d in systems.values()))
    print(f"[analysis] {len(systems)} systems, {len(common)} paired items")
    keys = sorted(common)

    table = {}
    for name, d in systems.items():
        rows = [d[k] for k in keys]
        n = len(rows)
        nv = sum(1 for r in rows if r["valid"])
        ne = sum(1 for r in rows if r["exact_recovery"])
        val = [r for r in rows if r["valid"]]
        tan = [r["tanimoto"] for r in val if r.get("tanimoto") is not None]
        sca = [r for r in val if r.get("scaffold_match") is not None]
        mcs = [r["mcs_frac"] for r in val if r.get("mcs_frac") is not None]
        lo_v, hi_v = wilson(nv, n)
        lo_e, hi_e = wilson(ne, n)
        table[name] = {
            "n": n,
            "validity": round(nv / n, 4), "validity_ci": [round(lo_v, 4), round(hi_v, 4)],
            "exact_recovery": round(ne / n, 4),
            "exact_recovery_ci": [round(lo_e, 4), round(hi_e, 4)],
            "tanimoto_given_valid": round(sum(tan) / len(tan), 4) if tan else None,
            "scaffold_given_valid": (round(sum(1 for r in sca if r["scaffold_match"]) / len(sca), 4)
                                     if sca else None),
            "mcs_given_valid": round(sum(mcs) / len(mcs), 4) if mcs else None,
            # the number the paper is about
            "validity_recovery_ratio": (round((nv / n) / (ne / n), 1) if ne else None),
        }

    tests = {}
    pairs = [tuple(c.split(":")) for c in args.compare if ":" in c]
    for a, b in pairs:
        if a not in systems or b not in systems:
            print(f"  skip test {a}:{b} (missing)"); continue
        ah = [systems[a][k]["exact_recovery"] for k in keys]
        bh = [systems[b][k]["exact_recovery"] for k in keys]
        nb, nc, p = mcnemar_exact(ah, bh)
        tests[f"{a}_vs_{b}"] = {"a_only": nb, "b_only": nc, "p_exact": p}
    if tests:
        m = len(tests)
        for v in tests.values():
            v["p_bonferroni"] = min(1.0, v["p_exact"] * m)

    # risk-coverage for systems that expose a confidence signal
    risk = {}
    for name, d in systems.items():
        rows = [d[k] for k in keys]
        sig = None
        if all(r.get("margin") is not None for r in rows if r.get("pred")):
            sig = "margin"
        elif any(r.get("agreement") is not None for r in rows):
            sig = "agreement"
        if not sig:
            continue
        scored = [(r.get(sig), r["exact_recovery"]) for r in rows
                  if r.get(sig) is not None]
        if len(scored) < 50:
            continue
        scored.sort(key=lambda t: -t[0])
        curve = []
        for frac in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            k = max(1, int(len(scored) * frac))
            sub = scored[:k]
            curve.append({"coverage": frac,
                          "precision": round(sum(1 for _, h in sub if h) / k, 4)})
        risk[name] = {"signal": sig, "curve": curve}

    report = {"n_paired": len(keys), "systems": table, "mcnemar": tests,
              "risk_coverage": risk}
    json.dump(report, open(args.out, "w"), indent=2)

    order = sorted(table, key=lambda s: -table[s]["exact_recovery"])
    w = max(len(s) for s in order) + 1
    print(f"\n{'system':<{w}} {'valid':>7} {'exact':>7} {'tanim':>6} {'scaf':>6} {'V/R':>6}")
    for s in order:
        t = table[s]
        print(f"{s:<{w}} {t['validity']*100:6.1f}% {t['exact_recovery']*100:6.1f}% "
              f"{(t['tanimoto_given_valid'] or 0):6.3f} "
              f"{(t['scaffold_given_valid'] or 0)*100:5.1f}% "
              f"{(t['validity_recovery_ratio'] or 0):6.1f}")
    if tests:
        print("\nMcNemar (exact recovery):")
        for k, v in tests.items():
            print(f"  {k}: +{v['a_only']} / -{v['b_only']}  p={v['p_exact']:.3g}  "
                  f"p_bonf={v['p_bonferroni']:.3g}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
