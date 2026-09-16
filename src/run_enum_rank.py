"""Enumerate--Rank: symbolic enumeration of the repair space, neural ranking.

The symbolic half emits every valid molecule one token edit from the corrupted
string (sound by construction, complete for single-edit corruptions). The neural
half ranks that set by chemical-language-model likelihood and returns the top
candidate. Validity is therefore guaranteed whenever the set is non-empty, and
all of the remaining difficulty is where it belongs: choosing WHICH valid
molecule was intended.

Reference policies over the same candidate set:
  uniform  -- pick uniformly at random (the informed non-learned baseline)
  shortest -- pick the lexicographically shortest canonical SMILES
No policy consults the ground truth.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import recovery_metrics as RM
import edit_ops as EO
from openai import OpenAI


def score_batch(cli, model, prompts, prefix, retries=3):
    """Total and mean token logprob for each prompt under the LM."""
    import time
    texts = [prefix + p for p in prompts]
    last = None
    for a in range(retries):
        try:
            r = cli.completions.create(model=model, prompt=texts, max_tokens=0,
                                       echo=True, logprobs=0, temperature=0)
            out = [None] * len(texts)
            for ch in r.choices:
                lps = [x for x in (ch.logprobs.token_logprobs or []) if x is not None]
                out[ch.index] = (sum(lps), (sum(lps) / len(lps)) if lps else 0.0)
            return out
        except Exception as e:
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"scoring failed: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True, help="output of enumerate_repairs.py")
    ap.add_argument("--policy", default="lm_sum",
                    choices=["lm_sum", "lm_mean", "uniform", "shortest"])
    ap.add_argument("--prefix", default="", help="scoring prefix, e.g. 'SMILES: '")
    ap.add_argument("--base_url", default="http://localhost:8022/v1")
    ap.add_argument("--model", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--prune", action="store_true",
                    help="constrain candidates by the RDKit diagnosis before ranking")
    ap.add_argument("--max_cands", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mcs_n", type=int, default=400)
    ap.add_argument("--out_dir", default="results_recovery")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.cands) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    needs_lm = args.policy.startswith("lm_")
    cli = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180.0) if needs_lm else None
    rng = random.Random(args.seed)

    def pick(i):
        r = rows[i]
        cands = r["candidates"]
        if args.prune and cands:
            cands, _ = EO.prune(r["corrupted"], cands, r["witness"], r.get("category"))
        if not cands:
            return i, None, 0, None
        truncated = len(cands) > args.max_cands
        if truncated:                      # deterministic truncation, no truth used
            cands = sorted(cands, key=len)[:args.max_cands]
        if args.policy == "uniform":
            return i, rng.choice(cands), len(cands), None
        if args.policy == "shortest":
            return i, min(cands, key=lambda c: (len(c), c)), len(cands), None
        scored = score_batch(cli, args.model, cands, args.prefix)
        j = 0 if args.policy == "lm_sum" else 1
        order = sorted(range(len(cands)), key=lambda t: -scored[t][j])
        best = cands[order[0]]
        # margin between top-1 and top-2 = leakage-free confidence signal
        margin = (scored[order[0]][j] - scored[order[1]][j]) if len(order) > 1 else None
        return i, best, len(cands), margin

    out = [None] * len(rows)
    if needs_lm:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(pick, i) for i in range(len(rows))]
            done = 0
            for f in as_completed(futs):
                i, best, nc, mg = f.result()
                out[i] = (best, nc, mg); done += 1
                if done % 250 == 0:
                    print(f"  {done}/{len(rows)}", flush=True)
    else:
        for i in range(len(rows)):
            _, best, nc, mg = pick(i)
            out[i] = (best, nc, mg)

    recs = []
    for i, (r, (best, nc, mg)) in enumerate(zip(rows, out)):
        sc = RM.score(r["ground_truth"], best, with_mcs=(i < args.mcs_n))
        recs.append({"ground_truth": r["ground_truth"], "corrupted": r["corrupted"],
                     "category": r.get("category"), "pred": best,
                     "n_cands": nc, "margin": mg,
                     "truth_in_candidates": r.get("truth_in_candidates"), **sc})

    agg = RM.aggregate(recs)
    agg.update({"policy": args.policy, "prefix": args.prefix, "model": args.model,
                "seed": args.seed,
                "empty_candidate_sets": sum(1 for r in recs if not r["n_cands"]),
                "oracle_truth_in_candidates": round(
                    sum(1 for r in recs if r.get("truth_in_candidates")) / len(recs), 4)})
    cats = {}
    for c in sorted({r["category"] for r in recs if r["category"]}):
        sub = [r for r in recs if r["category"] == c]
        a = RM.aggregate(sub)
        cats[c] = {"n": a["n"], "validity": a["validity"],
                   "exact_recovery": a["exact_recovery"],
                   "tanimoto_given_valid": a["tanimoto_given_valid"]}
    agg["per_category"] = cats

    tag = args.tag or f"enumrank_{args.policy}"
    with open(os.path.join(args.out_dir, f"{tag}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, f"{tag}.summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps({k: v for k, v in agg.items() if k != "per_category"}, indent=2))


if __name__ == "__main__":
    main()
