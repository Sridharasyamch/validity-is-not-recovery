"""Neuro-symbolic RECOVERY on the ground-truth corruption benchmark.

Differs from the earlier repair harness in two ways:
  1. Scoring is identity-aware (exact recovery / Tanimoto / scaffold), not just validity.
  2. Candidate selection is explicit. `min_edit` keeps every valid candidate and
     picks the one closest to the CORRUPTED INPUT -- never to the ground truth,
     which would be leakage. Selection therefore uses only observable signal.
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from concurrent.futures import ThreadPoolExecutor, as_completed

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import symbolic as S
import metrics as M
import recovery_metrics as RM
from repair import extract_smiles, SYSTEM, PROMPT_NO_FEEDBACK, PROMPT_FEEDBACK
from openai import OpenAI


def build_prompt(cur, use_feedback, use_rich):
    if not use_feedback:
        return PROMPT_NO_FEEDBACK.format(smi=cur), "(none)"
    diag = S.rich_feedback(cur) if use_rich else S.diagnose(cur).message
    return PROMPT_FEEDBACK.format(smi=cur, diag=diag), diag


class RecoveryEngine:
    def __init__(self, base_url, model, k=1, temperature=0.0, max_tokens=160,
                 timeout=90.0, retries=3, seed=None):
        self.cli = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model, self.k, self.retries, self.max_tokens = model, k, retries, max_tokens
        self.temperature = temperature if k == 1 else max(temperature, 0.7)
        self.seed = seed

    def _chat(self, prompt):
        last = None
        for a in range(self.retries):
            try:
                kw = dict(model=self.model,
                          messages=[{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}],
                          temperature=self.temperature, max_tokens=self.max_tokens,
                          n=self.k)
                if self.seed is not None:
                    kw["seed"] = self.seed
                r = self.cli.chat.completions.create(**kw)
                return [c.message.content or "" for c in r.choices]
            except Exception as e:
                last = e; time.sleep(1.5 * (a + 1))
        raise RuntimeError(f"LLM failed: {last}")

    def recover(self, smi, condition, select="first_valid", max_iters=3):
        use_feedback = not condition.startswith("no_feedback")
        use_rich = "rich" in condition
        iterate = condition.startswith("iterative")
        K = max_iters if iterate else 1
        cur, n_iter, n_calls = smi, 0, 0
        best = None
        n_valid = 0
        agreement = None      # self-consistency of the valid candidate set
        chosen_edit = None    # edit distance from the observed corrupted input
        for it in range(1, K + 1):
            if S.is_valid(cur):
                best = cur; break
            prompt, _ = build_prompt(cur, use_feedback, use_rich)
            cands = self._chat(prompt); n_calls += 1
            n_iter = it
            parsed = [extract_smiles(c) for c in cands]
            valid = [p for p in parsed if p and S.is_valid(p)]
            if valid:
                if select == "min_edit":
                    # closest to the OBSERVED corrupted input (no ground truth used)
                    best = min(valid, key=lambda p: M.token_edit_distance(smi, p))
                else:
                    best = valid[0]
                # --- leakage-free confidence signals (for abstention) ---
                canons = [M.canonical(p) for p in valid]
                canons = [c for c in canons if c]
                n_valid = len(canons)
                if canons:
                    bc = M.canonical(best)
                    agreement = sum(1 for c in canons if c == bc) / len(canons)
                chosen_edit = M.token_edit_distance(smi, best) if best else None
                break
            cur = parsed[0] if parsed else cur
        return {"pred": best, "iterations": n_iter, "n_calls": n_calls,
                "n_valid_cands": n_valid, "agreement": agreement,
                "chosen_edit": chosen_edit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--select", default="first_valid", choices=["first_valid", "min_edit"])
    ap.add_argument("--base_url", default="http://localhost:8022/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mcs_n", type=int, default=400)
    ap.add_argument("--out_dir", default="results_recovery")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.benchmark) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    eng = RecoveryEngine(args.base_url, args.model, k=args.k, seed=args.seed)

    out = [None] * len(rows)

    def work(i):
        r = rows[i]
        try:
            res = eng.recover(r["corrupted"], args.condition, args.select, args.max_iters)
        except Exception as e:
            res = {"pred": None, "iterations": 0, "n_calls": 0, "error": str(e)}
        sc = RM.score(r["ground_truth"], res["pred"], with_mcs=(i < args.mcs_n))
        return i, {"ground_truth": r["ground_truth"], "corrupted": r["corrupted"],
                   "category": r["observed_category"], **res, **sc}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(rows))]
        done = 0
        for f in as_completed(futs):
            i, rec = f.result(); out[i] = rec; done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(rows)}", flush=True)

    agg = RM.aggregate(out)
    agg.update({"condition": args.condition, "select": args.select, "k": args.k,
                "seed": args.seed, "model": args.model,
                "mean_calls": round(sum(r["n_calls"] for r in out) / len(out), 3)})
    cats = {}
    for c in sorted(set(r["category"] for r in out)):
        sub = [r for r in out if r["category"] == c]
        a = RM.aggregate(sub)
        cats[c] = {"n": a["n"], "validity": a["validity"],
                   "exact_recovery": a["exact_recovery"],
                   "tanimoto_given_valid": a["tanimoto_given_valid"]}
    agg["per_category"] = cats

    tag = args.tag or f"{args.model.split('/')[-1]}__{args.condition}__{args.select}__k{args.k}__s{args.seed}"
    with open(os.path.join(args.out_dir, f"{tag}.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, f"{tag}.summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps({k: v for k, v in agg.items() if k != "per_category"}, indent=2))


if __name__ == "__main__":
    main()
