"""Run one (model, condition) over the benchmark and dump per-item + summary.

Conditions:
  llm:no_feedback   llm:feedback   llm:iterative     (need --base_url + --model)
  rule              identity                          (no LLM)

Outputs (under --out_dir):
  <tag>.jsonl   one RepairResult per benchmark item
  <tag>.summary.json  aggregate metrics (overall + per error category)

Aggregate across tags later with analyze.py (cheap, runs on laptop).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import symbolic as S
import metrics as M


def load_benchmark(path: str, limit: int = 0) -> list[dict]:
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
            if limit and len(rows) >= limit:
                break
    return rows


def run_llm(rows, condition, base_url, model, workers, max_iters, temperature, n_samples=1):
    from repair import LLMRepairer
    rep = LLMRepairer(base_url=base_url, model=model, temperature=temperature, n_samples=n_samples)
    results = [None] * len(rows)

    def work(i):
        r = rep.repair(rows[i]["smiles"], condition=condition, max_iters=max_iters)
        return i, r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(rows))]
        done = 0
        for fut in as_completed(futs):
            i, r = fut.result()
            results[i] = r
            done += 1
            if done % 100 == 0:
                print(f"  [{condition}/{model}] {done}/{len(rows)}", flush=True)
    return results


def run_rule(rows):
    out = []
    for r in rows:
        rep = S.rule_repair(r["smiles"])
        out.append({"original": r["smiles"], "repaired": rep,
                    "valid": bool(rep) and M.is_valid(rep),
                    "condition": "rule", "iterations": 1, "latency_s": 0.0})
    return out


def to_record(r, row):
    """Normalize an LLM RepairResult or a rule dict into a serializable record."""
    if hasattr(r, "original"):
        d = dict(original=r.original, repaired=r.repaired, valid=r.valid,
                 condition=r.condition, iterations=r.iterations,
                 latency_s=round(r.latency_s, 3), error=r.error)
    else:
        d = dict(r)
    rep = d.get("repaired")
    d["category"] = row["category"]
    d["is_syntactic"] = row["is_syntactic"]
    if d["valid"] and rep:
        d["char_edit"] = M.edit_distance(d["original"], rep)
        d["token_edit"] = M.token_edit_distance(d["original"], rep)
        d["tanimoto_to_input_parse"] = None  # input is invalid, no fp; kept for schema
        d["repaired_canonical"] = M.canonical(rep)
        d["qed"] = M.qed(rep)
    else:
        d["char_edit"] = d["token_edit"] = None
        d["repaired_canonical"] = d["qed"] = None
    return d


def summarize(records: list[dict], tag: str) -> dict:
    n = len(records)
    valid = [r for r in records if r["valid"]]
    by_cat = defaultdict(lambda: {"n": 0, "fixed": 0})
    for r in records:
        c = by_cat[r["category"]]
        c["n"] += 1
        c["fixed"] += int(r["valid"])
    cat_rates = {k: {"n": v["n"], "fixed": v["fixed"],
                     "repair_rate": round(v["fixed"] / v["n"], 4)}
                 for k, v in by_cat.items()}
    syn = [r for r in records if r["is_syntactic"]]
    sem = [r for r in records if not r["is_syntactic"]]
    edits = [r["token_edit"] for r in valid if r["token_edit"] is not None]
    qeds = [r["qed"] for r in valid if r["qed"] is not None]
    lat = [r.get("latency_s", 0) for r in records if r.get("latency_s")]
    iters = [r.get("iterations", 0) for r in records]
    return {
        "tag": tag,
        "n": n,
        "repair_rate": round(len(valid) / n, 4) if n else 0,
        "n_fixed": len(valid),
        "syntactic_repair_rate": round(sum(r["valid"] for r in syn) / len(syn), 4) if syn else None,
        "semantic_repair_rate": round(sum(r["valid"] for r in sem) / len(sem), 4) if sem else None,
        "mean_token_edit_fixed": round(sum(edits) / len(edits), 3) if edits else None,
        "mean_qed_fixed": round(sum(qeds) / len(qeds), 3) if qeds else None,
        "mean_iterations": round(sum(iters) / len(iters), 3) if iters else None,
        "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
        "per_category": cat_rates,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--condition", required=True,
                    choices=["no_feedback", "feedback", "iterative",
                             "feedback_rich", "iterative_rich", "rule", "identity"])
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max_iters", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=1, help="pass@k candidates per round")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_benchmark(args.benchmark, args.limit)
    print(f"[eval] {len(rows)} items | condition={args.condition} model={args.model} samples={args.samples}")

    if args.condition in ("no_feedback", "feedback", "iterative", "feedback_rich", "iterative_rich"):
        assert args.model, "LLM conditions require --model"
        raw = run_llm(rows, args.condition, args.base_url, args.model,
                      args.workers, args.max_iters, args.temperature, args.samples)
    elif args.condition == "rule":
        raw = run_rule(rows)
    else:  # identity (do-nothing control)
        raw = [{"original": r["smiles"], "repaired": r["smiles"],
                "valid": False, "condition": "identity", "iterations": 0,
                "latency_s": 0.0} for r in rows]

    records = [to_record(r, rows[i]) for i, r in enumerate(raw)]
    tag = args.tag or f"{(args.model.split('/')[-1] or 'na')}__{args.condition}"
    with open(os.path.join(args.out_dir, f"{tag}.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = summarize(records, tag)
    with open(os.path.join(args.out_dir, f"{tag}.summary.json"), "w") as sf:
        json.dump(summary, sf, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
