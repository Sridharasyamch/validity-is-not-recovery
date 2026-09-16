"""Harvest a benchmark of *real* invalid SMILES from generated samples.

Crash-resilient: RDKit can SEGFAULT (not raise) on pathological strings that a
char-level generator emits. We therefore run diagnosis in a worker subprocess
that checkpoints its position before each item; a driver relaunches the worker
after any crash, recording and skipping the offending index. The normal path is
a single worker pass; subprocess restarts happen only on the rare crashers.

Usage:
    python harvest.py run --inputs a.smiles b.smiles --out_dir data/benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

import symbolic as S


def build_flat_list(inputs, min_len, max_len):
    items = []
    for path in inputs:
        tag = os.path.basename(path)
        with open(path) as f:
            for ln in f:
                s = ln.strip()
                if s and (min_len <= len(s) <= max_len):
                    items.append((s, tag))
    return items


# --------------------------------------------------------------------------- #
# Worker: process items[start:], checkpoint before each, append invalids to raw.
# --------------------------------------------------------------------------- #
def worker(args):
    items = build_flat_list(args.inputs, args.min_len, args.max_len)
    state_path = os.path.join(args.work_dir, "state.json")
    skip_path = os.path.join(args.work_dir, "skip.txt")
    raw_path = os.path.join(args.work_dir, "raw.jsonl")
    done_path = os.path.join(args.work_dir, "DONE")

    skip = set()
    if os.path.exists(skip_path):
        skip = {int(x) for x in open(skip_path).read().split() if x.strip()}

    if os.path.exists(state_path):
        st = json.load(open(state_path))
        i, total, valid = st["i"], st["total"], st["valid"]
    else:
        i, total, valid = 0, 0, 0

    raw = open(raw_path, "a")
    flush_every = 200
    while i < len(items):
        # checkpoint BEFORE touching item i, so a crash is attributable to i
        json.dump({"i": i, "total": total, "valid": valid}, open(state_path, "w"))
        if i in skip:
            i += 1
            continue
        smi, tag = items[i]
        d = S.diagnose(smi)          # <-- may segfault; that's why we checkpoint
        total += 1
        if d.valid:
            valid += 1
        else:
            raw.write(json.dumps({"smiles": smi, "category": d.category,
                                  "is_syntactic": d.is_syntactic,
                                  "message": d.message, "source": tag}) + "\n")
            raw.flush()   # persist each invalid immediately (crash-safe)
        i += 1
    raw.flush(); raw.close()
    json.dump({"i": len(items), "total": total, "valid": valid}, open(state_path, "w"))
    open(done_path, "w").write("DONE")
    print(f"[worker] finished: total={total} valid={valid}")


# --------------------------------------------------------------------------- #
# Driver: relaunch worker across crashes, then finalize.
# --------------------------------------------------------------------------- #
def run(args):
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    state_path = os.path.join(args.work_dir, "state.json")
    skip_path = os.path.join(args.work_dir, "skip.txt")
    done_path = os.path.join(args.work_dir, "DONE")
    # fresh start
    for p in (state_path, skip_path, done_path, os.path.join(args.work_dir, "raw.jsonl")):
        if os.path.exists(p):
            os.remove(p)

    base = [sys.executable, os.path.abspath(__file__), "worker",
            "--inputs", *args.inputs, "--work_dir", args.work_dir,
            "--min_len", str(args.min_len), "--max_len", str(args.max_len)]
    restarts = 0
    while not os.path.exists(done_path):
        rc = subprocess.call(base)
        if os.path.exists(done_path):
            break
        # worker died (segfault): record the crashing index and skip it
        if os.path.exists(state_path):
            crashed = json.load(open(state_path))["i"]
            with open(skip_path, "a") as f:
                f.write(f"{crashed}\n")
            restarts += 1
            print(f"[driver] worker died (rc={rc}); skipping crashing item {crashed} "
                  f"(restart {restarts})", flush=True)
        else:
            print(f"[driver] worker died with no state (rc={rc}); aborting"); return
    finalize(args, restarts)


def finalize(args, restarts=0):
    work = args.work_dir
    raw_path = os.path.join(work, "raw.jsonl")
    state = json.load(open(os.path.join(work, "state.json")))
    skip_n = 0
    if os.path.exists(os.path.join(work, "skip.txt")):
        skip_n = len([x for x in open(os.path.join(work, "skip.txt")).read().split() if x.strip()])

    seen, invalids = set(), []
    cat_counts = Counter()
    if os.path.exists(raw_path):
        for ln in open(raw_path):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r["smiles"] in seen:
                continue
            seen.add(r["smiles"])
            invalids.append(r)
            cat_counts[r["category"]] += 1

    if args.max_per_cat > 0:
        capped, c = [], Counter()
        for r in invalids:
            if c[r["category"]] < args.max_per_cat:
                capped.append(r); c[r["category"]] += 1
        invalids = capped
        cat_counts = Counter(r["category"] for r in invalids)

    bench_path = os.path.join(args.out_dir, "benchmark.jsonl")
    with open(bench_path, "w") as f:
        for r in invalids:
            f.write(json.dumps(r) + "\n")

    total, valid = state["total"], state["valid"]
    stats = {
        "total_diagnosed": total,
        "valid": valid,
        "overall_validity_rate": round(valid / max(total, 1), 4),
        "invalid_total": total - valid,
        "unique_invalid_in_benchmark": len(invalids),
        "benchmark_category_distribution": dict(cat_counts),
        "syntactic_vs_semantic_in_benchmark": {
            "syntactic": sum(v for k, v in cat_counts.items() if k in S.SYNTACTIC),
            "semantic": sum(v for k, v in cat_counts.items() if k in S.SEMANTIC),
        },
        "rdkit_crashers_skipped": skip_n,
        "worker_restarts": restarts,
    }
    json.dump(stats, open(os.path.join(args.out_dir, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\n[harvest] benchmark -> {bench_path} ({len(invalids)} unique invalid SMILES)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "worker"):
        p = sub.add_parser(name)
        p.add_argument("--inputs", nargs="+", required=True)
        p.add_argument("--out_dir", default="data/benchmark")
        p.add_argument("--work_dir", default="data/benchmark/_work")
        p.add_argument("--min_len", type=int, default=4)
        p.add_argument("--max_len", type=int, default=200)
        p.add_argument("--max_per_cat", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "worker":
        worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
