#!/usr/bin/env bash
# Run a list of conditions for one model LOCALLY (RDKit local) against the
# tunneled vLLM endpoint, strictly one at a time. Usage:
#   bash scripts/local_eval.sh <served_model> <port> <samples> <cond1> [cond2 ...]
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; PORT="$2"; SAMPLES="$3"; shift 3
BENCH=data/benchmark/benchmark_eval.jsonl
for COND in "$@"; do
  SUF=""; [ "$SAMPLES" -gt 1 ] && SUF="_pass${SAMPLES}"
  TAG="${MODEL}__${COND}${SUF}"
  echo "=== $TAG $(date +%H:%M:%S) ==="
  python3 -u src/evaluate.py --benchmark "$BENCH" --condition "$COND" \
    --base_url "http://localhost:${PORT}/v1" --model "$MODEL" \
    --workers 32 --samples "$SAMPLES" --out_dir results --tag "$TAG" \
    > "logs_${TAG}.log" 2>&1
  grep '"repair_rate"' "logs_${TAG}.log" | head -1 || true
done
echo "LOCAL_EVAL_DONE"
