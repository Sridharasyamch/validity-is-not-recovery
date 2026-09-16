#!/usr/bin/env bash
# For each model: serve with vLLM, run all repair conditions, then the
# model-independent baselines once. Run ON the Thunder GPU instance.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f env.sh ] && source env.sh   # LD_LIBRARY_PATH / TRITON_LIBCUDA_PATH for vLLM

BENCH_FULL=${BENCH_FULL:-data/benchmark/benchmark.jsonl}
BENCH=${BENCH:-data/benchmark/benchmark_eval.jsonl}
N_EVAL=${N_EVAL:-2000}        # size of the fixed, representative eval subset
LIMIT=0                       # BENCH is already the subset; evaluate all of it
WORKERS=${WORKERS:-64}

# Build a fixed, seeded, representative eval subset (identical across all runs).
if [ ! -f "$BENCH" ]; then
  python -c "import json,random; \
rows=[l for l in open('$BENCH_FULL') if l.strip()]; \
random.seed(42); random.shuffle(rows); \
open('$BENCH','w').writelines(rows[:$N_EVAL]); \
print(f'eval subset: {min(len(rows),$N_EVAL)} of {len(rows)} invalids')"
fi
PORT=${PORT:-8000}
MAXLEN=${MAXLEN:-2048}
# ungated by default; add Llama/Gemma if an HF token is exported
MODELS=${MODELS:-"Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct"}

wait_for_server() {
  echo "  waiting for vLLM..."
  for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then echo "  up"; return 0; fi
    sleep 5
  done
  echo "  server did not come up"; return 1
}

# model-independent baselines (once)
python src/evaluate.py --benchmark "$BENCH" --condition rule     --limit "$LIMIT" --out_dir results --tag rule
python src/evaluate.py --benchmark "$BENCH" --condition identity --limit "$LIMIT" --out_dir results --tag identity

for MODEL in $MODELS; do
  ALIAS=$(basename "$MODEL")
  echo "=== serving $MODEL ==="
  nohup vllm serve "$MODEL" --port "$PORT" --dtype bfloat16 \
    --max-model-len "$MAXLEN" --gpu-memory-utilization 0.90 \
    --served-model-name "$ALIAS" > "runs/vllm_${ALIAS}.log" 2>&1 &
  VPID=$!
  if wait_for_server; then
    for COND in no_feedback feedback iterative; do
      echo "--- $ALIAS / $COND ---"
      python src/evaluate.py --benchmark "$BENCH" --condition "$COND" \
        --base_url "http://localhost:${PORT}/v1" --model "$ALIAS" \
        --limit "$LIMIT" --workers "$WORKERS" --out_dir results \
        --tag "${ALIAS}__${COND}"
    done
  fi
  echo "  stopping server ($VPID)"
  kill "$VPID" 2>/dev/null; sleep 8; kill -9 "$VPID" 2>/dev/null || true
done

echo "run_eval_all done. results in results/"
