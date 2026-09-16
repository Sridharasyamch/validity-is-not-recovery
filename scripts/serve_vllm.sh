#!/usr/bin/env bash
# Serve a model with vLLM on the GPU box. Stops any running server first.
# Usage (ON the instance):  bash scripts/serve_vllm.sh <hf_model> <served_name> [max_model_len] [gpu_util]
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; NAME="$2"; MAXLEN="${3:-2048}"; UTIL="${4:-0.90}"

echo "stopping any running vLLM..."
pkill -9 -f "vllm.entrypoints" 2>/dev/null; sleep 5
mkdir -p logs
echo "starting vLLM for $MODEL (served as $NAME)..."
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port 8000 --dtype bfloat16 \
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" \
  --served-model-name "$NAME" > "logs/vllm_${NAME}.log" 2>&1 &
echo "launched PID $!  (log: logs/vllm_${NAME}.log)"
