#!/usr/bin/env bash
# Train the generator, sample at several temperatures, harvest the benchmark.
# Run ON the Thunder GPU instance from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT=${LIMIT:-1000000}     # use most of GuacaMol train on GPU
EPOCHS=${EPOCHS:-10}
NSAMP=${NSAMP:-100000}      # per temperature

echo "=== train generator (cuda) ==="
python src/gen_model.py train --data data/guacamol_train.smiles --out runs/lm \
    --limit "$LIMIT" --epochs "$EPOCHS" --batch 512 --hidden 768 --layers 3 --log_every 200

echo "=== sample at multiple temperatures ==="
for T in 1.0 1.2 1.5; do
    python src/gen_model.py sample --ckpt runs/lm/best.pt --n "$NSAMP" --temp "$T" \
        --batch 1024 --out "runs/gen_t${T}.smiles"
done

echo "=== harvest benchmark (crash-resilient) ==="
python src/harvest.py run --inputs runs/gen_t1.0.smiles runs/gen_t1.2.smiles runs/gen_t1.5.smiles \
    --out_dir data/benchmark --work_dir data/benchmark/_work

echo "run_generate done."
