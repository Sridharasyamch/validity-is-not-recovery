#!/usr/bin/env bash
# Run ON the Thunder GPU instance once, to install dependencies.
# Bakes in the proven Thunder fixes: libcuda linker path + pinned vLLM stack.
set -euo pipefail

echo "=== libcuda linker fix (Thunder base image) ==="
LIBDIR=/usr/lib/x86_64-linux-gnu
if [ ! -e "$LIBDIR/libcuda.so" ] && [ -e "$LIBDIR/libcuda.so.1" ]; then
  sudo ln -sf "$LIBDIR/libcuda.so.1" "$LIBDIR/libcuda.so"
fi
echo "$LIBDIR" | sudo tee /etc/ld.so.conf.d/thunder-cuda.conf >/dev/null
sudo ldconfig
export LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}"
export TRITON_LIBCUDA_PATH="$LIBDIR"
# persist for run scripts (repo-local env file, always writable)
printf 'export LD_LIBRARY_PATH=%s:${LD_LIBRARY_PATH:-}\nexport TRITON_LIBCUDA_PATH=%s\n' \
  "$LIBDIR" "$LIBDIR" > env.sh || true

echo "=== GPU ==="
nvidia-smi || { echo "no GPU visible"; exit 1; }

echo "=== pip installs (pinned, proven combo) ==="
pip install -q --upgrade pip
pip install -q "vllm==0.11.0" "transformers==4.56.2" "huggingface_hub<1.0" \
    accelerate rdkit "openai>=1.40" numpy pandas tqdm selfies

echo "=== versions ==="
python -c "import torch,vllm,rdkit,openai,transformers; \
print('torch',torch.__version__,'cuda',torch.cuda.is_available()); \
print('vllm',vllm.__version__,'transformers',transformers.__version__,'rdkit',rdkit.__version__)"

echo "=== download GuacaMol train (on instance, fast link) ==="
mkdir -p data runs results
if [ ! -s data/guacamol_train.smiles ]; then
  curl -sL -o data/guacamol_train.smiles "https://ndownloader.figshare.com/files/13612760"
fi
echo "train lines: $(wc -l < data/guacamol_train.smiles)"
echo "setup_remote done."
