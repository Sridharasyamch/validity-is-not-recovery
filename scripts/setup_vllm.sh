#!/usr/bin/env bash
# Minimal vLLM-only setup for a fresh Thunder box (eval runs locally, so no rdkit
# or data needed here). Applies the libcuda linker fix + pinned vLLM stack.
set -euo pipefail
LIBDIR=/usr/lib/x86_64-linux-gnu
if [ ! -e "$LIBDIR/libcuda.so" ] && [ -e "$LIBDIR/libcuda.so.1" ]; then
  sudo ln -sf "$LIBDIR/libcuda.so.1" "$LIBDIR/libcuda.so"
fi
echo "$LIBDIR" | sudo tee /etc/ld.so.conf.d/thunder-cuda.conf >/dev/null
sudo ldconfig
export LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}"; export TRITON_LIBCUDA_PATH="$LIBDIR"
nvidia-smi || { echo "no GPU"; exit 1; }
pip install -q --upgrade pip
pip install -q "vllm==0.11.0" "transformers==4.56.2" "huggingface_hub<1.0" accelerate
python -c "import vllm,torch;print('vllm',vllm.__version__,'cuda',torch.cuda.is_available())"
echo "setup_vllm done."
