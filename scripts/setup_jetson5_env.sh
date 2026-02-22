#!/usr/bin/env bash
# Setup Tree Line Follower conda env on Jetson 5 (Orin / JetPack 6).
# Run from project root after: conda activate tree-line-follower-jetson5
#
# Prerequisites on device:
#   sudo apt-get update
#   sudo apt-get install -y python3-pip libopenblas-dev
#
# Optional: set JP_VER to your JetPack version (e.g. 60 for 6.0, 62 for 6.2).
# Default: 60 (JetPack 6.0). For 6.2 we use Jetson AI Lab index for torch+torchvision.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

JP_VER="${JP_VER:-60}"
ARCH="$(uname -m)"

echo "=== Tree Line Follower – Jetson 5 env setup (JP_VER=${JP_VER}, arch=${ARCH}) ==="

# 1) numpy first (recommended before PyTorch on Jetson)
pip install --upgrade "numpy>=1.26,<3.0"

# 2) PyTorch: use NVIDIA Jetson wheels only on aarch64; on x86_64 use PyPI (dev machine)
if [[ "$ARCH" != "aarch64" ]]; then
  echo "Detected ${ARCH} (not Jetson aarch64). Installing PyTorch from PyPI..."
  pip install torch torchvision
elif [[ "$JP_VER" == "62" ]]; then
  echo "Installing PyTorch + torchvision for JetPack 6.2 (Jetson AI Lab index)..."
  pip install torch torchvision --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu126/
else
  # JetPack 6.0 / 6.1 on aarch64: use NVIDIA redist (v60). Override TORCH_WHEEL if you need a different build.
  BASE="https://developer.download.nvidia.com/compute/redist/jp/v${JP_VER}/pytorch"
  if [[ -z "${TORCH_WHEEL:-}" ]]; then
    TORCH_WHEEL="${BASE}/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl"
  fi
  echo "Installing PyTorch from NVIDIA wheel..."
  pip install --no-cache-dir "$TORCH_WHEEL"
  pip install "torchvision>=0.18,<0.20" 2>/dev/null || true
fi

# 3) Project deps (opencv-headless, scipy, numba, matplotlib, ultralytics, roboflow)
pip install -r requirements-jetson5.txt

echo "=== Done. Verify with: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\" ==="
