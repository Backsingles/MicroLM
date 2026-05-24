#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_DIR="$PROJECT_ROOT/reports"
LOG_PATH="$REPORT_DIR/vllm_wsl_setup.log"
VENV_DIR="${VLLM_VENV_DIR:-$HOME/.venvs/microlm-vllm}"

mkdir -p "$REPORT_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "== MicroLM vLLM Ubuntu setup =="
echo "Project root: $PROJECT_ROOT"
echo "Log path: $LOG_PATH"
echo "Venv: $VENV_DIR"
date

echo
echo "[1/6] System information"
uname -a
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found inside WSL. Install/update NVIDIA Windows driver with WSL CUDA support."
fi

echo
echo "[2/6] Installing OS packages"
if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi
$SUDO apt-get update
$SUDO apt-get install -y python3 python3-venv python3-pip build-essential git curl

echo
echo "[3/6] Creating Python virtual environment"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo
echo "[4/6] Installing vLLM"
python -m pip install vllm

echo
echo "[5/6] Verifying Python packages"
python - <<'PY'
import torch
import vllm

print("vllm", vllm.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

echo
echo "[6/6] Checking merged model files"
test -f "$PROJECT_ROOT/outputs/qwen_lora_merged_final/config.json"
test -f "$PROJECT_ROOT/outputs/qwen_lora_merged_final/model.safetensors"
echo "Merged model is ready: $PROJECT_ROOT/outputs/qwen_lora_merged_final"

echo
echo "vLLM setup finished."
echo "Start server from WSL with:"
echo "  cd $PROJECT_ROOT"
echo "  source $VENV_DIR/bin/activate"
echo "  vllm serve $PROJECT_ROOT/outputs/qwen_lora_merged_final --host 0.0.0.0 --port 8000 --max-model-len 4096 --dtype auto"
