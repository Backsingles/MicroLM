param(
    [string]$Distro = "MicroLM-Ubuntu",
    [int]$Port = 8000,
    [string]$HostName = "0.0.0.0",
    [int]$MaxModelLen = 4096,
    [string]$VenvDir = "/root/.venvs/microlm-vllm"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$WslProjectRoot = "/mnt/" + $ProjectRoot.Substring(0, 1).ToLowerInvariant() + $ProjectRoot.Substring(2).Replace("\", "/")
$ModelPath = "$WslProjectRoot/outputs/qwen_lora_merged_final"

$Command = @"
set -euo pipefail
cd "$WslProjectRoot"
source "$VenvDir/bin/activate"
export VLLM_NO_USAGE_STATS="`${VLLM_NO_USAGE_STATS:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="`${VLLM_USE_FLASHINFER_SAMPLER:-0}"
vllm serve "$ModelPath" --host "$HostName" --port "$Port" --max-model-len "$MaxModelLen" --dtype auto
"@

wsl.exe -d $Distro -- bash -lc $Command
