#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$ReportDir = Join-Path $ProjectRoot "reports"
$LogPath = Join-Path $ReportDir "vllm_wsl_admin_install.log"
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

Start-Transcript -Path $LogPath -Append | Out-Null
try {
    Write-Host "== MicroLM vLLM WSL bootstrap =="
    Write-Host "Project root: $ProjectRoot"
    Write-Host "Log path: $LogPath"

    Write-Host "`n[1/4] Enabling Windows Subsystem for Linux..."
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -All -NoRestart

    Write-Host "`n[2/4] Enabling Virtual Machine Platform..."
    Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart

    Write-Host "`n[3/4] Setting WSL default version to 2..."
    try {
        wsl.exe --update --web-download
    } catch {
        Write-Warning "wsl --update --web-download failed; continuing to default-version/install checks."
    }
    wsl.exe --set-default-version 2

    Write-Host "`n[4/4] Installing Ubuntu distribution if needed..."
    wsl.exe --install -d Ubuntu --web-download --no-launch

    Write-Host "`nWSL bootstrap command completed."
    Write-Host "If Windows asks for a reboot, restart first, then run:"
    Write-Host "  wsl -d Ubuntu -- bash -lc `"cd /mnt/e/MicroLM && bash scripts/setup_vllm_ubuntu.sh`""
} finally {
    Stop-Transcript | Out-Null
}
