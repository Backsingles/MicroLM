$ErrorActionPreference = "Continue"

$cmd = @(".\.venv\Scripts\python.exe", "-u", "scripts\train_sft.py", "--config", "configs\sft_lora.json")
$logPath = "reports\b3_sft_lora_0p5.log"
$mdPath = "reports\terminal_outputs.md"
$exitPath = "reports\b3_sft_lora_0p5.exitcode"
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"

$header = @"

## B3 SFT LoRA 0.5% valid run - $stamp

``````powershell
$($cmd -join " ")
``````

``````text
"@

Add-Content -Path $mdPath -Value $header -Encoding UTF8
& $cmd[0] $cmd[1..($cmd.Length - 1)] 2>&1 `
  | Tee-Object -FilePath $logPath `
  | Tee-Object -FilePath $mdPath -Append
$exitCode = $LASTEXITCODE

$footer = @"
``````

Exit code: $exitCode
"@

Add-Content -Path $mdPath -Value $footer -Encoding UTF8
Set-Content -Path $exitPath -Value $exitCode -Encoding UTF8
exit $exitCode
