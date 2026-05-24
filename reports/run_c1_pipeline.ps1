$ErrorActionPreference = "Continue"

$mdPath = "reports\terminal_outputs.md"
$logPath = "reports\c1_instructie_pipeline.log"
$exitPath = "reports\c1_instructie_pipeline.exitcode"
$steps = @(
  "scripts\01_normalize.py",
  "scripts\02_filter.py",
  "scripts\03_quality_tier.py",
  "scripts\04_derive_tasks.py",
  "scripts\05_stratified_sample.py",
  "scripts\06_to_chat_jsonl.py"
)

function Add-Markdown {
  param([string]$Text)
  for ($i = 0; $i -lt 10; $i++) {
    try {
      Add-Content -Path $mdPath -Value $Text -Encoding UTF8 -ErrorAction Stop
      return
    } catch {
      Start-Sleep -Milliseconds 500
    }
  }
  Add-Content -Path "reports\terminal_outputs_recovery.md" -Value $Text -Encoding UTF8
}

function Write-Both {
  param([string]$Text)
  Write-Output $Text
  Add-Content -Path $logPath -Value $Text -Encoding UTF8
  Add-Markdown -Text $Text
}

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Set-Content -Path $logPath -Value "" -Encoding UTF8

$header = @"

## C1 InstructIE six-step pipeline - $stamp

``````powershell
python scripts/01_normalize.py
python scripts/02_filter.py
python scripts/03_quality_tier.py
python scripts/04_derive_tasks.py
python scripts/05_stratified_sample.py
python scripts/06_to_chat_jsonl.py
``````

``````text
"@
Add-Markdown -Text $header

$finalExit = 0
foreach ($step in $steps) {
  $stepStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
  Write-Both ""
  Write-Both ">>> START $step at $stepStamp"
  $duration = Measure-Command {
    & .\.venv\Scripts\python.exe -u $step 2>&1 | ForEach-Object {
      Write-Both -Text $_.ToString()
    }
  }
  $code = $LASTEXITCODE
  Write-Both "<<< END $step exit=$code elapsed=$($duration.ToString())"
  if ($code -ne 0) {
    $finalExit = $code
    break
  }
}

$footer = @"
``````

Exit code: $finalExit
"@
Add-Markdown -Text $footer
Set-Content -Path $exitPath -Value $finalExit -Encoding UTF8
exit $finalExit
