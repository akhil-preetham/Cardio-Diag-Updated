param(
    [ValidateSet("preflight","audit","corrected","author","report","all-safe")]
    [string]$Mode = "all-safe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Config = Join-Path $ProjectRoot "config\runner_config.json"
$Python = Join-Path $ProjectRoot ".venv-repro\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }

Set-Location $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "runner"

function Invoke-Step([string]$Name, [string]$Script) {
    Write-Host "`n========== $Name ==========" -ForegroundColor Cyan
    & $Python (Join-Path $ProjectRoot $Script) $Config
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

switch ($Mode) {
    "preflight" { Invoke-Step "PREFLIGHT" "runner\preflight.py" }
    "audit" {
        Invoke-Step "PREFLIGHT" "runner\preflight.py"
        Invoke-Step "STATIC NOTEBOOK AUDIT" "runner\audit.py"
    }
    "corrected" {
        Invoke-Step "PREFLIGHT" "runner\preflight.py"
        Invoke-Step "CORRECTED VALIDATION" "runner\corrected_validation.py"
    }
    "author" {
        Invoke-Step "AUTHOR NOTEBOOK EXECUTION" "runner\execute_notebooks.py"
    }
    "report" {
        Invoke-Step "FINAL REPORT" "runner\finalize_report.py"
    }
    "all-safe" {
        Invoke-Step "PREFLIGHT" "runner\preflight.py"
        Invoke-Step "STATIC NOTEBOOK AUDIT" "runner\audit.py"
        Invoke-Step "DATASET VALIDATION" "runner\dataset_validation.py"
        Write-Host "`nAuthor notebook execution remains disabled by default; corrected validation is a separate explicit mode." -ForegroundColor Yellow
        Write-Host "Review audit reports before changing either track in config\runner_config.json."
    }
}

