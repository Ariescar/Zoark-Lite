# Zoark end to end on this computer: NVIDIA Nemotron on Nebius Token Factory -> NVIDIA Newton.
# Before running, in the same PowerShell window:   $env:NEBIUS_API_KEY = "<your Token Factory key>"
# Then:   powershell -ExecutionPolicy Bypass -File scripts\run_nemotron.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $env:NEBIUS_API_KEY) { Write-Error 'Set $env:NEBIUS_API_KEY first'; exit 1 }
$env:PYTHONUTF8 = "1"
function Step($title, [scriptblock]$cmd) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE) { Write-Error "step failed: $title"; exit $LASTEXITCODE }
}
Step "install"                     { python -m pip install -q -r requirements.txt }
Step "1. Nemotron models on Token Factory" { python -m zoark models }
Step "2. Extract (Nemotron reads a text)"  { python -m zoark extract --text examples/source_snow_leopard.txt --save --out examples/nemotron_extract.json }
Step "3. Compose + Build + Prove (Nemotron -> Newton)" {
    python -m zoark order "front-heavy quadruped with a long tail for balance on rough ground" -n 12 -o out/nemotron --save-structure examples/nemotron_compose.json }
Write-Host "`nDone. Send these back: examples\nemotron_extract.json, examples\nemotron_compose.json, out\nemotron\manifest.json, out\nemotron\results.csv" -ForegroundColor Green
