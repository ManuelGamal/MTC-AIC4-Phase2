param(
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $repoRoot 'tools\orin_live_simulator.py'

if ($Help) {
    Write-Host 'Runs the Orin contest simulator with the bundled preset.'
    Write-Host ''
    Write-Host 'Default command:'
    Write-Host '  python tools/orin_live_simulator.py --preset contest --simulate-camera --threshold-ms 30'
    Write-Host ''
    Write-Host 'You can append extra simulator flags after the wrapper, for example:'
    Write-Host '  .\tools\run_orin_simulator.ps1 --sequence dataset1/cows --display'
    exit 0
}

Set-Location $repoRoot

if (-not $env:ORIN_TF32) {
    $env:ORIN_TF32 = '1'
}

$arguments = @(
    $scriptPath
    '--preset', 'contest'
    '--simulate-camera'
    '--threshold-ms', '30'
)

if ($ExtraArgs.Count -gt 0) {
    $arguments += $ExtraArgs
}

python @arguments