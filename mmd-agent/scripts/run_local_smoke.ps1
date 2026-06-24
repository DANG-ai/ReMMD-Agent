# =============================================================================
# Windows / conda local smoke test for the unified MMD-Agent.
#
# This script is for LOCAL TESTING ONLY (the user said "本地运行的环境为
# conda 的 mmd 环境，只用于测试"). It runs the gpt-5.2 path against a small
# subset of ReMMDBench so you can sanity-check the pipeline end-to-end.
#
# Usage:
#     conda activate mmd
#     powershell -ExecutionPolicy Bypass -File .\scripts\run_local_smoke.ps1
#
# Optional arguments are forwarded to run_mmd_agent.py, e.g.:
#     .\scripts\run_local_smoke.ps1 -MaxSamples 1
#     .\scripts\run_local_smoke.ps1 -MaxSamples 3 -NumWorkers 2
# =============================================================================

param(
    [int]$MaxSamples = 2,
    [int]$NumWorkers = 1,
    [string]$Model = "gpt-5.2",
    [string]$BaseUrl = "http://YOUR_GPT_ENDPOINT/v1",
    [string]$ApiKey = "sk-YOUR_GPT_API_KEY_HERE",
    [int]$SerperKeyIndex = 5,
    [string]$BenchRoot = "C:\path\to\ReMMDBench",
    [string]$SerperKeyFile = "C:\path\to\serper_api.txt",
    [string]$DatasetName = "remmdbench_smoke",
    [string]$RunName = "",
    [int]$MaxImages = 6
)

$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $Here "..")).Path
$EvalRoot = Join-Path $ProjectRoot "eval"
$OutputRoot = Join-Path $ProjectRoot "outputs"

if (-not (Test-Path $BenchRoot)) {
    Write-Error "ReMMDBench root not found: $BenchRoot"
}
if (-not (Test-Path $SerperKeyFile)) {
    Write-Error "Serper key file not found: $SerperKeyFile"
}

if (-not $RunName) {
    $RunName = (Get-Date -Format "yyyyMMdd_HHmmss") + "_local_smoke"
}

Write-Host "=============================================================="
Write-Host "Local smoke test (mmd conda env)"
Write-Host "  model            : $Model"
Write-Host "  base_url         : $BaseUrl"
Write-Host "  bench_root       : $BenchRoot"
Write-Host "  serper_key_file  : $SerperKeyFile (index $SerperKeyIndex)"
Write-Host "  output_root      : $OutputRoot"
Write-Host "  max_samples      : $MaxSamples"
Write-Host "  num_workers      : $NumWorkers"
Write-Host "  run_name         : $RunName"
Write-Host "=============================================================="

Push-Location $EvalRoot
try {
    python .\run_mmd_agent.py `
        --sampled_root      $BenchRoot `
        --serper_key_file   $SerperKeyFile `
        --serper_key_index  $SerperKeyIndex `
        --model_name        $Model `
        --base_url          $BaseUrl `
        --api_key           $ApiKey `
        --answer_path       $OutputRoot `
        --dataset_name      $DatasetName `
        --run_name          $RunName `
        --max_samples       $MaxSamples `
        --max_images        $MaxImages `
        --num_workers       $NumWorkers `
        --image_detail      low
}
finally {
    Pop-Location
}
