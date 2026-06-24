# Entrypoint for the GPT-5.2 backend on Windows (PowerShell).
#
# Usage examples (from the t2-agent project root):
#   .\scripts\run_gpt5_2.ps1
#   .\scripts\run_gpt5_2.ps1 --smoke
#   .\scripts\run_gpt5_2.ps1 --limit 20
#   .\scripts\run_gpt5_2.ps1 --indices 0,5,10
#
# Any remaining flag is forwarded to scripts/run_remmdbench.py as-is.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ConfigFile = Join-Path $ProjectRoot "configs\gpt5_2.yaml"

if (-not $env:T2AGENT_CONDA_ENV) { $CondaEnv = "mmd" } else { $CondaEnv = $env:T2AGENT_CONDA_ENV }
if (-not $env:T2AGENT_MAX_WORKERS) { $MaxWorkers = "4" } else { $MaxWorkers = $env:T2AGENT_MAX_WORKERS }

Set-Location $ProjectRoot

$Forwarded = @()
if ($args.Count -gt 0) { $Forwarded = $args }

if (Get-Command conda -ErrorAction SilentlyContinue) {
    conda run -n $CondaEnv --no-capture-output python "scripts\run_remmdbench.py" `
        --config "$ConfigFile" `
        --max-workers $MaxWorkers `
        @Forwarded
} else {
    python "scripts\run_remmdbench.py" `
        --config "$ConfigFile" `
        --max-workers $MaxWorkers `
        @Forwarded
}
