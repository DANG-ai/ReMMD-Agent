# Entrypoint for the Qwen3.5-4B backend on Windows (PowerShell).
#
# Make sure configs\qwen3_5_4b.yaml has the correct ``api.model``,
# ``api.api_key`` and ``api.primary_base_url`` set before running.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ConfigFile = Join-Path $ProjectRoot "configs\qwen3_5_4b.yaml"

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
