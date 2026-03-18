param(
    [ValidateSet("dev", "stg", "pro")]
    [string]$Stage = "dev",

    [ValidateSet("windows", "vps-cpu", "vps-gpu")]
    [string]$Environment = "windows",

    [switch]$UseCpuProfile
)

$ErrorActionPreference = "Stop"

# Backward-compatible wrapper.
# Prefer from repo root: ./src/scripts/install/deploy.ps1 -Stage dev -Environment windows

if ($UseCpuProfile) {
    $Environment = "vps-cpu"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $RepoRoot

& (Join-Path $ScriptDir "deploy.ps1") -Stage $Stage -Environment $Environment
