param(
    [ValidateSet("dev", "stg", "pro")]
    [string]$Stage = "dev",

    [ValidateSet("windows", "windows-gpu", "vps-cpu", "vps-gpu")]
    [string]$Environment = "windows",

    [switch]$UseCpuProfile,

    [switch]$UseGpuProfile
)

$ErrorActionPreference = "Stop"

# Backward-compatible wrapper.
# Prefer from repo root: ./src/scripts/install/deploy.ps1 -Stage dev -Environment windows

if ($UseCpuProfile -and $UseGpuProfile) {
    throw "UseCpuProfile and UseGpuProfile cannot be used together."
}

if ($UseCpuProfile) {
    $Environment = "windows"
}

if ($UseGpuProfile) {
    $Environment = "windows-gpu"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $RepoRoot

if ($Environment -eq "windows-gpu") {
    & (Join-Path $ScriptDir "preflight_windows_gpu.ps1") -Stage $Stage -Environment $Environment
}

& (Join-Path $ScriptDir "deploy.ps1") -Stage $Stage -Environment $Environment
