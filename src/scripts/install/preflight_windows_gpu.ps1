param(
    [ValidateSet("dev", "stg", "pro")]
    [string]$Stage = "dev",

    [string]$Environment = "windows-gpu"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

function Get-EnvValue([string]$FilePath, [string]$Name) {
    if (-not (Test-Path $FilePath)) {
        throw "Distribution env file not found: $FilePath"
    }

    $match = Select-String -Path $FilePath -Pattern "^$Name=(.*)$" | Select-Object -First 1
    if (-not $match) {
        return $null
    }

    return $match.Matches[0].Groups[1].Value.Trim()
}

function Get-ModelDownloadHint([string]$ModelFileName) {
    switch ($ModelFileName) {
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf" { return "python -m ai_engine.llm.model_manager download qwen2.5-7b" }
        "Qwen2.5-3B-Instruct-Q4_K_M.gguf" { return "python -m ai_engine.llm.model_manager download qwen2.5-3b" }
        "Phi-3.5-mini-instruct-Q4_K_M.gguf" { return "python -m ai_engine.llm.model_manager download phi-3.5-mini" }
        default { return $null }
    }
}

function Write-Section([string]$Title) {
    Write-Host ""
    Write-Host "==> $Title"
}

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

Require-Command "docker"
Require-Command "wsl.exe"
Require-Command "nvidia-smi"

$distributionEnvFile = Join-Path $RepoRoot "distributions\$Stage\$Environment.env"
$modelsDirValue = Get-EnvValue -FilePath $distributionEnvFile -Name "AI_ENGINE_MODELS_DIR"
$modelFileValue = Get-EnvValue -FilePath $distributionEnvFile -Name "LLAMA_MODEL_FILE"

if (-not $modelsDirValue -or -not $modelFileValue) {
    throw "Unable to resolve AI_ENGINE_MODELS_DIR or LLAMA_MODEL_FILE from $distributionEnvFile"
}

$modelsDirPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $modelsDirValue))
$modelPath = Join-Path $modelsDirPath $modelFileValue

Write-Section "Host GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

Write-Section "Model file"
Write-Host "Distribution env: $distributionEnvFile"
Write-Host "Expected model: $modelPath"

if (-not (Test-Path $modelPath)) {
    $downloadHint = Get-ModelDownloadHint -ModelFileName $modelFileValue
    if ($downloadHint) {
        throw "Configured GGUF model not found: $modelPath`nDownload it first from $RepoRoot with:`n  $downloadHint"
    }

    throw "Configured GGUF model not found: $modelPath"
}

Write-Section "Docker daemon"
$dockerInfo = docker info --format '{{json .}}' | ConvertFrom-Json
if (-not $dockerInfo.ServerVersion) {
    throw "Docker daemon is not available. Start Docker Desktop first."
}

$cpuCount = [int]$dockerInfo.NCPU
$memoryBytes = [double]$dockerInfo.MemTotal
$memoryGiB = [math]::Round($memoryBytes / 1GB, 2)

Write-Host "Docker CPUs: $cpuCount"
Write-Host "Docker Memory GiB: $memoryGiB"

if ($cpuCount -lt 20) {
    Write-Warning "Docker exposes fewer than 20 CPUs. Recommended for windows-gpu profile: 20."
}

if ($memoryBytes -lt 40GB) {
    Write-Warning "Docker exposes less than 40 GB RAM. Recommended for windows-gpu profile: 40 GB."
}

Write-Section "WSL backend"
wsl.exe -d docker-desktop sh -lc "nproc && grep MemTotal /proc/meminfo"

Write-Section "GPU passthrough to containers"
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

Write-Section "Result"
Write-Host "Preflight completed. If warnings appeared above, increase Docker Desktop / WSL resources before running windows-gpu."