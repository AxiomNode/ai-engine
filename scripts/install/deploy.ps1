param(
    [ValidateSet("dev", "stg", "pro")]
    [string]$Stage = "dev",

    [ValidateSet("windows", "vps-cpu", "vps-gpu")]
    [string]$Environment = "windows"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $RepoRoot

$EnvFile = "distributions/$Stage/$Environment.env"
if (-not (Test-Path $EnvFile)) {
    throw "Distribution file not found: $EnvFile"
}

$Profile = "cpu"
if ($Environment -eq "vps-gpu") {
    $Profile = "gpu"
}

$ComposeArgs = @("--env-file", $EnvFile, "--profile", $Profile, "-f", "docker-compose.yml")

New-Item -ItemType Directory -Force -Path "models" | Out-Null
New-Item -ItemType Directory -Force -Path "data" | Out-Null

Write-Host "Validating compose config for $Stage/$Environment (profile=$Profile)..."
docker compose @ComposeArgs config | Out-Null

Write-Host "Starting stack for $Stage/$Environment..."
docker compose @ComposeArgs up -d --build

Write-Host "Services status:"
docker compose @ComposeArgs ps

Write-Host "Health checks:"
try {
    $stats = Invoke-RestMethod -Method Get -Uri "http://localhost:8000/health" -TimeoutSec 10
    $api = Invoke-RestMethod -Method Get -Uri "http://localhost:8001/health" -TimeoutSec 20
    Write-Host "Stats health:" ($stats | ConvertTo-Json -Compress)
    Write-Host "API health:" ($api | ConvertTo-Json -Compress)
} catch {
    Write-Warning "Health check failed. Services may still be warming up."
    Write-Warning $_.Exception.Message
}
