param(
    [ValidateSet("dev", "stg", "pro")]
    [string]$Stage = "dev",

    [ValidateSet("windows", "windows-gpu", "vps-cpu", "vps-gpu")]
    [string]$Environment = "windows"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $RepoRoot

function Get-DotEnvConfig {
    param(
        [string[]]$Paths
    )

    $config = @{}
    foreach ($path in $Paths) {
        foreach ($line in Get-Content $path) {
            $trimmed = $line.Trim()
            if (-not $trimmed -or $trimmed.StartsWith("#")) {
                continue
            }

            $separatorIndex = $trimmed.IndexOf("=")
            if ($separatorIndex -lt 1) {
                continue
            }

            $key = $trimmed.Substring(0, $separatorIndex).Trim()
            $value = $trimmed.Substring($separatorIndex + 1).Trim()
            $config[$key] = $value
        }
    }

    return $config
}

$EnvFile = "distributions/$Stage/$Environment.env"
if (-not (Test-Path $EnvFile)) {
    throw "Distribution file not found: $EnvFile"
}

$SecretsFile = ".env.secrets"
if (-not (Test-Path $SecretsFile)) {
    throw "Secrets file not found: $SecretsFile. Run: node ../secrets/scripts/prepare-runtime-secrets.mjs $Stage ai-engine"
}

$Config = Get-DotEnvConfig -Paths @($EnvFile, $SecretsFile)
$LlamaPort = if ($Config.ContainsKey("LLAMA_PORT") -and $Config["LLAMA_PORT"]) { $Config["LLAMA_PORT"] } else { "7002" }
$StatsPort = if ($Config.ContainsKey("STATS_PORT") -and $Config["STATS_PORT"]) { $Config["STATS_PORT"] } else { "7000" }
$ApiPort = if ($Config.ContainsKey("API_PORT") -and $Config["API_PORT"]) { $Config["API_PORT"] } else { "7001" }

$ComposeMode = "cpu"
if ($Environment -eq "vps-gpu" -or $Environment -eq "windows-gpu") {
    $ComposeMode = "gpu"
}

$ComposeArgs = @("--env-file", $EnvFile, "--env-file", $SecretsFile, "--profile", $ComposeMode, "-f", "docker-compose.yml")

New-Item -ItemType Directory -Force -Path "models" | Out-Null
New-Item -ItemType Directory -Force -Path "data" | Out-Null

Write-Host "Validating compose config for $Stage/$Environment (profile=$ComposeMode)..."
docker compose @ComposeArgs config | Out-Null

Write-Host "Starting stack for $Stage/$Environment..."
docker compose @ComposeArgs up -d --build

if ($Environment -like "windows*") {
    & (Join-Path $ScriptDir "configure_windows_public_ports.ps1") -Stage $Stage -Environment $Environment -LlamaPort $LlamaPort -StatsPort $StatsPort -ApiPort $ApiPort -Config $Config
}

Write-Host "Services status:"
docker compose @ComposeArgs ps

Write-Host "Health checks:"
try {
    $stats = Invoke-RestMethod -Method Get -Uri "http://localhost:$StatsPort/health" -TimeoutSec 10
    $api = Invoke-RestMethod -Method Get -Uri "http://localhost:$ApiPort/health" -TimeoutSec 20
    Write-Host "Stats health:" ($stats | ConvertTo-Json -Compress)
    Write-Host "API health:" ($api | ConvertTo-Json -Compress)
} catch {
    Write-Warning "Health check failed. Services may still be warming up."
    Write-Warning $_.Exception.Message
}

Write-Host "Published host ports: llama=$LlamaPort stats=$StatsPort api=$ApiPort"

if ($Environment -like "windows*") {
    $autoExposeVpsRelay = if ($Config.ContainsKey("AUTO_EXPOSE_VPS_RELAY")) { $Config["AUTO_EXPOSE_VPS_RELAY"] } else { "false" }
    if ($autoExposeVpsRelay -in @("true", "1", "yes", "on")) {
        & (Join-Path $ScriptDir "configure_vps_reverse_relay.ps1") -Stage $Stage -Environment $Environment -StatsPort $StatsPort -ApiPort $ApiPort -Config $Config
    }
}
