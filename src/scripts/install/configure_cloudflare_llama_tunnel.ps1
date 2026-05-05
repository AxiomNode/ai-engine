param(
    [ValidateSet("dev", "stg", "pro")]
    [string]$Stage,

    [ValidateSet("windows", "windows-gpu", "vps-cpu", "vps-gpu")]
    [string]$Environment,

    [string]$LlamaPort = "",

    [string]$ComposeMode = "",

    [string[]]$ComposeArgs = @(),

    [hashtable]$Config = @{},

    [switch]$RestartTunnel,

    [switch]$StopTunnel,

    [switch]$NoStartLlama
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $RepoRoot

function Get-DotEnvConfig {
    param([string[]]$Paths)

    $config = @{}
    foreach ($path in $Paths) {
        if (-not (Test-Path $path)) {
            continue
        }

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

if (-not $Stage) {
    $Stage = "stg"
}

if (-not $Environment) {
    $Environment = "windows-gpu"
}

$EnvFile = "distributions/$Stage/$Environment.env"
if (-not (Test-Path $EnvFile)) {
    throw "Distribution file not found: $EnvFile"
}

$SecretsFile = ".env.secrets"
if (-not $Config -or $Config.Count -eq 0) {
    $Config = Get-DotEnvConfig -Paths @($EnvFile, $SecretsFile)
}

if (-not $LlamaPort) {
    $LlamaPort = if ($Config.ContainsKey("LLAMA_PORT") -and $Config["LLAMA_PORT"]) { $Config["LLAMA_PORT"] } else { "7002" }
}

if (-not $ComposeMode) {
    $ComposeMode = "cpu"
    if ($Environment -eq "vps-gpu" -or $Environment -eq "windows-gpu") {
        $ComposeMode = "gpu"
    }
}

if (-not $ComposeArgs -or $ComposeArgs.Count -eq 0) {
    $ComposeArgs = @("--env-file", $EnvFile)
    if (Test-Path $SecretsFile) {
        $ComposeArgs += @("--env-file", $SecretsFile)
    }
    $ComposeArgs += @("--profile", $ComposeMode, "-f", "docker-compose.yml")
}

function Get-ConfigValue {
    param(
        [string]$Key,
        [string]$DefaultValue = ""
    )

    if ($Config.ContainsKey($Key) -and $Config[$Key]) {
        return $Config[$Key]
    }

    return $DefaultValue
}

function Invoke-Docker {
    param([string[]]$Arguments)

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & docker @Arguments 2>&1 | ForEach-Object { $_.ToString() }
        if ($LASTEXITCODE -ne 0) {
            throw (($output | Out-String).Trim())
        }

        return ($output | Out-String).Trim()
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Test-HttpOk {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -Method Get -Uri $Url -TimeoutSec 10 -UseBasicParsing
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 300
    } catch {
        return $false
    }
}

function Wait-LocalLlama {
    param([string]$Port)

    $url = "http://127.0.0.1:$Port/v1/models"
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        if (Test-HttpOk -Url $url) {
            return
        }

        Start-Sleep -Seconds 2
    }

    throw "Local llama did not become ready at $url. Check model availability and container logs."
}

function Get-TunnelUrlFromLogs {
    param([string]$ContainerName)

    $logs = Invoke-Docker -Arguments @("logs", $ContainerName)
    $match = [regex]::Match($logs, "https://[a-z0-9-]+\.trycloudflare\.com")
    if (-not $match.Success) {
        return $null
    }

    return $match.Value
}

$containerName = Get-ConfigValue -Key "CLOUDFLARE_LLAMA_TUNNEL_CONTAINER" -DefaultValue "axiomnode-llama-cloudflared"
$targetFile = Get-ConfigValue -Key "CLOUDFLARE_LLAMA_TARGET_FILE" -DefaultValue "data/llama-cloudflare-target.json"
$localLlamaUrl = "http://host.docker.internal:$LlamaPort"

if ($StopTunnel) {
    Invoke-Docker -Arguments @("rm", "-f", $containerName) | Out-Null
    Write-Host "Cloudflare llama tunnel stopped: $containerName"
    return
}

if (-not $NoStartLlama) {
    if (Test-HttpOk -Url "http://127.0.0.1:$LlamaPort/v1/models") {
        Write-Host "Local llama already responds at http://127.0.0.1:$LlamaPort/v1/models."
    } else {
        Write-Host "Starting llama server for $Stage/$Environment (profile=$ComposeMode)..."
        docker compose @ComposeArgs up -d "llama-server-$ComposeMode"
    }
}

Write-Host "Waiting for local llama at http://127.0.0.1:$LlamaPort/v1/models..."
Wait-LocalLlama -Port $LlamaPort

$existing = docker ps --filter "name=^/$containerName$" --format "{{.Names}}"
if ($existing -and $RestartTunnel) {
    Invoke-Docker -Arguments @("rm", "-f", $containerName) | Out-Null
    $existing = ""
}

if (-not $existing) {
    Write-Host "Starting Cloudflare quick tunnel for llama..."
    Invoke-Docker -Arguments @(
        "run",
        "-d",
        "--name", $containerName,
        "cloudflare/cloudflared:latest",
        "tunnel",
        "--url", $localLlamaUrl
    ) | Out-Null
} else {
    Write-Host "Cloudflare llama tunnel already running: $containerName"
}

$publicUrl = $null
for ($attempt = 1; $attempt -le 30; $attempt++) {
    $publicUrl = Get-TunnelUrlFromLogs -ContainerName $containerName
    if ($publicUrl) {
        break
    }

    Start-Sleep -Seconds 2
}

if (-not $publicUrl) {
    throw "Cloudflare tunnel URL was not found in container logs for $containerName."
}

$modelsUrl = "$publicUrl/v1/models"
for ($attempt = 1; $attempt -le 20; $attempt++) {
    if (Test-HttpOk -Url $modelsUrl) {
        break
    }

    if ($attempt -eq 20) {
        throw "Cloudflare tunnel is up but llama probe failed at $modelsUrl."
    }

    Start-Sleep -Seconds 2
}

$uri = [Uri]$publicUrl
$target = [ordered]@{
    protocol = $uri.Scheme
    host = $uri.Host
    port = 443
    modelsUrl = "$($uri.Scheme)://$($uri.Host):443/v1/models"
    container = $containerName
    stage = $Stage
    environment = $Environment
    updatedAt = (Get-Date).ToUniversalTime().ToString("o")
}

$targetPath = Join-Path (Get-Location) $targetFile
$targetDir = Split-Path -Parent $targetPath
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
($target | ConvertTo-Json -Depth 4) | Set-Content -Path $targetPath -Encoding UTF8

Write-Host "Cloudflare llama tunnel ready."
Write-Host "  protocol: $($target.protocol)"
Write-Host "  host: $($target.host)"
Write-Host "  port: $($target.port)"
Write-Host "  probe: $($target.modelsUrl)"
Write-Host "  target file: $targetFile"