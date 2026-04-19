param(
    [Parameter(Mandatory = $true)]
    [string]$Stage,

    [Parameter(Mandatory = $true)]
    [string]$Environment,

    [Parameter(Mandatory = $true)]
    [string]$StatsPort,

    [Parameter(Mandatory = $true)]
    [string]$ApiPort,

    [Parameter(Mandatory = $true)]
    [hashtable]$Config
)

$ErrorActionPreference = "Stop"

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

function Resolve-UserPath {
    param([string]$PathValue)

    if (-not $PathValue) {
        return $PathValue
    }

    if ($PathValue.StartsWith("~/") -or $PathValue.StartsWith("~\\")) {
        $home = $env:USERPROFILE
        return Join-Path $home $PathValue.Substring(2)
    }

    return $PathValue
}

function Invoke-NativeCommand {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [switch]$AllowNonZeroExit
    )

    $output = & $FilePath @Arguments 2>&1
    if (-not $AllowNonZeroExit -and $LASTEXITCODE -ne 0) {
        throw (($output | Out-String).Trim())
    }

    return ($output | Out-String).Trim()
}

function Ensure-ReverseTunnel {
    param(
        [string]$SshPath,
        [string]$SshKey,
        [string]$SshHost,
        [string]$StatsTunnelPort,
        [string]$ApiTunnelPort,
        [string]$StatsPort,
        [string]$ApiPort
    )

    $existingProcess = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match '^ssh(\.exe)?$' -and
            $_.CommandLine -like "*127.0.0.1:$StatsTunnelPort:127.0.0.1:$StatsPort*" -and
            $_.CommandLine -like "*127.0.0.1:$ApiTunnelPort:127.0.0.1:$ApiPort*" -and
            $_.CommandLine -like "*$SshHost*"
        } |
        Select-Object -First 1

    if ($existingProcess) {
        Write-Host "Reverse tunnel already running (PID=$($existingProcess.ProcessId))."
        return
    }

    $arguments = @(
        "-i", $SshKey,
        "-o", "ExitOnForwardFailure=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-N",
        "-R", "127.0.0.1:$StatsTunnelPort`:127.0.0.1:$StatsPort",
        "-R", "127.0.0.1:$ApiTunnelPort`:127.0.0.1:$ApiPort",
        $SshHost
    )

    $process = Start-Process -FilePath $SshPath -ArgumentList $arguments -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 2
    if ($process.HasExited) {
        throw "Reverse tunnel process exited immediately. Check SSH connectivity to $SshHost."
    }

    Write-Host "Reverse tunnel started (PID=$($process.Id))."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$relayScript = Join-Path $scriptDir "vps_tcp_relay.py"
$sshPath = (Get-Command ssh -ErrorAction Stop).Source
$scpPath = (Get-Command scp -ErrorAction Stop).Source

$sshHost = Get-ConfigValue -Key "VPS_RELAY_SSH_HOST" -DefaultValue "sebss@amksandbox.cloud"
$sshKey = Resolve-UserPath (Get-ConfigValue -Key "VPS_RELAY_SSH_KEY" -DefaultValue "~/.ssh/axiomnode_k3s_ci")
$publicHost = Get-ConfigValue -Key "VPS_RELAY_PUBLIC_HOST" -DefaultValue "195.35.48.40"
$statsPublicPort = Get-ConfigValue -Key "VPS_RELAY_STATS_PUBLIC_PORT" -DefaultValue "27000"
$apiPublicPort = Get-ConfigValue -Key "VPS_RELAY_API_PUBLIC_PORT" -DefaultValue "27001"
$statsTunnelPort = Get-ConfigValue -Key "VPS_RELAY_STATS_TUNNEL_PORT" -DefaultValue "27100"
$apiTunnelPort = Get-ConfigValue -Key "VPS_RELAY_API_TUNNEL_PORT" -DefaultValue "27101"

Ensure-ReverseTunnel -SshPath $sshPath -SshKey $sshKey -SshHost $sshHost -StatsTunnelPort $statsTunnelPort -ApiTunnelPort $apiTunnelPort -StatsPort $StatsPort -ApiPort $ApiPort

Invoke-NativeCommand -FilePath $scpPath -Arguments @("-i", $sshKey, "-o", "StrictHostKeyChecking=accept-new", $relayScript, "$sshHost:/tmp/axiomnode_ai_engine_tcp_relay.py") | Out-Null

$remoteCommand = @"
pkill -f 'python3 /tmp/axiomnode_ai_engine_tcp_relay.py $statsPublicPort $statsTunnelPort' >/dev/null 2>&1 || true
pkill -f 'python3 /tmp/axiomnode_ai_engine_tcp_relay.py $apiPublicPort $apiTunnelPort' >/dev/null 2>&1 || true
nohup python3 /tmp/axiomnode_ai_engine_tcp_relay.py $statsPublicPort $statsTunnelPort >/tmp/axiomnode-ai-relay-stats.log 2>&1 & echo \$! >/tmp/axiomnode-ai-relay-stats.pid
nohup python3 /tmp/axiomnode_ai_engine_tcp_relay.py $apiPublicPort $apiTunnelPort >/tmp/axiomnode-ai-relay-api.log 2>&1 & echo \$! >/tmp/axiomnode-ai-relay-api.pid
curl -fsS http://127.0.0.1:$statsPublicPort/health >/dev/null
curl -fsS http://127.0.0.1:$apiPublicPort/health >/dev/null
"@

Invoke-NativeCommand -FilePath $sshPath -Arguments @("-i", $sshKey, "-o", "StrictHostKeyChecking=accept-new", $sshHost, $remoteCommand) | Out-Null

Write-Host "VPS relay exposure ready."
Write-Host "  target host: $publicHost"
Write-Host "  stats port: $statsPublicPort -> this-pc:$StatsPort"
Write-Host "  api port: $apiPublicPort -> this-pc:$ApiPort"
Write-Host "Use this host and these ports as the STG ai-engine runtime target when the rest of the platform runs on the VPS cluster."