param(
    [Parameter(Mandatory = $true)]
    [string]$Stage,

    [Parameter(Mandatory = $true)]
    [string]$Environment,

    [Parameter(Mandatory = $true)]
    [string]$LlamaPort,

    [Parameter(Mandatory = $true)]
    [string]$StatsPort,

    [Parameter(Mandatory = $true)]
    [string]$ApiPort,

    [Parameter(Mandatory = $true)]
    [hashtable]$Config
)

$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Ensure-FirewallRule {
    param(
        [string]$DisplayName,
        [int]$Port
    )

    $existingRule = Get-NetFirewallRule -DisplayName $DisplayName -ErrorAction SilentlyContinue
    if (-not $existingRule) {
        New-NetFirewallRule -DisplayName $DisplayName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port | Out-Null
        return
    }

    Set-NetFirewallRule -DisplayName $DisplayName -Enabled True -Direction Inbound -Action Allow | Out-Null
}

function Ensure-PortProxyPrerequisites {
    $ipHelper = Get-Service -Name iphlpsvc -ErrorAction Stop
    if ($ipHelper.StartType -eq "Disabled") {
        Set-Service -Name iphlpsvc -StartupType Manual
    }

    if ($ipHelper.Status -ne "Running") {
        Start-Service -Name iphlpsvc
    }
}

function Ensure-PortProxy {
    param(
        [string]$Name,
        [int]$ListenPort,
        [int]$ConnectPort
    )

    if ($ListenPort -eq $ConnectPort) {
        return
    }

    $existingEntries = netsh interface portproxy show v4tov4
    $entryPattern = "0\.0\.0\.0\s+$ListenPort\s+127\.0\.0\.1\s+$ConnectPort"
    if ($existingEntries -match $entryPattern) {
        return
    }

    netsh interface portproxy delete v4tov4 listenport=$ListenPort listenaddress=0.0.0.0 | Out-Null
    netsh interface portproxy add v4tov4 listenport=$ListenPort listenaddress=0.0.0.0 connectport=$ConnectPort connectaddress=127.0.0.1 | Out-Null
    Write-Host "Configured public TCP proxy for $Name: 0.0.0.0:$ListenPort -> 127.0.0.1:$ConnectPort"
}

function Ensure-Service {
    param(
        [string]$Name,
        [string]$PrivatePort,
        [string]$PublicPortSetting
    )

    if (-not $PublicPortSetting) {
        return $null
    }

    $publicPort = [int]$PublicPortSetting
    $privatePort = [int]$PrivatePort
    $firewallRuleName = "AxiomNode ai-engine $Stage $Environment $Name public tcp $publicPort"
    Ensure-FirewallRule -DisplayName $firewallRuleName -Port $publicPort
    Ensure-PortProxy -Name $Name -ListenPort $publicPort -ConnectPort $privatePort
    return $publicPort
}

$autoExpose = $Config["AUTO_EXPOSE_PUBLIC_PORTS"]
if ($autoExpose -notin @("true", "1", "yes", "on")) {
    Write-Host "Skipping Windows public port exposure. Set AUTO_EXPOSE_PUBLIC_PORTS=true in the distribution env file to enable it."
    return
}

if (-not (Test-IsAdministrator)) {
    throw "Windows public port exposure requires an elevated PowerShell session. Re-run deploy.ps1 as Administrator."
}

Ensure-PortProxyPrerequisites

$llamaPublicPort = Ensure-Service -Name "llama" -PrivatePort $LlamaPort -PublicPortSetting $Config["LLAMA_PUBLIC_PORT"]
$statsPublicPort = Ensure-Service -Name "stats" -PrivatePort $StatsPort -PublicPortSetting $Config["STATS_PUBLIC_PORT"]
$apiPublicPort = Ensure-Service -Name "api" -PrivatePort $ApiPort -PublicPortSetting $Config["API_PUBLIC_PORT"]

Write-Host "Windows public exposure ready."
if ($llamaPublicPort) {
    Write-Host "  llama public port: $llamaPublicPort -> localhost:$LlamaPort"
}
if ($statsPublicPort) {
    Write-Host "  stats public port: $statsPublicPort -> localhost:$StatsPort"
}
if ($apiPublicPort) {
    Write-Host "  api public port: $apiPublicPort -> localhost:$ApiPort"
}
Write-Host "If this workstation is behind a router, you still need matching NAT/port-forward rules on the router for external clients to reach these ports."