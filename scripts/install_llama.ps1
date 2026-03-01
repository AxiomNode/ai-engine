Param(
  [Parameter(Mandatory=$true)][string]$hf_repo,
  [Parameter(Mandatory=$true)][string]$hf_file,
  [string]$dest_dir = "models\llama"
)

if (-not $env:HUGGINGFACE_TOKEN) {
  Write-Error "Please set HUGGINGFACE_TOKEN environment variable with your HF token."
  exit 1
}

New-Item -ItemType Directory -Force -Path $dest_dir | Out-Null
$url = "https://huggingface.co/$hf_repo/resolve/main/$hf_file"
$out = Join-Path $dest_dir ([IO.Path]::GetFileName($hf_file))

Write-Host "Downloading $url to $out"
Invoke-RestMethod -Headers @{ Authorization = "Bearer $env:HUGGINGFACE_TOKEN" } -Uri $url -OutFile $out
Write-Host "Done. Verify model integrity and move to your inference server as needed."
