param(
  [string]$Data = "datasets",
  [string]$Bag = ""
)
# One command on Windows (Docker Desktop): builds the image if it is missing, starts the detector and RViz,
# opens RViz in the browser and leaves a console in which recordings are played with `play`.
#   .\run.ps1                            recordings from .\datasets
#   .\run.ps1 D:\bags                     recordings from another folder
#   .\run.ps1 D:\bags example_bag ... and play that recording at once
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Port = if ($env:METRO_PORT) { $env:METRO_PORT } else { "6080" }
$Image = if ($env:METRO_IMAGE) { $env:METRO_IMAGE } else { "metro-lidar" }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw "Docker не найден: установите Docker Desktop." }
if (-not (Test-Path $Data)) { throw "Папка с записями не найдена: $Data" }
$Data = (Resolve-Path $Data).Path
cmd /c "docker image inspect $Image >nul 2>&1"  # PowerShell 5.1 with Stop turns docker's stderr into an error
if ($LASTEXITCODE -ne 0) {
  if (Test-Path "metro-lidar.tar.gz") {
    Write-Host "Загружаю готовый образ из metro-lidar.tar.gz..."
    docker load -i metro-lidar.tar.gz
  } else {
    Write-Host "Собираю образ $Image (первый раз несколько минут)..."
    docker build -t $Image .
  }
  if ($LASTEXITCODE -ne 0) { throw "Не удалось получить образ $Image" }
}
cmd /c "docker rm -f metro >nul 2>&1"
$envArgs = @()
if ($Bag) {
  if (-not $Bag.StartsWith("/")) { $Bag = "/data/" + ($Bag -replace "\\", "/") }
  $envArgs += @("-e", "AUTOPLAY=$Bag")
  if ($env:PLAY_RATE) { $envArgs += @("-e", "PLAY_RATE=$($env:PLAY_RATE)") }
  if ($env:PLAY_ARGS) { $envArgs += @("-e", "PLAY_ARGS=$($env:PLAY_ARGS)") }
}
$Url = "http://localhost:$Port/vnc.html?autoconnect=1&resize=scale"
Start-Job -ScriptBlock {
  param($Port, $Url)
  for ($i = 0; $i -lt 120; $i++) {
    try { Invoke-WebRequest -UseBasicParsing "http://localhost:$Port/vnc.html" -TimeoutSec 2 | Out-Null; Start-Process $Url; break } catch { Start-Sleep -Seconds 1 }
  }
} -ArgumentList $Port, $Url | Out-Null
Write-Host "RViz откроется в браузере: $Url"
docker run --rm -it --name metro --shm-size=512m -p "127.0.0.1:${Port}:6080" -v "${Data}:/data:ro" @envArgs $Image studio
