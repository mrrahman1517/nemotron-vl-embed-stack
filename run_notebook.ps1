param(
    [switch]$Launch
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$packagesPath = Join-Path $projectRoot ".packages"
$tempPath = Join-Path $projectRoot ".tmp"

New-Item -ItemType Directory -Force -Path $packagesPath | Out-Null
New-Item -ItemType Directory -Force -Path $tempPath | Out-Null

$env:TEMP = $tempPath
$env:TMP = $tempPath
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$packagesPath;$env:PYTHONPATH" } else { $packagesPath }

py -3.13 -m pip install --upgrade --target $packagesPath -r (Join-Path $projectRoot "requirements-notebook.txt")
py -3.13 -m pip install --upgrade --target $packagesPath torch

if ($Launch) {
    py -3.13 -m notebook (Join-Path $projectRoot "nemotron_testing.ipynb")
}
