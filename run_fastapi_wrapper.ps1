param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8010
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

py -3 -m pip install -r (Join-Path $projectRoot "requirements-api.txt")
py -3 -m uvicorn fastapi_wrapper:app --host $Host --port $Port --app-dir $projectRoot
