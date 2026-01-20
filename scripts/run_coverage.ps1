$ErrorActionPreference = "Stop"

Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -Path ".."

$timestamp = Get-Date -Format "yyyyMMddHHmmss"
$dataFile = "manual_tmp/coverage/.coverage.$timestamp.$PID"

New-Item -ItemType Directory -Force "manual_tmp/coverage" | Out-Null

python -m coverage run --rcfile ".coveragerc" --data-file "$dataFile" -m pytest -q -p no:pytest_cov -p no:tmpdir
python -m coverage report --rcfile ".coveragerc" --data-file "$dataFile" --fail-under 90
