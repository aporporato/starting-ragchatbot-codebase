# Format all frontend files in place with Prettier (Windows / PowerShell).
#
# Usage:  ./scripts/format.ps1
$ErrorActionPreference = "Stop"

Write-Host "Formatting frontend files with Prettier..." -ForegroundColor Cyan
npx prettier --write "frontend/**/*.{js,css,html}"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done." -ForegroundColor Green
