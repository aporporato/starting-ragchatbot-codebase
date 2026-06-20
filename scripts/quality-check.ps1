# Run frontend quality checks WITHOUT modifying files (Windows / PowerShell).
# Fails (non-zero exit) if any file is not formatted -- suitable for CI / pre-commit.
#
# Usage:  ./scripts/quality-check.ps1
$ErrorActionPreference = "Stop"

Write-Host "Checking frontend formatting with Prettier..." -ForegroundColor Cyan
npx prettier --check "frontend/**/*.{js,css,html}"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Formatting issues found. Run ./scripts/format.ps1 to fix." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "All frontend files are properly formatted." -ForegroundColor Green
