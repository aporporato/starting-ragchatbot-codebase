#!/usr/bin/env bash
# Run frontend quality checks WITHOUT modifying files (Linux / macOS).
# Fails (non-zero exit) if any file is not formatted -- suitable for CI / pre-commit.
#
# Usage:  ./scripts/quality-check.sh
set -euo pipefail

echo "Checking frontend formatting with Prettier..."
if ! npx prettier --check "frontend/**/*.{js,css,html}"; then
    echo "Formatting issues found. Run ./scripts/format.sh to fix."
    exit 1
fi
echo "All frontend files are properly formatted."
