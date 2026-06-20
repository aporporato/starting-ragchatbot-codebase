#!/usr/bin/env bash
# Format all frontend files in place with Prettier (Linux / macOS).
#
# Usage:  ./scripts/format.sh
set -euo pipefail

echo "Formatting frontend files with Prettier..."
npx prettier --write "frontend/**/*.{js,css,html}"
echo "Done."
