#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./package_paranal_simulator_context.sh [BALDRAPP_REPOSITORY] [OUTPUT_ZIP]
#
# If omitted, BALDRAPP_REPOSITORY defaults to the current directory and the
# archive is written there with a timestamped name.

repo_root="${1:-$PWD}"
repo_root="$(cd "$repo_root" && pwd -P)"

paranal_dir="baldrapp/apps/paranal_simulator"
common_dir="baldrapp/common"

for required_dir in "$paranal_dir" "$common_dir"; do
    if [[ ! -d "$repo_root/$required_dir" ]]; then
        printf 'Error: expected directory not found: %s\n' "$repo_root/$required_dir" >&2
        printf 'Run this from the BaldrApp repository root, or pass that path as argument 1.\n' >&2
        exit 1
    fi
done

if ! command -v zip >/dev/null 2>&1; then
    printf 'Error: the "zip" command is not installed.\n' >&2
    printf 'On Ubuntu/Debian, install it with: sudo apt install zip\n' >&2
    exit 1
fi

timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
output_zip="${2:-$repo_root/baldrapp_paranal_context_${timestamp}.zip}"

if [[ "$output_zip" != /* ]]; then
    output_zip="$PWD/$output_zip"
fi

output_parent="$(dirname "$output_zip")"
if [[ ! -d "$output_parent" ]]; then
    printf 'Error: output directory does not exist: %s\n' "$output_parent" >&2
    exit 1
fi

if [[ -e "$output_zip" ]]; then
    printf 'Error: refusing to overwrite existing archive: %s\n' "$output_zip" >&2
    exit 1
fi

(
    cd "$repo_root"
    zip -r -y "$output_zip" "$paranal_dir" "$common_dir" \
        -x '*/__pycache__/*' \
           '*/.pytest_cache/*' \
           '*/.mypy_cache/*' \
           '*/.ruff_cache/*' \
           '*/build/*' \
           '*/dist/*' \
           '*.py[co]' \
           '*.o' \
           '*.obj' \
           '*.so' \
           '*.a' \
           '*.dylib' \
           '*.exe' \
           '*.zip'
)

printf '\nCreated: %s\n' "$output_zip"
printf 'Contents:\n'
unzip -l "$output_zip"

