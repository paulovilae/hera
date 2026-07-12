#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${HOME}/.local/bin"
TARGET_PATH="${TARGET_DIR}/claude-hera"
SOURCE_PATH="${SCRIPT_DIR}/claude-hera"

mkdir -p "${TARGET_DIR}"
chmod +x "${SOURCE_PATH}"
ln -sfn "${SOURCE_PATH}" "${TARGET_PATH}"

echo "Installed claude-hera -> ${TARGET_PATH}"
