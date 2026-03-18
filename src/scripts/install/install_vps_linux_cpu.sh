#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper.
# Prefer from repo root: ./src/scripts/install/deploy.sh <dev|stg|pro> vps-cpu

STAGE="${1:-dev}"

case "${STAGE}" in
  dev|stg|pro) ;;
  *) echo "Invalid stage: ${STAGE}. Use dev|stg|pro."; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

exec "${SCRIPT_DIR}/deploy.sh" "${STAGE}" "vps-cpu"
