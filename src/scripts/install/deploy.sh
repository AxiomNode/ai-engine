#!/usr/bin/env bash
set -euo pipefail

# Unified deploy script for Linux hosts.
# Usage from repo root: ./src/scripts/install/deploy.sh <dev|stg|pro> <windows|vps-cpu|vps-gpu>

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <dev|stg|pro> <windows|vps-cpu|vps-gpu>"
  exit 1
fi

STAGE="$1"
ENVIRONMENT="$2"

case "${STAGE}" in
  dev|stg|pro) ;;
  *) echo "Invalid stage: ${STAGE}"; exit 1 ;;
esac

case "${ENVIRONMENT}" in
  windows|vps-cpu|vps-gpu) ;;
  *) echo "Invalid environment: ${ENVIRONMENT}"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_FILE="distributions/${STAGE}/${ENVIRONMENT}.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Distribution file not found: ${ENV_FILE}"
  exit 1
fi

PROFILE="cpu"
if [[ "${ENVIRONMENT}" == "vps-gpu" ]]; then
  PROFILE="gpu"
fi

COMPOSE_ARGS=(--env-file "${ENV_FILE}" --profile "${PROFILE}" -f docker-compose.yml)

mkdir -p models data

echo "Validating compose config for ${STAGE}/${ENVIRONMENT} (profile=${PROFILE})..."
docker compose "${COMPOSE_ARGS[@]}" config >/dev/null

echo "Starting stack for ${STAGE}/${ENVIRONMENT}..."
docker compose "${COMPOSE_ARGS[@]}" up -d --build

echo "Services status:"
docker compose "${COMPOSE_ARGS[@]}" ps

echo "Health checks:"
curl -fsS http://localhost:8000/health && echo
curl -fsS http://localhost:8001/health && echo
