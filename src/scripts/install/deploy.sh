#!/usr/bin/env bash
set -euo pipefail

# Unified deploy script for Linux hosts.
# Usage from repo root: ./src/scripts/install/deploy.sh <dev|stg|pro> <windows|windows-gpu|vps-cpu|vps-gpu>
#
# AI_ENGINE_DEPLOY_MODE (default: build)
#   build  Build ai-api/ai-stats on this host (`up -d --build`). Legacy path, kept as fallback.
#   image  Pull the pinned GHCR images from src/.env.images (override the path with
#          AI_ENGINE_IMAGES_FILE) and start them with `up -d --no-build`. Nothing is built here.

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <dev|stg|pro> <windows|windows-gpu|vps-cpu|vps-gpu>"
  exit 1
fi

STAGE="$1"
ENVIRONMENT="$2"

case "${STAGE}" in
  dev|stg|pro) ;;
  *) echo "Invalid stage: ${STAGE}"; exit 1 ;;
esac

case "${ENVIRONMENT}" in
  windows|windows-gpu|vps-cpu|vps-gpu) ;;
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

SECRETS_FILE=".env.secrets"
if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Secrets file not found: ${SECRETS_FILE}"
  echo "Run: node ../secrets/scripts/prepare-runtime-secrets.mjs ${STAGE} ai-engine"
  exit 1
fi

PROFILE="cpu"
if [[ "${ENVIRONMENT}" == "vps-gpu" || "${ENVIRONMENT}" == "windows-gpu" ]]; then
  PROFILE="gpu"
fi

DEPLOY_MODE="${AI_ENGINE_DEPLOY_MODE:-build}"
case "${DEPLOY_MODE}" in
  build|image) ;;
  *) echo "Invalid AI_ENGINE_DEPLOY_MODE: ${DEPLOY_MODE} (expected build or image)"; exit 1 ;;
esac

COMPOSE_ARGS=(--env-file "${ENV_FILE}" --env-file "${SECRETS_FILE}")
if [[ "${DEPLOY_MODE}" == "image" ]]; then
  IMAGES_FILE="${AI_ENGINE_IMAGES_FILE:-.env.images}"
  if [[ ! -f "${IMAGES_FILE}" ]]; then
    echo "Image pin file not found: ${IMAGES_FILE}"
    echo "Copy distributions/examples/images.env.example to ${IMAGES_FILE} and set the image tags or digests."
    exit 1
  fi
  COMPOSE_ARGS+=(--env-file "${IMAGES_FILE}")
fi
COMPOSE_ARGS+=(--profile "${PROFILE}" -f docker-compose.yml)
if [[ "${DEPLOY_MODE}" == "image" ]]; then
  COMPOSE_ARGS+=(-f docker-compose.ghcr.yml)
fi

file_env_value() {
  local file="$1"
  local key="$2"
  local default_value="$3"
  local value
  value="$(awk -F= -v key="${key}" '
    $0 !~ /^[[:space:]]*#/ && $1 == key {
      sub(/^[^=]*=/, "")
      sub(/\r$/, "")
      print
      exit
    }
  ' "${file}" 2>/dev/null || true)"
  if [[ -n "${value}" ]]; then
    printf '%s' "${value}"
  else
    printf '%s' "${default_value}"
  fi
}

env_value() {
  file_env_value "${ENV_FILE}" "$1" "$2"
}

secret_value() {
  file_env_value "${SECRETS_FILE}" "$1" "$2"
}

wait_for_health() {
  local name="$1"
  local url="$2"
  local attempts="${3:-30}"
  local delay_seconds="${4:-5}"

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl -fsS "${url}" >/dev/null; then
      echo "${name} healthy at ${url}"
      return 0
    fi
    echo "Waiting for ${name} (${attempt}/${attempts}) at ${url}..."
    sleep "${delay_seconds}"
  done

  echo "${name} did not become healthy at ${url}"
  return 1
}

wait_for_rag() {
  local url="$1"
  local attempts="${2:-30}"
  local delay_seconds="${3:-5}"
  local api_key=""
  local curl_args=(-fsS)

  api_key="$(secret_value AI_ENGINE_API_KEY "")"
  if [[ -z "${api_key}" ]]; then
    api_key="$(secret_value AI_ENGINE_BRIDGE_API_KEY "")"
  fi
  if [[ -z "${api_key}" ]]; then
    api_key="$(secret_value AI_ENGINE_GAMES_API_KEY "")"
  fi
  if [[ -n "${api_key}" ]]; then
    curl_args+=(-H "X-API-Key: ${api_key}")
  fi

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    local payload=""
    local chunks="0"
    local coverage="unknown"

    payload="$(curl "${curl_args[@]}" "${url}" 2>/dev/null || true)"
    chunks="$(printf '%s' "${payload}" | sed -n 's/.*"total_chunks"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -n 1)"
    coverage="$(printf '%s' "${payload}" | sed -n 's/.*"coverage_level"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
    chunks="${chunks:-0}"
    coverage="${coverage:-unknown}"

    if [[ "${chunks}" -gt 0 && "${coverage}" != "empty" ]]; then
      echo "RAG ready at ${url}: chunks=${chunks}, coverage=${coverage}"
      return 0
    fi

    echo "Waiting for RAG index (${attempt}/${attempts}) at ${url}: chunks=${chunks}, coverage=${coverage}..."
    sleep "${delay_seconds}"
  done

  echo "RAG index did not become ready at ${url}"
  return 1
}

mkdir -p models data

LLAMA_MODEL_FILE="$(env_value LLAMA_MODEL_FILE Qwen2.5-3B-Instruct-Q4_K_M.gguf)"
if [[ ! -f "models/${LLAMA_MODEL_FILE}" ]]; then
  echo "Model file not found: models/${LLAMA_MODEL_FILE}"
  echo "Download it before deploying, for example:"
  echo "  cd src && python -m ai_engine.llm.model_manager download qwen2.5-7b"
  exit 1
fi

STATS_PORT_VALUE="$(env_value STATS_PORT 7000)"
API_PORT_VALUE="$(env_value API_PORT 7001)"

echo "Validating compose config for ${STAGE}/${ENVIRONMENT} (profile=${PROFILE})..."
docker compose "${COMPOSE_ARGS[@]}" config >/dev/null

if [[ "${DEPLOY_MODE}" == "image" ]]; then
  echo "Pulling pinned images for ${STAGE}/${ENVIRONMENT}..."
  if ! docker compose "${COMPOSE_ARGS[@]}" pull ai-api ai-stats; then
    echo "Image pull failed. If the GHCR packages are private, run: docker login ghcr.io (token with read:packages)."
    exit 1
  fi
  echo "Starting stack for ${STAGE}/${ENVIRONMENT} from pulled images (no build)..."
  docker compose "${COMPOSE_ARGS[@]}" up -d --no-build
else
  echo "Starting stack for ${STAGE}/${ENVIRONMENT} (building on this host)..."
  docker compose "${COMPOSE_ARGS[@]}" up -d --build
fi

echo "Services status:"
docker compose "${COMPOSE_ARGS[@]}" ps

echo "Health checks:"
wait_for_health "ai-stats" "http://localhost:${STATS_PORT_VALUE}/health"
wait_for_health "ai-api" "http://localhost:${API_PORT_VALUE}/health"
wait_for_rag "http://localhost:${API_PORT_VALUE}/diagnostics/rag/stats"
