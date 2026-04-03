#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for ai-engine segmented API keys.
# Usage from repo root:
#   bash src/scripts/install/smoke_test.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[ERROR] Missing env file: ${ENV_FILE}"
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

required_vars=(
  AI_ENGINE_GAMES_API_KEY
  AI_ENGINE_BRIDGE_API_KEY
  AI_ENGINE_STATS_API_KEY
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "[ERROR] Missing required variable in .env: ${var_name}"
    exit 1
  fi
done

API_BASE="${API_BASE:-http://localhost:7001}"
STATS_BASE="${STATS_BASE:-http://localhost:7000}"
LLAMA_BASE="${LLAMA_BASE:-http://localhost:7002}"

pass_count=0
fail_count=0

run_case() {
  local name="$1"
  local method="$2"
  local url="$3"
  local expected_code="$4"
  local api_key="${5:-}"
  local body="${6:-}"

  local response_file
  response_file="$(mktemp)"

  local code
  if [[ "${method}" == "GET" ]]; then
    if [[ -n "${api_key}" ]]; then
      code="$(curl -s -o "${response_file}" -w '%{http_code}' -H "X-API-Key: ${api_key}" "${url}")"
    else
      code="$(curl -s -o "${response_file}" -w '%{http_code}' "${url}")"
    fi
  else
    if [[ -n "${api_key}" ]]; then
      code="$(curl -s -o "${response_file}" -w '%{http_code}' -X "${method}" -H 'Content-Type: application/json' -H "X-API-Key: ${api_key}" -d "${body}" "${url}")"
    else
      code="$(curl -s -o "${response_file}" -w '%{http_code}' -X "${method}" -H 'Content-Type: application/json' -d "${body}" "${url}")"
    fi
  fi

  if [[ "${code}" == "${expected_code}" ]]; then
    printf '[PASS] %s -> %s\n' "${name}" "${code}"
    pass_count=$((pass_count + 1))
  else
    printf '[FAIL] %s -> got %s expected %s\n' "${name}" "${code}" "${expected_code}"
    echo "       response: $(tr -d '\n' < "${response_file}" | head -c 220)"
    fail_count=$((fail_count + 1))
  fi

  rm -f "${response_file}"
}

echo "Running ai-engine smoke tests..."

authless_generate_body='{"query":"agua"}'
ingest_empty_body='{"documents":[]}'
empty_event_body='{}'

run_case "api health public" GET "${API_BASE}/health" 200
run_case "generate without key denied" POST "${API_BASE}/generate" 401 "" "${authless_generate_body}"
run_case "generate with games key" POST "${API_BASE}/generate" 200 "${AI_ENGINE_GAMES_API_KEY}" "${authless_generate_body}"
run_case "ingest with games key denied" POST "${API_BASE}/ingest" 403 "${AI_ENGINE_GAMES_API_KEY}" "${ingest_empty_body}"
run_case "ingest with bridge key" POST "${API_BASE}/ingest" 200 "${AI_ENGINE_BRIDGE_API_KEY}" "${ingest_empty_body}"
run_case "stats without key denied" GET "${STATS_BASE}/stats" 401
run_case "stats with bridge key" GET "${STATS_BASE}/stats" 200 "${AI_ENGINE_BRIDGE_API_KEY}"
run_case "events wrong key denied" POST "${STATS_BASE}/events" 403 "wrong" "${empty_event_body}"
run_case "events with stats key" POST "${STATS_BASE}/events" 200 "${AI_ENGINE_STATS_API_KEY}" "${empty_event_body}"
run_case "llama models public" GET "${LLAMA_BASE}/v1/models" 200

echo
printf 'Summary: %d passed, %d failed\n' "${pass_count}" "${fail_count}"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi

echo "Smoke tests passed."
