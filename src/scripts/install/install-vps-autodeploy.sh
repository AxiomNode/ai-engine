#!/usr/bin/env bash
set -euo pipefail

# Install a systemd timer that keeps ai-engine updated on a single VPS.
# Usage from repo root:
#   ./src/scripts/install/install-vps-autodeploy.sh <stg|pro> <vps-cpu|vps-gpu> [branch] [interval]

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <stg|pro> <vps-cpu|vps-gpu> [branch] [interval]"
  echo "Example: $0 pro vps-cpu main 5min"
  exit 1
fi

STAGE="$1"
ENVIRONMENT="$2"
BRANCH="${3:-main}"
INTERVAL="${4:-5min}"

case "${STAGE}" in
  stg|pro) ;;
  *) echo "Invalid stage for VPS autodeploy: ${STAGE}"; exit 1 ;;
esac

case "${ENVIRONMENT}" in
  vps-cpu|vps-gpu) ;;
  *) echo "Invalid VPS environment: ${ENVIRONMENT}"; exit 1 ;;
esac

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl is required. Run this on the target VPS host."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SERVICE_NAME="axiomnode-ai-engine-autodeploy"
RUNNER_PATH="/usr/local/bin/${SERVICE_NAME}"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
TIMER_PATH="/etc/systemd/system/${SERVICE_NAME}.timer"
RUN_USER="$(id -un)"
RUN_GROUP="$(id -gn)"
LOCK_PATH="/tmp/${SERVICE_NAME}.lock"

echo "Installing ai-engine VPS autodeploy:"
echo "  repo: ${REPO_ROOT}"
echo "  stage/environment: ${STAGE}/${ENVIRONMENT}"
echo "  branch: ${BRANCH}"
echo "  interval: ${INTERVAL}"
echo "  service user: ${RUN_USER}:${RUN_GROUP}"

sudo tee "${RUNNER_PATH}" >/dev/null <<EOF
#!/usr/bin/env bash
set -euo pipefail

exec 9>"${LOCK_PATH}"
flock -n 9 || {
  echo "Another ai-engine autodeploy run is already active."
  exit 0
}

cd "${REPO_ROOT}"

echo "Fetching ${BRANCH}..."
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}"

if [[ -d "../secrets" && -f "../secrets/scripts/prepare-runtime-secrets.mjs" ]]; then
  echo "Preparing runtime secrets for ${STAGE}/ai-engine..."
  node ../secrets/scripts/prepare-runtime-secrets.mjs "${STAGE}" ai-engine
else
  echo "Secrets repo not found next to ai-engine; keeping existing src/.env.secrets."
fi

./src/scripts/install/deploy.sh "${STAGE}" "${ENVIRONMENT}"
EOF

sudo chmod 0755 "${RUNNER_PATH}"

sudo tee "${SERVICE_PATH}" >/dev/null <<EOF
[Unit]
Description=AxiomNode ai-engine VPS autodeploy
Wants=network-online.target docker.service
After=network-online.target docker.service

[Service]
Type=oneshot
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${REPO_ROOT}
ExecStart=${RUNNER_PATH}
TimeoutStartSec=1800
Nice=5
IOSchedulingClass=best-effort
IOSchedulingPriority=6
EOF

sudo tee "${TIMER_PATH}" >/dev/null <<EOF
[Unit]
Description=Run AxiomNode ai-engine VPS autodeploy periodically

[Timer]
OnBootSec=2min
OnUnitActiveSec=${INTERVAL}
AccuracySec=30s
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}.timer"

echo "Installed ${SERVICE_NAME}.timer"
echo "Run once now with: sudo systemctl start ${SERVICE_NAME}.service"
echo "Inspect logs with: journalctl -u ${SERVICE_NAME}.service -f"