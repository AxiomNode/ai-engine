#!/usr/bin/env bash
set -euo pipefail

# Simple helper to download a model file from Hugging Face using an access token.
# Usage: ./install_llama.sh <hf_repo> <file_path_in_repo> <dest_dir>
# Example: ./install_llama.sh meta-llama/Llama-3-1-8b-q4_k_m model.gguf models/llama-3-1-8b-q4_k_m

HF_REPO=${1:-}
HF_FILE=${2:-}
DEST_DIR=${3:-models/llama}

if [ -z "$HF_REPO" ] || [ -z "$HF_FILE" ]; then
  echo "Usage: $0 <hf_repo> <file_path_in_repo> <dest_dir>"
  echo "Example: $0 meta-llama/Llama-3-1-8b-q4_k_m/ gguf/model.gguf models/llama-3-1-8b-q4_k_m"
  exit 1
fi

if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
  echo "Please set HUGGINGFACE_TOKEN environment variable with your HF token."
  exit 1
fi

mkdir -p "$DEST_DIR"
URL="https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILE}"

echo "Downloading ${URL} to ${DEST_DIR}/$(basename "$HF_FILE")"
curl -L -H "Authorization: Bearer ${HUGGINGFACE_TOKEN}" -o "${DEST_DIR}/$(basename "$HF_FILE")" "$URL"
echo "Done. Verify model integrity and move to your inference server as needed."
