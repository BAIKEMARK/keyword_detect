#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-all}"
MODEL_ROOT="${2:-/mnt/workspace/models}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_speech_backbones.sh [all|wavlm-large|hubert-large] [model_root]

Examples:
  bash scripts/download_speech_backbones.sh all
  bash scripts/download_speech_backbones.sh wavlm-large /mnt/workspace/models
EOF
}

if [[ "${MODEL_NAME}" != "all" \
      && "${MODEL_NAME}" != "wavlm-large" \
      && "${MODEL_NAME}" != "hubert-large" ]]; then
  usage >&2
  exit 2
fi

if ! command -v modelscope >/dev/null 2>&1; then
  echo "modelscope CLI not found." >&2
  echo "Install it with: python3 -m pip install -U modelscope" >&2
  exit 1
fi

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
mkdir -p "${MODEL_ROOT}"

has_weights() {
  local directory="$1"
  local weight
  weight="$(find "${directory}" -maxdepth 2 -type f \
    \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) \
    -print -quit 2>/dev/null || true)"
  [[ -s "${directory}/config.json" && -n "${weight}" ]]
}

download_model() {
  local repo_id="$1"
  local local_name="$2"
  local output_dir="${MODEL_ROOT}/${local_name}"

  if has_weights "${output_dir}"; then
    echo "skip complete model: ${output_dir}"
  else
    mkdir -p "${output_dir}"
    echo "download ${repo_id} -> ${output_dir}"
    modelscope download \
      --model "${repo_id}" \
      --local_dir "${output_dir}"
  fi

  if ! has_weights "${output_dir}"; then
    echo "incomplete model directory: ${output_dir}" >&2
    exit 1
  fi

  python3 - "${output_dir}" <<'PY'
import sys
from transformers import AutoConfig

path = sys.argv[1]
config = AutoConfig.from_pretrained(path, local_files_only=True)
print(
    f"verified: {path} "
    f"model_type={config.model_type} "
    f"hidden_size={getattr(config, 'hidden_size', 'unknown')} "
    f"layers={getattr(config, 'num_hidden_layers', 'unknown')}"
)
PY
}

case "${MODEL_NAME}" in
  all)
    download_model "microsoft/wavlm-large" "wavlm-large"
    download_model "facebook/hubert-large-ll60k" "hubert-large-ll60k"
    ;;
  wavlm-large)
    download_model "microsoft/wavlm-large" "wavlm-large"
    ;;
  hubert-large)
    download_model "facebook/hubert-large-ll60k" "hubert-large-ll60k"
    ;;
esac

echo "done. Use these local paths with --model-id:"
[[ "${MODEL_NAME}" == "all" || "${MODEL_NAME}" == "wavlm-large" ]] \
  && echo "  ${MODEL_ROOT}/wavlm-large"
[[ "${MODEL_NAME}" == "all" || "${MODEL_NAME}" == "hubert-large" ]] \
  && echo "  ${MODEL_ROOT}/hubert-large-ll60k"
