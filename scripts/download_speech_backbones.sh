#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-all}"
MODEL_ROOT="${2:-/mnt/workspace/models}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_speech_backbones.sh [all|new|wavlm-large|hubert-large|w2v-bert-2|parakeet|whisper-large-v3] [model_root]

Examples:
  bash scripts/download_speech_backbones.sh all
  bash scripts/download_speech_backbones.sh new /mnt/workspace/models
  bash scripts/download_speech_backbones.sh wavlm-large /mnt/workspace/models
EOF
}

if [[ "${MODEL_NAME}" != "all" \
      && "${MODEL_NAME}" != "new" \
      && "${MODEL_NAME}" != "wavlm-large" \
      && "${MODEL_NAME}" != "hubert-large" \
      && "${MODEL_NAME}" != "w2v-bert-2" \
      && "${MODEL_NAME}" != "parakeet" \
      && "${MODEL_NAME}" != "whisper-large-v3" ]]; then
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
  weight="$(find "${directory}" -maxdepth 3 -type f \
    \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \
       -o -name '*.nemo' -o -name '*.ckpt' -o -name '*.pt' \) \
    -print -quit 2>/dev/null || true)"
  [[ -s "${directory}/config.json" && -n "${weight}" ]] || [[ -n "${weight}" ]]
}

download_model() {
  local repo_id="$1"
  local local_name="$2"
  local verify_kind="${3:-transformers}"
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

  if [[ "${verify_kind}" == "files" ]]; then
    echo "verified: ${output_dir} model files present"
    return
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
  new)
    download_model "${W2V_BERT_REPO:-facebook/w2v-bert-2.0}" \
      "w2v-bert-2"
    download_model "${PARAKEET_REPO:-nvidia/parakeet-tdt-0.6b-v3}" \
      "parakeet-tdt-0.6b-v3" files
    download_model "${WHISPER_REPO:-openai/whisper-large-v3}" \
      "whisper-large-v3"
    ;;
  wavlm-large)
    download_model "microsoft/wavlm-large" "wavlm-large"
    ;;
  hubert-large)
    download_model "facebook/hubert-large-ll60k" "hubert-large-ll60k"
    ;;
  w2v-bert-2)
    download_model "${W2V_BERT_REPO:-facebook/w2v-bert-2.0}" "w2v-bert-2"
    ;;
  parakeet)
    download_model "${PARAKEET_REPO:-nvidia/parakeet-tdt-0.6b-v3}" \
      "parakeet-tdt-0.6b-v3" files
    ;;
  whisper-large-v3)
    download_model "${WHISPER_REPO:-openai/whisper-large-v3}" \
      "whisper-large-v3"
    ;;
esac

echo "done. Use these local paths with --model-id:"
[[ "${MODEL_NAME}" == "all" || "${MODEL_NAME}" == "wavlm-large" ]] \
  && echo "  ${MODEL_ROOT}/wavlm-large"
[[ "${MODEL_NAME}" == "all" || "${MODEL_NAME}" == "hubert-large" ]] \
  && echo "  ${MODEL_ROOT}/hubert-large-ll60k"
[[ "${MODEL_NAME}" == "new" || "${MODEL_NAME}" == "w2v-bert-2" ]] \
  && echo "  ${MODEL_ROOT}/w2v-bert-2"
[[ "${MODEL_NAME}" == "new" || "${MODEL_NAME}" == "parakeet" ]] \
  && echo "  ${MODEL_ROOT}/parakeet-tdt-0.6b-v3"
[[ "${MODEL_NAME}" == "new" || "${MODEL_NAME}" == "whisper-large-v3" ]] \
  && echo "  ${MODEL_ROOT}/whisper-large-v3"
