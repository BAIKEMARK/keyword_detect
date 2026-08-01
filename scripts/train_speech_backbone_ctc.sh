#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-}"
MODE="${2:-smoke}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/workspace/models}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_speech_backbone_ctc.sh MODEL [smoke|screen]

MODEL:
  w2v-bert-2 | whisper-large-v3 | parakeet

Examples:
  bash scripts/train_speech_backbone_ctc.sh w2v-bert-2 smoke
  nohup bash scripts/train_speech_backbone_ctc.sh whisper-large-v3 screen \
    > logs/whisper_large_v3_screen.log 2>&1 &

Optional environment overrides:
  MODEL_ROOT, SUBSET, EPOCHS, BS, WORKERS, DEVICE, NOISE_DIR
EOF
}

case "${MODEL_NAME}" in
  w2v-bert-2)
    MODEL_ID="${MODEL_ROOT}/w2v-bert-2"
    BACKBONE="w2v-bert"
    RUN_NAME="w2v_bert_2"
    DEFAULT_BS=16
    ;;
  whisper-large-v3)
    MODEL_ID="${MODEL_ROOT}/whisper-large-v3"
    BACKBONE="whisper"
    RUN_NAME="whisper_large_v3"
    DEFAULT_BS=8
    ;;
  parakeet)
    MODEL_ID="${MODEL_ROOT}/parakeet-tdt-0.6b-v3"
    BACKBONE="parakeet"
    RUN_NAME="parakeet_tdt_0_6b_v3"
    DEFAULT_BS=16
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

case "${MODE}" in
  smoke)
    DEFAULT_SUBSET=256
    DEFAULT_EPOCHS=1
    LOG_EVERY=1
    ;;
  screen)
    DEFAULT_SUBSET=100000
    DEFAULT_EPOCHS=3
    LOG_EVERY=100
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

SUBSET="${SUBSET:-${DEFAULT_SUBSET}}"
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
BS="${BS:-${DEFAULT_BS}}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
NOISE_DIR="${NOISE_DIR:-noise/DEMAND_16k/wav}"
if [[ "${MODE}" == "smoke" && "${SUBSET}" == "256" \
      && "${EPOCHS}" == "1" ]]; then
  SUFFIX="smoke"
elif [[ "${SUBSET}" == "100000" ]]; then
  SUFFIX="100k_e${EPOCHS}"
else
  SUFFIX="${SUBSET}_e${EPOCHS}"
fi
OUT="baseline/checkpoints/${RUN_NAME}_phoneme_temporal_hardneg_${SUFFIX}.pt"

if [[ ! -d "${MODEL_ID}" && ! -f "${MODEL_ID}" ]]; then
  echo "model not found: ${MODEL_ID}" >&2
  exit 1
fi

if [[ "${BACKBONE}" == "parakeet" ]]; then
  python3 -c 'from nemo.collections.asr.models import ASRModel' 2>/dev/null || {
    echo "Parakeet runtime is missing." >&2
    echo "Run: bash scripts/install_parakeet_runtime.sh" >&2
    exit 1
  }
fi

export NLTK_DATA="${NLTK_DATA:-/mnt/workspace/nltk_data}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p baseline/checkpoints logs

echo "model=${MODEL_NAME} backbone=${BACKBONE} mode=${MODE}"
echo "subset=${SUBSET} epochs=${EPOCHS} bs=${BS} out=${OUT}"

exec python3 -u baseline/train_wavlm_ctc.py \
  --model-id "${MODEL_ID}" \
  --backbone "${BACKBONE}" \
  --units phoneme \
  --head temporal \
  --adapter-dim 256 \
  --adapter-layers 2 \
  --hard-negative-weight 0.25 \
  --hard-negative-margin 0.5 \
  --subset "${SUBSET}" \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --workers "${WORKERS}" \
  --log-every "${LOG_EVERY}" \
  --device "${DEVICE}" \
  --noise-prob 0.5 \
  --noise-dir "${NOISE_DIR}" \
  --out "${OUT}"
