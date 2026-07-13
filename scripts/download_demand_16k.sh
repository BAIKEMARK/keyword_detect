#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/noise/DEMAND_16k}"
ZIP_DIR="${OUT_DIR}/zips"
WAV_DIR="${OUT_DIR}/wav"
BASE_URL="https://zenodo.org/records/1227121/files"

SCENES=(
  TCAR_16k
  TBUS_16k
  TMETRO_16k
  STRAFFIC_16k
  PCAFETER_16k
  OMEETING_16k
  OOFFICE_16k
  DKITCHEN_16k
  DLIVING_16k
)

mkdir -p "${ZIP_DIR}" "${WAV_DIR}"

CURL_ARGS=(-L --fail --retry 3 --continue-at -)
if [[ -n "${HTTPS_PROXY:-}" ]]; then
  CURL_ARGS+=(--proxy "${HTTPS_PROXY}")
elif [[ -n "${HTTP_PROXY:-}" ]]; then
  CURL_ARGS+=(--proxy "${HTTP_PROXY}")
fi

for scene in "${SCENES[@]}"; do
  zip_path="${ZIP_DIR}/${scene}.zip"
  url="${BASE_URL}/${scene}.zip?download=1"
  if [[ -s "${zip_path}" ]]; then
    echo "skip existing ${zip_path}"
  else
    echo "download ${scene}"
    curl "${CURL_ARGS[@]}" -o "${zip_path}" "${url}"
  fi

  echo "test ${scene}"
  unzip -tq "${zip_path}" >/dev/null

  scene_dir="${scene%_16k}"
  if [[ -d "${WAV_DIR}/${scene_dir}" ]]; then
    echo "skip extracted ${WAV_DIR}/${scene_dir}"
  else
    echo "extract ${scene}"
    unzip -q "${zip_path}" -d "${WAV_DIR}"
  fi
done

count="$(find "${WAV_DIR}" -type f -name '*.wav' | wc -l | tr -d ' ')"
echo "done: ${count} wav files under ${WAV_DIR}"
