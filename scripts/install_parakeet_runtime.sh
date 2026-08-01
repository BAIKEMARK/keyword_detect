#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install "nemo_toolkit[asr]"

python3 - <<'PY'
from nemo.collections.asr.models import ASRModel

print("NeMo ASR runtime ready:", ASRModel.__module__)
PY
