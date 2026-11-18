#!/usr/bin/env bash
set -euo pipefail
demo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$demo_root"
exec env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  panda_calligraphy/venv/bin/python panda_calligraphy/callirewrite_replay.py \
  --input panda_calligraphy/docs/callirewrite_demo_refined.npz \
  --target panda_calligraphy/docs/callirewrite_demo_input.png \
  --output .local_setup/verification/integration/presentation_replay \
  --serve --port 8084 "$@"
