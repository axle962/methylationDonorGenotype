#!/bin/bash
# Runs stages 03, 04, 06, 07 and 10 (both ladders, docs/PREREG.md section 4 and amendment A1) on the synthetic
# fixture of tests/make_fixture.py, on CPU. No eGTEx data, CpGPT or Slurm needed (a laptop check, not a result).
#   tests/run_fixture.sh [root]     (default: a fresh temporary directory; PY selects the interpreter)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:-$(mktemp -d)}"
PY="${PY:-python3}"
O="$ROOT/outputs"
E="$ROOT/egtex"
$PY tests/make_fixture.py "$ROOT"
stage() { echo "--- $*"; "$PY" "$@" | tail -n "${TAIL:-25}"; }
stage scripts/03_train_predictor.py --smoke --enc "$O/encodings" --out "$O/predictor" --metadata "$E/metadata.csv" --epochs 5
stage scripts/03_train_predictor.py --smoke --beta-space --enc "$O/encodings" --out "$O/predictor" --metadata "$E/metadata.csv" --epochs 5
stage scripts/04_evaluate.py --smoke --beta-space --encodings "$O/encodings" --pred "$O/predictor" --egtex-dir "$E"
stage scripts/04_evaluate.py --smoke --encodings "$O/encodings" --pred "$O/predictor" --egtex-dir "$E"
for space in "" _beta; do
  stage scripts/06_covariates.py --smoke --encodings "$O/encodings$space" --pred "$O/predictor$space" --egtex-dir "$E" --out "$O/experiments$space/covariates"
  stage scripts/07_mqtl_deep.py --smoke --encodings "$O/encodings$space" --pred "$O/predictor$space" --egtex-dir "$E" --out "$O/experiments$space/mqtl_deep"
done
TAIL=60 stage scripts/10_decide.py --smoke --outputs "$O"
TAIL=60 stage scripts/10_decide.py --smoke --outputs "$O" --space beta
echo "fixture run OK: $ROOT"
