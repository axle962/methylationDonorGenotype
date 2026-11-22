#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=n2c48m24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=jepa_smoke
#SBATCH --output=slurm/logs/%x_%j.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/smoke.sbatch [script args ...]
# CPU smoke test of the whole pipeline on small data: stage 1 on 50 samples (>= 5 lung-colon donor pairs), 02-08 and 10 with --smoke (20 samples x 5 blocks), 09 with --smoke (2 epochs). FRESH=1 also rebuilds data/smoke.

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
S=outputs/smoke
rm -rf "$S"
if [[ "${FRESH:-0}" == 1 ]]; then rm -rf data/smoke; fi
run_stage scripts/01_prepare_egtex.py --n-samples 50 --min-lung-colon-pairs 5 --arrow data/smoke/egtex.arrow --processed-dir data/smoke/processed
run_stage scripts/02_encode_blocks.py --smoke --include-x --processed-dir data/smoke/processed --out "$S/encodings"
run_stage scripts/03_train_predictor.py --smoke --enc "$S/encodings" --out "$S/predictor"
run_stage scripts/03_train_predictor.py --smoke --beta-space --enc "$S/encodings" --out "$S/predictor"
run_stage scripts/04_evaluate.py --smoke --beta-space --encodings "$S/encodings" --pred "$S/predictor"
run_stage scripts/04_evaluate.py --smoke --encodings "$S/encodings" --pred "$S/predictor"
run_stage scripts/05_ablations.py --smoke --encodings "$S/encodings" --pred "$S/predictor" --out "$S/experiments/ablations" --size-encodings "$S/encodings_{mb}mb"
run_stage scripts/06_covariates.py --smoke --encodings "$S/encodings" --pred "$S/predictor" --out "$S/experiments/covariates"
run_stage scripts/07_mqtl_deep.py --smoke --encodings "$S/encodings" --pred "$S/predictor" --out "$S/experiments/mqtl_deep"
run_stage scripts/08_structure.py --smoke --encodings "$S/encodings" --pred "$S/predictor" --out "$S/experiments/structure"
run_stage scripts/10_decide.py --smoke
run_stage scripts/09_train_lewm.py --smoke --encodings "$S/encodings" --out-root "$S/lewm" --lam 0.1
