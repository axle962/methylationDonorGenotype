# Brief for the autonomous session (started 2026-11-22)

You are a Claude Code session running headless inside a Slurm CPU job (`slurm/agent_session.sbatch`)
on NYU HPC Cloud Bursting. Nobody is watching live. The user (Aditya) has authorised you to finish the
work below, including submitting the CPU and GPU Slurm jobs it needs, **within a hard budget of 50
GPU-hours** for the whole session. Read `CLAUDE.md`, `docs/PREREG.md`, `docs/plan.md` and this file
before doing anything. Your memory directory also holds the user's standing rules; follow them.

## Where things stand

- Repo `/scratch/am16489/methylationDonorGenotype`, branch `jepa/genotype-probe`. Skanda's Phase A/B
  pipeline was imported from `CpGPT_AMG_Lab@jepa_predictor` and ported to Cloud Bursting (see git log).
- `docs/PREREG.md` is **approved and frozen**: primary endpoint, comparators, success-ladder thresholds,
  3D test, leakage guard. You may not change any threshold or rule in it.
- Data are in place: `data/egtex/` (betas, covariates, mQTL tables), `data/annotations/` (Zhou EPIC hg38
  mask), CpGPT small in the HF cache (`slurm/local.env`).
- Uncommitted code written just before this session (commit it once the smoke test confirms it):
  - joint deep-sets predictor kinds `mlp_joint` (primary) and `linear_joint` (`src/cpgpt_jepa/predictor.py`);
    stages 05-08 default to `--kind mlp_joint`;
  - stage 03 `--split-seed` (fixed 0: donor split, held-out blocks, fixed sources) separate from `--seed`
    (training);
  - stage 04 inference: permutation p vs chance, paired bootstrap of the residual vs `Z - tissue mean` and
    vs the beta-space residual, rung-1 held-out MSE bootstrap vs `block_mean`;
  - stage 02 `--probe-mask` (default: Zhou `M_general`, 44,339 probes; `none` = unmasked secondary);
  - `slurm/smoke.sbatch` now also runs 03/04 `--beta-space`.
- Already committed: the session infrastructure (`slurm/gpu_budget.sh` and the guard in `slurm/submit.sh`,
  `slurm/wait_jobs.sh`, `slurm/agent_session.sbatch`, this brief, the CLAUDE.md exception).
- **Smoke job 39533** (CPU) was submitted to test that code. Its log: `slurm/logs/jepa_smoke_39533.out`.

## Tasks, in order (finish each, commit, update the report, then move on)

1. **Smoke test.** Wait for 39533 (`slurm/wait_jobs.sh 39533`). Every stage must end `SANITY_OK`; note
   stage 08 prints it twice and 09 runs 03-07 internally. Fix failures in our code (never in `cpgpt/`),
   re-test single stages with `slurm/smoke_stage.sbatch`, then commit the new code (logical commits).
2. **Decision script** `scripts/10_decide.py`: reads every output the PREREG needs (stage 03 metrics,
   04 evaluation.json incl. `inference`, 05-08 results, all three seeds, beta-space, unmasked) and applies
   docs/PREREG.md mechanically: each rung PASS / FAIL / NOT EVALUATED with the numbers behind it, the
   beta-space rule, the 3D rule, the leakage comparison. Criteria use the mean over seeds 0-2 where the
   PREREG says so. Writes `outputs/decision.json` and prints a table. Add a smoke variant and test it via
   a CPU job.
3. **Wire the full chain** (`slurm/run_phase_a.sh`, keep `DRY_RUN=1` working), roughly:
   01 -> 02 (masked, `--include-x`) -> 03 seeds 0,1,2 (seed 0 -> `outputs/predictor`, others
   `outputs/predictor_seed{1,2}`) -> 03 `--beta-space` -> 04 `--beta-space` -> 04 for each seed -> 05-08
   on seed 0 -> 10. In parallel after 01: 02 unmasked (`--probe-mask none`, `outputs/encodings_unmasked`)
   -> 03/04 on it; 02b 5 and 20 Mb encodings before 05. Check every stage's paths and flags actually line
   up (the smoke chain only exercises some of them).
4. **3D tracks.** PREREG section 5 requires a tissue-matched ENCODE compartment track per tested tissue
   (lung: a lung cell type such as IMR-90 or lung tissue; colon: a colon cell type or tissue), built with
   `experiments/threed/build_annotations.py` (+ circular shifts via `make_shift_control.py`). Check what
   ENCODE actually has, record the cell type per tissue in the report, and make stage 08 use the right track
   for each tissue's samples. If ENCODE has no usable track for a tissue, say so and stop that part; do not
   substitute one cell type for all tissues.
5. **Full run.** Submit the chain. Time the first GPU job (stage 02) and tighten `--time` on the remaining
   GPU jobs so the committed budget stays well under 50 GPU-h (`slurm/gpu_budget.sh` shows used and
   committed). Fix and resubmit failures; three failed attempts at the same step means stop and report.
6. **Report** against docs/PREREG.md: run `scripts/10_decide.py`, then write the final section of the
   report (below). Negative results are results: if rung 2 fails, say plainly that frozen CpGPT block
   residuals do not identify donors.

Out of scope: Phase B (stage 09 full run), GoDMC, GEO whole blood, the `large` model, any new dataset or
stage beyond the above, and any change to PREREG thresholds.

## Rules for this session

- **Jobs:** submit only through `slurm/submit.sh` (or `slurm/run_phase_a.sh`, which uses it). Never call
  `sbatch`, `srun` or `salloc` directly, never bypass or edit the budget guard, and never raise
  `GPU_BUDGET_HOURS`. This CLAUDE.md rule ("never submit a Slurm job unless the user says so") is waived
  for this session by the user's authorisation above; the budget is not.
- **Compute:** pipeline stages and anything heavier than a quick check run as Slurm jobs, not inside this
  session's process.
- **Waiting:** `slurm/wait_jobs.sh <ids>` blocks up to 9 minutes and exits 2 while jobs are still queued
  or running; call it again. Do not end your turn while your jobs are still running.
- **Budget watchdog:** this session's job cancels all `jepa_*` GPU jobs if real usage reaches 50 GPU-h and
  creates `logs/agent/BUDGET_EXHAUSTED`. If that file exists, or the guard refuses a job, stop GPU work,
  finish what CPU work you can, and report.
- **Pre-registration:** apply it, never edit its rules. A purely procedural problem (a path, a file that
  does not exist) may be logged in its amendment table **only if you have not seen the result it
  affects**; anything touching a threshold, endpoint or comparator is a question for the user: write it
  under "Questions for Aditya" in the report and stop that line of work.
- **Git:** commit on `jepa/genotype-probe` after each working step, with clear messages ending in
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push. Never commit `data/`, `outputs/`,
  `slurm/logs/`, `slurm/local.env`.
- **Files:** never delete anything you did not create in this session (see the user's global CLAUDE.md).
- **Writing:** plain English, no em dashes.

## Report

Keep `docs/agent/REPORT.md` current: after every task, append a section in the project journal's shape
(what we wanted, what we did, what we did not do, results with the numbers, what next and why). End with a
summary: ladder rung by rung with PASS/FAIL and the numbers, the beta-space verdict, the 3D verdict, the
leakage comparison, GPU-hours used, open questions. If you stop early for any reason, the last section says
exactly why and what the next step would be.
