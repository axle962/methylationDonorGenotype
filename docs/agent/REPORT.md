# Autonomous session report (started 2026-11-22)

Session job 39539 (`slurm/agent_session.sbatch`), branch `jepa/genotype-probe`. Brief: `docs/agent/BRIEF.md`.
Rules applied: `docs/PREREG.md` (frozen). GPU budget: 50 GPU-h (`slurm/gpu_budget.sh`).

## Task 1: smoke test

**What we wanted.** Confirm that the code written just before the session (joint predictors, `--split-seed`,
stage 04 inference, `--probe-mask`, beta-space in the smoke chain) runs end to end on CPU, then commit it.

**What we did.** Waited for smoke job 39533. Stages 01 to 07 (including 03/04 `--beta-space`) ended
`SANITY_OK`. Stage 08 failed: its pair-error matrix called the predictor with one source per row, but the
joint (deep-sets) model expects a set axis (`RuntimeError: Tensors must have same number of dimensions`).
Fix: for a joint model, each single source is passed as a set of size 1, so `E[a, b]` stays the one-source
error. Re-ran stage 08 (job 39540: `SANITY_OK` twice, as expected) and stage 09 (job 39541, which skipped
in 39533 because 08 failed first: `SANITY_OK` six times, 03-07 inside it included).

**What we did not do.** No change to `cpgpt/`. Smoke numbers are not interpreted (20 samples, 5 blocks).

**Result.** All stages pass. Committed as five commits (447fe7c joint predictors + 08 fix, a5aebb2
split seed, d095e77 stage 04 inference, ed60ba1 probe mask, 75cfb07 smoke chain).

**Next.** Task 2, the decision script. While reading the stage outputs against the PREREG, check that each
rule has the numbers it needs.

## Tasks 2 to 4: decision script, full chain, 3D tracks

The autonomous session (job 39539) did these before it was stopped at 05:21 on 2026-11-22 by a SIGTERM
from the cluster: `scripts/10_decide.py` (dfa5587), the PREREG fixes it needed (8b99809 mQTL background and
odds-ratio CI, f80431d smoking and the permutation test, de4f694 beta-space PCA on the split seed), a
tissue-matched ENCODE compartment track per tissue for stage 08 (6e5821b), and the full chain
`slurm/run_phase_a.sh` (2f29d1c). The smoke chain (job 39554) passed every stage. Tracks used by stage 08:
lung `lower_lobe_of_left_lung`, colon `transverse_colon` (both ENCODE tissue tracks, coverage 1.0).

## Task 5: full run (continued interactively, 2026-11-22 to 11-24)

**What we wanted.** Run Phase A end to end on the real data.

**What got in the way, and what we changed.**
- Partitions. The course account accepts only `n2c48m24` for CPU (24 CPUs, 22 GB) and, among GPU partitions,
  `g2-standard-12` (L4) and `c12m85-a100-1` (A100). CPU stages moved to `n2c48m24` at 22 GB (7f116bc).
- `g2-standard-12` nodes failed to boot for every user from about 2026-11-22 23:00 (NODE_FAIL, requeue every
  10 minutes; 26 restarts of one job). GPU jobs moved to `c12m85-a100-1` (ff095ee).
- Stage 01. `CpGPTDataSaver` builds one pandas frame of 987 x 754,110 columns; it ran 4 h at the 22 GB limit
  and timed out (job 40423). Stage 01 now has a native writer that applies the same column filter and writes
  the same files in column chunks (7da25a9). On the smoke subset its files are byte-identical to the
  datasaver's (job 41067). The full run took 4 minutes and 11 GB (job 41068).
- Budget guard. `sacct` does not list jobs waiting on a dependency, so the guard undercounted; it now also
  counts queued jobs from `squeue` (386e5fd).
- Kills. Four jobs were cancelled by uid 0 at :21 or :51 past the hour after 32 to 34 minutes: the first
  agent session and three runs of stage 08 seed 0, each while streaming Hi-C from ENCODE. Each 08 run kept its
  cached contact maps (71, 132, 188, then 231 of 231), so the fourth run finished. One stage 04 job lost its
  node (NODE_FAIL) and was resubmitted.

**What we did not do.** No change to `cpgpt/` or to PREREG thresholds. Phase B (stage 09) not run.

**Result.** Every stage of Phase A ended `SANITY_OK` for seeds 0, 1 and 2. Stage 01: 987 samples,
754,110 sites, all with a DNA embedding, no missing values; chrX has 3 sites and chrY 1, so the analysis is
autosomal. Stage 02 (masked): 278 blocks of 10 Mb, 741,370 sites, 20 minutes on one A100; decode check
r = 0.92 to 0.94 (0.32 to 0.36 with site means removed). GPU-hours used: about 1.7 of 50.

## Task 6: decision (scripts/10_decide.py, job 42547, `outputs/decision.json`)

Kind `mlp_joint`, seeds 0 to 2, test donors only (153 lung + colon donors, chance top-1 1/153 = 0.0065).

| Rung | Result | Numbers |
|---|---|---|
| 1 Predictor non-trivial | PASS | held-out MSE 17.5% below `block_mean`, diff CI [-0.224, -0.205] |
| 2 Residual identifies the donor | **FAIL** | top-1 lung->colon 0.015 (perm p 0.198), colon->lung 0.035 (perm p 0.001); minimum 0.033 both ways; not above `Z - tissue mean` (CI [-0.089, -0.002] and [-0.070, 0.022]) |
| 3 Survives covariate removal | FAIL | top-1 after regression 0.026 and 0.020, Holm p 0.034 and 0.074 |
| 4 Genetic in the specific sense | FAIL | mQTL CpGs enriched (pct 0.554 vs 0.467, top-5% OR 1.58 [1.55, 1.62]), but the age/tissue negative set is enriched more (OR 2.37 [2.29, 2.45]) |
| 5 Structured where it should be | FAIL | Spearman rho 0.061, p 0.38 over 278 blocks |
| 6 Not 3D structure | PASS | `mlp_joint_struct` keeps 100% of `mlp_joint` top-1 |

**Beta-space rule: about methylation data.** The same predictor on per-block PCA of the betas
(k = 128, 96.8% of train variance) gives residual top-1 0.745 (lung->colon) and 0.830 (colon->lung).
CpGPT minus beta-space: -0.730 [-0.802, -0.658] and -0.795 [-0.861, -0.725]. Raw betas: 0.039 and 0.020;
tissue-centred raw betas: 0.255 and 0.222.

**3D test (PREREG section 5): not detected.** Partial Spearman of log pair error vs compartment similarity
(control: gene density): colon rho 0.010 (shift null 95% [-0.070, 0.032], p 0.76), lung rho 0.000
([-0.051, 0.030], p 0.98). Hi-C rho -0.096 (colon) and -0.088 (lung).

**Leakage guard: no significant excess.** Unmasked minus masked top-1: 0.007 [0.000, 0.020] lung->colon,
0.007 [-0.013, 0.033] colon->lung.

**Block size (stage 05, seed 0).** Residual top-1 lung->colon / colon->lung: 5 Mb 0.007 / 0.007,
10 Mb 0.013 / 0.033, 20 Mb 0.026 / 0.013. No size rescues matching.

**Interpretation (fixed in advance).** Rung 2 fails: frozen CpGPT block residuals do not identify donors.
The beta-space control shows the signal is in the data (about 80% top-1 across tissues with the same
predictor and residual), so CpGPT's 128-dimensional block embedding discards nearly all of it.

**Next.** Phase B: train a block encoder that keeps local variation (stage 09, LeWM) and run the same
ladder on it. Out of this brief's scope; needs the user's go-ahead.
