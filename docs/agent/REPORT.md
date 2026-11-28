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

## Task 7: other frozen encoders (2026-11-24 to 11-25, `slurm/run_encoder_chain.sh`)

Perceiver small (job 42723, CPU, 37 min) and Perceiver large (job 42740, A100, 26 min) encoded through stage 02
`--bundle`, then 03 to 08 for seeds 0 to 2 and 10 (jobs 42739, 42756), all `SANITY_OK`. Unmasked leakage guard
and the 5/20 Mb ablation were not run for them (NOT EVALUATED).

| Encoder | Rung 1 (MSE below `block_mean`) | Rung 2 top-1 L->C / C->L | Rungs 3-5 | Rung 6 | 3D |
|---|---|---|---|---|---|
| CpGPT small | PASS, 17% | 0.015 / 0.035 | FAIL | PASS | not detected |
| Perceiver small | PASS, 93% | 0.007 / 0.013 | FAIL | PASS | not detected |
| Perceiver large | PASS, 94% | 0.007 / 0.013 | FAIL | PASS | lung rho -0.098 (p 0.005), "detectable but negligible" |

Both Perceivers fail rung 2 at chance. Their near-complete rung-1 predictability says little is local to a block.

## Task 8: Phase B, LeWM block encoder (stage 09)

**Global SIGReg (array 42697 and resubmission 42944, A100, about 47 min per lambda).** Tasks 1 and 2 of 42697
requeued about 30 times on `c12m85-a100-1` (node stuck in CONFIGURING about 7 min, then requeued) and were
resubmitted; `c24m170-a100-2` and the other GPU partitions reject the course account. All three lambdas train
but fail stage 03: within-block variance of g is 0.0000 (0.0002 at lambda 1), so `block_mean` predicts
everything and the residual equals standardised Z. SIGReg on z_B pooled over random blocks is satisfied by
spreading blocks apart, so samples collapse inside a block.

| lambda | decoder R^2 | Z - tissue mean top-1 L->C / C->L | after full covariates (M4) |
|---|---|---|---|
| 0.01 | 0.30 | 0.020 / 0.039 | - |
| 0.1 | 0.56 | 0.046 / 0.065 | - |
| 1 | 0.56 | 0.281 / 0.268 | 0.072 / 0.052 |

At lambda 1 most of the 0.28 is shared sex/age/smoking: M1 (tissue, sex, age, smoking) leaves 0.118 / 0.111.
Only 2 of 153 lung/colon pairs share a Sentrix chip and 24 a plate, so the drop at M3 is not chip sharing.

**Within-block SIGReg (`--sigreg-mode within`, 399e0e4; smoke 43140, full array 43141).** Batches are 4 target
blocks x 16 samples and SIGReg acts on z_B minus its block's batch mean. Collapse is fixed (within-block
var(z_B) at epoch 20: 0.61 at lambda 0.1, 0.91 at lambda 1; stage 03 passes), but the residual carries little:

| lambda | residual top-1 L->C / C->L | after M4 | mQTL OR / negative-set OR (top 5%) |
|---|---|---|---|
| 0.1 within | 0.065 / 0.052 | 0.033 / 0.007 | 1.55 / 6.26 |
| 1 within | 0.039 / 0.052 | 0.033 / 0.033 | 1.56 / 4.81 |

## Task 9: beta-space ladder (PREREG amendment A1, jobs 43253-43264, `outputs/decision_beta.json`)

Thresholds approved by the user before submission. All 12 CPU jobs `SANITY_OK`.

| Rung | Result | Numbers |
|---|---|---|
| B1 | FAIL | rel. MSE reduction 0.038 (needs 0.10) |
| B2 | FAIL (comparator only) | top-1 0.745 / 0.830, perm p 0.0001; R - (Z - tissue mean) 0.007 [-0.013, 0.033] and 0.013 [-0.020, 0.046] |
| B3 | PASS | after full covariates 0.719 / 0.719; 129 donors sharing neither plate nor chip 0.729 / 0.736 |
| B4 | FAIL | variance-matched mQTL OR 0.946 [0.926, 0.967]; negative set OR 1.153 [1.105, 1.203] |
| B4c | pass (B4 fails overall) | median rho lung vs colon: mQTL 0.600 vs control 0.082, MW p 1.7e-59; raw betas 0.731 vs 0.032 |
| B5 | PASS | Spearman rho 0.193, p 0.002 over 278 blocks |
| B6 | PASS | struct keeps 100% |
| Leakage | UNMASKED EXCESS | +0.026 [0.007, 0.052] lung->colon, 0.000 colon->lung |

The printed interpretation in that run ("do not identify donors") was misleading; `10_decide.py` now says when
rung 2 fails on its comparator only. Reading: a strong donor signal in tissue-centred betas, not batch, partly
from SNP-affected probes, genetic at the strongest mQTL CpGs (B4c, B5) but not mQTL-specific site by site; the
predictor's residual adds nothing over tissue-centring.

## Task 10: reconstruction test (stage 11, job 44115; 8cc84d4)

y_hat = z_B . w_i for each run at the 200 strongest mQTL CpGs and variance-matched controls, 153 test donors.
(Job 43572 was cancelled by uid 0 after an hour pending, never started; resubmitted.)

| Representation | Concordance mQTL / control | Fidelity, lung, mQTL / control | Matching L->C, 200 mQTL sites |
|---|---|---|---|
| raw betas | 0.73 / 0.03 | - | 1.00 (controls 0.05) |
| beta-space PCA | 0.56 / 0.05 | 0.89 / 0.76 | 0.97 |
| CpGPT small | 0.11 / 0.12 | 0.16 / 0.36 | 0.01 |
| Perceiver large | 0.09 / 0.04 | 0.29 / 0.59 | 0.12 |
| LeWM lambda 1 within | 0.05 / 0.06 | 0.11 / 0.28 | 0.05 |

CpGPT's reconstructions keep no mQTL-vs-control concordance gap and track mQTL sites worse than controls.

## Task 11: genotype fingerprint (stage 12, job 45167; 45239bd)

SNP annotation: Zhou et al. EPIC hg38 common-SNP table (InfiniumAnnotationV1 `Anno/EPIC/EPIC.hg38.commonsnp.tsv.gz`,
in `data/annotations/snp_dl/`), MAF >= 1% globally or in Europeans. 20% of block sites and 38% of the top 200
mQTL CpGs overlap a common SNP.

- **SNP-free sites.** Raw-beta concordance 0.71 / 0.04; lung->colon top-1 at 200 sites 1.00 (controls 0.03).
- **Fingerprint size (SNP-free, L->C).** 5 sites 0.47, 10 sites 0.67, 20 sites 0.87, 50 sites 0.99; controls 0.01-0.03.
- **Tissue pairs.** All 20 ordered pairs with >= 20 shared test donors match at 0.88-1.00 on SNP-free sites
  (testis pairs lowest, 24 donors).
- **Cluster shape (raw betas).** SNP-free mQTL sites prefer 3 components at 55% (lung) / 39% (colon) of sites;
  lung-vs-colon spread difference 0.057 (SNP probes 0.033): SNP probes are a little more tissue-invariant.
- **Whole-sample embedding.** CpGPT `Z_full` reconstructions: concordance 0.14 / 0.14, matching 0.01-0.03 at
  every size. Perceiver `Z_full` also at chance. Per-block PCA keeps 0.96.

**Conclusion.** The eGTEx betas carry a genotype fingerprint (20 SNP-free mQTL CpGs re-identify 87% of donors
across tissues). Frozen CpGPT, block-wise or whole-sample, discards it, and so do the Perceivers and LeWM; a
linear per-block PCA keeps it. The JEPA residual adds nothing on any representation.
