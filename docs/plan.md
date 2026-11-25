# Experiment plan

Companion to `docs/proposal.md`. Phase A analyses the residual of the latent block predictor trained on
frozen CpGPT block embeddings (stages 01–04). Phase B replaces the frozen encoder with a block encoder
trained end to end. Every Phase A experiment reads `outputs/encodings` and `outputs/predictor/mlp`, uses
test donors only, writes JSON and CSV under `outputs/experiments/<name>/`, and prints one summary table.

## Phase A experiments

### 05 — Ablations (`scripts/05_ablations.py`)
The residual is rebuilt with K = 1, 4, 8 and 32 source blocks, and donor matching is repeated for each K.
Block size is varied over 5, 10 and 20 Mb by re-partitioning the same X.mmap. The 5 and 20 Mb encodings come
from a separate GPU job (`slurm/02b_encode_sizes.sbatch`), so the ablation skips any size whose encodings
are absent and says so. Donor matching is extended to every pair of tissues that shares at least 20 donors,
reported as a matrix. Finally, each donor's residual is averaged over all of that donor's tissues and
retrieval is run leave-one-tissue-out.

### 06 — Covariates (`scripts/06_covariates.py`)
Within each block the residual is regressed on tissue, sex, age (from the GTEx covariates), plate,
sentrix_id, and the PEER factors in `data/egtex/metadata/<Tissue>.covariates.txt`, aligned by donor. PEER
factors are defined per tissue, so that part of the fit is done within tissue. Donor matching is re-run on
the cleaned residual and reported before and after.

### 07 — mQTL in depth (`scripts/07_mqtl_deep.py`)
Enrichment is computed against lung-only, colon-only and union mQTL sets. A negative CpG set is formed from
the top 5% age- or tissue-associated CpGs (ANOVA on betas within train donors) that are not mQTLs. Per-block
matching accuracy, measured as each block's leave-one-block-out contribution, is compared with per-block mQTL
density by Spearman correlation, with the scatter saved as CSV. Tri-modality: for the 200 strongest mQTL CpGs
(smallest q-value, largest |slope|), the residual contribution g across test donors is fitted with Gaussian
mixtures of 1 and 3 components and the fraction of sites whose BIC prefers 3 is reported. chrX blocks are
encoded as a positive control, added to the block table behind the flag `--include-x`.

### 08 — Structure (`scripts/08_structure.py`)
`experiments/threed/build_annotations.py` is run at the current block size to give each block its
A-compartment fraction, Hi-C contact and gene density. A pair-error matrix E[a, b] is computed from the
predictor on test donors. Partial Spearman correlations relate log E to compartment similarity and to log
Hi-C contact, controlling for gene density. The null distribution comes from circular shifts of the
compartment track via `make_shift_control.py` (200 shifts). A predictor `mlp_struct` is trained with the
compartment profile appended to c_B (`scripts/03_train_predictor.py --struct`) and its donor matching is
reported.

## Success ladder

1. Predictor is non-trivial: test-donor MSE on held-out blocks clearly below the block-mean baseline.
2. Residual identifies the donor: lung→colon top-1 well above chance (1/153) AND above the tissue-centred Z baseline.
3. Survives covariate removal: matching holds after regressing out tissue, sex, age, plate, sentrix_id, PEER factors.
4. Genetic in the specific sense: residual-driving sites enriched for cis-mQTL CpGs (percentile > 0.5, OR > 1 in top 5%), and NOT enriched for the matched age/tissue-associated non-mQTL set.
5. Structured where it should be: per-block matching accuracy correlates with per-block mQTL density.
6. Not 3D structure: matching survives the compartment profile appended to c_B.

Pass 1–3 → "a donor-specific subspace exists"; pass 4–5 → "it is genetic"; fail at 2 → negative result, Phase B becomes the question.

## Beta-space ladder (docs/PREREG.md amendment A1, Plan A)

The CpGPT ladder failed at rung 2 while the beta-space residual matched donors, so the same ladder is run on the
beta-space residual (`outputs/encodings_beta`, `outputs/predictor*_beta`) with three added criteria: B3 matching on
donors whose lung and colon share no plate or chip (stage 06 `batch`), B4 mQTL and negative-set enrichment with the
percentile taken within (block, train-beta variance decile) (stage 07 `enrichment_variance_matched`), and B4c
cross-tissue concordance of the residual at the strongest mQTL CpGs vs variance-matched controls (stage 07
`concordance`). `slurm/run_phase_a_beta.sh` (CPU only) runs 03/04 `--beta-space` on the unmasked encodings, 06, 07
and 08 on the beta-space residual for seeds 0-2, then `10_decide.py --space beta` -> `outputs/decision_beta.json`.
`tests/run_fixture.sh` runs 03, 04, 06, 07 and 10 on synthetic data on a laptop.

## Phase B — end-to-end block encoder (`src/cpgpt_jepa/lewm.py`, `scripts/09_train_lewm.py`)

The frozen CpGPT encoder is replaced by a block encoder trained end to end in the LeWorldModel recipe
(arXiv 2603.19312). The encoder g reads a block's sites; the input for each site is CpGPT's
Nucleotide-Transformer sequence embedding (a frozen lookup) concatenated with the beta value. g is a 4-layer
set transformer with d = 256 and mean pooling. The predictor f(g(x_A), c_B) is the one used in Phase A. The
loss is the prediction MSE plus λ·SIGReg on g(x_B), where SIGReg is the sketched isotropic-Gaussian
regulariser from LeJEPA (256 random directions, Epps–Pulley statistic). There is no EMA and no stop-gradient
beyond what the paper uses. λ is swept over 0.01, 0.1 and 1 as an array job. The splits and the residual
construction are the same as in Phase A, and stages 04–07 are run on the resulting embeddings through an
`--encodings outputs/lewm/<run>/` flag. Linear probes on g's latent predict sex, age, tissue and the first
five genotype PCs from the covariate files. The entry point is left runnable but is not run.

## SLURM

Jobs follow the conventions of `slurm/common.sh`. There are sbatch files for 02b (GPU), 05–08 (CPU) and 09
(GPU array over λ). `slurm/run_phase_a.sh` chains 01 → 02 → 03 → 04 → {05, 06, 07, 08} with
`--dependency=afterok`; `slurm/run_phase_b.sh` chains 09 after 02. Every job ends with SANITY_OK or
SANITY_FAIL. `slurm/smoke.sbatch` (CPU, 4 cores, 16 GB, 2 h) runs stage 1 on a 50-sample subset that
includes at least 5 lung–colon donor pairs, then 02–08 with `--smoke` (20 samples × 5 blocks, on CPU), then
09 with `--smoke` for 2 epochs on CPU.
