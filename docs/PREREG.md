# Pre-registration: Phase A decision rules

**Status: APPROVED 2026-11-22 by Aditya Maddur Guruprakash; every [proposed] value accepted as written.**
Written 2026-11-22, before any full-scale run. Only the 50-sample
smoke test (outputs/smoke, 5 blocks, 5 lung-colon donors) has been run, and its numbers are not used here.
Once approved, this file is frozen: changes go in the amendment log at the end, dated, with the reason,
and only before the result they affect has been seen.

Items marked **[proposed]** were proposals in the draft. All were accepted at approval and are now binding.

## 1. Fixed analysis choices

| Choice | Value |
|---|---|
| Encoder | released CpGPT `small` (d = 128), frozen, eval mode, CLS embedding |
| Block size (primary) | 10 Mb; 5 and 20 Mb are ablations only |
| Predictor (primary) | `mlp` from stage 03; `linear` and `block_mean` are baselines |
| Residual (primary) | **[proposed]** one predictor that takes all K = 8 source blocks jointly, replacing the K-averaged residual (verification.md §5b) |
| Splits | donor-disjoint; all 153 lung+colon donors are test donors; standardisation uses train-donor statistics only; X-chromosome blocks are eval-only |
| Probe set (primary) | **[proposed]** all retained `cg` probes **minus** the Zhou et al. 2017 EPIC general mask (SNP at or near the CpG, cross-hybridising). The unmasked set is reported as a secondary analysis. |
| Seeds | predictor seeds 0, 1, 2; each criterion is evaluated on the mean over seeds |

No `rs` genotyping probes are present: all 754,110 retained probes are `cg` (checked in the smoke subset).

## 2. Primary endpoint

**Lung-to-colon donor retrieval, top-1, on the residual R**, over the 153 test donors (chance = 1/153 ≈ 0.0065),
reported together with colon-to-lung. Both directions must pass for a criterion that uses it.

Inference (code to be added to stage 04 before the full run):
- **Against chance:** permutation test, shuffling donor labels within the colon gallery, 10,000 permutations, one-sided.
- **Against a baseline representation:** paired bootstrap over test donors (10,000 resamples). The difference
  in top-1 is significant if its 95% CI excludes 0.

Everything else is secondary. Secondary p-values are Holm-corrected within each rung.

## 3. Comparators (all on the same test donors and the same probe set)

| Comparator | Role |
|---|---|
| `Z - tissue mean` (tissue-centred block embeddings) | baseline that rung 2 must beat |
| `Z_full` (one 10k-site genome-wide embedding) | context: what the usual CpGPT embedding carries |
| **Beta-space residual** (`--beta-space`: same predictors on a per-block PCA of the betas) | **decisive control**: does the result depend on CpGPT? |
| Raw-beta matching on the same sites | ceiling: how much donor signal the inputs themselves carry |

**Beta-space decision rule [proposed]:** if the beta-space residual's top-1 is not significantly below the
CpGPT residual's (paired bootstrap CI includes 0 or favours beta space), the finding is reported as being
about methylation data, not about CpGPT.

## 4. Success ladder (from docs/plan.md, with thresholds)

| Rung | Claim | Pass criterion [proposed] |
|---|---|---|
| 1 | Predictor is non-trivial | On held-out blocks and test donors, `mlp` test MSE is at least 10% below `block_mean`, and the paired bootstrap CI over donors for the difference excludes 0 |
| 2 | Residual identifies the donor | top-1 ≥ 5/153 (≈ 0.033, about 5× chance) in both directions, permutation p < 0.01, **and** significantly above `Z - tissue mean` |
| 3 | Survives covariate removal | after stage 06 regression (tissue, sex, age, plate, sentrix_id, PEER, **smoking** [proposed addition]), top-1 still meets rung 2's chance criterion and keeps ≥ 50% of its pre-regression value |
| 4 | Genetic in the specific sense | residual-driving sites enriched for cis-mQTL CpGs: Mann-Whitney p < 0.01 with mean percentile > 0.5, and top-5% odds ratio > 1 with Fisher p < 0.01; **and** the age/tissue-associated non-mQTL negative set is not enriched (OR CI includes 1, or OR significantly below the mQTL set's) |
| 5 | Structured where it should be | per-block matching contribution vs per-block mQTL density: Spearman ρ > 0, p < 0.05, over autosomal blocks |
| 6 | Not 3D structure | with the compartment profile appended to c_B (`mlp_struct`), top-1 keeps ≥ 80% of the `mlp` value |

Interpretation, fixed in advance: passing rungs 1-3 means "a donor-specific subspace exists". Passing 4-5 as
well means "it is genetic". Failing rung 2 is a negative result: frozen CpGPT block residuals do not identify
donors, and it is reported as such.

**mQTL background [proposed]:** the non-mQTL comparison set is restricted to CpGs present in the eGTEx mQTL
tables (tested and not significant), not all other array CpGs. Base rates are high (in the smoke subset,
46% of block sites are in the any-tissue mQTL list), so enrichment is reported as an odds ratio with a CI,
not only a p-value.

## 5. 3D-structure test (stage 08)

Run after rungs 1-5, on the same predictor.

- **Statistic:** partial Spearman between log pair error E[a, b] and compartment similarity, controlling for
  gene density. Separately, partial Spearman between log E and log inter-chromosomal Hi-C contact
  (10 Mb bins, observed/expected).
- **Null:** 200 circular shifts of the compartment track along each chromosome. The effect is real if the
  observed ρ is more extreme than the 97.5th percentile of the shift null (two-sided, α = 0.05).
- **Expected direction:** more similar compartments, lower error (ρ < 0).
- **Expected size, stated now:** small. Earlier work in this project found compartment structure in
  CpGPT reconstruction errors (8 of 8 windows), but adding compartments as model input gained only 0.14% of
  MAE over a gene-density arm. **[proposed]** |ρ| < 0.1 is reported as "detectable but negligible",
  whatever its p-value.
- **Tracks [proposed]:** a tissue-matched ENCODE compartment track per tested tissue: IMR-90 or another lung
  cell type for lung, and a colon cell type (e.g. transverse colon tissue or Caco-2) for colon. The cell type
  is recorded per tissue. Using one cell type for every tissue is an analysis error, not a robustness check.

## 6. Leakage guard (before rung 2 is read)

- Matching is reported with and without the Zhou mask (§1). If unmasked matching is much higher than
  masked, the excess is attributed to SNP-affected probes, not to CpGPT.
- **Ground-truth donor check:** the rs-probe genotype caller (`same_person.py`) cannot be used, because no
  rs probes are in the data. **[proposed]** donor identity comes from the eGTEx metadata only.

## 7. Amendment log

| Date | Change | Reason | Seen any result it affects? |
|---|---|---|---|
| | | | |
