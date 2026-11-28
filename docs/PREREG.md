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
| 2026-11-25 | Amendment A1 (section 8): a second, beta-space ladder, with three added criteria (B3 cross-batch, B4 variance-matched enrichment, B4c cross-tissue concordance). The CpGPT ladder above is unchanged. | The CpGPT ladder failed at rung 2 while the beta-space residual matched donors (REPORT.md task 6), so the question becomes whether that residual is genetic. | Rungs 1-2 in beta space: yes (top-1 0.745 / 0.830 reported in REPORT.md). B3, B4, B4c, B5, B6 and the beta-space leakage guard: no, none of them has been computed. |

## 8. Amendment A1 (2026-11-25): beta-space ladder

**Status: written 2026-11-25 on the user's go-ahead for "Plan A". Thresholds approved by the user on
2026-11-25, unchanged, before `slurm/run_phase_a_beta.sh` was submitted (jobs 43253-43264, same day); none of
the results they govern had been computed at approval. Outcome: `outputs/decision_beta.json` (REPORT.md task 9).**

**Why.** The CpGPT ladder failed at rung 2. The decisive control (section 3), the same predictor and residual on a
per-block PCA of the betas, matched donors at top-1 0.745 (lung to colon) and 0.830 (colon to lung). That rule
says the finding is about methylation data. This amendment sets out in advance how that beta-space residual is
tested for being batch-driven, SNP-artefact-driven or genetic. Sections 1 to 7 stay as written and apply to the
CpGPT result, which is reported as the negative control.

**Representation.** The `mlp_joint` residual on the beta-space encodings (`outputs/encodings_beta`, k = 128 PCA
components per block fit on train samples, split seed 0), predictor seeds 0, 1, 2 (`outputs/predictor*_beta`).
Everything else in section 1 (splits, probe mask, seeds, mean over seeds) is unchanged.
Applied mechanically by `scripts/10_decide.py --space beta` -> `outputs/decision_beta.json`.

| Rung | Pass criterion |
|---|---|
| B1 | as rung 1, on the beta-space predictors |
| B2 | as rung 2; the comparator is the beta-space `Z - tissue mean` (tissue-centred PCA scores) |
| B3 | as rung 3, **and** batch: restricted to queries whose lung and colon samples share neither a plate nor a Sentrix chip (if fewer than 20 such donors: those that share no chip; if still fewer than 20, NOT EVALUATED), top-1 after the full covariate model is >= 5/153 and >= 50% of the all-query top-1 after the same model, both directions. The same-donor vs different-donor plate and chip sharing rates are reported. |
| B4 | as rung 4, with the residual score's percentile taken within (block, decile of the site's train-sample beta variance) instead of within block, for both the any-tissue mQTL set (background: tested CpGs) and the age/tissue negative set. The within-block version is reported. |
| B4c | cross-tissue concordance: for the 200 strongest mQTL CpGs (stage 07 part 3) and one variance-matched control per CpG (tested, not significant in any table, same block, nearest train-beta variance), Spearman over lung+colon test donors between the donor's lung and colon residual contribution g. Pass: median rho at mQTL CpGs > 0, above the controls' median, one-sided Mann-Whitney p < 0.01. The same statistic on raw betas is reported as the input reference. Part of B4: B4 passes only if B4c passes. |
| B5 | as rung 5 |
| B6 | as rung 6 (stage 08 on the beta-space encodings) |
| Leakage | as section 6: beta-space residual on the unmasked encodings vs the masked ones (seed 0), same rule |

Interpretation, fixed in advance: B1 to B3 pass means "a donor-specific subspace exists in methylation residuals and
is not plate, chip or measured covariates". B4 (with B4c) and B5 passing as well means "it is genetic". A leakage
excess means SNP-affected probes carry part of it and is reported next to whatever the ladder says. Failing B3
means the beta-space match is not shown to be donor biology, and is reported as such.
