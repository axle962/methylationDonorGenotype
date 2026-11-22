# Static verification of the pipeline (2026-10-15)

Scope: scripts 01–09, `src/cpgpt_jepa/`, `slurm/`, checked against the CpGPT source and against the LeWM /
LeJEPA papers. No code was executed and no job was submitted. Only clear bugs were fixed; design judgments are
marked CONCERN. Line numbers refer to the tree at commit 8e850db (CpGPT files are unmodified upstream).

Legend: **PASS** = matches the design; **FIXED** = changed in this review (diff summary given);
**CONCERN** = needs a decision.

## 1. Encode path (02, blocks.py)

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1.1 | NaN handling of meth | PASS | Ours `src/cpgpt_jepa/blocks.py:120-121` (`mask_na = torch.isnan(meth)`, `nan_to_num(meth, nan=0.0)`) is the same as CpGPT `cpgpt/model/cpgpt_module.py:768-769`. NaN sites are then dropped per sample and padded with meth 0, seq 0, chroms −1, positions −1, mask True (`blocks.py:131-146`), as in `cpgpt_module.py:797` (NaN removed from the input mask) and `cpgpt_module.py:842-855` (same padding values, `pad_sequence` at :851). |
| 1.2 | meth dtype / shape | PASS | float32 (batch, n_sites): `blocks.py:171` from `X` float32 (`02_encode_blocks.py:96`); CpGPT builds `torch.tensor(X, dtype=torch.float32)` (`cpgpt/data/components/cpgpt_dataset.py:446`). |
| 1.3 | chroms / positions | PASS | Taken from var.mmap int32 (`02_encode_blocks.py:85`, `blocks.py:167-168`); CpGPT: `torch.tensor(var, dtype=torch.int32)` (`cpgpt_dataset.py:447`). Order within a block is ascending position (`blocks.py:93`), which is what `sorted_chromosome` produces for a single chromosome (`cpgpt_dataset.py:175`). |
| 1.4 | DNA embedding lookup | PASS | `blocks.py:54` looks up `"{reverse_vocab[chrom]}:{pos}"`, identical to `cpgpt_dataset.py:317`. |
| 1.5 | Sequence embeddings shared across the batch | PASS | `encode_sequence` once per block on (1, L, 1024) (`02_encode_blocks.py:153`, `blocks.py:149-154`), expanded over the batch (`blocks.py:177`). CpGPT computes it per sample (`cpgpt_module.py:631`), but `dna_encoder` is row-wise, so the values are identical. |
| 1.6 | W from the same tensor | PASS | `w = net.meth_decoder(seq)[0]` uses the same `seq` (`02_encode_blocks.py:154-155`); CpGPT's `query_methylation` uses `meth_decoder(sequence_embeddings)` (`cpgpt/model/components/model.py:485-488`). |
| 1.7 | W stored as fp16 | PASS | On CUDA `meth_decoder` runs under fp16 autocast, so `w` is already fp16 and `.float().astype(float16)` (`02_encode_blocks.py:157`) round-trips exactly: **0 relative error vs the computed w**. On the CPU smoke path w is fp32, and rounding to fp16 gives max relative error 2⁻¹¹ ≈ 4.9×10⁻⁴ per element for \|w\| ≥ 6.1×10⁻⁵ (smaller values are subnormal and lose relative precision; irrelevant for ranking). |
| 1.8 | Autocast / eval / no grad | PASS | fp16 autocast only on CUDA, nullcontext on CPU (`blocks.py:17-21`); CpGPT's trainer uses `16-mixed` (`cpgpt/trainer/cpgpt_trainer.py:12`). `model.to(device).eval().requires_grad_(False)` (`02_encode_blocks.py:74`); `@torch.no_grad()` on both encode helpers (`blocks.py:149,157`) and around the decoder calls (`02_encode_blocks.py:154,172`). `CpGPTInferencer(...)` calls `enable_legacy_numerics()` (`cpgpt/infer/cpgpt_inferencer.py:45`), and we construct it (`02_encode_blocks.py:71`). |
| 1.9 | Power norm / CLS | PASS | small.yaml has `use_power_norm: false` and `sample_embedding_method: cls`; in eval mode `encode_sample` returns the CLS token (`model.py:449-450`). |
| 1.10 | binarize_input | PASS (warn) | small: false. The large model was trained with `binarize_input: true`; CpGPT would Bernoulli-sample the input, while we pass betas and print a warning (`02_encode_blocks.py:79-80`). |
| 1.11 | Pearson sanity check | PASS | `query_methylation(z, seq, m_or_beta="beta")` on the same block sites (`02_encode_blocks.py:172-176`), NaNs excluded (`:177-178`). FAIL if any r < 0.3 (`:230-232`). |
| 1.12 | Z_full | PASS / CONCERN (minor) | Fixed-seed draw of 10,000 sites across the array, same for all samples (`02_encode_blocks.py:128-131`), same encode path (`:196-198`). Two differences from CpGPT's dataset: (a) chromosomes are ordered ascending, whereas CpGPT shuffles chromosome order per item (`cpgpt_dataset.py:181`); rotary encoding is index-based, so order can matter slightly. (b) One draw for all samples, whereas CpGPT draws per `__getitem__` (`cpgpt_dataset.py:417`). Both are deliberate for a deterministic baseline. |
| 1.13 | Site subset | PASS | CpGPT subsamples to `max_length` (`cpgpt_dataset.py:417`) and selects `n_sites // n_splits` sites (`cpgpt_module.py:795`). We use every observed site of a block, equal to n_splits = 1 with blocks below max_length (20,000; rotary `max_seq_len=20001`). |

Our encode path next to CpGPT's (differences marked ◆):

```
CpGPT (_predict_forward_step via CpGPTDataset)          | Ours (02 + blocks.py)
1  item = dataset[i]: random <=max_length sites,         | 1 ◆ cols = all sites of one block, sorted by position;
   sorted_chromosome (pos within chrom, chroms shuffled) |     the same set for every sample (no random subset)
2  dna = emb_mmap[index["chrom:pos"]]  (per sample)      | 2   dna = emb_mmap[index["chrom:pos"]]  (once per block)
3  batch = collate(items)  (pad meth with NaN)           | 3   meth = X[batch, cols]  (float32, equal lengths)
4  mask_na = isnan(meth); meth = nan_to_num(meth, 0)     | 4   mask_na = isnan(meth); meth = nan_to_num(meth, 0)
5  input_mask = randperm(n)[: n // n_splits] & ~mask_na  | 5   input = all sites & ~mask_na   (= n_splits 1)
6  seq = net.encode_sequence(dna)   (B x L x d)          | 6 ◆ seq = net.encode_sequence(dna[None]) (1 x L x d), expanded
7  per sample: select input_mask; pad 0/0/-1/-1/True     | 7   per sample: drop NaN sites; pad 0/0/-1/-1/True
8  z = net.encode_sample(meth, seq, chroms, pos, mask)   | 8   z = net.encode_sample(meth, seq, chroms, pos, mask)
9  precision 16-mixed (CUDA autocast fp16), eval,        | 9 ◆ fp16 autocast on CUDA; fp32 on CPU (smoke only);
   inference_mode                                        |     eval, requires_grad False, no_grad
10 pred = bmm(meth_decoder(seq), z)  (M-values)          | 10  w = meth_decoder(seq) once; W.mmap fp16 (lossless on CUDA)
11 —                                                     | 11  check: query_methylation(z, seq, "beta") vs betas, r >= 0.3
12 Z_full analogue: random sites per item, chroms shuffled | 12 ◆ 10k sites, one fixed-seed draw, chroms ascending
```

## 2. Splits and training (03, predictor.py)

Counts from `data/egtex/metadata.csv`. These were computed with Python's `csv` module because the host Python has
no pandas and the venv only runs inside the container; the logic is the same as `make_splits`.

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 2.1 | All lung+colon donors in test; donor-disjoint | PASS | `predictor.py:79-80` puts every donor with both tissues in test, and `:84` assigns samples by donor (`np.isin(donor_ids, test)`), so no donor can be split. Counts: 424 donors, 153 lung+colon donors, 25% target 106 so fill = 0, **test 153 donors / 503 samples, train 271 donors / 484 samples, 0 donors in both**. Test by tissue: Lung 153, Colon 153, Prostate 56, Ovary 35, Whole Blood 30, Testis 24, Kidney 19, Breast 17, Muscle 16. No test donor has two lung or two colon samples. |
| 2.2 | mu/sd from train samples only | PASS | `03_train_predictor.py:98-99` uses `Z[train_s][:, auto_blocks]`. |
| 2.3 | block_mean from train samples only | PASS | `predictor.py:146` passes `Zs[train_s_t]`; `predictor.py:46-49` takes the table mean over those, and blocks that are never targets get the mean over train blocks. |
| 2.4 | Held-out blocks never targets | PASS | Targets are drawn from `train_b` only (`predictor.py:155`); `train_blocks` excludes held-out and chrX blocks (`predictor.py:86-87,98`). |
| 2.5 | chrom(A) ≠ chrom(B) | PASS | Rejection sampling until no same-chromosome pair remains (`predictor.py:105-111`), source pool = autosomal blocks (`03_train_predictor.py:120` passes `auto_blocks`). |
| 2.6 | Fixed K sources, same for every sample | PASS | `fixed_sources` draws K sources per target once, from other chromosomes (`predictor.py:113-120`), saved in splits.json; `predict` uses `sources[b]` for all samples (`predictor.py:173-188`); residual `R = Zs − P` (`03_train_predictor.py:124-125`). |
| 2.7 | MSE in standardised space, no per-sample normalisation | PASS | `mse_loss(pred, Zs[...])` (`predictor.py:162`); the only normalisations are per-dimension over train samples (`03:98-103`) and per-dimension over blocks for c_B (`predictor.py:123-129`, no sample information). |
| 2.8 | Dropout off when predicting | PASS | `model.eval()` in `predict` (`predictor.py:177`). |
| 2.9 | Test fraction | CONCERN | The 25% donor target cannot be met: the 153 required test donors are 36% of donors and 51% of samples. That follows from the design; it leaves 484 training samples. |

## 3. Evaluation (04, evaluation.py)

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 3.1 | Query/gallery restricted to test donors | PASS | `evaluation.py:145-151`: lung/colon sets are `run.is_test & tissue`; queries are test samples of lung+colon donors; galleries are all test samples of the other tissue. |
| 3.2 | Cosine on the residual flattened over blocks | PASS | `evaluation.py:154-162` flattens the autosomal blocks; `:127-129` L2-normalises and takes `q @ g.T`. |
| 3.3 | top-1 / top-5 / MRR per query, chance, both directions | PASS | Rank of the best correct gallery item (`evaluation.py:132-136`, strict ">" so ties count in the query's favour; ties are essentially impossible with float cosines). Metrics per query, averaged (`:139-142`); chance = 1/n_gallery = 1/153; both directions (`:161-162`). |
| 3.4 | Tissue-centred Z baseline uses train tissue means | PASS | `evaluation.py:165-172` (`ref = run.train_s[...]`). |
| 3.5 | Unstandardise before the dot product, same sd as 03 | PASS | `g = (R[samples, b] * run.sd) @ W[rows].T` (`evaluation.py:184`); `run.sd` is read from 03's `standardise.npz` (`evaluation.py:107`). mu cancels in a residual. |
| 3.6 | Per-site score, within-block percentiles | PASS | Variance over the chosen test samples (`evaluation.py:186`); within-block ranks `(rank − 0.5)/n` (`:201-206`). 04 uses all test samples plus each tissue's test samples (`04_evaluate.py:63-65`). |
| 3.7 | Mann–Whitney direction | PASS | `alternative="greater"` for set vs other (`evaluation.py:220`). |
| 3.8 | Fisher orientation | PASS | `[[top∧mQTL, top∧¬mQTL], [¬top∧mQTL, ¬top∧¬mQTL]]` (`evaluation.py:216-217`) gives OR = (a·d)/(b·c) > 1 when mQTL sites are over-represented in the top 5%; `alternative="greater"` (`:218`). |
| 3.9 | Probe → X column path | PASS | A cg id in `mqtl_sites_any.txt` is matched by `np.isin(run.site_probes, …)` (`04_evaluate.py:64`). `site_probes` = `site_probe_ids.npy` = `probe_ids[W_index[:,0]]` (`02_encode_blocks.py:192`), so W row r ↔ X column `W_index[r,0]` (`02:158`) ↔ probe `probe_ids.npy[col]`. That file is written by 01 (`01_prepare_egtex.py:155`) after re-deriving the datasaver's column filter and asserting the resulting (chrom, pos) equal `var.mmap` (`01:145-152`). |
| 3.10 | Expected overlap | PASS (estimate) | `mqtl_sites_any.txt` 286,152; in the model vocab 286,147; with a manifest location 286,147 (286,146 autosomal, 1 chrX). Lung 166,239 and Colon 156,416 in vocab; lung-only 67,249, colon-only 57,426, union 223,668. The exact count inside blocks needs stage 2 (sites with a DNA embedding in blocks of ≥ 500 sites); expect slightly below 286,146. |
| 3.11 | mQTL background | CONCERN | Non-mQTL = all other block sites. The eGTEx tables tested 754,054 CpGs (≈ the whole array), so this is close to "tested and not significant", but CpGs absent from the tables are counted as non-mQTL. |

## 4. Data contract

See `docs/data_contract.md`. Every file a consumer reads has a producer. **FIXED**: before this review, Phase B
and beta-space encodings had no `W.mmap`, so the mQTL test was skipped for them; both now write one.
CONCERN: 09 hard-codes the DNA-embedding width 1024 (`09_train_lewm.py:99`); correct for the model in use.

## 5. Math

**(a) MSE ⇒ conditional mean.** For any f, E‖z_B − f(z_A,c_B)‖² = E‖z_B − m‖² + E‖m − f‖² with
m = E[z_B | z_A, c_B]; the cross term vanishes because E[z_B − m | z_A, c_B] = 0. The minimiser is f = m:

  argmin_f E‖z_B − f(z_A,c_B)‖² = E[z_B | z_A, c_B].

The code minimises exactly this empirical risk (`predictor.py:162`) over fresh (s, A, B) pairs, so f
approximates m to the extent capacity, data (484 training samples) and regularisation allow.

**(b) Residual ⟂ any function of z_A.** With r = z_B − m, for any h, E[r·h(z_A)] = E[ h(z_A) · E[z_B − m | z_A, c_B] ] = 0:

  E[(z_B − E[z_B|z_A,c_B]) h(z_A)] = 0.

CONCERN (design, not a bug): our residual averages K = 8 sources, r = z_B − (1/K) Σ_k f(z_{A_k}, c_B)
(`predictor.py:186`). It is orthogonal to functions of each z_{A_k} only insofar as each f(z_{A_k}) already
equals E[z_B | all K sources]; in general the average is a variance-reduced, not exact, projection.

**(c) cis effects on B cannot enter E[z_B | z_A].** Let G_B be the genotype at B's cis-mQTLs, with
z_B = φ(state) + β G_B + ε, and A on another chromosome. Then G_B is independent of z_A apart from ancestry or
relatedness (no LD across chromosomes), so

  E[β G_B | z_A] = β E[G_B] (+ an ancestry term, see (e)),

a constant absorbed into the intercept; the donor-specific deviation β(G_B − E[G_B | ancestry]) stays in r.
The code enforces chrom(A) ≠ chrom(B) both in training (`predictor.py:105-111`) and for the residual sources
(`predictor.py:118`).

**(d) Non-genetic quantities that can survive and are shared across a donor's tissues.** The predictor removes
whatever is predictable from z_A. Age, sex, batch (plate, chip) and smoking are sample-wide and largely
predictable, but the part of their effect on B that z_A does not capture survives, and it is shared by a
donor's tissues:

  r ⊇ (γ_age,B − E[γ_age,B·age | z_A]) + … .

06 regresses these out, in this order:
- tissue, sex and age: M1 (`06_covariates.py:43-45`, model list just below them);
- plate: M2 (`:46`);
- sentrix_id (chip): M3 (`:47`);
- PEER factors within each tissue: M4 (`06_covariates.py:55-68`).

Age is the decade midpoint from the SOFT file (`evaluation.py:49-50`). CONCERN: smoking status is parsed
(`evaluation.py`, `load_metadata`) but not regressed; sex also enters through chrX, which is excluded from
matching (`evaluation.py:154-158` uses `auto_blocks`).

**(e) Ancestry is absorbed into the prediction.** Ancestry Π shifts allele frequencies genome-wide, so it
leaves a signature in z_A (A's own cis-mQTLs) and in z_B. Hence

  E[G_B | z_A] ≈ E[G_B | Π̂(z_A)],

and the ancestry-predictable part of B's genetic effect is predicted from z_A rather than left in r. What
remains is the within-ancestry deviation. The code does not regress genotype PCs (06 docstring, line 9),
which is consistent: PCs are ancestry, and removing them would be redundant with (e).

No code step contradicts (a)–(e); the K-averaging in (b) and smoking in (d) are listed as CONCERNs.

## 6. Phase B (09, lewm.py) against LeWM (arXiv 2603.19312) and LeJEPA (arXiv 2511.08544)

Sources read: the LeJEPA HTML (SIGReg definition, Definition 2, Epps–Pulley eq., reference listing) and the
LeWM HTML (eq. L_LeWM = L_pred + λ SIGReg(Z), "M=1024 projections and λ=0.1", projector description, "We do
not employ stop-gradient, exponential moving averages, or additional stabilization heuristics").

| # | Item | Paper | Before | After | Verdict |
|---|---|---|---|---|---|
| 6.1 | Directions | uniform on the sphere; normalised Gaussians seeded by step (LeJEPA listing) | same (`lewm.py:29-31`) | unchanged | PASS |
| 6.2 | Epps–Pulley form | T = N ∫ w(t) \|φ̂(t) − e^{−t²/2}\|² dt; listing: t ∈ [−5,5], 17 knots, trapz, × N | real form `(cos̄ − e^{−t²/2})² + sin̄²` (equal to \|φ̂ − φ\|² since φ is real), trapz, × N (`lewm.py:32-37`) | unchanged | PASS |
| 6.3 | Weight w(t) | listing e^{−t²/2}; LeJEPA text e^{−t²/σ²} (σ=1); LeWM text e^{−t²/(2σ²)} | e^{−t²/2} | unchanged | PASS (follows the code listing) |
| 6.4 | Knots | LeJEPA listing [−5,5] × 17; LeWM text "[0.2, 4]" | [−5,5] × 17 | unchanged | CONCERN (papers disagree; both report insensitivity) |
| 6.5 | Average over directions | SIGReg = (1/M) Σ_m T(m) | `.mean()` over directions (`lewm.py:37`) | unchanged | PASS |
| 6.6 | M default | LeWM M = 1024 (LeJEPA listing 256, text recommends 1024) | `n_directions: int = 256` (lewm.py), `--n-directions 256` (09) | 1024 (`lewm.py:27`, `09_train_lewm.py:56`) | FIXED |
| 6.7 | λ default | LeWM λ = 0.1 | without `--lam` and outside an array job: `lams[0]` = 0.01 | `lam = 0.1` (`09_train_lewm.py:64-69`); the array still sweeps 0.01/0.1/1 | FIXED |
| 6.8 | Loss form | LeWM: L_pred + λ·SIGReg (LeJEPA instead uses (1−λ)·pred + λ·SIGReg, λ = 0.05) | `pred_loss + lam * reg` | unchanged (`09_train_lewm.py:144`) | PASS (LeWM form) |
| 6.9 | SIGReg target | encoder embeddings, all of them, per time step | `sigreg(z_b)` only | `0.5 * (sigreg(z_a) + sigreg(z_b))` (`09_train_lewm.py:143`) | FIXED |
| 6.10 | Encoder projector | "1-layer MLP with Batch Normalization", because a final LayerNorm blocks SIGReg | pooled LayerNorm output, no projector | `Projector = Linear + BatchNorm1d` after pooling (`lewm.py:40-50,64,72`) | FIXED |
| 6.11 | Predictor projector | "followed by a projector network with the same implementation" | MLPPredictor output | `LeWMPredictor = MLPPredictor + Projector` (`lewm.py:75-86`, used at `09_train_lewm.py:120`) | FIXED |
| 6.12 | EMA / stop-grad | none | none | unchanged | PASS |
| 6.13 | BatchNorm with 1-example batches | — | would raise in training | batches < 2 skipped (`09_train_lewm.py:136-137`) | FIXED |
| 6.14 | w_i for the mQTL test | — | none (04/07 skipped mQTL for Phase B) | post-hoc RidgeCV decoder raw g → block betas on train samples, written as W.mmap (`beta_space.py` `ridge_decoder`, `09_train_lewm.py:179-180`) | FIXED |
| 6.15 | Projector width / activation | LeWM gives no hidden width | Linear(d,d) + BN, no activation | — | CONCERN (interpretation of "1-layer MLP") |

Diff summary (6.6–6.14):

```diff
- def sigreg(z, n_directions: int = 256, ...)
+ def sigreg(z, n_directions: int = 1024, ...)
- lam = args.lam if args.lam is not None else lams[int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))]
+ lam = args.lam if given; else lams[SLURM_ARRAY_TASK_ID] in an array job; else 0.1
- reg = sigreg(z_b.float(), args.n_directions, step)
+ reg = 0.5 * (sigreg(z_a.float(), M, step) + sigreg(z_b.float(), M, step))
- return (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)            # BlockEncoder
+ return self.proj((x * keep).sum(1) / keep.sum(1).clamp(min=1.0))  # Projector = Linear + BatchNorm1d
- f = MLPPredictor(args.d, Cn.size(1))
+ f = LeWMPredictor(args.d, Cn.size(1))                            # MLPPredictor + Projector
+ if len(idx) < 2: continue                                         # BatchNorm needs >= 2 examples
+ W_dec, decoder_r2 = ridge_decoder(Z, betas, offsets, train_s, test_s); write_site_vectors(run_dir, W_dec, ...)
```

## 7. Additions

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 7.1 | `--beta-space` control | FIXED (added) | `beta_space.build_beta_encodings`: per block, PCA of the betas fit on the stage-3 train samples, k = min(d, n_train − 1, smallest block) (= 128 for the full run). W.mmap holds the PCA loadings, so the mQTL test is unchanged. 03 `--beta-space` (`03_train_predictor.py:63-69`) writes `<enc>_beta`, `<out>_beta`; 04 `--beta-space` reads them. Run with `slurm/submit.sh slurm/03_train.sbatch --beta-space`, then `slurm/04_evaluate.sbatch --beta-space`. |
| 7.2 | pyBigWig, hic-straw | FIXED (declared) / CONCERN | Added to `pyproject.toml` `dependencies`. `uv.lock` and `.venv` are **not** updated, because this step allowed no execution: run `uv lock && uv sync --all-groups` in the container before 08. |
| 7.3 | CLAUDE.md tracked | FIXED | Removed from `.gitignore` together with AGENTS.md; both committed (AGENTS.md is a symlink). |
| 7.4 | Success ladder | FIXED | Filled in `docs/plan.md` with the six rungs and the pass rules exactly as given. |

## 8. Open questions

1. Run `uv lock && uv sync --all-groups` (in the container) for pyBigWig / hic-straw? Without it, a full 08 run ends SANITY_FAIL.
2. K-averaged residual (§5 b): keep, or define the residual from a single fixed source / a predictor that takes all K sources jointly?
3. Regress smoking status in 06 as well (it is parsed but unused)?
4. Z_full chromosome order: keep ascending (deterministic) or shuffle with a fixed seed like CpGPT's dataset?
5. SIGReg knots: LeJEPA listing [−5, 5] (current) vs LeWM text [0.2, 4]?
6. Projector: Linear + BatchNorm without activation (current) or Linear–BN–GELU–Linear?
7. mQTL background: restrict non-mQTL sites to CpGs present in the eGTEx tables?
8. The smoke test has still not run, so all of the above is static.

## Checks

```
$ ruff check --isolated --target-version py310 scripts/0*.py src/cpgpt_jepa/
All checks passed!
$ ruff check --isolated --select F,E9,B006,B023 --target-version py310 scripts/0*.py src/cpgpt_jepa/
All checks passed!
$ python3 -m py_compile scripts/*.py src/cpgpt_jepa/*.py
py_compile: 18 files OK
$ bash -n slurm/*.sh slurm/*.sbatch scripts/download_egtex.sh
bash -n: 17 files OK
```

Commits made in this step:

```
c85383f docs(jepa): static verification report and data contract
8e850db feat(jepa): success ladder in docs/plan.md
1430c61 feat(jepa): declare pyBigWig and hic-straw
df09371 feat(jepa): track CLAUDE.md and AGENTS.md
d7c7526 fix(jepa): Phase B against LeWM / LeJEPA
714fe9c feat(jepa): beta-space control (--beta-space in 03/04)
```

Not yet done: `uv lock && uv sync --all-groups` for pyBigWig / hic-straw (§7.2). The success-ladder text came from the user's message, since the plan.md placeholder had none. Commits are local and not pushed.
