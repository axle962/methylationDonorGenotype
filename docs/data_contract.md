# Data contract — files read and written by scripts 01–09

Symbols: N = samples (987 full run; 50 in stage 1 / 20 from stage 2 on in the smoke test), P = probes in the
model vocab written to the arrow file, S = sites retained by CpGPTDataSaver (X.mmap columns), B = blocks
(autosomal, ≥ 500 sites, plus chrX blocks with `--include-x`), S_B = sites inside blocks (= W rows), d = 128
(CpGPT small; 256 for Phase B; k for beta space), K = fixed sources per target (8). Stage 1 has not run, so S,
B and S_B are not known yet. "HF" = Hugging Face cache (`$HF_HOME/hub/...`, see CLAUDE.md).

## Inputs that exist before stage 1

| Path | Shape / dtype | Read by |
|---|---|---|
| `data/egtex/geo/GSE213478_betas.txt.gz` | ~754k probes × 987 samples, comma-separated text; header `"",GTEX-…-SM-…` | 01 (line 54 header, lines 104–108 stream) |
| `data/egtex/geo/GSE213478_family.soft.gz` | GEO SOFT text | `build_metadata.py` (called by 01); `evaluation.load_metadata` (age, smoker) in 04–09 |
| `data/egtex/mqtl/*.regular.perm.fdr.txt` | 9 TSVs, cols `cpg_id … slope … qval` | `build_metadata.py`; 07 (line 191, strongest mQTLs) |
| `data/egtex/metadata/<Tissue>.covariates.txt` | covariates × donors TSV (PC1–5, pcr, platform, sex, PEER1–20) | 06 (`covariates.peer_factors`), 09 (`covariates.genotype_pcs`) |
| HF `vocab/small.json` | `{"input": [1,187,190 probe ids]}` | 01 (line 95) |
| HF `config/small.yaml`, `weights/small.ckpt` | model config / Lightning checkpoint | 02 (lines 69–74) |
| HF deps `illumina_metadata.db` | sqlitedict `{species: {probe: "chrom:pos"}}` | 01 (CpGPTDataSaver via prober) |
| HF deps `ensembl_metadata.db` | sqlitedict (vocab, reverse_vocab, `{llm: {2001: {"chrom:pos": row}}}`) | 01, 02, 08 (locus index), 09 (path only) |
| HF deps `dna_embeddings/.../2001bp_dna_embeddings.mmap` | float32 (len(index), 1024), raw memmap | 02 (`blocks.open_dna_embeddings`), 09 (line 99, reshape(-1, 1024)) |

## Produced by the pipeline

| Path | Producer | Consumers | Shape / dtype | Shape determined by |
|---|---|---|---|---|
| `data/egtex/metadata.csv` | `build_metadata.py` (via 01) | 01, 02 (`--smoke`), 03, 08, 09, beta_space; `evaluation.load_metadata` | 987 rows × 7 str cols | GEO SOFT |
| `data/egtex/mqtl_sites_{Tissue,any}.txt` | `build_metadata.py` | 04, 07 | one cg id per line (any: 286,152) | q < 0.05 in the eGTEx tables |
| `data/egtex/arrow/egtex.arrow` | 01 (line 127) | CpGPTDataSaver (01) | N rows × (2 + P) cols: `sample_id` str, `species` str, P float32 | probes in vocab |
| `data/processed/<ds>/X.mmap` | CpGPTDataSaver (01) | 01 report, 02, 07, 09, beta_space | float32 (N, S), raw memmap | `X_shape.npy` |
| `data/processed/<ds>/var.mmap` | CpGPTDataSaver | 01 (check), 02 | int32 (S, 2): chrom index, GRCh38 pos | (S, len(var_names)) |
| `data/processed/<ds>/{var_names,obs_names,species,X_shape}.npy` | CpGPTDataSaver | 02 (`blocks.open_processed`) | `["chrom","position"]`; object (N,); str; int (2,) | — |
| `data/processed/<ds>/probe_ids.npy` | 01 (line 155) | 02 | str (S,) | re-derived from the datasaver's column filter; asserted equal to var.mmap (01 lines 145–152) |
| `data/processed/{dataset_metrics,genomic_locations}.db` | CpGPTDataSaver | CpGPTDataSaver (skip already-processed files) | sqlitedict | — |
| `outputs/encodings/Z.npy` | 02 (line 143) | 03, 04–09, beta_space (no), 09 probes | float32 (N, B, d), .npy | blocks × d_embedding |
| `outputs/encodings/C.npy` | 02 (line 190) | 03, 05, 08, 09, beta_space (copied) | float32 (B, d) | — |
| `outputs/encodings/W.mmap` | 02 (lines 145, 157) | 04, 07, 08 (`run.W_block`) via `evaluation.load_run` | float16 (S_B, d), raw memmap | `encode_meta.json["W_shape"]` |
| `outputs/encodings/W_index.npy` | 02 (line 191) | load_run, 07 (line 82), 09, beta_space | int64 (S_B, 2): X column, block id; rows contiguous by block | — |
| `outputs/encodings/site_probe_ids.npy` | 02 (line 192) | load_run (04, 07), beta_space/09 (copied) | str (S_B,) | probe_ids[W_index[:,0]] |
| `outputs/encodings/site_embedding_rows.npy` | 02 (line 193) | 08 (line 81), 09, beta_space (copied) | int64 (S_B,) | DNA-embedding row = annotation row |
| `outputs/encodings/blocks.npy` | 02 (line 123) | 03, 04–09, beta_space | object (B,) of dicts {block_id, chrom, start, end, n_sites, site_cols} | — |
| `outputs/encodings/obs_names.npy` | 02 (line 124) | 03, 04–09 | object (N,) | samples encoded |
| `outputs/encodings/Z_full.npy`, `Z_full_sites.npy` | 02 (lines 199–200) | 04 (`Z_full` baseline) | float32 (N, d); int64 (10,000) | `--full-sites` |
| `outputs/encodings/encode_meta.json` | 02 (line 211) | load_run, 05 (`window_bp`), 07/09/beta_space (`dataset_dir`, `sample_idx`) | JSON | — |
| `outputs/encodings_{5,20}mb/*` | 02b (02 `--window-mb`) | 05 (line 99) | as above, different B | `--window-mb` |
| `outputs/predictor/splits.json` | 03 | 04–08 via load_run | JSON: donors, sample/block indices, `fixed_sources` (B × K) | `make_splits`, `fixed_sources` |
| `outputs/predictor/standardise.npz` | 03 (line 102) | load_run (`mu`, `sd`) | float32 (d,) each | train samples × autosomal blocks |
| `outputs/predictor/<kind>/R.npy`, `P.npy` | 03 (lines 124–125) | 04–08 | float32 (N, B, d), standardised space | same as Z |
| `outputs/predictor/<kind>/model.pt` | 03 | 05 (line 69), 08 | state dict | — |
| `outputs/predictor/<kind>/metrics.json`, `metrics.json` | 03 | 08 (mlp_struct comparison) | JSON | — |
| `outputs/predictor/<kind>/evaluation.json` | 04 | — | JSON | — |
| `outputs/encodings_beta/*` | 03 `--beta-space` (`beta_space.build_beta_encodings`) | 03/04 `--beta-space` | Z float32 (N, B, k); W.mmap float16 (S_B, k); other files as stage 2 | k = min(d, n_train − 1, smallest block) |
| `outputs/predictor_beta/*` | 03 `--beta-space` | 04 `--beta-space` | as outputs/predictor | — |
| `outputs/experiments/ablations/*` | 05 | — | result.json, summary.csv, tissue_pairs*.csv | — |
| `outputs/experiments/covariates/*` | 06 | — | result.json, summary.csv | — |
| `outputs/experiments/mqtl_deep/*` | 07 | — | result.json, summary.csv, block_contribution_vs_mqtl_density.csv | — |
| `data/3d/annotations/<cell>.{mmap,json}` | `experiments/threed/build_annotations.py` (via 08) | 08, `make_shift_control.py` | float32 (len(index), 2) [eigenvector scaled, covered] | JSON `shape` |
| `data/3d/hic_cache/inter_*.npz` | 08 (`structure.interchrom_contacts`) | 08 | bx, by int64; counts float64 | records streamed |
| `outputs/experiments/structure/*` | 08 | 03 (`struct_profile.npz` via `--struct`) | pair_error.npy float64 (B, B); struct_profile.npz `profile` (B, 2); CSVs; result.json | — |
| `outputs/predictor/mlp_struct/*` | 03 `--struct` (called by 08) | 08, 04 if re-run | as other kinds | — |
| `outputs/lewm/lam<λ>/{Z,C,blocks,obs_names}.npy`, `encode_meta.json` | 09 (lines 165–171) | 03–07 run by 09 | Z float32 (N, B, 256) | `--d` |
| `outputs/lewm/lam<λ>/W.mmap` (+ W_index, site_probe_ids, site_embedding_rows) | 09 (lines 179–180, post-hoc ridge decoder) | 04, 07 on Phase B | float16 (S_B, 256) | encode_meta `W_shape` |
| `outputs/lewm/lam<λ>/lewm.pt`, `probes/*`, `predictor/*`, `experiments/*` | 09 | — | — | — |

## Mismatches found

| Item | Status |
|---|---|
| Consumer expects a file no producer writes | None found. Before this review, Phase B and beta-space encodings had no `W.mmap`, so 04/07 skipped the mQTL test; both now write one (`beta_space.write_site_vectors`). |
| `var_names.npy` as probe map | Not usable: CpGPTDataSaver writes `["chrom","position"]` (`cpgpt/data/components/cpgpt_datasaver.py:617`). Probe ids come from `probe_ids.npy` / `site_probe_ids.npy`. |
| W rows contiguous by block | Relied on by `beta_space.block_offsets` and 09; guaranteed by 02's write loop (lines 150–160 append block by block). |
| DNA-embedding width | 09 reshapes the memmap with a hard-coded 1024; correct for nucleotide-transformer-v2-500m (`dna_llm_embedder.py` size table), would break for another DNA LLM. |
| 02b encodings | Built without `--include-x`; 05 recomputes splits with chrX eval-only per encoding, so B differs but nothing is misaligned. |
| dtype | Z/R/P float32 throughout; W float16 everywhere (lossless for CUDA runs, see verification.md §1). No disagreement. |
| pyBigWig / hic-straw | Declared in pyproject.toml; uv.lock and .venv not updated yet (needs `uv lock && uv sync`). |
