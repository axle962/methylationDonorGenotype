# Latent Block Prediction on CpGPT — does a methylation foundation model encode donor genotype?

Course project, Computational Genomics (CSCI-GA 3033-135), NYU. Skanda Shreesha Prasad, Aditya Maddur Guruprakash.

## Idea
CpGPT is a frozen transformer that maps a set of (CpG site, methylation value) pairs to one sample
embedding. We split the genome into fixed blocks (10 Mb; stage 2 falls back to 20 Mb if stage 1 retains
fewer than 600k probes — not yet run), encode each block separately to get
z_B per (sample, block), and train a small JEPA-style predictor f(z_A, c_B) -> z_B between blocks on
different chromosomes (c_B = mean sequence embedding of B's sites). What f recovers is sample-wide
state (tissue, age, sex); the residual r = z_B - f(z_A, c_B) is what is local to B. Hypothesis: the
residual is dominated by local genetic effects (mQTLs), hence donor-specific and consistent across a
donor's tissues. Tests: (1) sites driving the residual are enriched for catalogued mQTL CpGs;
(2) a donor's lung residual retrieves the same donor's colon sample. Scope (Nov 2026, PDF proposal in
genomics-2.pdf): eGTEx core (01-07) first, then the 3D compartment/Hi-C control (08); GoDMC and GEO blood later.

## CpGPT facts we rely on (verified in source)
- Sample embedding = CLS output of `net.encode_sample(meth, sequence_embeddings, chroms, positions, mask_na)`.
- Site sequence embedding = `net.encode_sequence(dna_embeddings)`; dna_embeddings are Nucleotide-Transformer
  vectors of 2001 bp of *reference* sequence (precomputed memmap in the dependencies bundle; identical for all donors).
- Reconstruction is a dot product: `query_methylation(z, seq_emb) = bmm(meth_decoder(seq_emb), z)`, so a
  residual projects onto sites via w_i = meth_decoder(seq_emb_i). The output is an M-value, not a beta.
- Model is trained on random <=10k-site subsets, so encoding one block is in-distribution.
- `CpGPTDataset` subsamples sites randomly — never use it for block encoding; read X.mmap/var.mmap directly.
- Processed layout (CpGPTDataSaver): `<processed>/<dataset>/X.mmap` (samples x sites float32, NaN=missing),
  `var.mmap` (sites x 2 int32: chrom idx, GRCh38 pos), `var_names.npy`, `obs_names.npy`, `species.npy`.
  `var_names.npy` is only `["chrom", "position"]`; the probe id of each X column is in `probe_ids.npy`,
  written next to it by `scripts/01_prepare_egtex.py`.

## Data
- eGTEx EPIC arrays only (GEO GSE213478; Oliva et al. 2023 Nat Genet): 987 samples, 424 donors, 9 tissues,
  754,054 CpGs (number retained in CpGPT vocab not yet known — stage 1 has not run), noob+BMIQ betas.
  153 donors have both Lung and Colon - Transverse.
- Beta columns are named `<donor_id>-<site>-<sample_id>` (e.g. `GTEX-11EM3-1326-SM-GW24C`); metadata
  `sample_id` is the `SM-...` suffix. The beta file is comma-separated.
- `data/egtex/metadata.csv` (sample_id, gsm, donor_id, tissue, sex, plate, sentrix_id);
  `data/egtex/mqtl_sites_<Tissue>.txt` and `mqtl_sites_any.txt` (q<0.05 cis-mQTL CpGs);
  `data/egtex/mqtl/*.regular.perm.fdr.txt` (source tables).
- CpGPT bundle, in the Hugging Face cache (`HF_HOME=/scratch/am16489/hf_cache` on NYU Cloud Bursting; set in `slurm/local.env`), default model `small`:
  - checkpoint `$HF_HOME/hub/models--lucascamillomd--cpgpt-models/snapshots/dd4d31e8579123ebfe77b6d5ad1049203b4d0a6d/weights/small.ckpt`
  - config `.../snapshots/dd4d31e8579123ebfe77b6d5ad1049203b4d0a6d/config/small.yaml`
  - vocab `.../snapshots/dd4d31e8579123ebfe77b6d5ad1049203b4d0a6d/vocab/small.json` (1,187,190 probe ids)
  - dependencies `$HF_HOME/hub/models--lucascamillomd--cpgpt-human-dependencies/snapshots/e66bbe467b3404c1a783b1361e565c5fe6c7235f/`
    (`dna_embeddings/homo_sapiens/nucleotide-transformer-v2-500m-multi-species/2001bp_dna_embeddings.mmap`,
    `ensembl_metadata.db`, `illumina_metadata.db`)
  - d_embedding = 128 (`small`). `large.ckpt` / `large.yaml` / `large.json` sit beside them: d_embedding = 512,
    trained with `binarize_input: true`.
  - Resolve these in code with `cpgpt.downloads.resolve_cached_model("small")` and `resolve_cached_dependencies("human")`.
- `data/`, `outputs/`, `dependencies/`, `slurm/logs/` are never committed.

## Layout
- `cpgpt/`, `configs/`, `tutorials/`: upstream CpGPT (read-only for us).
- `src/cpgpt_jepa/`: our package (block encoding, predictors, evaluation, covariates, structure, Phase B LeWM encoder).
- `scripts/01..09_*.py`: pipeline and experiment entry points; `scripts/` is gitignored upstream, so new files are force-added.
- `slurm/`: job scripts (`common.sh`, `submit.sh`, `*.sbatch`, `run_phase_a.sh`, `run_phase_b.sh`, `smoke.sbatch`).
- `docs/proposal.md` (proposal), `docs/plan.md` (Phase A / Phase B experiment plan).
- `experiments/h1/`: earlier reproduction work (H1 annotation-arm training pipeline).
- `experiments/threed/`: 3D-structure tooling. `build_annotations.py` builds per-locus ENCODE compartment
  annotations (memmap row-aligned to the DNA-embedding index, channels = oriented eigenvector, covered) plus
  log locus density as the gene-density proxy; `make_shift_control.py` builds the circular-shift null of a track.
  Reuse both (scripts/08_structure.py aggregates them per block); the Hi-C stream helper is in `h3_attention_vs_contacts.py`.

## Pipeline (run via SLURM: slurm/run_phase_a.sh for 01-08, slurm/run_phase_b.sh for 09, slurm/smoke.sbatch for the CPU smoke test; plan in docs/plan.md)
1. `scripts/01_prepare_egtex.py`  betas -> arrow -> CpGPTDataSaver -> data/processed/
2. `scripts/02_encode_blocks.py`  frozen CpGPT per (sample, block) -> outputs/encodings/{Z,C,Z_full}.npy, W.mmap, blocks.npy
3. `scripts/03_train_predictor.py` mlp / linear / block_mean -> outputs/predictor/<kind>/{metrics.json,R.npy,P.npy}, outputs/predictor/splits.json
   `--beta-space` (03 and 04): the same predictors/tests on per-block PCA of the betas -> outputs/encodings_beta, outputs/predictor_beta
4. `scripts/04_evaluate.py`       donor matching + mQTL enrichment -> outputs/predictor/<kind>/evaluation.json
5-8. `scripts/05_ablations.py`, `06_covariates.py`, `07_mqtl_deep.py`, `08_structure.py` -> outputs/experiments/<name>/
9. `scripts/09_train_lewm.py`     Phase B LeWM encoder -> outputs/lewm/lam<lambda>/ (+ 03-07 on it)

## Rules for agents
- One agreed step at a time. Do not add stages, datasets, or models beyond the current instruction; stop and report after each step.
- This repo is a fork of CpGPT. Never modify upstream modules under `cpgpt/`, `configs/`, `tutorials/`; ours live in `src/cpgpt_jepa/`, `scripts/`, `slurm/`.
- Splits are donor-disjoint; all lung+colon donors are test donors. Standardisation uses train-donor stats only.
- Jobs go through `slurm/*.sbatch` (conventions copied from the predplan repo); every job log ends with SANITY_OK / SANITY_FAIL.
- Never submit a Slurm job unless the user says to submit, in their own words, at that moment.
  Exception: the autonomous session started by `slurm/agent_session.sbatch` has the user's written
  authorisation (2026-11-22, `docs/agent/BRIEF.md`) to submit the jobs its brief needs, through
  `slurm/submit.sh` only, within the GPU-hour budget that `slurm/gpu_budget.sh` enforces.
- Python 3.10+, type hints, ruff; no new heavy dependencies without asking.
