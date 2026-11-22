# methylationDonorGenotype

Latent block prediction on frozen CpGPT: does a methylation foundation model encode donor genotype?
Proposal: `genomics-2.pdf` (Prasad and Maddur Guruprakash). Agent context: `CLAUDE.md`, `genotype_jepa_handoff.md`.

The pipeline (`src/cpgpt_jepa/`, `scripts/01..09`, `slurm/`, `docs/`, `experiments/threed/`) was written by
Skanda Prasad on `axle962/CpGPT_AMG_Lab@jepa_predictor` (73f9981, 2026-10-15) and imported here, then ported
from Torch/Singularity to NYU HPC Cloud Bursting.

## Running on Cloud Bursting

```bash
cp slurm/local.env.example slurm/local.env      # account, venv, HF cache
bash scripts/download_egtex.sh data/egtex       # eGTEx betas, covariates, mQTL tables (~6.4 GB)
slurm/submit.sh slurm/smoke.sbatch              # CPU smoke test of 01-09 (n2c48m24)
DRY_RUN=1 slurm/run_phase_a.sh                  # print the Phase A chain without submitting
```

CPU stages run on `n2d32p` (32 CPU, 128 GB); GPU stages on `g2-standard-12` (1x L4). CpGPT comes from the
`CpGPT_AMG_Lab` venv (`VENV` in `local.env`). Stage 08 also needs `pyBigWig` and `hic-straw` in that venv.
