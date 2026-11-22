#!/usr/bin/env bash
# Download the eGTEx EPIC methylation data (GEO GSE213478 betas + GTEx-bucket mQTLs/metadata).
#   usage: bash scripts/download_egtex.sh [DEST_DIR]   (default: ./data/egtex)   — resumable
set -euo pipefail
DEST="${1:-./data/egtex}"
mkdir -p "$DEST/geo" "$DEST/mqtl" "$DEST/metadata"
GEO_BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213478"
GCS_BASE="https://storage.googleapis.com/egtex/methylation/epic-arrays"
TISSUES="BreastMammaryTissue ColonTransverse KidneyCortex Lung MuscleSkeletal Ovary Prostate Testis WholeBlood"

fetch() { echo ">> $2"; curl -L --fail --retry 5 --retry-delay 10 -C - -o "$2" "$1"; }

# beta matrix, 6.0 GB: noob+BMIQ betas, 754,054 CpGs x 987 samples
fetch "$GEO_BASE/suppl/GSE213478_methylation_DNAm_noob_final_BMIQ_all_tissues_987.txt.gz" "$DEST/geo/GSE213478_betas.txt.gz"
# GEO per-sample metadata (donor id, tissue, sex)
fetch "$GEO_BASE/soft/GSE213478_family.soft.gz" "$DEST/geo/GSE213478_family.soft.gz"
# GTEx-portal metadata + covariates
fetch "$GCS_BASE/metadata/eGTExDNA_Pierce_Jan18.09-11-2021.tsv" "$DEST/metadata/eGTExDNA_Pierce_Jan18.09-11-2021.tsv"
fetch "$GCS_BASE/metadata/MethylationEPIC_GTEX_Batch01_11_SampleSheet.15-12-2021.1000_samples.csv" "$DEST/metadata/sample_sheet_1000.csv"
for T in $TISSUES; do fetch "$GCS_BASE/metadata/${T}.covariates.txt" "$DEST/metadata/${T}.covariates.txt"; done
# per-CpG mQTL FDR tables (~75 MB each). NOT the *.regular.txt.gz files (40-55 GB each).
for T in $TISSUES; do fetch "$GCS_BASE/mQTLs/${T}.regular.perm.fdr.txt" "$DEST/mqtl/${T}.regular.perm.fdr.txt"; done

echo; du -sh "$DEST"/*
echo; echo "beta file header/first row:"; zcat "$DEST/geo/GSE213478_betas.txt.gz" | head -2 | cut -c1-200
