# 3D structure prior arm

Tests whether CpGPT benefits from 3D genome organisation, as per-locus
annotation (H1) and later as contact-biased attention (H2). Full hypothesis,
objections and staged plan: `../../3d_genome_hypothesis.md`.

## Why this design

`dna_encoder.input_adapter` is the single (128, 1024) linear layer where DNA
embeddings enter the model. Widening it and zero-padding the new columns leaves
the model numerically identical to the pretrained checkpoint, so the annotation
arm **starts exactly at baseline**. Whatever training puts into those channels
then measures how much the model actually wants the prior, with no
initialisation confound. Verified: `max abs diff 0.00e+00`.

## Data

ENCODE "genome compartments" bigWigs, **GRCh38-native**, so no hg19 liftOver and
none of the silent coordinate shifts that would produce a null result for the
wrong reason. 1,064 files across 25+ biosamples, 1-3 MB each.

Cell types relevant here: GM12878 (44 files), CD4/CD8 T cells, CD14 monocytes
(matching the blood cohorts), mammary epithelial cell (matching GSE307770),
IMR-90, K562, HepG2. Cell-type-specific tracks are what make the compartment
prior testable against the deconvolution angle.

Also available and not yet used: ENCODE Repli-seq (104 experiments) for
replication timing, and full `.hic` contact matrices (~95 GB, readable over HTTP
range requests via `hic-straw`, so no full download) for H2.

## Pipeline

```bash
# 1. per-locus annotations, row-aligned to CpGPT's DNA embedding memmap
python experiments/threed/build_annotations.py --cell-type "CD14-positive monocyte" --name CD14_mono

# 1b. controls: shifted copy of a real track, and gene density alone
python experiments/threed/make_shift_control.py --source CD14_mono --seed 0
python experiments/threed/make_shift_control.py --kind density

# 2. widen a checkpoint; verifies equivalence itself
python experiments/threed/widen_checkpoint.py --k 2 --out data/3d/small_k2.ckpt

# 3. confirm the arm starts identical to baseline on real data
python experiments/threed/verify_pipeline.py
```

Then pass `annotations_path=` to `CpGPTDataModule`; the dataset concatenates the
channels onto each DNA embedding.

## H3 result: attention does not track contacts (24 September 2026)

`h3_attention_vs_contacts.py`, chr1:30,000,000-50,000,000, GM12878 intact Hi-C
streamed from ENCODE, 2 samples, 2,995 complete sites at **648 bp median
spacing** against the ~310,000 bp that genome-wide sampling gives. All 8 layers
x 8 heads, 4.48M site pairs per test.

| | value |
| --- | --- |
| Best distance-controlled rho, any layer/head | **+0.031** (layer 0, head 6) |
| Range across 64 layer-head combinations | -0.046 to +0.031 |
| Combinations with abs(rho) > 0.05 | **0 of 64** |
| Linear distance alone vs contact | **+0.381** |

**The model's attention carries essentially no information about 3D contacts.**
The +0.381 for distance alone is the positive control: the measurement detects
real structure when it is there, so the flat attention result is not an artefact
of an insensitive test. Raw correlations are near zero too, so this is not a case
of distance masking a real signal.

**H3 is false.** CpGPT has not learned genome folding unsupervised. There is no
interpretation paper here, and no redundancy argument against adding the prior.

Caveats: one window on one chromosome; Hi-C is GM12878 while the methylation is
sorted immune cells, a cell-type mismatch that biases toward the null; only two
samples, though attention is nearly sample-independent here.

## Status

Working: annotation build (1,185,347 loci, 99.7% coverage), checkpoint surgery,
dataset plumbing, equivalence verification.

Not done: training runs, the contact-bias arm (H2), Repli-seq channels, and the
region-restricted high-density design. On measured CPU cost (~0.25 s/step at
1,000 sites, batch 4) a 100k-step ablation is roughly 10-14 hours, so a
region-restricted run is an overnight job rather than a cluster job.

## Annotation provenance (entry 09)

`pool5` is what entries 07-08 called "GM12878": an ENCODE aggregate pooling
GM12878, IMR-90, MCF 10A, HUVEC and mammary epithelium. The builder now takes
only single-cell-type files, orients each chromosome so the eigenvector rises
with locus density (ENCODE signs flip between files and chromosomes), and picks
the best-scoring eigenvector among candidates. Sets built: `GM12878`,
`CD14_mono`, `CD4_T`, `CD8_T`, `B_cell`, `NK`, `HL60`, `HL60_RA`, plus controls
`CD14_mono_shift0` and `density`.

## Sanity check worth keeping

A-compartment fraction at array loci is 71.8%, against **44.3% genome-wide**
(an earlier note here said "roughly 50%", which was not measured). Illumina
arrays are biased toward gene-rich open chromatin, so the enrichment is the
expected direction and suggests the track is aligned correctly. Orientation
checks out too: chrX is only 24.9% positive, and GM12878 is female with a
largely inactive X, which should read as B.

**Do not re-centre these values.** A vs B is the sign of the eigenvector, so
subtracting a mean moves the A/B boundary. The builder scales by standard
deviation only. An earlier version z-scored them and silently corrupted every
same-compartment comparison downstream.
