# Latent Block Prediction on CpGPT: Does a Methylation Foundation Model Encode Donor Genotype?
Skanda Shreesha Prasad (sp9002@nyu.edu), Aditya Maddur Guruprakash (am16489@nyu.edu)

## 1. Background and Motivation
CpGPT is a transformer pretrained on more than 150,000 DNA methylation profiles. It summarises a profile as a single sample embedding, from which the methylation level at any CpG site can be reconstructed. Methylation, however, has two sources of variation: the overall state of the sample (tissue, age, sex), which affects sites genome-wide, and local genetic variants (mQTLs), which act on particular sites and are specific to the donor rather than the tissue. Whether the embedding retains this genetic component has not been studied.

A third source is the 3D organisation of the nucleus. Megabase-scale regions are partitioned into open (A) and closed (B) compartments, and regions in the same compartment share chromatin state even across chromosomes. In our earlier analysis of CpGPT, its reconstruction errors were organised by compartment in 8 of 8 genomic windows, while its attention showed no relationship with measured Hi-C contacts. Spatial structure therefore shapes the signal the embedding must explain, yet the model does not appear to represent it directly. We propose a self-supervised method to isolate the genetic component without genotype data, and to measure how much of the information shared between distant regions reflects 3D organisation.

## 2. Proposed Method
**Properties of CpGPT.** Our design follows from three features of the released code of CpGPT.
- Each CpG site is represented by a DNA-sequence embedding of its surrounding region combined with its methylation value; the sample embedding is the model's CLS token.
- CpGPT never processes a full array. Every forward pass encodes a random subset of at most 10,000 sites, so the model is already trained to produce an embedding from partial input.
- Methylation at a site is reconstructed as a dot product between a fixed, sequence-derived vector for that site and the sample embedding. Any difference between individuals must therefore be carried by the embedding itself.

**Predictor objective.** We adapt the joint-embedding predictive architecture (JEPA) to methylation data, keeping CpGPT frozen throughout.
1. The genome is divided into blocks of roughly 10 Mb, each containing a few thousand array sites.
2. For each sample, CpGPT encodes the sites of one block alone to give an embedding z_A, and the sites of a second block, on a different chromosome, to give z_B.
3. A small predictor network f is trained to estimate z_B from z_A, given a descriptor c_B of the target block (the average sequence embedding of its sites). The descriptor tells the predictor which region is being asked about, as the positional token does in the I-JEPA paper.
4. The training loss is the squared distance between prediction and target, L = ||f(z_A, c_B) − z_B||². Prediction takes place entirely in embedding space; as the target encoder is fixed, representational collapse is not a concern.

**Interpreting the residual.** Our interest lies less in the prediction than in what it leaves unexplained, r = z_B − f(z_A, c_B).
- The predictor can only recover information that is shared between distant regions of the genome. This corresponds to the overall state of the sample.
- The residual contains what is specific to the target block and cannot be inferred from elsewhere. We hypothesise that this is largely the local genetic component, and should therefore be characteristic of the donor and consistent across that donor's tissues.
- Because reconstruction is a dot product, the residual can be projected back onto individual sites, indicating which CpGs account for the unexplained signal.

**Accounting for 3D structure.** Information shared between two blocks on different chromosomes need not be sample-wide: blocks that occupy the same compartment, or that contact each other in the nucleus, may co-vary for structural reasons. For every block we record its A-compartment fraction from public ENCODE compartment tracks, and for every block pair its inter-chromosomal Hi-C contact frequency. This serves two purposes.
- As a test. If CpGPT encodes 3D organisation, prediction error should fall for block pairs with similar compartment composition or higher contact, beyond what gene density alone explains.
- As a control. A residual that tracks compartment structure would otherwise be mistaken for genetic signal. We therefore compare predictors with and without the block's compartment profile appended to c_B; whatever part of the residual this removes was structural rather than genetic.

## 3. Evaluation
- **Agreement with known mQTLs.** We will test whether the sites contributing most to the residual are enriched for catalogued mQTL sites (GoDMC, eGTEx) compared with other sites in the same block.
- **Donor matching across tissues.** About 150 eGTEx donors have both lung and colon samples. We will test whether a donor's lung residual identifies the same donor's colon sample, and compare this against the original embedding.
- **3D structure.** We relate prediction error to compartment similarity and Hi-C contact between block pairs, controlling for gene density, and repeat the analysis with compartment tracks circularly shifted along each chromosome, which preserves their smoothness but breaks their alignment to the genome. Sites that drive the residual are tested for compartment enrichment alongside mQTL enrichment.
- **Baselines.** A linear predictor, a block-average predictor, and a variant predicting methylation values directly. Training and test sets share no donors, and some blocks are held out.

## 4. Data and Feasibility
All data are public: the eGTEx methylation arrays (987 samples, nine tissues, about 424 donors) and whole-blood datasets from GEO. Compartment and Hi-C tracks are public ENCODE files on the same genome build (GRCh38), and the pipeline that maps them onto CpGPT's sites already exists from our earlier work. Each training example is a (sample, source block, target block) combination, giving millions of examples for a small predictor, and computing embeddings takes minutes on a single GPU. Should the residual carry no genetic signal, we would conclude that CpGPT embeddings do not encode genotype, a useful result in its own right; likewise, if prediction error is unrelated to 3D structure, the embedding does not capture nuclear organisation even at the scale of whole blocks.

## Scope decision (Oct 2026)
In the interest of time we use eGTEx only. The GoDMC replication, whole-blood GEO data, and the ENCODE compartment/Hi-C control are deferred; the compartment control is the first thing to add back if time allows.
