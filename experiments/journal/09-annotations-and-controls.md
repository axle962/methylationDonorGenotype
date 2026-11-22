# 09. Fixing the compartment labels and building the controls

**Date:** 25 September 2026 · **Status:** done · **Code:** `experiments/threed/build_annotations.py`, `make_shift_control.py`, `h3d_replication.py --annotation`

## What we wanted to do

Get the inputs for the H1 training run ready: compartment labels from the cell
types actually in blood, since the methylation is whole blood, and the control
arm entry 08 asked for, so a gain can be pinned on 3D structure rather than on
"any extra smooth input".

## What we did

Built compartment labels for every blood cell type ENCODE has: CD14 monocytes,
CD4 and CD8 T cells, B cells, NK cells, and HL-60 in two states. ENCODE has no
neutrophils, which are about 60% of whole blood, so HL-60 treated with retinoic
acid (pushed toward neutrophil-like cells) stands in for them. Also rebuilt a
genuine GM12878 set.

Scored each set against the same eight windows of blood residuals from entry 08.

Built two controls and scored them the same way:

- **Shifted:** the real CD14 track slid along each chromosome by a random 20-80%
  of its length. Same signal, same smoothness, wrong places.
- **Density:** how many array spots sit in each 100 kb stretch. No 3D
  information at all, just "is this a gene-rich neighbourhood".

## What we did not do

No training. The shift control has one seed; small chromosomes can line up with
the real track by chance (per-chromosome correlation up to +0.36), so the
training run should use two or three. HL-60 is a leukaemia line, not a
neutrophil, and its labels are weak on chromosomes 18 and 21.

## What the results were

**Three problems with the labels, found and fixed.**

1. **The "GM12878" labels were never GM12878.** They came from an ENCODE
   aggregate that pools GM12878 with four non-blood cell types (IMR-90, MCF 10A,
   HUVEC, mammary epithelium). Several other cell types' first-listed files are
   pools too (B + NK, monocyte + dendritic cell). The builder now accepts only
   files from exactly one cell type. The old set is kept as `pool5` so entries
   07-08 reproduce exactly.
2. **ENCODE eigenvectors are not consistently oriented.** Whether positive means
   A or B flips between files, and between chromosomes of the same file
   (chromosome 19: r = -0.83 between two tracks; chromosome 21 was flipped even
   in `pool5`). Entries 07-08 are unaffected, because "same compartment" within
   one window does not care about a whole-chromosome flip. A model reading the
   value does. Each chromosome is now oriented so the value rises with gene
   density, the standard convention.
3. **Some ENCODE eigenvectors are poor.** The first GM12878 experiment tried
   barely tracked compartments on half the chromosomes. The builder now scores
   every candidate and keeps the best.

**Every blood cell type's compartments explain the errors**, at spots varying
at least 0.05, averaged over 8 windows:

| labels | gap | 95% interval | positive in |
| --- | --- | --- | --- |
| B cell | +0.0190 | [+0.0100, +0.0280] | 8 of 8 |
| CD8 T | +0.0187 | [+0.0085, +0.0290] | 7 of 8 |
| CD14 monocyte | +0.0178 | [+0.0106, +0.0250] | 8 of 8 |
| NK | +0.0167 | [+0.0097, +0.0238] | 8 of 8 |
| GM12878 (genuine) | +0.0161 | [+0.0083, +0.0239] | 7 of 8 |
| HL-60 + retinoic acid | +0.0156 | [+0.0089, +0.0223] | 8 of 8 |
| CD4 T | +0.0127 | [+0.0072, +0.0182] | 7 of 8 |
| pool5 (entries 07-08) | +0.0117 | [+0.0069, +0.0165] | 8 of 8 |
| HL-60 untreated | +0.0107 | [+0.0079, +0.0136] | 8 of 8 |
| *control: shifted CD14* | *+0.0029* | *[-0.0009, +0.0067]* | *5 of 8* |
| *control: gene density* | *+0.0058* | *[+0.0028, +0.0088]* | *7 of 8* |

Proper single-cell-type labels beat the pooled ones the earlier entries used,
though the intervals overlap too much to crown a winner.

**The shifted control is flat.** Good: it is a fair control.

**The density control is not flat, and that changes how to read the earlier
numbers.** At the broader cut (spots varying at least 0.03), gene density alone
gives +0.0059, as much as any compartment track (+0.0045 to +0.0073). So at that
cut, "compartment" was mostly telling us "gene-rich neighbourhood". Only at the
most variable spots does compartment clearly beat density, by about 2-3x.

## What we recommend next and why

**Training arms for H1: baseline, CD14 compartments, shifted CD14 (2-3 seeds),
and gene density.** The density arm is not optional. Without it, a compartment
win could be a gene-density win, and this entry shows that is a real risk.

CD14 over B or CD8 despite near-identical scores: monocytes are myeloid, the
same lineage as the neutrophils that dominate blood, its eigenvector is among
the cleanest (no weak chromosomes, replicate agreement r = +0.97), and picking
the top scorer on the same windows we test on would be quietly overfitting.

Still open: the compute budget. That is at least 5 arms, about 10-14 hours
each at 100k steps on this laptop, before adding seeds.

## Lesson

"GM12878 compartments" was a file listed under GM12878, not a GM12878
measurement. Check what a public file actually contains, not what the search
that found it was called.
