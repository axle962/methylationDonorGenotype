# 07. Are the model's mistakes 3D-shaped?

**Date:** 24 September 2026 · **Status:** done, answer is yes at compartment scale only · **Code:** `experiments/threed/h3b_residual_structure.py`, `h3c_residual_3d_contrast.py`

## What we wanted to do

H3 showed the model has not learned folding. That alone does not justify
training: if the model's leftover mistakes have nothing to do with folding
either, then telling it about folding cannot help, and the honest outcome is a
negative result written up.

So: are the model's remaining errors organised by 3D structure?

## What we did

Took 100 blood samples, hid half the measurement spots in a 20-million-letter
window, asked the model to fill them in, and recorded how wrong it was at each
spot for each person. Then asked whether spots that are close in 3D tend to be
wrong **together**, after accounting for simple closeness along the string.

## What we did not do

We did not accept the first answer. The first run said "no structure at all",
including for neighbouring spots, which we know are biologically linked. That
was a broken measurement, not a finding, and reporting it would have been wrong.

The cause: pairwise correlations estimated from 100 samples are mostly noise, and
a rank correlation over a million pairs drowns the real signal in them. Switching
to group-mean comparisons within distance bands recovered it.

We also did not use matched data. The folding information is from a
lymphoblastoid cell line while the methylation is whole blood, and it is one
window on one chromosome.

## What the results were

**The model's errors are spatially structured**, once measured properly:

| spot pairs | true methylation | model's errors |
| --- | --- | --- |
| near (under 5 kb apart) | +0.102 | +0.040 |
| far (over 5 Mb apart) | +0.043 | +0.003 |
| near, high-variance spots | +0.331 | +0.279 |

Then, splitting pairs by 3D relationship within matched distance bands:

| effect | spots varying >= 0.03 | spots varying >= 0.05 |
| --- | --- | --- |
| **Contact strength** (fine detail) | +0.0002 [-0.0017, +0.0021] | +0.0021 [-0.0026, +0.0066] |
| **Same compartment** (coarse) | +0.0010 [-0.0014, +0.0032] | **+0.0096 [+0.0037, +0.0150]** |

**Correction, same day.** The first version of this table read +0.0034 and
+0.0123 for the compartment rows, and both looked solid. They were computed with
a broken definition of compartment membership: the annotation builder z-scored
the values, and z-scoring subtracts the mean, but A versus B is defined by the
**sign** of the value. Subtracting a nonzero mean moved the A/B boundary, so the
test was really splitting pairs at the median rather than at A/B. Rebuilt with
sign preserved, the effect survives only at the high-variance subset.

Square brackets are 95% confidence intervals. An interval that excludes zero is
the effect being real rather than chance.

**Fine-grained contact structure explains nothing.** Both intervals include zero.

**Compartment membership does, but only at the most variable spots.** The
interval excludes zero for spots varying at least 0.05 (238 spots), and includes
zero for the broader set. So the evidence is one subset, in one window, on one
chromosome.

Honest note on size: +0.0096 is a small effect resting on a thin slice of data.
This says some structure is probably there, not that it is large or established.

## What we recommend next and why

**Replicate before training. Then do H1, not H2.**

After the correction the compartment effect holds in only one of two variance
cuts, from 238 spots in a single window. That is too thin to spend a night of
training on, even a cheap night. The cheap fix is replication across several
windows and chromosomes, which costs under an hour and either firms the effect
up or kills it.

On the H1 versus H2 split the evidence is clearer, and the correction did not
change it. The signal lives at the coarse
compartment scale, not the fine contact scale. That is exactly what we predicted
for a specific reason: the model normally sees spots 310,000 letters apart,
which is too sparse to resolve contact-level detail, while compartments span
millions of letters and survive that sparsity.

So H1 is targeting the scale where the leftover structure actually is, and H2 is
targeting a scale where we now have direct evidence there is nothing to gain.

Chain of reasoning across three tests: the model does not know folding (06); its
errors show some compartment-scale organisation but only in the most variable
spots (07); therefore replicate that across windows first, and if it holds,
compartment information is worth an overnight run while contact maps are not.

Lesson worth keeping: the bug was caught only because someone asked what the
numbers actually were before committing to use them. A standardisation step that
looks routine destroyed the meaning of a sign.
