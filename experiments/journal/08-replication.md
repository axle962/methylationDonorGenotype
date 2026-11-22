# 08. Does it replicate? (and a conclusion I got wrong)

**Date:** 25 September 2026 · **Status:** done, effect replicates · **Code:** `experiments/threed/h3d_replication.py`, `h3e_conditional.py`

## What we wanted to do

Entry 07 found that the model's leftover errors line up with compartment
structure, but from 238 spots in one window on one chromosome. One window is one
draw. Before spending a night training, check whether the same thing happens
elsewhere in the genome.

The bar was set before looking: the effect had to point the same way in at least
6 of 8 windows, and the average across windows had to exclude zero.

## What we did

Repeated the whole test on 20-million-letter windows on eight different
chromosomes, deliberately mixing gene-rich ones (17, 19) with gene-poor ones
(4, 8). Each window is an independent run.

Then, because "regions that touch" and "regions in the same compartment" overlap
heavily, we checked whether each effect survives once the other is held fixed.

## What we did not do

Chromosome 8 errored after its predictions and was dropped, so this is 7 windows
rather than 8 (recovered later, see the update below). Everything still uses one cell type's folding data against blood
methylation, and 100 people.

## What the results were

**The compartment effect replicates.**

| spots varying at least | average gap | 95% interval | direction positive in |
| --- | --- | --- | --- |
| 0.03 | +0.0055 | [+0.0025, +0.0085] | **7 of 7** |
| 0.05 | +0.0124 | [+0.0071, +0.0177] | **7 of 7** |

Worth noting: chromosome 1, the original window, turned out to be one of the
**weakest**. Chromosomes 2 and 19 are far stronger. The first result was not a
lucky draw, it was a mediocre one.

**A conclusion from entry 07 was wrong.** We reported the contact effect as null
and said H2 was not worth pursuing. That rested on a single window. Across seven,
contact averages +0.0040 [+0.0013, +0.0066] and points positive in 7 of 7.

**Both effects are real and partly independent.** Holding one fixed:

| test | spots >= 0.03 | spots >= 0.05 |
| --- | --- | --- |
| contact, within same compartment | +0.0041 real | +0.0061 real |
| contact, within different compartment | +0.0040 real | +0.0039 not distinguishable from zero |
| compartment, among touching pairs | +0.0037 real | +0.0117 real |
| compartment, among non-touching pairs | +0.0033 real | +0.0081 real |

Neither is simply a stand-in for the other. At the spots that vary most, the
compartment effect is about twice the contact effect.

## What we recommend next and why

**Run H1 as the training experiment. Keep H2 alive but second.**

H1 first for a practical reason, not a scientific one. In normal use the model
sees spots averaging 310,000 letters apart. Compartments span millions of
letters, so a compartment label is meaningful at that spacing. Contact detail is
not: we only measured it here because we packed the spots into one window.

So H1 works in the model's normal operating mode, while H2 only pays off if the
sampling changes too, which is a bigger redesign.

The training run needs a control arm with random annotations matched for spatial
smoothness. Without it, a gain could just mean "any smoothly varying extra
channel helps", which would prove nothing about 3D.

Also fix the chromosome 8 failure, so the replication is 8 of 8.

## Update: chromosome 8 recovered, 8 of 8

**Date:** 25 September 2026 · **Code:** `h3_attention_vs_contacts.py` (`fetch_contacts`), `h3e_conditional.py`

The chromosome 8 predictions had finished fine; what failed was the step that
streams its folding data from ENCODE over the network. Rerun, it worked, so the
cause was a network hiccup rather than anything about chromosome 8. Folding data
is now saved locally after the first download and retried on failure, so a blip
can no longer silently drop a window. The conditional test also had its own
copy of the window list, missing chromosome 8; it now uses the shared list.

With all eight windows:

| spots varying at least | effect | average gap | 95% interval | positive in |
| --- | --- | --- | --- | --- |
| 0.03 | compartment | +0.0052 | [+0.0025, +0.0079] | **8 of 8** |
| 0.03 | contact | +0.0039 | [+0.0015, +0.0062] | **8 of 8** |
| 0.05 | compartment | +0.0117 | [+0.0069, +0.0165] | **8 of 8** |
| 0.05 | contact | +0.0052 | [+0.0017, +0.0087] | 7 of 8 |

Holding one effect fixed, **all eight conditional tests now exclude zero**. The
one that did not before (contact, among different-compartment pairs, at the
most variable spots) is now +0.0037 [+0.0004, +0.0069], which is barely real.

Nothing changes in the recommendation. Chromosome 8 sits below the average for
compartment (+0.0069 at the high-variance cut), so the averages moved slightly
down, not up. The compartment effect is still about twice the contact effect at
the spots that vary most.

## Lesson

Two calls in two days were made from a single window and both needed revising.
The cost of replication here was about 90 minutes. The cost of not replicating
would have been a night of training pointed at the wrong hypothesis, or worse,
telling Michal that contact structure was a dead end when it is not.
