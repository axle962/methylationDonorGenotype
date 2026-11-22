# 06. Has the model already learned folding? (H3)

**Date:** 24 September 2026 · **Status:** done, answer is no · **Code:** `experiments/threed/h3_attention_vs_contacts.py`

## What we wanted to do

Check whether the model has already worked out DNA folding by itself, without
being told. Large language models trained on protein sequences famously learn
3D contacts unsupervised, so the same could be true here.

This matters because it decides everything else. If the model already knows,
telling it again is pointless. Best of all, the check needs no training.

## What we did

Compared the model's internal "attention" (which spots it looks at when
processing each spot) against real measured folding data.

Two design choices made this meaningful:

**Packed the spots together.** Normally the model sees spots averaged 310,000
letters apart, far too spread out to see folding detail. We restricted it to a
single 20-million-letter region so its spots sat 648 letters apart instead.

**Controlled for simple closeness.** Spots near each other along the string both
touch more often and get more attention, so a raw correlation would be
meaningless. We compared against a distance-matched background and measured
within distance bands.

Streamed the folding data directly from ENCODE without downloading the 95 GB
file. Tested all 8 layers and all 8 attention heads, 4.5 million spot pairs each.

## What we did not do

One region on one chromosome. The folding data is from a lymphoblastoid cell
line while the methylation is from immune cells, a mismatch that makes finding
nothing more likely. Only two samples.

Also, to run this at all we had to fix a bug in the authors' code: asking for
per-head attention crashed. We needed per-head rather than averaged, because a
single folding-tracking head would vanish in an average of eight.

## What the results were

**No. The model's attention carries essentially no folding information.**

| | value |
| --- | --- |
| Best result across all 64 layer-head combinations | **+0.031** |
| Range | -0.046 to +0.031 |
| Combinations above a trivial 0.05 threshold | **0 of 64** |
| Simple distance alone, same data | **+0.381** |

That last row is what makes this believable. The identical measurement on the
identical pairs finds a strong +0.381 for plain distance. So the test can detect
real structure; the model's attention simply has none.

## What we recommend next and why

A flat result here is not yet a green light. It says the model has not learned
folding, but not whether folding could help. If the model's remaining errors
have nothing to do with folding either, then adding it cannot help much and the
honest outcome is a written-up negative result.

So the next test asks exactly that: are the model's leftover mistakes organised
by folding? That one number decides between "worth training" and "worth writing
up as negative".
