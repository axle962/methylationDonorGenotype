# 05. Setting up the 3D structure experiment

**Date:** 24 September 2026 · **Status:** built and verified · **Branch:** `3dprior/contact-matrix` · **Code:** `experiments/threed/`

## What we wanted to do

Test whether the model would do better if it knew how DNA is folded.

DNA is two metres of string packed into a tiny nucleus, so spots far apart along
the string can touch in space. Methylation is applied by enzymes floating in
that space, so touching spots plausibly behave alike. The model currently knows
only the 2,000 DNA letters around each spot and its position number. Neither
says anything about folding, and folding cannot be worked out from 2,000 local
letters, because it is a property of regions millions of letters wide.

## What we did

Found that ENCODE publishes **compartment tracks** free: a label for each region
saying whether it sits in the open busy part of the nucleus or the closed packed
part. They are 1 to 3 MB each, already on the same genome version the model
uses, and available separately for many cell types.

Looked up which compartment each of the model's 1.2 million spots falls in, and
widened the model so each spot carries two extra numbers.

The important trick: the new inputs start with **weights of exactly zero**, so
the widened model is bit-for-bit identical to the original. Verified, difference
0.00. That gives a clean experiment later, because the 3D version starts as a
perfect copy and anything it gains can only come from the new information.

## What we did not do

No training has happened, so there is no result yet about whether 3D helps. We
also did not build the contact-map version of the idea, which needs two bugs
fixed in the authors' unused code first.

## What the results were

Pipeline works: 1.2 million spots labelled, 99.7% coverage, widened model
verified identical at the start.

One reassuring sign: 71.8% of measured spots sit in the open compartment versus
about 50% of the genome. Array designers deliberately targeted active regions,
so that skew is expected and suggests the coordinates line up correctly.

An unexpected bonus: because these tracks avoid a genome-version conversion the
plan had budgeted one to two days for, a whole step of risk disappeared. That
conversion is the classic way to get a meaningless null result.

## What we recommend next and why

Before spending a night training, check whether the model has already figured
out folding on its own. If it has, adding the information is pointless. That
check costs nothing.
