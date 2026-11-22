"""Stage 4: donor matching and mQTL enrichment on the test donors, for each predictor kind.

  python scripts/04_evaluate.py [--encodings outputs/encodings] [--pred outputs/predictor] [--egtex-dir data/egtex]

Writes <pred>/<kind>/evaluation.json and prints one table comparing kinds x representations.
Donor matching flattens the autosomal blocks. mQTL enrichment needs the per-site decoder vectors
(W.mmap: CpGPT's meth_decoder in stage 2, PCA loadings for --beta-space, a post-hoc decoder in Phase B);
encodings without them skip it and say so.
--beta-space: evaluate the beta-space control (<encodings>_beta, <pred>_beta) written by 03 --beta-space.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cpgpt_jepa import finish  # noqa: E402
from cpgpt_jepa.evaluation import (  # noqa: E402
    COLON,
    LUNG,
    donor_matching,
    enrichment,
    fmt,
    lung_colon_sets,
    load_run,
    paired_bootstrap,
    percentile_within_block,
    read_set,
    site_scores,
    tissue_centred,
)

p = argparse.ArgumentParser()
p.add_argument("--encodings", "--enc", dest="enc", default="outputs/encodings")
p.add_argument("--pred", default="outputs/predictor")
p.add_argument("--egtex-dir", default="data/egtex")
p.add_argument("--beta-space", action="store_true", help="evaluate <encodings>_beta / <pred>_beta")
p.add_argument("--smoke", action="store_true")
p.add_argument("--n-perm", type=int, default=10_000, help="permutations for the test against chance (smoke: 200)")
p.add_argument("--n-boot", type=int, default=10_000, help="paired bootstrap resamples over test donors (smoke: 200)")
args = p.parse_args()
if args.smoke:
    args.n_perm, args.n_boot = min(args.n_perm, 200), min(args.n_boot, 200)
t0 = time.time()
egtex = Path(args.egtex_dir)
if args.beta_space:
    args.enc, args.pred = f"{args.enc}_beta", f"{args.pred}_beta"

run = load_run(args.enc, args.pred, egtex)
kinds = run.kinds()
if not kinds:
    finish(False, f"no residuals (R.npy) under {run.pred}")
n_samples, n_blocks, d = run.Z.shape

# --- A. donor matching -------------------------------------------------------------------------
NP = args.n_perm
shared_reps = {"Z (standardised)": donor_matching(run, run.Zs, n_perm=NP),
               "Z - tissue mean": donor_matching(run, tissue_centred(run, run.Zs), n_perm=NP)}
if run.Z_full is not None:
    shared_reps["Z_full"] = donor_matching(run, run.Z_full, n_perm=NP)
DIRS = ("lung_to_colon", "colon_to_lung")


def compare(a: dict, b: dict) -> dict:
    """Paired bootstrap a - b in top-1, per direction (docs/PREREG.md section 2)."""
    return {d: paired_bootstrap(a[d], b[d], n_boot=args.n_boot) for d in DIRS}


def heldout_mse_bootstrap(kind_a: str, kind_b: str) -> dict:
    """Rung 1: per-sample MSE of the residual on held-out blocks, test donors, a - b; CI from resampling
    test donors (a donor's samples move together). Negative = kind_a predicts better."""
    held = np.array(run.splits["heldout_blocks"])
    if len(held) == 0:
        return {"skipped": "no held-out blocks"}
    mse = {k: (run.residual(k)[run.test_s][:, held] ** 2).mean((1, 2)) for k in (kind_a, kind_b)}
    donors = run.donor[run.test_s]
    uniq = np.unique(donors)
    by_donor = np.array([(mse[kind_a][donors == u] - mse[kind_b][donors == u]).sum() for u in uniq])
    n_by_donor = np.array([(donors == u).sum() for u in uniq])
    idx = np.random.default_rng(0).integers(0, len(uniq), size=(args.n_boot, len(uniq)))
    boot = by_donor[idx].sum(1) / n_by_donor[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    a, b = float(mse[kind_a].mean()), float(mse[kind_b].mean())
    return {"kinds": [kind_a, kind_b], "mse_a": a, "mse_b": b, "rel_reduction": 1 - a / b,
            "diff": a - b, "ci95": [float(lo), float(hi)], "n_donors": int(len(uniq)), "n_boot": args.n_boot}

# --- B. mQTL enrichment ------------------------------------------------------------------------
mqtl_lists = {name: read_set(egtex / f"mqtl_sites_{name}.txt") for name in ("any", "Lung", "ColonTransverse")}
sample_sets = {"all": run.test_s,
               "Lung": np.flatnonzero(run.is_test & (run.tissue == LUNG)),
               "ColonTransverse": np.flatnonzero(run.is_test & (run.tissue == COLON))}
has_w = run.W is not None
membership = {}
if has_w:
    membership = {name: np.isin(run.site_probes, sorted(ids)) for name, ids in mqtl_lists.items()}

results = {}
for kind in kinds:
    R = run.residual(kind)
    dm_r = donor_matching(run, R, n_perm=NP)
    res = {"kind": kind, "donor_matching": {"residual R": dm_r, **shared_reps}, "mqtl": {},
           "inference": {"R vs Z - tissue mean": compare(dm_r, shared_reps["Z - tissue mean"])}}
    if not args.beta_space:  # the decisive control, once 03/04 --beta-space have run
        beta_eval = Path(f"{run.pred}_beta") / kind / "evaluation.json"
        if beta_eval.exists():
            beta_r = json.loads(beta_eval.read_text())["donor_matching"]["residual R"]
            res["inference"]["R vs beta-space R"] = compare(dm_r, beta_r)
        else:
            res["inference"]["R vs beta-space R"] = f"skipped: {beta_eval} not found (run 03/04 --beta-space first)"
    if "block_mean" in kinds and kind != "block_mean":
        res["inference"]["rung1 heldout MSE vs block_mean"] = heldout_mse_bootstrap(kind, "block_mean")
    if has_w:
        for set_name, samples in sample_sets.items():
            if len(samples) < 2:
                continue
            rows, var_g, abs_g = site_scores(run, R, samples)
            pct_var = percentile_within_block(var_g, run.W_block[rows])
            pct_abs = percentile_within_block(abs_g, run.W_block[rows])
            for list_name, is_mqtl in membership.items():
                if set_name not in ("all", list_name):
                    continue
                res["mqtl"][f"{list_name} list / {set_name} test samples"] = {
                    "n_samples": len(samples), "variance": enrichment(pct_var, is_mqtl[rows]),
                    "mean_abs": enrichment(pct_abs, is_mqtl[rows])}
    else:
        res["mqtl"] = "skipped: these encodings have no per-site decoder vectors (W.mmap)"
    (run.pred / kind / "evaluation.json").write_text(json.dumps(res, indent=2))
    results[kind] = res
    print(f"[{kind}] evaluated ({time.time() - t0:.0f}s)", flush=True)

# --- report ------------------------------------------------------------------------------------
s = lung_colon_sets(run)
print("\n=== STAGE 4 REPORT (test donors only) ===")
print(f"A. Donor matching by cosine similarity, representation flattened over {len(run.auto_blocks)} autosomal blocks.")
print(f"   lung->colon: {len(s['lung_q'])} queries, {len(s['colon'])} colon gallery;  "
      f"colon->lung: {len(s['colon_q'])} queries, {len(s['lung'])} lung gallery;  "
      f"chance {fmt(1 / max(len(s['colon']), 1), '.4f')} / {fmt(1 / max(len(s['lung']), 1), '.4f')}")
print(f"{'kind':<12s}{'representation':<18s}|{'top1':>7s}{'top5':>7s}{'MRR':>7s} |{'top1':>7s}{'top5':>7s}{'MRR':>7s}")
print(f"{'':<30s}|{'lung -> colon':^21s} |{'colon -> lung':^21s}")


def dm_row(kind: str, name: str, m: dict) -> str:
    a, b = m["lung_to_colon"], m["colon_to_lung"]
    return (f"{kind:<12s}{name:<18s}|{fmt(a['top1']):>7s}{fmt(a['top5']):>7s}{fmt(a['mrr']):>7s} |"
            f"{fmt(b['top1']):>7s}{fmt(b['top5']):>7s}{fmt(b['mrr']):>7s}")


for kind in kinds:
    print(dm_row(kind, "residual R", results[kind]["donor_matching"]["residual R"]))
for name, m in shared_reps.items():
    print(dm_row("(any)", name, m))

print(f"\nInference: permutation p vs chance ({args.n_perm} perms), and paired-bootstrap top-1 difference "
      f"with 95% CI ({args.n_boot} resamples over test donors).")
print(f"{'kind':<14s}{'test':<26s}|{'lung->colon':^32s}|{'colon->lung':^32s}")
for kind in kinds:
    dm = results[kind]["donor_matching"]["residual R"]
    print(f"{kind:<14s}{'R: perm p vs chance':<26s}|{fmt(dm['lung_to_colon'].get('perm_p_top1'), '.4f'):^32s}"
          f"|{fmt(dm['colon_to_lung'].get('perm_p_top1'), '.4f'):^32s}")
    for name, cmp_ in results[kind]["inference"].items():
        if name.startswith("rung1"):
            if "diff" in cmp_:
                print(f"{kind:<14s}{'heldout MSE vs block_mean':<26s}| rel. reduction {cmp_['rel_reduction']:+.3f}, "
                      f"diff {cmp_['diff']:+.4f} [{cmp_['ci95'][0]:+.4f}, {cmp_['ci95'][1]:+.4f}]")
            continue
        if isinstance(cmp_, str):
            print(f"{kind:<14s}{name:<26s}| {cmp_}")
            continue
        cells = [f"{c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]" for c in (cmp_[d] for d in DIRS)]
        print(f"{kind:<14s}{name:<26s}|{cells[0]:^32s}|{cells[1]:^32s}")

if has_w:
    print("\nB. mQTL enrichment of the residual's per-site score (variance of g over samples), percentile within block.")
    print("   pct = mean percentile (mQTL / other sites); MW p = one-sided Mann-Whitney; OR, Fisher p = mQTL in the top 5%.")
    print(f"{'kind':<12s}{'mQTL list / test samples':<34s}{'n':>5s}{'pct mQTL':>10s}{'pct other':>10s}"
          f"{'MW p':>10s}{'OR top5%':>10s}{'Fisher p':>10s}")
    for kind in kinds:
        for name, m in results[kind]["mqtl"].items():
            v = m["variance"]
            print(f"{kind:<12s}{name:<34s}{m['n_samples']:>5d}{fmt(v['mean_pct_set'], '.4f'):>10s}"
                  f"{fmt(v['mean_pct_other'], '.4f'):>10s}{fmt(v['mannwhitney_p'], '.2e'):>10s}"
                  f"{fmt(v['top5_odds_ratio']):>10s}{fmt(v['top5_fisher_p'], '.2e'):>10s}")
    print("   (secondary score, mean |g|, is in each evaluation.json)")
    print(f"   sites in blocks: {len(run.site_probes)}; in mQTL lists: "
          + ", ".join(f"{k} {int(v.sum())}" for k, v in membership.items()))
else:
    print("\nB. mQTL enrichment skipped: no per-site decoder vectors (W.mmap) in these encodings.")
print(f"elapsed: {time.time() - t0:.0f}s")
finish(True)
