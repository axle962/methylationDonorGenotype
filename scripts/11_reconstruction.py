"""Stage 11: can each encoder's block embedding reproduce genotype-driven methylation?

For every run (frozen CpGPT, other encoders, LeWM, the beta-space PCA reference) the per-site reconstruction is
y_hat[s, i] = z[s, B(i)] . w_i, with w_i the run's per-site vector (W.mmap: CpGPT's meth_decoder(seq_emb_i),
LeWM's post-hoc ridge decoder, the PCA loadings for beta space). Every statistic below is a Spearman correlation
or a cosine retrieval on tissue-centred values, so per-site monotone maps (M-value vs beta, a missing intercept)
do not matter. Sites: the strongest autosomal cis-mQTL CpGs (stage 07 order: q-value, then |slope|) and one
variance-matched control each (tested in the eGTEx tables, not significant in any, same block, nearest
train-sample beta variance; docs/PREREG.md amendment A1, rung B4c). Test donors only.

1. Concordance: per site, Spearman over lung+colon test donors between the donor's lung and colon y_hat;
   median at mQTL vs control sites, one-sided Mann-Whitney. Raw betas give the input reference (B4c: 0.73 vs 0.03).
2. Fidelity: per site and tissue, Spearman over that tissue's test samples between y_hat and the observed beta;
   how much of the between-donor variation at the site the embedding keeps.
3. Donor matching (the genetic share of the beta-space signal): lung<->colon retrieval on tissue-centred raw
   betas and on tissue-centred y_hat, restricted to the N strongest mQTL CpGs vs their N controls, with a
   paired bootstrap of the top-1 difference.

  python scripts/11_reconstruction.py [--runs label=encodings_dir:predictor_dir ...] [--out ...] [--smoke]
The predictor dir is read only for its splits.json / standardise.npz (load_run); all runs must share samples,
sites and test split.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cpgpt_jepa import finish  # noqa: E402
from cpgpt_jepa.blocks import open_processed  # noqa: E402
from cpgpt_jepa.evaluation import (  # noqa: E402
    COLON,
    LUNG,
    cross_tissue_concordance,
    fmt,
    lung_colon_sets,
    load_run,
    paired_bootstrap,
    read_set,
    retrieval,
    tested_cpgs,
    variance_matched_controls,
    write_outputs,
)

DEFAULT_RUNS = [
    "cpgpt_small=outputs/encodings:outputs/predictor",
    "perceiver_small=outputs/perceiver_small/encodings:outputs/perceiver_small/predictor",
    "perceiver_large=outputs/perceiver_large/encodings:outputs/perceiver_large/predictor",
    "lewm_lam0.1_within=outputs/lewm/lam0.1_within:outputs/lewm/lam0.1_within/predictor",
    "lewm_lam1_within=outputs/lewm/lam1_within:outputs/lewm/lam1_within/predictor",
    "beta_pca=outputs/encodings_beta:outputs/predictor_beta",
]

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="label=encodings_dir:predictor_dir")
p.add_argument("--egtex-dir", default="data/egtex")
p.add_argument("--out", default="outputs/experiments/reconstruction")
p.add_argument("--n-strongest", type=int, default=200, help="sites for concordance and fidelity (as stage 07)")
p.add_argument("--n-match", default="200,2000", help="site counts for the donor-matching comparison")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--smoke", action="store_true", help="20 sites; for the smoke encodings")
args = p.parse_args()
t0 = time.time()
egtex = Path(args.egtex_dir)
log = lambda m: print(m, flush=True)  # noqa: E731
n_conc = 20 if args.smoke else args.n_strongest
n_match = [20] if args.smoke else sorted({int(x) for x in args.n_match.split(",")})

# --- runs: same samples, sites and test split ---------------------------------------------------
runs = {}
for spec in args.runs:
    label, paths = spec.split("=", 1)
    enc, pred = paths.split(":", 1)
    if not (Path(enc) / "W.mmap").exists():
        log(f"skip {label}: no {enc}/W.mmap")
        continue
    runs[label] = load_run(enc, pred, egtex)
if not runs:
    finish(False, "no run with per-site vectors (W.mmap)")
ref = next(iter(runs.values()))
for label, r in runs.items():
    same = (np.array_equal(r.obs_names, ref.obs_names) and np.array_equal(r.site_probes, ref.site_probes)
            and np.array_equal(r.W_block, ref.W_block) and np.array_equal(r.test_s, ref.test_s))
    if not same:
        finish(False, f"{label} differs from {next(iter(runs))} in samples, sites, blocks or test split")
log(f"runs: {', '.join(runs)} ({time.time() - t0:.0f}s)")

# --- sites: strongest mQTL CpGs and variance-matched controls ------------------------------------
X_mm, _, _, _ = open_processed(Path(ref.enc_meta["dataset_dir"]))
sample_rows = np.array(ref.enc_meta.get("sample_idx", np.arange(X_mm.shape[0])))
W_index = np.load(ref.enc / "W_index.npy")
auto_rows = np.flatnonzero(np.isin(ref.W_block, ref.auto_blocks))
Xtr = np.asarray(X_mm[sample_rows[ref.train_s]])[:, W_index[auto_rows, 0]]
var_of_row = np.full(len(ref.W_block), np.nan)  # train-sample variance of each site's betas (as stage 07)
var_of_row[auto_rows] = np.nanvar(Xtr, axis=0)
del Xtr

tables = [pd.read_csv(f, sep="\t", usecols=["cpg_id", "slope", "qval"]) for f in sorted((egtex / "mqtl").glob("*.regular.perm.fdr.txt"))]
strong = pd.concat(tables)
strong["abs_slope"] = strong.slope.abs()
strong = strong.sort_values(["qval", "abs_slope"], ascending=[True, False]).drop_duplicates("cpg_id")
row_of = {pr: r for r, pr in zip(auto_rows, ref.site_probes[auto_rows])}
strong_rows = np.array([row_of[c] for c in strong.cpg_id if c in row_of], dtype=int)
any_set = read_set(egtex / "mqtl_sites_any.txt")
tested = np.isin(ref.site_probes, sorted(tested_cpgs(egtex)))
cand = auto_rows[tested[auto_rows] & ~np.isin(ref.site_probes[auto_rows], sorted(any_set))]
site_sets = {}
for n in sorted({n_conc, *n_match}):
    mq = strong_rows[:n]
    ctrl = variance_matched_controls(mq, cand, ref.W_block, var_of_row, np.random.default_rng(args.seed))
    site_sets[n] = (mq, ctrl)
    log(f"N = {n}: {len(mq)} mQTL sites, {len(ctrl)} variance-matched controls")
all_rows = np.unique(np.concatenate([np.concatenate(v) for v in site_sets.values()]))
pos = {r: j for j, r in enumerate(all_rows)}

# observed betas at those sites, every sample; NaN -> train-sample mean of the site
B = np.stack([np.asarray(X_mm[r, W_index[all_rows, 0]], dtype=np.float32) for r in sample_rows])
B = np.where(np.isnan(B), np.nanmean(B[ref.train_s], axis=0), B)
log(f"betas {B.shape} ({time.time() - t0:.0f}s)")


def reconstruct(r) -> np.ndarray:
    """y_hat (samples x all_rows) = z[s, block(i)] . w_i, from the raw (unstandardised) block embeddings."""
    out = np.empty((len(r.obs_names), len(all_rows)), dtype=np.float32)
    for b in np.unique(r.W_block[all_rows]):
        sel = np.flatnonzero(r.W_block[all_rows] == b)
        out[:, sel] = r.Z[:, b].astype(np.float32) @ np.asarray(r.W[all_rows[sel]], dtype=np.float32).T
    return out


def tissue_centre(x: np.ndarray) -> np.ndarray:
    """Subtract each tissue's train-sample mean (no test donor enters the means), as evaluation.tissue_centred."""
    out = x.copy()
    for t in np.unique(ref.tissue):
        tr = ref.train_s[ref.tissue[ref.train_s] == t]
        if len(tr):
            out[ref.tissue == t] -= x[tr].mean(0)
    return out


# --- donors: one lung/colon test pair each (as stage 07 part 5); retrieval sets as stage 04 ---------
s = lung_colon_sets(ref)
pairs, seen = [], set()
for li in s["lung_q"]:
    m = s["colon"][ref.donor[s["colon"]] == ref.donor[li]]
    if len(m) and ref.donor[li] not in seen:
        seen.add(ref.donor[li])
        pairs.append((li, m[0]))
li_idx, ci_idx = np.array([a for a, _ in pairs], int), np.array([b for _, b in pairs], int)
lung_t = np.flatnonzero(ref.is_test & (ref.tissue == LUNG))
colon_t = np.flatnonzero(ref.is_test & (ref.tissue == COLON))
if len(pairs) < 5:
    finish(False, f"only {len(pairs)} lung/colon test donor pairs")


def concordance(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    cols = [pos[r] for r in rows]
    return cross_tissue_concordance(x[li_idx][:, cols], x[ci_idx][:, cols])


def fidelity(yhat: np.ndarray, rows: np.ndarray, samples: np.ndarray) -> np.ndarray:
    rho = np.full(len(rows), np.nan)
    for j, r in enumerate(rows):
        a, b = yhat[samples, pos[r]], B[samples, pos[r]]
        if np.ptp(a) > 0 and np.ptp(b) > 0:
            rho[j] = spearmanr(a, b).statistic
    return rho


def matching(x_tc: np.ndarray, rows: np.ndarray) -> dict:
    rep = x_tc[:, [pos[r] for r in rows]]
    return {"lung_to_colon": retrieval(rep, ref.donor, s["lung_q"], s["colon"]),
            "colon_to_lung": retrieval(rep, ref.donor, s["colon_q"], s["lung"])}


def mw_greater(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(mannwhitneyu(a, b, alternative="greater").pvalue) if len(a) and len(b) else float("nan")


def strip(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("ranks", "query_donor")}


# --- evaluate: raw betas first (the input reference), then each run's reconstruction --------------
result = {"n_donor_pairs": len(pairs), "n_strongest": n_conc, "n_match": n_match, "seed": args.seed,
          "control": "tested, not significant in any table, same block, nearest train-beta variance",
          "sites": {str(n): {"n_mqtl": len(mq), "n_control": len(c)} for n, (mq, c) in site_sets.items()},
          "representations": {}}
rows_csv = []
mq_c, ctrl_c = site_sets[n_conc]
reps = {"raw betas": B, **{label: None for label in runs}}
for label in reps:
    x = B if label == "raw betas" else reconstruct(runs[label])
    if not np.isfinite(x).all():
        finish(False, f"non-finite reconstruction for {label}")
    res = {}
    rho_m, rho_c = concordance(x, mq_c), concordance(x, ctrl_c)
    res["concordance"] = {"median_rho_mqtl": float(np.nanmedian(rho_m)), "median_rho_control": float(np.nanmedian(rho_c)),
                          "mannwhitney_p_mqtl_gt_control": mw_greater(rho_m, rho_c),
                          "rho_mqtl": rho_m.tolist(), "rho_control": rho_c.tolist()}
    if label != "raw betas":
        res["fidelity"] = {}
        for tname, samples in (("lung", lung_t), ("colon", colon_t)):
            f_m, f_c = fidelity(x, mq_c, samples), fidelity(x, ctrl_c, samples)
            res["fidelity"][tname] = {"median_rho_mqtl": float(np.nanmedian(f_m)), "median_rho_control": float(np.nanmedian(f_c)),
                                      "n_samples": int(len(samples))}
    x_tc = tissue_centre(x)
    res["matching"] = {}
    for n in n_match:
        mq, ctrl = site_sets[n]
        m_m, m_c = matching(x_tc, mq), matching(x_tc, ctrl)
        res["matching"][str(n)] = {
            d: {"mqtl": strip(m_m[d]), "control": strip(m_c[d]), "mqtl_minus_control": paired_bootstrap(m_m[d], m_c[d], seed=args.seed)}
            for d in ("lung_to_colon", "colon_to_lung")}
        for d in ("lung_to_colon", "colon_to_lung"):
            rows_csv.append({"representation": label, "analysis": "matching", "n_sites": n, "direction": d,
                             "top1_mqtl": m_m[d]["top1"], "top1_control": m_c[d]["top1"],
                             "diff": res["matching"][str(n)][d]["mqtl_minus_control"]["diff"],
                             "ci_lo": res["matching"][str(n)][d]["mqtl_minus_control"]["ci95"][0],
                             "ci_hi": res["matching"][str(n)][d]["mqtl_minus_control"]["ci95"][1]})
    rows_csv.append({"representation": label, "analysis": "concordance", "n_sites": n_conc,
                     "median_rho_mqtl": res["concordance"]["median_rho_mqtl"],
                     "median_rho_control": res["concordance"]["median_rho_control"],
                     "p": res["concordance"]["mannwhitney_p_mqtl_gt_control"]})
    for tname, fv in res.get("fidelity", {}).items():
        rows_csv.append({"representation": label, "analysis": f"fidelity {tname}", "n_sites": n_conc,
                         "median_rho_mqtl": fv["median_rho_mqtl"], "median_rho_control": fv["median_rho_control"]})
    result["representations"][label] = res
    log(f"[{label}] done ({time.time() - t0:.0f}s)")
write_outputs(Path(args.out), result, rows_csv)

# --- report ------------------------------------------------------------------------------------
print(f"\n=== 11 RECONSTRUCTION (test donors; {len(pairs)} lung/colon pairs; {n_conc} strongest mQTL CpGs "
      f"vs variance-matched controls) ===")
print("1-2. Median Spearman: concordance = lung vs colon over donors; fidelity = y_hat vs observed beta within tissue")
print(f"{'representation':<20s}{'conc mQTL':>10s}{'conc ctrl':>10s}{'MW p':>10s}{'fid lung m/c':>16s}{'fid colon m/c':>16s}")
for label, res in result["representations"].items():
    c, f = res["concordance"], res.get("fidelity", {})
    fl = f"{fmt(f['lung']['median_rho_mqtl'], '.2f')}/{fmt(f['lung']['median_rho_control'], '.2f')}" if f else "-"
    fc = f"{fmt(f['colon']['median_rho_mqtl'], '.2f')}/{fmt(f['colon']['median_rho_control'], '.2f')}" if f else "-"
    print(f"{label:<20s}{fmt(c['median_rho_mqtl']):>10s}{fmt(c['median_rho_control']):>10s}"
          f"{fmt(c['mannwhitney_p_mqtl_gt_control'], '.1e'):>10s}{fl:>16s}{fc:>16s}")
print("\n3. Donor matching top-1 on tissue-centred values at N sites (chance "
      f"{fmt(1 / max(len(s['colon']), 1), '.4f')}): mQTL / control, mQTL - control [95% CI]")
for n in n_match:
    print(f"   N = {n}")
    for label, res in result["representations"].items():
        parts = []
        for d, short in (("lung_to_colon", "L->C"), ("colon_to_lung", "C->L")):
            m = res["matching"][str(n)][d]
            bt = m["mqtl_minus_control"]
            parts.append(f"{short} {fmt(m['mqtl']['top1'])} / {fmt(m['control']['top1'])}, "
                         f"{bt['diff']:+.3f} [{bt['ci95'][0]:+.3f}, {bt['ci95'][1]:+.3f}]")
        print(f"   {label:<20s}" + "   ".join(parts))
print(f"elapsed: {time.time() - t0:.0f}s")
finish(True)
