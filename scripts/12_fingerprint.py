"""Stage 12: the genotype fingerprint in the betas, and what each encoder keeps of it.

Follows stage 11 (same runs, reconstructions y_hat = z . w_i, mQTL order and variance-matched controls), and adds:
  1a. SNP-free sites: mQTL CpGs and controls restricted to probes with no common SNP (Zhou et al. EPIC hg38
      common-SNP table, MAF >= --snp-maf globally or in Europeans), next to all sites and to SNP probes only.
  1b. Cluster shape (raw betas): per site and tissue, Gaussian mixture 1 vs 3 components (BIC) and, where 3 win,
      the spread of the cluster means. A hybridisation artefact gives wide, tissue-invariant clusters; a
      genuine mQTL a tissue-dependent shift. Reported for SNP probes vs SNP-free probes.
  2.  CpGPT's standard whole-sample embedding: y_hat = Z_full . w_i (runs with Z_full.npy), next to the block one.
  3.  Fingerprint size: lung<->colon top-1 on tissue-centred values at N = --n-sites strongest mQTL CpGs vs N
      controls, with a paired bootstrap of mQTL - control.
  4.  Every ordered tissue pair with >= --min-pair-donors shared test donors (N = --n-main sites).
  5.  Tri-modality of y_hat at the same sites (1b's fit on each reconstruction), and an example figure.
Writes <out>/result.json, summary.csv, tissue_pairs.csv, shape.csv, examples.npz (+ examples.png if matplotlib).

  python scripts/12_fingerprint.py [--runs label=enc:pred ...] [--snp-table ...] [--out ...] [--smoke]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
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
from cpgpt_jepa.fingerprint import (  # noqa: E402
    common_snp_probes,
    pair_matching,
    reconstruct,
    strongest_mqtl_order,
    tissue_centre,
)

DEFAULT_RUNS = [
    "cpgpt_small=outputs/encodings:outputs/predictor",
    "perceiver_small=outputs/perceiver_small/encodings:outputs/perceiver_small/predictor",
    "perceiver_large=outputs/perceiver_large/encodings:outputs/perceiver_large/predictor",
    "lewm_lam0.1_within=outputs/lewm/lam0.1_within:outputs/lewm/lam0.1_within/predictor",
    "lewm_lam1_within=outputs/lewm/lam1_within:outputs/lewm/lam1_within/predictor",
    "beta_pca=outputs/encodings_beta:outputs/predictor_beta",
]
SHAPE_REPS = ("raw betas", "cpgpt_small", "cpgpt_small_full", "perceiver_large", "beta_pca")

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="label=encodings_dir:predictor_dir")
p.add_argument("--egtex-dir", default="data/egtex")
p.add_argument("--snp-table", default="data/annotations/snp_dl/EPIC.hg38.commonsnp.tsv.gz")
p.add_argument("--snp-maf", type=float, default=0.01)
p.add_argument("--out", default="outputs/experiments/fingerprint")
p.add_argument("--n-sites", default="5,10,20,50,100,200,2000", help="fingerprint sizes (part 3)")
p.add_argument("--n-main", type=int, default=200, help="sites for concordance, fidelity, shape, tissue pairs")
p.add_argument("--min-pair-donors", type=int, default=20)
p.add_argument("--n-examples", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
t0 = time.time()
egtex, out = Path(args.egtex_dir), Path(args.out)
log = lambda m: print(m, flush=True)  # noqa: E731
n_main = 20 if args.smoke else args.n_main
n_sites = [5, 20] if args.smoke else sorted({int(x) for x in args.n_sites.split(",")} | {n_main})

# --- runs (as stage 11), plus whole-sample variants ---------------------------------------------
runs = {}
for spec in args.runs:
    label, paths = spec.split("=", 1)
    enc, pred = paths.split(":", 1)
    if (Path(enc) / "W.mmap").exists():
        runs[label] = load_run(enc, pred, egtex)
    else:
        log(f"skip {label}: no {enc}/W.mmap")
if not runs:
    finish(False, "no run with per-site vectors (W.mmap)")
ref = next(iter(runs.values()))
for label, r in runs.items():
    if not (np.array_equal(r.obs_names, ref.obs_names) and np.array_equal(r.site_probes, ref.site_probes)
            and np.array_equal(r.W_block, ref.W_block) and np.array_equal(r.test_s, ref.test_s)):
        finish(False, f"{label} differs from {next(iter(runs))} in samples, sites, blocks or test split")
# (label, run, full): block reconstructions, then whole-sample ones for the frozen encoders that saved Z_full
recon = [(lab, r, False) for lab, r in runs.items()]
recon += [(f"{lab}_full", r, True) for lab, r in runs.items()
          if r.Z_full is not None and r.enc_meta.get("model") not in ("lewm",) and "beta" not in lab]
log(f"representations: raw betas, {', '.join(lab for lab, _, _ in recon)} ({time.time() - t0:.0f}s)")

# --- site pools ----------------------------------------------------------------------------------
X_mm, _, _, _ = open_processed(Path(ref.enc_meta["dataset_dir"]))
sample_rows = np.array(ref.enc_meta.get("sample_idx", np.arange(X_mm.shape[0])))
W_index = np.load(ref.enc / "W_index.npy")
auto_rows = np.flatnonzero(np.isin(ref.W_block, ref.auto_blocks))
Xtr = np.asarray(X_mm[sample_rows[ref.train_s]])[:, W_index[auto_rows, 0]]
var_of_row = np.full(len(ref.W_block), np.nan)  # train-sample variance of each site's betas (as stage 07)
var_of_row[auto_rows] = np.nanvar(Xtr, axis=0)
del Xtr

snp = common_snp_probes(args.snp_table, args.snp_maf)
has_snp = np.isin(ref.site_probes, sorted(snp))
row_of = {pr: r for r, pr in zip(auto_rows, ref.site_probes[auto_rows])}
strong_all = np.array([row_of[c] for c in strongest_mqtl_order(egtex) if c in row_of], dtype=int)
any_set = read_set(egtex / "mqtl_sites_any.txt")
tested = np.isin(ref.site_probes, sorted(tested_cpgs(egtex)))
cand_all = auto_rows[tested[auto_rows] & ~np.isin(ref.site_probes[auto_rows], sorted(any_set))]
pools = {  # variant -> (mQTL rows in strength order, control candidates)
    "all": (strong_all, cand_all),
    "snp_free": (strong_all[~has_snp[strong_all]], cand_all[~has_snp[cand_all]]),
    "snp_probe": (strong_all[has_snp[strong_all]], cand_all),
}
snp_info = {"table": args.snp_table, "maf": args.snp_maf, "n_snp_probes_table": len(snp),
            "frac_block_sites_with_snp": float(has_snp[auto_rows].mean()),
            "frac_top_n_main_mqtl_with_snp": float(has_snp[strong_all[:n_main]].mean()),
            "n_mqtl_ranked": {v: int(len(m)) for v, (m, _) in pools.items()}}
site_sets = {}  # (variant, N) -> (mQTL rows, control rows)
for v, (mq_pool, cand) in pools.items():
    for n in n_sites:
        if v == "snp_probe" and n > n_main:
            continue  # only the main size is used for the SNP-probe comparison
        mq = mq_pool[:n]
        site_sets[(v, n)] = (mq, variance_matched_controls(mq, cand, ref.W_block, var_of_row,
                                                           np.random.default_rng(args.seed)))
log(f"SNP probes: {snp_info['frac_block_sites_with_snp']:.3f} of block sites, "
    f"{snp_info['frac_top_n_main_mqtl_with_snp']:.3f} of the top {n_main} mQTL CpGs")
all_rows = np.unique(np.concatenate([np.concatenate(v) for v in site_sets.values()]))
pos = {r: j for j, r in enumerate(all_rows)}
B = np.stack([np.asarray(X_mm[r, W_index[all_rows, 0]], dtype=np.float32) for r in sample_rows])
B = np.where(np.isnan(B), np.nanmean(B[ref.train_s], axis=0), B)
log(f"betas {B.shape} at {len(all_rows)} sites ({time.time() - t0:.0f}s)")

# --- samples -------------------------------------------------------------------------------------
s = lung_colon_sets(ref)
pairs, seen = [], set()
for li in s["lung_q"]:
    m = s["colon"][ref.donor[s["colon"]] == ref.donor[li]]
    if len(m) and ref.donor[li] not in seen:
        seen.add(ref.donor[li])
        pairs.append((li, m[0]))
if len(pairs) < 5:
    finish(False, f"only {len(pairs)} lung/colon test donor pairs")
li_idx, ci_idx = np.array([a for a, _ in pairs], int), np.array([b for _, b in pairs], int)
by_tissue = {t: np.flatnonzero(ref.is_test & (ref.tissue == t)) for t in (LUNG, COLON)}
tissues = sorted(set(ref.tissue))
tissue_pairs = []
for ta in tissues:
    for tb in tissues:
        if ta != tb:
            da = set(ref.donor[ref.is_test & (ref.tissue == ta)])
            db = set(ref.donor[ref.is_test & (ref.tissue == tb)])
            if len(da & db) >= args.min_pair_donors:
                tissue_pairs.append((ta, tb, len(da & db)))
log(f"{len(pairs)} lung/colon pairs; {len(tissue_pairs)} ordered tissue pairs with >= {args.min_pair_donors} donors")


def cols(rows: np.ndarray) -> list[int]:
    return [pos[r] for r in rows]


def mw_greater(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(mannwhitneyu(a, b, alternative="greater").pvalue) if len(a) and len(b) else float("nan")


def strip(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("ranks", "query_donor")}


def shape_stats(x: np.ndarray, rows: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per site: does BIC prefer 3 Gaussian components over 1, and the spread (max - min) of the 3 means
    (NaN when 1 wins). Values are standardised per site for the fit, the spread is on the original scale."""
    from sklearn.mixture import GaussianMixture

    pref, spread = np.zeros(len(rows), bool), np.full(len(rows), np.nan)
    for j, c in enumerate(cols(rows)):
        v = x[samples, c].astype(np.float64)
        sd = v.std()
        if sd == 0:
            continue
        z = ((v - v.mean()) / sd)[:, None]
        g1 = GaussianMixture(1, random_state=args.seed).fit(z)
        g3 = GaussianMixture(3, n_init=3, random_state=args.seed, reg_covar=1e-4).fit(z)
        if g3.bic(z) < g1.bic(z):
            pref[j] = True
            mu = g3.means_.ravel() * sd + v.mean()
            spread[j] = mu.max() - mu.min()
    return pref, spread


# --- evaluate each representation -------------------------------------------------------------------
result = {"n_donor_pairs": len(pairs), "n_main": n_main, "n_sites": n_sites, "seed": args.seed, "snp": snp_info,
          "control": "tested, not significant in any table, same block, nearest train-beta variance",
          "site_sets": {f"{v}/{n}": {"n_mqtl": len(a), "n_control": len(b)} for (v, n), (a, b) in site_sets.items()},
          "tissue_pairs": [{"query": a, "gallery": b, "n_donors": n} for a, b, n in tissue_pairs],
          "representations": {}}
rows_csv, pair_rows, shape_rows = [], [], []
examples = {}
reps = [("raw betas", None, False)] + recon
for label, run, full in reps:
    x = B if run is None else reconstruct(run, all_rows, full=full)
    if not np.isfinite(x).all():
        finish(False, f"non-finite values for {label}")
    x_tc = tissue_centre(x, ref.tissue, ref.train_s)
    res = {"concordance": {}, "fidelity": {}, "matching": {}, "tissue_pairs": {}, "shape": {}}
    for v in pools:
        mq, ctrl = site_sets[(v, n_main)]
        # concordance and fidelity at the main size
        rho_m = cross_tissue_concordance(x[li_idx][:, cols(mq)], x[ci_idx][:, cols(mq)])
        rho_c = cross_tissue_concordance(x[li_idx][:, cols(ctrl)], x[ci_idx][:, cols(ctrl)])
        res["concordance"][v] = {"median_rho_mqtl": float(np.nanmedian(rho_m)),
                                 "median_rho_control": float(np.nanmedian(rho_c)),
                                 "mannwhitney_p_mqtl_gt_control": mw_greater(rho_m, rho_c)}
        rows_csv.append({"representation": label, "variant": v, "analysis": "concordance", "n_sites": n_main,
                         "mqtl": res["concordance"][v]["median_rho_mqtl"],
                         "control": res["concordance"][v]["median_rho_control"],
                         "p": res["concordance"][v]["mannwhitney_p_mqtl_gt_control"]})
        if run is not None:
            res["fidelity"][v] = {}
            for t, smp in by_tissue.items():
                fid = {}
                for name, rr in (("mqtl", mq), ("control", ctrl)):
                    rho = [spearmanr(x[smp, c], B[smp, c]).statistic if np.ptp(x[smp, c]) > 0 else np.nan
                           for c in cols(rr)]
                    fid[name] = float(np.nanmedian(rho))
                res["fidelity"][v][t] = fid
                rows_csv.append({"representation": label, "variant": v, "analysis": f"fidelity {t}",
                                 "n_sites": n_main, "mqtl": fid["mqtl"], "control": fid["control"]})
        # fingerprint size (lung <-> colon)
        res["matching"][v] = {}
        for n in n_sites:
            if (v, n) not in site_sets:
                continue
            mq_n, ctrl_n = site_sets[(v, n)]
            entry = {}
            for d, q, g in (("lung_to_colon", s["lung_q"], s["colon"]), ("colon_to_lung", s["colon_q"], s["lung"])):
                a = retrieval(x_tc[:, cols(mq_n)], ref.donor, q, g)
                b = retrieval(x_tc[:, cols(ctrl_n)], ref.donor, q, g)
                bt = paired_bootstrap(a, b, seed=args.seed)
                entry[d] = {"mqtl": strip(a), "control": strip(b), "mqtl_minus_control": bt}
                rows_csv.append({"representation": label, "variant": v, "analysis": f"matching {d}", "n_sites": n,
                                 "mqtl": a["top1"], "control": b["top1"], "diff": bt["diff"],
                                 "ci_lo": bt["ci95"][0], "ci_hi": bt["ci95"][1]})
            res["matching"][v][str(n)] = entry
        # every tissue pair, main size, all and SNP-free sites
        if v in ("all", "snp_free"):
            res["tissue_pairs"][v] = {}
            for ta, tb, nd in tissue_pairs:
                a = pair_matching(x_tc[:, cols(mq)], ref.donor, ref.is_test, ref.tissue, ta, tb)
                b = pair_matching(x_tc[:, cols(ctrl)], ref.donor, ref.is_test, ref.tissue, ta, tb)
                res["tissue_pairs"][v][f"{ta} -> {tb}"] = {"mqtl": strip(a), "control": strip(b), "n_donors": nd}
                pair_rows.append({"representation": label, "variant": v, "query": ta, "gallery": tb, "n_donors": nd,
                                  "n_query": a["n_query"], "n_gallery": a["n_gallery"], "top1_mqtl": a["top1"],
                                  "top1_control": b["top1"], "chance": a["chance"]})
        # cluster shape (1b on raw betas, 5 on reconstructions)
        if label in SHAPE_REPS:
            res["shape"][v] = {}
            for t, smp in by_tissue.items():
                pref, spread = shape_stats(x, mq, smp)
                res["shape"][v][t] = {"frac_prefer_3": float(pref.mean()), "median_spread": float(np.nanmedian(spread))
                                      if pref.any() else float("nan"), "n_sites": int(len(mq))}
                for j, r in enumerate(mq):
                    shape_rows.append({"representation": label, "variant": v, "tissue": t, "probe": ref.site_probes[r],
                                       "prefer_3": bool(pref[j]), "spread": spread[j]})
            if label == "raw betas":  # tissue invariance of the spread: |lung - colon| over sites where both prefer 3
                sl = np.array([rw["spread"] for rw in shape_rows if rw["representation"] == label
                               and rw["variant"] == v and rw["tissue"] == LUNG], float)
                sc = np.array([rw["spread"] for rw in shape_rows if rw["representation"] == label
                               and rw["variant"] == v and rw["tissue"] == COLON], float)
                ok = np.isfinite(sl) & np.isfinite(sc)
                res["shape"][v]["median_abs_spread_diff_lung_colon"] = float(np.median(np.abs(sl[ok] - sc[ok]))) if ok.any() else float("nan")
                res["shape"][v]["n_sites_3_in_both"] = int(ok.sum())
    if label in SHAPE_REPS:  # example values for the figure: the strongest SNP-free mQTL CpGs, lung test samples
        ex_rows = site_sets[("snp_free", n_main)][0][: args.n_examples]
        examples[label] = x[by_tissue[LUNG]][:, cols(ex_rows)]
        examples["probes"] = ref.site_probes[ex_rows]
    result["representations"][label] = res
    log(f"[{label}] done ({time.time() - t0:.0f}s)")

write_outputs(out, result, rows_csv)
for name, rws in (("tissue_pairs.csv", pair_rows), ("shape.csv", shape_rows)):
    with open(out / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rws[0]) if rws else ["empty"])
        w.writeheader()
        w.writerows(rws)
np.savez(out / "examples.npz", **{k: np.asarray(v) for k, v in examples.items()})
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [k for k in SHAPE_REPS if k in examples]
    fig, axes = plt.subplots(len(show), args.n_examples, figsize=(3 * args.n_examples, 2.2 * len(show)), squeeze=False)
    for i, k in enumerate(show):
        for j in range(min(args.n_examples, examples[k].shape[1])):
            axes[i, j].hist(examples[k][:, j], bins=30, color="#4c72b0")
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(str(examples["probes"][j]), fontsize=9)
        axes[i, 0].set_ylabel(k, fontsize=9)
    fig.suptitle("Lung test samples at the strongest SNP-free mQTL CpGs: observed betas vs reconstructions", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=130)
    log(f"figure: {out / 'examples.png'}")
except ImportError:
    log("matplotlib not available: examples.npz only")

# --- report ------------------------------------------------------------------------------------
R = result["representations"]
print(f"\n=== 12 FINGERPRINT (test donors; {len(pairs)} lung/colon pairs; controls variance-matched) ===")
print(f"SNP table {args.snp_table} (MAF >= {args.snp_maf}): {snp_info['frac_block_sites_with_snp']:.3f} of block sites "
      f"and {snp_info['frac_top_n_main_mqtl_with_snp']:.3f} of the top {n_main} mQTL CpGs overlap a common SNP")
print(f"\n1a/2. Concordance (median Spearman, lung vs colon over donors) at {n_main} sites, mQTL / control")
print(f"{'representation':<22s}" + "".join(f"{v:>22s}" for v in pools))
for label in R:
    print(f"{label:<22s}" + "".join(f"{fmt(R[label]['concordance'][v]['median_rho_mqtl'], '.2f') + ' / ' + fmt(R[label]['concordance'][v]['median_rho_control'], '.2f'):>22s}" for v in pools))
print(f"\n3. Fingerprint size: lung->colon top-1, mQTL / control (chance {fmt(1 / len(s['colon']), '.4f')})")
for v in ("all", "snp_free"):
    print(f"   sites: {v}")
    print(f"   {'representation':<22s}" + "".join(f"{'N=' + str(n):>14s}" for n in n_sites))
    for label in R:
        cells = []
        for n in n_sites:
            e = R[label]["matching"][v].get(str(n))
            cells.append(f"{fmt(e['lung_to_colon']['mqtl']['top1'], '.2f')}/{fmt(e['lung_to_colon']['control']['top1'], '.2f')}" if e else "-")
        print(f"   {label:<22s}" + "".join(f"{c:>14s}" for c in cells))
print(f"   SNP-probe mQTL only, N={n_main}: " + ", ".join(
    f"{lab} {fmt(R[lab]['matching']['snp_probe'][str(n_main)]['lung_to_colon']['mqtl']['top1'], '.2f')}" for lab in R))
print(f"\n4. Tissue pairs (N={n_main} mQTL sites, top-1 mQTL / control), raw betas")
for v in ("all", "snp_free"):
    for k, e in R["raw betas"]["tissue_pairs"][v].items():
        print(f"   {v:<9s} {k:<48s} {fmt(e['mqtl']['top1'], '.2f')} / {fmt(e['control']['top1'], '.2f')}  "
              f"({e['mqtl']['n_query']} queries, chance {fmt(e['mqtl']['chance'], '.3f')})")
print(f"\n1b/5. Cluster shape at {n_main} mQTL sites: fraction preferring 3 components (median spread of the means)")
for label in R:
    if R[label]["shape"]:
        cells = [f"{v}: " + "  ".join(f"{t.split(' ')[0]} {fmt(R[label]['shape'][v][t]['frac_prefer_3'], '.2f')} "
                                      f"({fmt(R[label]['shape'][v][t]['median_spread'], '.2f')})" for t in by_tissue)
                 for v in pools]
        print(f"   {label:<20s} " + " | ".join(cells))
for v in pools:
    sh = R["raw betas"]["shape"][v]
    print(f"   raw betas, {v}: median |spread lung - spread colon| {fmt(sh['median_abs_spread_diff_lung_colon'])} "
          f"over {sh['n_sites_3_in_both']} sites trimodal in both")
print(f"elapsed: {time.time() - t0:.0f}s")
finish(True)
