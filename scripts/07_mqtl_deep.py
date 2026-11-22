"""Phase A / 07: mQTL analyses of the residual in depth (test donors).

  python scripts/07_mqtl_deep.py [--encodings outputs/encodings] [--pred outputs/predictor] [--out outputs/experiments/mqtl_deep]

1. Enrichment of the per-site residual score for lung-only, colon-only, union (lung | colon) and any-tissue
   mQTL CpGs, plus a negative set: top-5% age- or tissue-associated CpGs (one-way ANOVA on betas, train
   donors) that are not mQTLs.
2. Per-block leave-one-block-out contribution to donor matching vs per-block mQTL density (Spearman, CSV).
3. Tri-modality: residual contribution g of the 200 strongest mQTL CpGs across test donors; Gaussian
   mixture BIC, 1 vs 3 components; 200 random non-mQTL sites from the same blocks as control.
4. chrX positive control (stage 2 --include-x): donor matching and sex prediction from chrX vs autosomal blocks.
Needs the frozen-CpGPT decoder vectors (W.mmap); Phase B encodings skip parts 1 and 3.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cpgpt_jepa import finish  # noqa: E402
from cpgpt_jepa.blocks import open_processed  # noqa: E402
from cpgpt_jepa.evaluation import (  # noqa: E402
    COLON,
    LUNG,
    donor_matching,
    enrichment,
    fmt,
    lung_colon_sets,
    load_run,
    percentile_within_block,
    rank_of_true,
    ranks_to_metrics,
    read_set,
    site_g,
    site_scores,
    write_outputs,
)

p = argparse.ArgumentParser()
p.add_argument("--encodings", "--enc", dest="enc", default="outputs/encodings")
p.add_argument("--pred", default="outputs/predictor")
p.add_argument("--egtex-dir", default="data/egtex")
p.add_argument("--out", default="outputs/experiments/mqtl_deep")
p.add_argument("--kind", default="mlp_joint", help="predictor kind (primary: mlp_joint, docs/PREREG.md)")
p.add_argument("--n-strongest", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
t0 = time.time()
egtex = Path(args.egtex_dir)
out = Path(args.out)

run = load_run(args.enc, args.pred, egtex)
if not (run.pred / args.kind / "R.npy").exists():
    finish(False, f"no residuals for {args.kind} under {run.pred}")
R = run.residual(args.kind)
n_samples, n_blocks, d = run.Z.shape
result, rows = {"kind": args.kind}, []
has_w = run.W is not None
test = run.test_s
lung_t = np.flatnonzero(run.is_test & (run.tissue == LUNG))
colon_t = np.flatnonzero(run.is_test & (run.tissue == COLON))

# --- 1. enrichment against mQTL sets and a negative set ----------------------------------------
if has_w:
    lung_set, colon_set = read_set(egtex / "mqtl_sites_Lung.txt"), read_set(egtex / "mqtl_sites_ColonTransverse.txt")
    any_set = read_set(egtex / "mqtl_sites_any.txt")
    sets = {"lung-only": lung_set - colon_set, "colon-only": colon_set - lung_set,
            "union (lung|colon)": lung_set | colon_set, "any tissue": any_set}

    # negative set: top-5% tissue- or age-associated CpGs by one-way ANOVA F on the betas (train donors), not mQTL
    X_mm, _, _, _ = open_processed(Path(run.enc_meta["dataset_dir"]))
    sample_rows = np.array(run.enc_meta.get("sample_idx", np.arange(X_mm.shape[0])))
    auto_rows = np.flatnonzero(np.isin(run.W_block, run.auto_blocks))
    site_cols = np.load(run.enc / "W_index.npy")[auto_rows, 0]
    Xtr = np.asarray(X_mm[sample_rows[run.train_s]])[:, site_cols]  # float32, train samples x block sites
    col_mean = np.nanmean(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), col_mean, Xtr)

    def anova_f(groups: np.ndarray) -> np.ndarray:
        labels = [g for g in pd.unique(groups) if g != "NA"]
        ok = np.isin(groups, labels)
        x, grp = Xtr[ok], groups[ok]
        grand = x.mean(0)
        ssb = sum((grp == g).sum() * (x[grp == g].mean(0) - grand) ** 2 for g in labels)
        ssw = sum(((x[grp == g] - x[grp == g].mean(0)) ** 2).sum(0) for g in labels)
        k, n = len(labels), len(x)
        return (ssb / max(k - 1, 1)) / (ssw / max(n - k, 1) + 1e-12)

    f_tissue = anova_f(run.tissue[run.train_s])
    f_age = anova_f(run.meta.age.fillna("NA").to_numpy()[run.train_s])
    top_t = f_tissue >= np.quantile(f_tissue, 0.95)
    top_a = f_age >= np.quantile(f_age, 0.95)
    probes_auto = run.site_probes[auto_rows]
    negative = (top_t | top_a) & ~np.isin(probes_auto, sorted(any_set))
    neg_probes = set(probes_auto[negative])
    sets["negative (age/tissue, not mQTL)"] = neg_probes
    result["negative_set"] = {"n_tissue_top5": int(top_t.sum()), "n_age_top5": int(top_a.sum()),
                              "n_negative": int(negative.sum())}

    sample_sets = {"all test": test, "lung test": lung_t, "colon test": colon_t}
    pairs = [(s, "all test") for s in sets] + [("lung-only", "lung test"), ("union (lung|colon)", "lung test"),
                                               ("colon-only", "colon test"), ("union (lung|colon)", "colon test"),
                                               ("negative (age/tissue, not mQTL)", "lung test"),
                                               ("negative (age/tissue, not mQTL)", "colon test")]
    pct_cache = {}
    result["enrichment"] = {}
    for set_name, samp_name in pairs:
        samples = sample_sets[samp_name]
        if len(samples) < 2:
            continue
        if samp_name not in pct_cache:
            w_rows, var_g, _ = site_scores(run, R, samples)
            pct_cache[samp_name] = (w_rows, percentile_within_block(var_g, run.W_block[w_rows]))
        w_rows, pct = pct_cache[samp_name]
        e = enrichment(pct, np.isin(run.site_probes[w_rows], sorted(sets[set_name])))
        result["enrichment"][f"{set_name} / {samp_name}"] = e
        rows.append({"analysis": "enrichment", "set": set_name, "samples": samp_name, **{k: v for k, v in e.items() if k != "top5_table"}})
    print(f"[1] enrichment done ({time.time() - t0:.0f}s)", flush=True)
else:
    result["enrichment"] = "skipped: no per-site decoder vectors (W.mmap)"

# --- 2. leave-one-block-out contribution vs mQTL density ---------------------------------------
s = lung_colon_sets(run)
auto = run.auto_blocks


def per_block_parts(q_idx: np.ndarray, g_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, g = R[q_idx][:, auto].astype(np.float64), R[g_idx][:, auto].astype(np.float64)
    dots = np.einsum("qbd,gbd->bqg", q, g)
    return dots, (q ** 2).sum(-1), (g ** 2).sum(-1)  # (B, nq, ng), (nq, B), (ng, B)


def mrr_from(dots: np.ndarray, qn: np.ndarray, gn: np.ndarray, q_idx: np.ndarray, g_idx: np.ndarray) -> float:
    sim = dots / (np.sqrt(qn)[:, None] * np.sqrt(gn)[None, :] + 1e-12)
    return ranks_to_metrics(rank_of_true(sim, run.donor[q_idx], run.donor[g_idx]), len(g_idx))["mrr"]


contrib = {}
for direction, (q_idx, g_idx) in {"l2c": (s["lung_q"], s["colon"]), "c2l": (s["colon_q"], s["lung"])}.items():
    q_idx = q_idx[np.isin(run.donor[q_idx], run.donor[g_idx])]
    if len(q_idx) == 0:
        continue
    dots, qn, gn = per_block_parts(q_idx, g_idx)
    full = mrr_from(dots.sum(0), qn.sum(1), gn.sum(1), q_idx, g_idx)
    loo = np.array([mrr_from(dots.sum(0) - dots[b], qn.sum(1) - qn[:, b], gn.sum(1) - gn[:, b], q_idx, g_idx)
                    for b in range(len(auto))])
    single = np.array([mrr_from(dots[b], qn[:, b], gn[:, b], q_idx, g_idx) for b in range(len(auto))])
    contrib[direction] = {"full_mrr": full, "contribution": full - loo, "single_block_mrr": single}
block_rows = []
if contrib:
    any_set = read_set(egtex / "mqtl_sites_any.txt")
    for j, b in enumerate(auto):
        blk = run.blocks[b]
        dens = float("nan")
        if has_w:
            dens = float(np.isin(run.site_probes[run.W_block == b], sorted(any_set)).mean())
        row = {"block_id": int(b), "chrom": blk["chrom"], "start": blk["start"], "n_sites": blk["n_sites"],
               "mqtl_density": dens}
        for direction, c in contrib.items():
            row[f"contribution_{direction}"] = float(c["contribution"][j])
            row[f"single_block_mrr_{direction}"] = float(c["single_block_mrr"][j])
        row["contribution_mean"] = float(np.mean([row[f"contribution_{k}"] for k in contrib]))
        block_rows.append(row)
    bdf = pd.DataFrame(block_rows)
    out.mkdir(parents=True, exist_ok=True)
    bdf.to_csv(out / "block_contribution_vs_mqtl_density.csv", index=False)
    ok = np.isfinite(bdf.mqtl_density) & np.isfinite(bdf.contribution_mean)
    rho, pval = (spearmanr(bdf.mqtl_density[ok], bdf.contribution_mean[ok]) if ok.sum() > 2 else (np.nan, np.nan))
    rho_s, pval_s = (spearmanr(bdf.mqtl_density[ok], bdf[[f"single_block_mrr_{k}" for k in contrib]].mean(1)[ok])
                     if ok.sum() > 2 else (np.nan, np.nan))
    result["block_contribution"] = {"full_mrr": {k: c["full_mrr"] for k, c in contrib.items()},
                                    "spearman_contribution_vs_density": {"rho": float(rho), "p": float(pval)},
                                    "spearman_single_block_mrr_vs_density": {"rho": float(rho_s), "p": float(pval_s)},
                                    "n_blocks": int(ok.sum())}
    rows.append({"analysis": "block_contribution", "set": "LOO contribution vs mQTL density", "samples": "lung/colon",
                 "rho": float(rho), "p": float(pval)})
print(f"[2] block contributions done ({time.time() - t0:.0f}s)", flush=True)

# --- 3. tri-modality of the strongest mQTL CpGs ------------------------------------------------
if has_w:
    from sklearn.mixture import GaussianMixture

    tables = [pd.read_csv(f, sep="\t", usecols=["cpg_id", "slope", "qval"]) for f in sorted((egtex / "mqtl").glob("*.regular.perm.fdr.txt"))]
    strong = pd.concat(tables)
    strong["abs_slope"] = strong.slope.abs()
    strong = strong.sort_values(["qval", "abs_slope"], ascending=[True, False]).drop_duplicates("cpg_id")
    auto_rows = np.flatnonzero(np.isin(run.W_block, run.auto_blocks))
    row_of = {pr: r for r, pr in zip(auto_rows, run.site_probes[auto_rows])}
    strong = strong[strong.cpg_id.isin(list(row_of))].head(args.n_strongest)
    mq_rows = np.array([row_of[c] for c in strong.cpg_id], dtype=int)
    any_set = read_set(egtex / "mqtl_sites_any.txt")
    rng = np.random.default_rng(args.seed)
    cand = auto_rows[np.isin(run.W_block[auto_rows], np.unique(run.W_block[mq_rows]))
                     & ~np.isin(run.site_probes[auto_rows], sorted(any_set))]
    ctrl_rows = rng.choice(cand, size=min(len(mq_rows), len(cand)), replace=False) if len(cand) else np.array([], int)

    def prefers_three(samples: np.ndarray, w_rows: np.ndarray) -> tuple[float, int]:
        if len(samples) < 5 or len(w_rows) == 0:
            return float("nan"), 0
        g = site_g(run, R, samples, w_rows)
        votes = []
        for j in range(g.shape[1]):
            x = g[:, j : j + 1].astype(np.float64)
            bic = [GaussianMixture(k, n_init=3, random_state=args.seed, reg_covar=1e-6).fit(x).bic(x) for k in (1, 3)]
            votes.append(bic[1] < bic[0])
        return float(np.mean(votes)), len(votes)

    result["trimodality"] = {}
    for samp_name, samples in (("lung test", lung_t), ("colon test", colon_t)):
        for set_name, w_rows in (("strongest mQTL", mq_rows), ("random non-mQTL", ctrl_rows)):
            frac, n = prefers_three(samples, w_rows)
            result["trimodality"][f"{set_name} / {samp_name}"] = {"fraction_prefer_3": frac, "n_sites": n,
                                                                 "n_samples": int(len(samples))}
            rows.append({"analysis": "trimodality", "set": set_name, "samples": samp_name, "fraction_prefer_3": frac,
                         "n_sites": n})
    print(f"[3] tri-modality done ({time.time() - t0:.0f}s)", flush=True)
else:
    result["trimodality"] = "skipped: no per-site decoder vectors (W.mmap)"

# --- 4. chrX positive control ------------------------------------------------------------------
x_blocks = np.flatnonzero(run.block_chrom == "X")
if len(x_blocks):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(args.seed)
    rand_auto = rng.choice(run.auto_blocks, size=min(len(x_blocks), len(run.auto_blocks)), replace=False)
    result["chrX"] = {"n_x_blocks": int(len(x_blocks))}
    for label, blocks in (("chrX blocks", x_blocks), (f"{len(rand_auto)} random autosomal blocks", rand_auto),
                          ("all autosomal blocks", run.auto_blocks)):
        entry = {"donor_matching": donor_matching(run, R, blocks=blocks)}
        feats = R[:, blocks].reshape(n_samples, -1)
        sex = run.meta.sex.to_numpy()
        tr, te = run.train_s, run.test_s
        if len(set(sex[tr])) > 1 and len(te):
            clf = LogisticRegression(max_iter=2000).fit(feats[tr], sex[tr])
            entry["sex_accuracy_test"] = float((clf.predict(feats[te]) == sex[te]).mean())
        result["chrX"][label] = entry
        dm = entry["donor_matching"]
        rows.append({"analysis": "chrX", "set": label, "samples": "test",
                     "l2c_top1": dm["lung_to_colon"]["top1"], "c2l_top1": dm["colon_to_lung"]["top1"],
                     "sex_accuracy": entry.get("sex_accuracy_test", float("nan"))})
else:
    result["chrX"] = "skipped: encodings were built without --include-x"

write_outputs(out, result, rows)

# --- report ------------------------------------------------------------------------------------
print("\n=== 07 mQTL DEEP (test donors) ===")
if isinstance(result["enrichment"], dict):
    print("1. Enrichment of the residual score (variance of g), percentile within block")
    print(f"   {'set / samples':<46s}{'n set':>8s}{'pct set':>9s}{'pct oth':>9s}{'MW p':>10s}{'OR top5%':>10s}{'Fisher p':>10s}")
    for k, e in result["enrichment"].items():
        print(f"   {k:<46s}{e['n_set']:>8d}{fmt(e['mean_pct_set'], '.4f'):>9s}{fmt(e['mean_pct_other'], '.4f'):>9s}"
              f"{fmt(e['mannwhitney_p'], '.2e'):>10s}{fmt(e['top5_odds_ratio']):>10s}{fmt(e['top5_fisher_p'], '.2e'):>10s}")
    print(f"   negative set: {json.dumps(result['negative_set'])}")
else:
    print(f"1. {result['enrichment']}")
bc = result.get("block_contribution")
if bc:
    print(f"2. Leave-one-block-out contribution to MRR vs mQTL density over {bc['n_blocks']} blocks: "
          f"Spearman rho {fmt(bc['spearman_contribution_vs_density']['rho'])} (p {fmt(bc['spearman_contribution_vs_density']['p'], '.2e')}); "
          f"single-block MRR vs density rho {fmt(bc['spearman_single_block_mrr_vs_density']['rho'])}; "
          f"full MRR {', '.join(f'{k} {fmt(v)}' for k, v in bc['full_mrr'].items())}")
else:
    print("2. block contributions: no lung/colon query-gallery pairs")
if isinstance(result["trimodality"], dict):
    print("3. Gaussian-mixture BIC, fraction of sites preferring 3 components over 1:")
    for k, v in result["trimodality"].items():
        print(f"   {k:<34s} {fmt(v['fraction_prefer_3'])}  ({v['n_sites']} sites, {v['n_samples']} samples)")
else:
    print(f"3. {result['trimodality']}")
if isinstance(result["chrX"], dict):
    print("4. chrX positive control (residual R):")
    for k, v in result["chrX"].items():
        if k == "n_x_blocks":
            continue
        dm = v["donor_matching"]
        print(f"   {k:<32s} top1 l->c {fmt(dm['lung_to_colon']['top1'])}  c->l {fmt(dm['colon_to_lung']['top1'])}"
              f"  sex accuracy {fmt(v.get('sex_accuracy_test', float('nan')))}")
else:
    print(f"4. {result['chrX']}")
print(f"elapsed: {time.time() - t0:.0f}s")
finish(True)
