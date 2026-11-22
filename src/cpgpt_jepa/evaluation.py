"""Shared loading and evaluation helpers for stages 04-09: donor matching, mQTL enrichment, outputs."""

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, rankdata

LUNG, COLON = "Lung", "Colon - Transverse"
# metadata tissue name -> eGTEx file stem (mqtl_sites_<stem>.txt, metadata/<stem>.covariates.txt)
TISSUE_STEM = {
    "Breast - Mammary Tissue": "BreastMammaryTissue",
    "Colon - Transverse": "ColonTransverse",
    "Kidney - Cortex": "KidneyCortex",
    "Lung": "Lung",
    "Muscle - Skeletal": "MuscleSkeletal",
    "Ovary": "Ovary",
    "Prostate": "Prostate",
    "Testis": "Testis",
    "Whole Blood": "WholeBlood",
}


def load_metadata(egtex_dir: str | Path, obs_names: np.ndarray) -> pd.DataFrame:
    """metadata.csv in obs_names order, plus age (decade midpoint) and smoker status from the GEO SOFT
    file, which build_metadata.py does not keep."""
    egtex_dir = Path(egtex_dir)
    meta = pd.read_csv(egtex_dir / "metadata.csv", dtype=str).set_index("sample_id")
    extra, cur = {}, {}
    with gzip.open(egtex_dir / "geo" / "GSE213478_family.soft.gz", "rt") as f:
        for line in f:
            if line.startswith("^SAMPLE"):
                if "sample_id" in cur:
                    extra[cur.pop("sample_id")] = cur
                cur = {}
            elif line.startswith("!Sample_characteristics_ch1"):
                key, _, value = line.split("=", 1)[1].strip().partition(":")
                key, value = key.strip(), value.strip()
                if key in ("sample_id", "age", "smoker_status"):
                    cur[key] = value
    if "sample_id" in cur:
        extra[cur.pop("sample_id")] = cur
    extra = pd.DataFrame.from_dict(extra, orient="index")
    meta = meta.join(extra, how="left")
    bounds = meta["age"].str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
    meta["age_years"] = bounds.mean(axis=1)
    return meta.loc[list(obs_names)]


@dataclass
class Run:
    """Everything an evaluation script needs from one encodings + predictor run."""

    enc: Path
    pred: Path
    Z: np.ndarray  # samples x blocks x d (raw)
    Zs: np.ndarray  # standardised with train-donor stats
    mu: np.ndarray
    sd: np.ndarray
    C: np.ndarray
    blocks: np.ndarray
    block_chrom: np.ndarray  # chromosome name per block
    auto_blocks: np.ndarray  # indices of autosomal blocks
    obs_names: np.ndarray
    meta: pd.DataFrame
    donor: np.ndarray
    tissue: np.ndarray
    splits: dict
    train_s: np.ndarray
    test_s: np.ndarray
    is_test: np.ndarray
    enc_meta: dict
    W: np.memmap | None  # per-site decoder vectors (frozen CpGPT only)
    W_block: np.ndarray | None
    site_probes: np.ndarray | None
    Z_full: np.ndarray | None

    def kinds(self) -> list[str]:
        return sorted(p.parent.name for p in self.pred.glob("*/R.npy"))

    def residual(self, kind: str) -> np.ndarray:
        return np.load(self.pred / kind / "R.npy")


def load_run(enc: str | Path, pred: str | Path, egtex_dir: str | Path) -> Run:
    enc, pred = Path(enc), Path(pred)
    Z = np.load(enc / "Z.npy")
    blocks = np.load(enc / "blocks.npy", allow_pickle=True)  # written by stage 2 / 09 (object array of dicts)
    obs_names = np.load(enc / "obs_names.npy", allow_pickle=True)  # written by stage 2 / 09
    enc_meta = json.loads((enc / "encode_meta.json").read_text())
    splits = json.loads((pred / "splits.json").read_text())
    stats = np.load(pred / "standardise.npz")
    meta = load_metadata(egtex_dir, obs_names)
    block_chrom = np.array([b["chrom"] for b in blocks])
    W = W_block = site_probes = None
    if (enc / "W.mmap").exists():
        W = np.memmap(enc / "W.mmap", dtype=np.float16, mode="r", shape=tuple(enc_meta["W_shape"]))
        W_block = np.load(enc / "W_index.npy")[:, 1]
        site_probes = np.load(enc / "site_probe_ids.npy")
    is_test = np.zeros(len(obs_names), dtype=bool)
    is_test[splits["test_samples"]] = True
    return Run(
        enc=enc, pred=pred, Z=Z, Zs=(Z - stats["mu"]) / stats["sd"], mu=stats["mu"], sd=stats["sd"],
        C=np.load(enc / "C.npy"), blocks=blocks, block_chrom=block_chrom,
        auto_blocks=np.flatnonzero(block_chrom != "X"), obs_names=obs_names, meta=meta,
        donor=meta.donor_id.to_numpy(), tissue=meta.tissue.to_numpy(), splits=splits,
        train_s=np.array(splits["train_samples"]), test_s=np.array(splits["test_samples"]),
        is_test=is_test, enc_meta=enc_meta, W=W, W_block=W_block, site_probes=site_probes,
        Z_full=np.load(enc / "Z_full.npy") if (enc / "Z_full.npy").exists() else None,
    )


# --- donor matching ------------------------------------------------------------------------------
def retrieval(rep: np.ndarray, donor: np.ndarray, query_idx: np.ndarray, gallery_idx: np.ndarray,
              n_perm: int = 0, seed: int = 0) -> dict:
    """Cosine retrieval of the same donor's sample among gallery samples. rep: samples x features.

    The result keeps each query's rank and donor ("ranks", "query_donor"), so two representations can
    be compared query by query with paired_bootstrap. With n_perm > 0 it adds "perm_p_top1", a one-sided
    permutation p-value against chance: gallery donor labels are shuffled n_perm times and top-1
    recomputed on the same similarity matrix (docs/PREREG.md section 2)."""
    nan = {"top1": float("nan"), "top5": float("nan"), "mrr": float("nan"),
           "chance": 1 / max(len(gallery_idx), 1), "n_query": 0, "n_gallery": len(gallery_idx),
           "ranks": [], "query_donor": []}
    gallery_donors = set(donor[gallery_idx])
    query_idx = np.array([i for i in query_idx if donor[i] in gallery_donors], dtype=int)
    if len(query_idx) == 0 or len(gallery_idx) == 0:
        return nan
    q, g = rep[query_idx].astype(np.float64), rep[gallery_idx].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    sim, q_donor, g_donor = q @ g.T, donor[query_idx], donor[gallery_idx]
    rank = rank_of_true(sim, q_donor, g_donor)
    out = {**ranks_to_metrics(rank, len(gallery_idx)), "ranks": rank.tolist(), "query_donor": q_donor.tolist()}
    if n_perm > 0:
        rng = np.random.default_rng(seed)
        obs = int((rank <= 1).sum())
        hits = sum(int((rank_of_true(sim, q_donor, rng.permutation(g_donor)) <= 1).sum()) >= obs
                   for _ in range(n_perm))
        out["perm_p_top1"] = (1 + hits) / (1 + n_perm)
        out["n_perm"] = n_perm
    return out


def paired_bootstrap(a: dict, b: dict, n_boot: int = 10_000, seed: int = 0, metric: str = "top1") -> dict:
    """Difference a - b in top-1 (or MRR) between two retrieval results over the same queries, with a
    95% percentile CI from resampling query donors (docs/PREREG.md section 2). Queries are aligned by
    donor; only donors present in both enter."""
    ra = dict(zip(a.get("query_donor", []), a.get("ranks", [])))
    rb = dict(zip(b.get("query_donor", []), b.get("ranks", [])))
    shared = sorted(set(ra) & set(rb))
    if not shared:
        return {"metric": metric, "diff": float("nan"), "ci95": [float("nan"), float("nan")], "n_query": 0}
    xa, xb = np.array([ra[d] for d in shared], float), np.array([rb[d] for d in shared], float)
    score = (lambda r: (r <= 1).astype(float)) if metric == "top1" else (lambda r: 1 / r)
    delta = score(xa) - score(xb)
    idx = np.random.default_rng(seed).integers(0, len(delta), size=(n_boot, len(delta)))
    boot = delta[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"metric": metric, "diff": float(delta.mean()), "ci95": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0), "n_query": len(shared), "n_boot": n_boot}


def rank_of_true(sim: np.ndarray, query_donor: np.ndarray, gallery_donor: np.ndarray) -> np.ndarray:
    """Rank (1 = top) of the best-ranked gallery item from the query's donor, per query row."""
    same = query_donor[:, None] == gallery_donor[None, :]
    best_true = np.where(same, sim, -np.inf).max(axis=1)
    return 1 + (np.where(same, -np.inf, sim) > best_true[:, None]).sum(axis=1)


def ranks_to_metrics(rank: np.ndarray, n_gallery: int) -> dict:
    return {"top1": float((rank <= 1).mean()), "top5": float((rank <= 5).mean()),
            "mrr": float((1 / rank).mean()), "chance": 1 / n_gallery,
            "n_query": int(len(rank)), "n_gallery": int(n_gallery)}


def lung_colon_sets(run: Run) -> dict[str, np.ndarray]:
    """Queries are test samples of lung+colon donors; galleries are all test samples of the other tissue."""
    both = sorted(run.splits["lung_colon_donors"])
    lung = np.flatnonzero(run.is_test & (run.tissue == LUNG))
    colon = np.flatnonzero(run.is_test & (run.tissue == COLON))
    return {"lung": lung, "colon": colon,
            "lung_q": lung[np.isin(run.donor[lung], both)], "colon_q": colon[np.isin(run.donor[colon], both)]}


def donor_matching(run: Run, rep: np.ndarray, blocks: np.ndarray | None = None, n_perm: int = 0,
                   seed: int = 0) -> dict:
    """Lung->colon and colon->lung matching. rep is samples x blocks x d (flattened over `blocks`,
    default the autosomal blocks) or samples x features."""
    if rep.ndim == 3:
        rep = rep[:, run.auto_blocks if blocks is None else blocks]
    rep = rep.reshape(len(rep), -1)
    s = lung_colon_sets(run)
    return {"lung_to_colon": retrieval(rep, run.donor, s["lung_q"], s["colon"], n_perm, seed),
            "colon_to_lung": retrieval(rep, run.donor, s["colon_q"], s["lung"], n_perm, seed)}


def tissue_centred(run: Run, Zs: np.ndarray) -> np.ndarray:
    """Zs minus each tissue's mean over the train samples (no test donor enters the means)."""
    out = Zs.copy()
    for t in np.unique(run.tissue):
        ref = run.train_s[run.tissue[run.train_s] == t]
        if len(ref):
            out[run.tissue == t] -= Zs[ref].mean(0)
    return out


# --- mQTL enrichment -----------------------------------------------------------------------------
def site_scores(run: Run, R: np.ndarray, samples: np.ndarray, blocks: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """g[s, i] = w_i . (r[s, B] * sd) for site i in block B (residual unstandardised first).
    Returns (W rows used, per-site variance of g over samples, per-site mean |g|)."""
    blocks = run.auto_blocks if blocks is None else blocks
    rows_all, var_all, abs_all = [], [], []
    for b in blocks:
        rows = np.flatnonzero(run.W_block == b)
        g = (R[samples, b] * run.sd).astype(np.float32) @ np.asarray(run.W[rows], dtype=np.float32).T
        rows_all.append(rows)
        var_all.append(g.var(0))
        abs_all.append(np.abs(g).mean(0))
    return np.concatenate(rows_all), np.concatenate(var_all), np.concatenate(abs_all)


def site_g(run: Run, R: np.ndarray, samples: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """g for selected W rows (any blocks): samples x len(rows)."""
    out = np.empty((len(samples), len(rows)), dtype=np.float32)
    for b in np.unique(run.W_block[rows]):
        sel = np.flatnonzero(run.W_block[rows] == b)
        w = np.asarray(run.W[rows[sel]], dtype=np.float32)
        out[:, sel] = (R[samples, b] * run.sd).astype(np.float32) @ w.T
    return out


def percentile_within_block(score: np.ndarray, block_of: np.ndarray) -> np.ndarray:
    pct = np.empty(len(score))
    for b in np.unique(block_of):
        m = block_of == b
        pct[m] = (rankdata(score[m]) - 0.5) / m.sum()
    return pct


def enrichment(pct: np.ndarray, is_set: np.ndarray, top: float = 0.95) -> dict:
    """Mean percentile in vs out of the set, one-sided Mann-Whitney, and Fisher test for the top 5%."""
    out = {"n_set": int(is_set.sum()), "n_other": int((~is_set).sum())}
    if out["n_set"] == 0 or out["n_other"] == 0:
        return {**out, "mean_pct_set": float("nan"), "mean_pct_other": float("nan"),
                "mannwhitney_p": float("nan"), "top5_odds_ratio": float("nan"), "top5_fisher_p": float("nan")}
    hi = pct >= top
    table = [[int((hi & is_set).sum()), int((hi & ~is_set).sum())],
             [int((~hi & is_set).sum()), int((~hi & ~is_set).sum())]]
    odds, fisher_p = fisher_exact(table, alternative="greater")
    return {**out, "mean_pct_set": float(pct[is_set].mean()), "mean_pct_other": float(pct[~is_set].mean()),
            "mannwhitney_p": float(mannwhitneyu(pct[is_set], pct[~is_set], alternative="greater").pvalue),
            "top5_odds_ratio": float(odds), "top5_fisher_p": float(fisher_p), "top5_table": table}


def read_set(path: Path) -> set[str]:
    return set(path.read_text().split()) if path.exists() else set()


# --- outputs -------------------------------------------------------------------------------------
def write_outputs(out_dir: Path, result: dict, rows: list[dict] | None = None, csv_name: str = "summary.csv") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))
    if rows:
        with open(out_dir / csv_name, "w", newline="") as f:
            # Rows from different sections can carry different extra keys (e.g. 05's n_blocks), so the
            # header is the union of keys in first-seen order and missing cells are left blank.
            fields = list(dict.fromkeys(k for r in rows for k in r))
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def fmt(x: float, spec: str = ".3f") -> str:
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)
