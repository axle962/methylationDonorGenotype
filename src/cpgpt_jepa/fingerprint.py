"""Helpers for the genotype-fingerprint analyses (scripts/11_reconstruction.py, scripts/12_fingerprint.py).

Sites are W rows (one per block site, encodings layout). A reconstruction is y_hat[s, i] = z . w_i with z the
sample's block embedding (Z[s, block(i)]) or its whole-sample embedding (Z_full[s]); w_i is the run's per-site
vector from W.mmap. All statistics built on these are rank- or cosine-based on tissue-centred values, so a
per-site monotone map (M-value vs beta, a missing intercept) does not change them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cpgpt_jepa.evaluation import Run, retrieval


def common_snp_probes(path: str | Path, maf: float = 0.01) -> set[str]:
    """Probes overlapping a common SNP in Zhou et al.'s EPIC hg38 common-SNP table (InfiniumAnnotationV1,
    Anno/EPIC/EPIC.hg38.commonsnp.tsv.gz): minor-allele frequency >= maf globally or in Europeans (GTEx is
    mostly of European ancestry). Any overlap counts, wherever the SNP sits in the probe."""
    t = pd.read_csv(path, sep="\t", usecols=["probeID", "GAF", "EUR_AF"])
    gmaf = np.minimum(t.GAF, 1 - t.GAF)
    emaf = np.minimum(t.EUR_AF, 1 - t.EUR_AF)
    return set(t.probeID[(gmaf >= maf) | (emaf >= maf)])


def strongest_mqtl_order(egtex: Path) -> list[str]:
    """CpG ids of every eGTEx cis-mQTL table, strongest first: q-value, then |slope| (stage 07, part 3)."""
    tables = [pd.read_csv(f, sep="\t", usecols=["cpg_id", "slope", "qval"])
              for f in sorted((egtex / "mqtl").glob("*.regular.perm.fdr.txt"))]
    strong = pd.concat(tables)
    strong["abs_slope"] = strong.slope.abs()
    strong = strong.sort_values(["qval", "abs_slope"], ascending=[True, False]).drop_duplicates("cpg_id")
    return strong.cpg_id.tolist()


def reconstruct(run: Run, rows: np.ndarray, full: bool = False) -> np.ndarray:
    """y_hat (samples x len(rows)) from the raw block embeddings, or from Z_full when full=True."""
    out = np.empty((len(run.obs_names), len(rows)), dtype=np.float32)
    if full:
        w = np.asarray(run.W[rows], dtype=np.float32)
        return (run.Z_full.astype(np.float32) @ w.T).astype(np.float32)
    for b in np.unique(run.W_block[rows]):
        sel = np.flatnonzero(run.W_block[rows] == b)
        out[:, sel] = run.Z[:, b].astype(np.float32) @ np.asarray(run.W[rows[sel]], dtype=np.float32).T
    return out


def tissue_centre(x: np.ndarray, tissue: np.ndarray, train_s: np.ndarray) -> np.ndarray:
    """Subtract each tissue's train-sample mean (no test donor enters the means)."""
    out = x.copy()
    for t in np.unique(tissue):
        tr = train_s[tissue[train_s] == t]
        if len(tr):
            out[tissue == t] -= x[tr].mean(0)
    return out


def pair_matching(rep: np.ndarray, donor: np.ndarray, is_test: np.ndarray, tissue: np.ndarray,
                  t_query: str, t_gallery: str) -> dict:
    """Retrieval of the same donor's t_gallery sample from each t_query test sample; gallery = all t_gallery
    test samples. Queries are restricted to donors that have a t_gallery test sample."""
    gal = np.flatnonzero(is_test & (tissue == t_gallery))
    q = np.flatnonzero(is_test & (tissue == t_query))
    q = q[np.isin(donor[q], donor[gal])]
    return retrieval(rep, donor, q, gal)
