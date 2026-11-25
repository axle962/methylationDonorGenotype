"""GTEx covariates: per-tissue covariate files (genotype PCs, sex, PEER factors) and design matrices."""

from pathlib import Path

import numpy as np
import pandas as pd

from cpgpt_jepa.evaluation import TISSUE_STEM


def tissue_covariates(egtex_dir: str | Path, tissue: str) -> pd.DataFrame | None:
    """data/egtex/metadata/<Tissue>.covariates.txt as donors x covariates (PC1-5, pcr, platform, sex,
    PEER1-20), or None if the file is missing."""
    path = Path(egtex_dir) / "metadata" / f"{TISSUE_STEM.get(tissue, tissue)}.covariates.txt"
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t", index_col=0).T.apply(pd.to_numeric, errors="coerce")


def genotype_pcs(egtex_dir: str | Path, donors: np.ndarray, n_pcs: int = 5) -> np.ndarray:
    """Genotype PC1..n per donor, taken from whichever tissue file lists the donor (NaN if none)."""
    table = {}
    for tissue in TISSUE_STEM:
        cov = tissue_covariates(egtex_dir, tissue)
        if cov is None:
            continue
        for dn, row in cov[[f"PC{i}" for i in range(1, n_pcs + 1)]].iterrows():
            table.setdefault(dn, row.to_numpy(dtype=float))
    return np.stack([table.get(dn, np.full(n_pcs, np.nan)) for dn in donors])


def peer_factors(egtex_dir: str | Path, tissue: str, donors: np.ndarray, n_peer: int = 20) -> np.ndarray:
    """PEER1..n for each donor from the tissue's covariate file; NaN rows for donors it does not list."""
    cov = tissue_covariates(egtex_dir, tissue)
    cols = [f"PEER{i}" for i in range(1, n_peer + 1)]
    if cov is None:
        return np.full((len(donors), n_peer), np.nan)
    cols = [c for c in cols if c in cov.columns]
    return cov.reindex(donors)[cols].to_numpy(dtype=float)


def one_hot(values: np.ndarray) -> np.ndarray:
    """Treatment-coded dummies (first level dropped); empty if only one level."""
    levels = sorted(set(values))
    return np.stack([values == lv for lv in levels[1:]], axis=1).astype(float) if len(levels) > 1 \
        else np.zeros((len(values), 0))


def numeric(values: np.ndarray) -> np.ndarray:
    """Column with missing values set to the mean."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, copy=True)
    v[np.isnan(v)] = np.nanmean(v) if np.isfinite(v).any() else 0.0
    return v[:, None]


def categorical_or_numeric(values: np.ndarray) -> np.ndarray:
    """numeric() if every non-missing value parses as a number, else one_hot() of the labels (missing as its own
    level). Sex in metadata.csv may be text ("Male"/"Female"), which numeric() would turn into a zero column."""
    s = pd.Series(values)
    present = s.notna()
    if pd.to_numeric(s[present], errors="coerce").notna().all():
        return numeric(values)
    return one_hot(s.fillna("NA").astype(str).to_numpy())


def residualise(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Y minus its least-squares fit on [1, X] (minimum-norm solution, so collinear dummies are fine)."""
    design = np.column_stack([np.ones(len(Y)), X])
    beta, *_ = np.linalg.lstsq(design, Y, rcond=None)
    return Y - design @ beta
