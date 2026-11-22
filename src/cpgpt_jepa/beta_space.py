"""Per-site vectors w_i for representations other than frozen CpGPT, in the stage-2 layout
(W.mmap fp16 + W_index.npy + site_probe_ids.npy + site_embedding_rows.npy), so 04/07 can run the mQTL test:

- beta space (03/04 --beta-space): each block's betas PCA-reduced to k dimensions, fit on the train samples;
  w_i is site i's PCA loading, so g = w_i . r is the residual's contribution to site i's beta.
- Phase B (09): a post-hoc ridge decoder from the block embedding to the block's betas, fit on train samples.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cpgpt_jepa.blocks import open_processed
from cpgpt_jepa.predictor import make_splits


def block_offsets(W_index: np.ndarray, n_blocks: int) -> np.ndarray:
    """Stage 2 writes W rows block by block, so block b's sites are rows offsets[b]:offsets[b + 1]."""
    return np.searchsorted(W_index[:, 1], np.arange(n_blocks + 1))


def site_betas(enc: Path) -> np.ndarray:
    """Betas of the encodings' samples at the block sites, in W-row order: samples x n_sites_total."""
    enc_meta = json.loads((enc / "encode_meta.json").read_text())
    X_mm, _, _, _ = open_processed(Path(enc_meta["dataset_dir"]))
    sample_rows = np.array(enc_meta.get("sample_idx", np.arange(X_mm.shape[0])))
    cols = np.load(enc / "W_index.npy")[:, 0]
    return np.asarray(X_mm[sample_rows])[:, cols]


def fill_with_train_mean(Y: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Missing betas set to the site's train-sample mean (0 if never observed in train)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> nan, set to 0 below
        mean = np.nan_to_num(np.nanmean(Y[train], axis=0))
    return np.where(np.isnan(Y), mean, Y), mean


def write_site_vectors(out: Path, W: np.ndarray, source_enc: Path, kind: str) -> None:
    """W (n_sites_total x d) as W.mmap fp16, with the source encodings' site index files copied alongside."""
    mm = np.memmap(out / "W.mmap", dtype=np.float16, mode="w+", shape=W.shape)
    mm[:] = W.astype(np.float16)
    mm.flush()
    for name in ("W_index.npy", "site_probe_ids.npy", "site_embedding_rows.npy"):
        np.save(out / name, np.load(source_enc / name))
    meta_path = out / "encode_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"W_shape": list(W.shape), "W_dtype": "float16", "W_kind": kind})
    meta_path.write_text(json.dumps(meta, indent=2))


def build_beta_encodings(frozen_enc: Path, out: Path, metadata: str, d: int | None = None, seed: int = 0,
                         log=print) -> Path:
    """Beta-space control: per block, PCA of the betas fit on the train samples (same splits as stage 3),
    k = min(d, n_train - 1, smallest block) components for every block. Writes the encodings layout to `out`."""
    frozen_enc, out = Path(frozen_enc), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    enc_meta = json.loads((frozen_enc / "encode_meta.json").read_text())
    blocks = np.load(frozen_enc / "blocks.npy", allow_pickle=True)  # written by stage 2 (object array of dicts)
    obs_names = np.load(frozen_enc / "obs_names.npy", allow_pickle=True)  # written by stage 2
    C = np.load(frozen_enc / "C.npy")
    W_index = np.load(frozen_enc / "W_index.npy")
    offsets = block_offsets(W_index, len(blocks))
    meta = pd.read_csv(metadata, dtype=str).set_index("sample_id").loc[list(obs_names)]
    chrom = np.array([b["chrom"] for b in blocks])
    splits = make_splits(meta.donor_id.to_numpy(), meta.tissue.to_numpy(), len(blocks), seed=seed,
                         eval_only_blocks=np.flatnonzero(chrom == "X"))
    train = np.array(splits["train_samples"])
    betas = site_betas(frozen_enc)
    k = min(d or C.shape[1], len(train) - 1, int(np.diff(offsets).min()))
    Z = np.zeros((len(obs_names), len(blocks), k), dtype=np.float32)
    W = np.zeros((len(W_index), k), dtype=np.float32)
    explained = []
    for b in range(len(blocks)):
        sl = slice(offsets[b], offsets[b + 1])
        Y, mean = fill_with_train_mean(betas[:, sl].astype(np.float64), train)
        Yc = Y - mean
        _, S, Vt = np.linalg.svd(Yc[train], full_matrices=False)
        V = Vt[:k].T  # sites x k
        Z[:, b] = Yc @ V
        W[sl] = V
        explained.append(float((S[:k] ** 2).sum() / max((S ** 2).sum(), 1e-12)))
    np.save(out / "Z.npy", Z)
    np.save(out / "C.npy", C)
    np.save(out / "blocks.npy", blocks, allow_pickle=True)
    np.save(out / "obs_names.npy", obs_names, allow_pickle=True)
    (out / "encode_meta.json").write_text(json.dumps({
        **{key: enc_meta[key] for key in ("window_bp", "dataset_dir", "sample_idx", "n_x_blocks", "dna_llm",
                                          "dna_context_len") if key in enc_meta},
        "model": "beta_pca", "d": k, "n_samples": len(obs_names), "n_blocks": len(blocks),
        "frozen_encodings": str(frozen_enc), "pca_fit": "train samples of the stage-3 split",
        "mean_explained_variance_train": float(np.mean(explained))}, indent=2))
    write_site_vectors(out, W, frozen_enc, "PCA loadings (beta units)")
    log(f"beta-space encodings: {len(blocks)} blocks, k = {k} components, mean train variance explained "
        f"{np.mean(explained):.3f} -> {out}")
    return out


def ridge_decoder(Z: np.ndarray, betas: np.ndarray, offsets: np.ndarray, train: np.ndarray,
                  test: np.ndarray) -> tuple[np.ndarray, float]:
    """Per block, RidgeCV from the (raw) block embedding to the block's betas, fit on train samples.
    Returns W (n_sites_total x d) and the mean test-sample R^2 over blocks."""
    from sklearn.linear_model import RidgeCV

    W = np.zeros((betas.shape[1], Z.shape[2]), dtype=np.float32)
    r2 = []
    for b in range(Z.shape[1]):
        sl = slice(offsets[b], offsets[b + 1])
        Y, _ = fill_with_train_mean(betas[:, sl].astype(np.float64), train)
        reg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Z[train, b], Y[train])
        W[sl] = reg.coef_
        if len(test):
            pred = reg.predict(Z[test, b])
            ss_res = ((Y[test] - pred) ** 2).sum()
            ss_tot = ((Y[test] - Y[test].mean(0)) ** 2).sum()
            r2.append(1 - ss_res / max(ss_tot, 1e-12))
    return W, float(np.mean(r2)) if r2 else float("nan")
