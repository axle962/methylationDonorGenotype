"""H3: does pretrained CpGPT attention already recapitulate 3D genome contacts?

Zero training required. Restricts the input to one genomic window so the sampled
sites sit kilobases apart rather than the ~310 kb apart that genome-wide sampling
gives, extracts per-head attention from every layer, and correlates it against
real Hi-C contacts from the same window.

The test is a **distance-controlled** one. Attention and contact frequency both
fall off steeply with linear genomic distance, so a raw correlation between them
is nearly guaranteed and means nothing. Contacts are converted to
observed/expected against a distance-matched background, and correlations are
computed within log-distance strata and averaged, so what survives is structure
beyond linear proximity.

Hi-C is streamed from ENCODE over HTTP range requests; the .hic file is ~95 GB
and is never downloaded. It is GRCh38/hg38 native, so no liftOver is involved.

Usage:
    python experiments/threed/h3_attention_vs_contacts.py --chrom 1 \
        --start 30000000 --end 50000000
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cpgpt import download_cpgpt
from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer

ROOT = Path(__file__).resolve().parents[2]
HIC_URL = ("https://www.encodeproject.org/files/ENCFF916GWS/@@download/"
           "ENCFF916GWS.hic")  # GM12878 intact Hi-C, GRCh38


def log(m):
    print(m, flush=True)


def fetch_contacts(chrom, start, end, resolution, norm="VC", retries=3):
    """Contact matrix for one window, streamed from ENCODE and cached on disk.

    The stream is plain HTTP range requests, so a network blip raises mid-read.
    That once silently dropped a whole window from the replication, hence the
    retries and the cache.
    """
    cache = (Path(__file__).resolve().parents[2] / "data/3d/hic_cache"
             / f"chr{chrom}_{start}_{end}_{resolution}_{norm}.npy")
    if cache.exists():
        return np.load(cache)

    import hicstraw

    for attempt in range(1, retries + 1):
        try:
            hic = hicstraw.HiCFile(HIC_URL)
            mzd = hic.getMatrixZoomData(f"chr{chrom}", f"chr{chrom}", "observed", norm,
                                        "BP", resolution)
            recs = mzd.getRecords(start, end, start, end)
            break
        except Exception as e:  # noqa: BLE001
            log(f"[hic] attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(10 * attempt)
    log(f"[hic] {len(recs)} records, {norm} @ {resolution} bp")
    n = (end - start) // resolution + 1
    mat = np.zeros((n, n), dtype="float64")
    for r in recs:
        i, j = (r.binX - start) // resolution, (r.binY - start) // resolution
        if 0 <= i < n and 0 <= j < n and np.isfinite(r.counts):
            mat[i, j] = mat[j, i] = r.counts
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, mat)
    return mat


def observed_over_expected(mat):
    """Divide each diagonal by its own mean, removing the distance decay."""
    n = mat.shape[0]
    oe = np.full_like(mat, np.nan)
    for d in range(n):
        diag = np.diagonal(mat, d)
        m = diag[diag > 0].mean() if (diag > 0).any() else np.nan
        if not np.isfinite(m) or m == 0:
            continue
        idx = np.arange(n - d)
        oe[idx, idx + d] = oe[idx + d, idx] = diag / m
    return oe


def stratified_spearman(x, y, dist, n_strata=10, min_per_stratum=50):
    """Spearman within log-distance strata, averaged. Controls for linear distance."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(dist) & (dist > 0)
    if ok.sum() < min_per_stratum * 2:
        return np.nan, 0
    x, y, d = x[ok], y[ok], np.log10(dist[ok])
    edges = np.quantile(d, np.linspace(0, 1, n_strata + 1))
    rhos, weights = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (d >= a) & (d < b)
        if m.sum() < min_per_stratum:
            continue
        if np.std(x[m]) == 0 or np.std(y[m]) == 0:
            continue
        r = spearmanr(x[m], y[m]).statistic
        if np.isfinite(r):
            rhos.append(r)
            weights.append(m.sum())
    if not rhos:
        return np.nan, 0
    return float(np.average(rhos, weights=weights)), int(np.sum(weights))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", default="1")
    ap.add_argument("--start", type=int, default=30_000_000)
    ap.add_argument("--end", type=int, default=50_000_000)
    ap.add_argument("--resolution", type=int, default=25_000)
    ap.add_argument("--gse", default="GSE182215")
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--max-sites", type=int, default=3000)
    args = ap.parse_args()

    res = download_cpgpt(model="small", species="human")
    deps = str(res.dependencies_path)
    emb = DNALLMEmbedder(dependencies_dir=deps)
    prober = IlluminaMethylationProber(dependencies_dir=deps, embedder=emb)

    df = pd.read_feather(ROOT / f"data/geo/processed/{args.gse}_betas.arrow").set_index("GSM_ID")
    vocab = set(json.load(open(res.vocab_path))["input"])
    df = df.loc[:, df.columns.isin(vocab)]

    log("[probes] locating probes in window")
    all_probes = list(df.columns)
    locs = prober.locate_probes(all_probes, "homo_sapiens")
    keep, positions = [], []
    for probe, loc in zip(all_probes, locs):
        c, p = loc.split(":")
        if c == args.chrom and args.start <= int(p) < args.end:
            keep.append(probe)
            positions.append(int(p))
    order = np.argsort(positions)
    keep = [keep[i] for i in order]
    positions = np.array([positions[i] for i in order])
    if len(keep) > args.max_sites:
        sel = np.linspace(0, len(keep) - 1, args.max_sites).round().astype(int)
        keep = [keep[i] for i in sel]
        positions = positions[sel]
    spacing = np.diff(positions)
    log(f"[probes] {len(keep)} sites in chr{args.chrom}:{args.start}-{args.end}, "
        f"median spacing {np.median(spacing):,.0f} bp "
        f"(genome-wide sampling gives ~310,000 bp)")

    sub = df[keep].iloc[: args.n_samples].dropna(axis=1, how="any")
    keep = list(sub.columns)
    pos_map = dict(zip([k for k in sub.columns], positions[[keep.index(k) for k in sub.columns]]))
    log(f"[data] {sub.shape[0]} samples x {sub.shape[1]} complete sites")

    run = ROOT / f"data/3d/h3_chr{args.chrom}_{args.start}"
    (run / "raw").mkdir(parents=True, exist_ok=True)
    sub.reset_index().to_feather(run / "raw/b.arrow")
    CpGPTDataSaver(data_paths=str(run / "raw/b.arrow"),
                   processed_dir=str(run / "proc")).process_files(prober, emb)

    inf = CpGPTInferencer(dependencies_dir=str(ROOT / "dependencies"), data_dir=str(ROOT / "data"))
    cfg = inf.load_cpgpt_config(str(res.config_path))
    model = inf.load_cpgpt_model(cfg, model_ckpt_path=str(res.checkpoint_path),
                                 strict_load=True)
    n_layers = cfg.model.net.n_layers
    dm = CpGPTDataModule(
        predict_dir=str(run / "proc"), dependencies_dir=deps, batch_size=1, num_workers=0,
        max_length=max(5000, len(keep) + 10), dna_llm=cfg.data.dna_llm,
        dna_context_len=cfg.data.dna_context_len,
        sorting_strategy=cfg.data.sorting_strategy, pin_memory=False)

    mat = fetch_contacts(args.chrom, args.start, args.end, args.resolution)
    oe = observed_over_expected(mat)

    trainer = CpGPTTrainer(accelerator="cpu", devices=1, enable_progress_bar=False)
    rows = []
    for layer in range(n_layers):
        out = trainer.predict(
            model=model, datamodule=dm, predict_mode="attention",
            aggregate_heads="none", layer_index=layer,
            return_keys=["attention_weights", "chroms", "positions"])
        attn = out["attention_weights"].float().numpy()   # (B, H, L+1, L+1)
        site_pos = out["positions"].float().numpy()[0]
        attn = attn[:, :, 1:, 1:]                          # drop CLS row/col
        n = min(attn.shape[-1], len(site_pos))
        attn, site_pos = attn[:, :, :n, :n], site_pos[:n]

        bins = ((site_pos - args.start) // args.resolution).astype(int)
        valid = (bins >= 0) & (bins < oe.shape[0])
        iu = np.triu_indices(n, k=1)
        pair_ok = valid[iu[0]] & valid[iu[1]]
        contact = oe[bins[iu[0]][pair_ok], bins[iu[1]][pair_ok]]
        dist = np.abs(site_pos[iu[0]][pair_ok] - site_pos[iu[1]][pair_ok])

        for head in range(attn.shape[1]):
            a = attn[:, head].mean(axis=0)
            a = (a + a.T) / 2
            vals = a[iu][pair_ok]
            rho_raw = spearmanr(vals, contact,
                                nan_policy="omit").statistic if np.std(vals) else np.nan
            rho_ctrl, npairs = stratified_spearman(vals, contact, dist)
            rows.append({"layer": layer, "head": head, "raw_rho": rho_raw,
                         "dist_controlled_rho": rho_ctrl, "n_pairs": npairs})
        log(f"[layer {layer}] best distance-controlled rho "
            f"{max(r['dist_controlled_rho'] for r in rows if r['layer']==layer):+.4f}")

    out_df = pd.DataFrame(rows)
    RES = ROOT / "results"
    RES.mkdir(exist_ok=True)
    out_df.to_csv(RES / "h3_attention_vs_contacts.csv", index=False)

    # Reference: how well does linear distance alone predict contact?
    rho_dist = spearmanr(-np.log10(dist), contact, nan_policy="omit").statistic
    log("")
    log(f"{'':6}{'raw rho':>12}{'distance-controlled':>22}")
    piv = out_df.groupby("layer")[["raw_rho", "dist_controlled_rho"]].agg(["mean", "max"])
    for layer in sorted(out_df.layer.unique()):
        s = out_df[out_df.layer == layer]
        log(f"L{layer:<5}{s.raw_rho.mean():>7.4f}/{s.raw_rho.max():>6.4f}"
            f"{s.dist_controlled_rho.mean():>12.4f}/{s.dist_controlled_rho.max():>8.4f}")
    log("")
    log(f"linear distance alone vs contact: rho {rho_dist:+.4f}")
    log(f"best head, distance-controlled:   rho {out_df.dist_controlled_rho.max():+.4f} "
        f"(layer {out_df.loc[out_df.dist_controlled_rho.idxmax(),'layer']}, "
        f"head {out_df.loc[out_df.dist_controlled_rho.idxmax(),'head']})")
    log(f"[write] {RES / 'h3_attention_vs_contacts.csv'}")


if __name__ == "__main__":
    main()
