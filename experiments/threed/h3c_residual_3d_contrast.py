"""Does the model's leftover error line up with 3D structure, beyond linear distance?

Re-analyses the cached residuals from h3b using a contrast statistic instead of a
rank correlation. Pairwise residual correlations estimated from ~100 samples are
mostly noise, so a Spearman over a million pairs drowns any real effect. Binning
by distance and comparing group means recovers it.

Within each log-distance stratum, pairs are split by whether their Hi-C contact
is above or below the stratum median, and by whether both sites sit in the same
A/B compartment. A positive gap means the model errs together at 3D-proximal
sites even after linear distance is accounted for, which is the headroom that
would justify training an explicit 3D prior.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h3_attention_vs_contacts import fetch_contacts, observed_over_expected

ROOT = Path(__file__).resolve().parents[2]
RES = 25_000


def bootstrap_gap(a, b, n_boot=2000, seed=0):
    """Difference in means with a bootstrap CI, since pair correlations are noisy."""
    rng = np.random.default_rng(seed)
    obs = a.mean() - b.mean()
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = (rng.choice(a, len(a), replace=True).mean()
                    - rng.choice(b, len(b), replace=True).mean())
    return obs, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def analyse(run_dir, CHROM, START, END, min_sds=(0.03, 0.05), quiet=False,
            annotation="pool5"):
    rows = []
    d = np.load(Path(run_dir) / "residuals.npz")
    resid, truth, pos = d["resid"], d["truth"], d["pos"]

    for min_sd in min_sds:
        keep = truth.std(axis=0) >= min_sd
        if keep.sum() < 60:
            continue
        r = resid[:, keep] - resid[:, keep].mean(axis=0, keepdims=True)
        p = pos[keep]
        corr = np.corrcoef(r.T)
        iu = np.triu_indices(corr.shape[0], k=1)
        rc, dist = corr[iu], np.abs(p[iu[0]] - p[iu[1]]).astype(float)

        mat = fetch_contacts(CHROM, START, END, RES)
        oe = observed_over_expected(mat)
        bins = ((p - START) // RES).astype(int)
        ok = (bins >= 0) & (bins < oe.shape[0])
        pair_ok = ok[iu[0]] & ok[iu[1]]
        contact = np.full(len(rc), np.nan)
        contact[pair_ok] = oe[bins[iu[0]][pair_ok], bins[iu[1]][pair_ok]]

        ann_p = ROOT / f"data/3d/annotations/{annotation}.mmap"
        meta = json.loads(ann_p.with_suffix(".json").read_text())
        ann = np.memmap(ann_p, dtype="float32", mode="r", shape=tuple(meta["shape"]))
        from cpgpt import download_cpgpt
        from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
        res_ = download_cpgpt(model="small", species="human")
        emb = DNALLMEmbedder(dependencies_dir=str(res_.dependencies_path))
        index = emb.ensembl_metadata_dict["homo_sapiens"][
            "nucleotide-transformer-v2-500m-multi-species"][2001]
        eig = np.array([ann[index[f"{CHROM}:{int(x)}"], 0] for x in p])
        same_comp = np.sign(eig[iu[0]]) == np.sign(eig[iu[1]])

        if not quiet:
            print("=" * 72)
            print(f"sites with SD >= {min_sd}: {keep.sum()}   pairs: {len(rc):,}")
            print("=" * 72)

        valid = np.isfinite(rc) & np.isfinite(dist) & (dist > 0)
        ld = np.log10(dist[valid])
        edges = np.quantile(ld, np.linspace(0, 1, 6))
        hi_all, lo_all, sc_all, dc_all = [], [], [], []
        if not quiet:
            print(f"{'distance band':>22}{'n':>9}{'hi-contact':>12}{'lo-contact':>12}{'gap':>9}")
        for a_, b_ in zip(edges[:-1], edges[1:]):
            m = valid.copy()
            m[valid] = (ld >= a_) & (ld < b_)
            mm = m & np.isfinite(contact)
            if mm.sum() < 200:
                continue
            med = np.nanmedian(contact[mm])
            hi = rc[mm & (contact > med)]
            lo = rc[mm & (contact <= med)]
            if len(hi) < 50 or len(lo) < 50:
                continue
            hi_all.append(hi); lo_all.append(lo)
            sc_all.append(rc[mm & same_comp]); dc_all.append(rc[mm & ~same_comp])
            if not quiet:
                print(f"{10**a_/1000:>9.0f}-{10**b_/1000:<8.0f}kb{mm.sum():>9,}"
                      f"{hi.mean():>12.4f}{lo.mean():>12.4f}{hi.mean()-lo.mean():>+9.4f}")

        if hi_all:
            hi, lo = np.concatenate(hi_all), np.concatenate(lo_all)
            gap, lo_ci, hi_ci = bootstrap_gap(hi, lo)
            print(f"\npooled contact gap (high minus low, distance-matched): "
                  f"{gap:+.4f}  95% CI [{lo_ci:+.4f}, {hi_ci:+.4f}]")
            sc, dc = np.concatenate(sc_all), np.concatenate(dc_all)
            g2, l2, h2 = bootstrap_gap(sc, dc)
            print(f"pooled compartment gap (same minus different):          "
                  f"{g2:+.4f}  95% CI [{l2:+.4f}, {h2:+.4f}]")
            rows.append({"chrom": CHROM, "start": START, "min_sd": min_sd,
                         "n_sites": int(keep.sum()), "n_pairs": len(rc),
                         "contact_gap": gap, "contact_lo": lo_ci, "contact_hi": hi_ci,
                         "comp_gap": g2, "comp_lo": l2, "comp_hi": h2})
            if not quiet:
                print(f"\ncontact effect excludes zero: {'YES' if lo_ci > 0 else 'no'}   "
                      f"compartment effect excludes zero: {'YES' if l2 > 0 else 'no'}\n")
    return rows


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "data/3d/h3b_GSE40279_chr1"))
    ap.add_argument("--chrom", default="1")
    ap.add_argument("--start", type=int, default=30_000_000)
    ap.add_argument("--end", type=int, default=50_000_000)
    a = ap.parse_args()
    analyse(a.run_dir, a.chrom, a.start, a.end)


if __name__ == "__main__":
    main()
