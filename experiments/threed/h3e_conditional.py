"""Is the contact effect independent of compartment, or just a proxy for it?

Both replicated across windows, but high-contact pairs usually share a
compartment, so one effect can borrow the other's signal. This conditions each
on the other: within same-compartment pairs only, does contact strength still
separate residual correlation, and vice versa.

Whichever survives conditioning is the one carrying real information, and that
decides whether H2 has anything to offer beyond H1.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h3_attention_vs_contacts import fetch_contacts, observed_over_expected
from h3c_residual_3d_contrast import bootstrap_gap
from h3d_replication import WINDOWS

ROOT = Path(__file__).resolve().parents[2]
RES = 25_000


def window_pairs(chrom, start, end, min_sd, annotation="pool5"):
    run = ROOT / f"data/3d/h3b_GSE40279_chr{chrom}_{start//1_000_000}"
    if not (run / "residuals.npz").exists():
        return None
    d = np.load(run / "residuals.npz")
    resid, truth, pos = d["resid"], d["truth"], d["pos"]
    keep = truth.std(axis=0) >= min_sd
    if keep.sum() < 60:
        return None
    r = resid[:, keep] - resid[:, keep].mean(axis=0, keepdims=True)
    p = pos[keep]
    corr = np.corrcoef(r.T)
    iu = np.triu_indices(corr.shape[0], k=1)
    rc = corr[iu]
    dist = np.abs(p[iu[0]] - p[iu[1]]).astype(float)

    oe = observed_over_expected(fetch_contacts(chrom, start, end, RES))
    bins = ((p - start) // RES).astype(int)
    ok = (bins >= 0) & (bins < oe.shape[0])
    pair_ok = ok[iu[0]] & ok[iu[1]]
    contact = np.full(len(rc), np.nan)
    contact[pair_ok] = oe[bins[iu[0]][pair_ok], bins[iu[1]][pair_ok]]

    meta = json.loads((ROOT / f"data/3d/annotations/{annotation}.json").read_text())
    ann = np.memmap(ROOT / f"data/3d/annotations/{annotation}.mmap", dtype="float32",
                    mode="r", shape=tuple(meta["shape"]))
    from cpgpt import download_cpgpt
    from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
    res_ = download_cpgpt(model="small", species="human")
    emb = DNALLMEmbedder(dependencies_dir=str(res_.dependencies_path))
    index = emb.ensembl_metadata_dict["homo_sapiens"][
        "nucleotide-transformer-v2-500m-multi-species"][2001]
    eig = np.array([ann[index[f"{chrom}:{int(x)}"], 0] for x in p])
    same = np.sign(eig[iu[0]]) == np.sign(eig[iu[1]])
    return rc, dist, contact, same


def banded_gap(rc, dist, split_hi, split_lo, n_bands=5):
    """Mean difference between two pair groups, matched on log distance."""
    hi_all, lo_all = [], []
    v = np.isfinite(rc) & np.isfinite(dist) & (dist > 0)
    ld = np.log10(dist[v])
    edges = np.quantile(ld, np.linspace(0, 1, n_bands + 1))
    for a, b in zip(edges[:-1], edges[1:]):
        band = v.copy()
        band[v] = (ld >= a) & (ld < b)
        h, l = rc[band & split_hi], rc[band & split_lo]
        h, l = h[np.isfinite(h)], l[np.isfinite(l)]
        if len(h) >= 40 and len(l) >= 40:
            hi_all.append(h)
            lo_all.append(l)
    if not hi_all:
        return None
    return np.concatenate(hi_all), np.concatenate(lo_all)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", default="pool5")
    args = ap.parse_args()
    for min_sd in (0.03, 0.05):
        pools = {k: ([], []) for k in
                 ("contact|same", "contact|diff", "comp|hi-contact", "comp|lo-contact")}
        for chrom, start, end in WINDOWS:
            got = window_pairs(chrom, start, end, min_sd, args.annotation)
            if got is None:
                continue
            rc, dist, contact, same = got
            fin = np.isfinite(contact)
            med = np.nanmedian(contact)
            hi_c, lo_c = fin & (contact > med), fin & (contact <= med)

            for key, (a_mask, b_mask) in {
                "contact|same": (hi_c & same, lo_c & same),
                "contact|diff": (hi_c & ~same, lo_c & ~same),
                "comp|hi-contact": (hi_c & same, hi_c & ~same),
                "comp|lo-contact": (lo_c & same, lo_c & ~same),
            }.items():
                out = banded_gap(rc, dist, a_mask, b_mask)
                if out:
                    pools[key][0].append(out[0])
                    pools[key][1].append(out[1])

        print("=" * 70)
        print(f"CONDITIONAL TEST, sites varying >= {min_sd}")
        print("=" * 70)
        for key, (a_list, b_list) in pools.items():
            if not a_list:
                continue
            a, b = np.concatenate(a_list), np.concatenate(b_list)
            gap, lo, hi = bootstrap_gap(a, b, n_boot=2000)
            sig = "YES" if lo > 0 else "no "
            label = {
                "contact|same": "contact effect, within same compartment",
                "contact|diff": "contact effect, within different compartment",
                "comp|hi-contact": "compartment effect, among high-contact pairs",
                "comp|lo-contact": "compartment effect, among low-contact pairs",
            }[key]
            print(f"  {label:<48}{gap:+.4f} [{lo:+.4f},{hi:+.4f}]  excludes zero: {sig}")
        print()


if __name__ == "__main__":
    main()
