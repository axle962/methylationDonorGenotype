"""Synthetic Phase A inputs for running stages 03, 04, 06, 07 and 10 without the eGTEx data or CpGPT.

  python tests/make_fixture.py <root>

Writes, under <root>:
  egtex/       metadata.csv, geo/GSE213478_family.soft.gz (age, smoker_status), mqtl/*.regular.perm.fdr.txt,
               mqtl_sites_{Lung,ColonTransverse,any}.txt, metadata/<stem>.covariates.txt (PEER1-20)
  processed/   X.mmap, X_shape.npy, var.mmap, obs_names.npy, species.npy (stage 01 layout)
  outputs/encodings/  a stage 02 stand-in: Z = random projection of each block's betas, C, blocks, W.mmap, ...
Betas = tissue profile + age slope + donor genotype at mQTL CpGs (0/1/2 dosage, same in every tissue of the
donor) + a plate offset + noise, so donor matching, the batch check and the concordance test have signal.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
rng = np.random.default_rng(0)
egtex, proc, enc = root / "egtex", root / "processed", root / "outputs" / "encodings"
for d in (egtex / "geo", egtex / "mqtl", egtex / "metadata", proc, enc):
    d.mkdir(parents=True, exist_ok=True)

# --- samples ---------------------------------------------------------------------------------------
n_both, n_single = 40, 40
donors = [f"GTEX-D{i:03d}" for i in range(n_both + n_single)]
rows = []
for i, dn in enumerate(donors):
    tissues = ["Lung", "Colon - Transverse"] if i < n_both else [rng.choice(["Lung", "Colon - Transverse", "Whole Blood"])]
    for t in tissues:
        rows.append({"donor_id": dn, "tissue": t})
meta = pd.DataFrame(rows)
n = len(meta)
meta["sample_id"] = [f"SM-{i:05d}" for i in range(n)]
meta["gsm"] = [f"GSM{i:07d}" for i in range(n)]
sex_of = {dn: rng.choice(["male", "female"]) for dn in donors}
meta["sex"] = meta.donor_id.map(sex_of)
plates = [f"P{k}" for k in range(6)]
meta["plate"] = rng.choice(plates, size=n)
meta["sentrix_id"] = [f"20{rng.integers(0, 30):04d}" for _ in range(n)]
# a quarter of the lung+colon donors get both samples on one plate and chip (batch-sharing pairs)
for dn in donors[: n_both // 4]:
    idx = meta.index[meta.donor_id == dn]
    meta.loc[idx, ["plate", "sentrix_id"]] = meta.loc[idx[0], ["plate", "sentrix_id"]].to_numpy()
meta[["sample_id", "gsm", "donor_id", "tissue", "sex", "plate", "sentrix_id"]].to_csv(egtex / "metadata.csv", index=False)
age_of = {dn: int(rng.integers(2, 7)) * 10 for dn in donors}
with gzip.open(egtex / "geo" / "GSE213478_family.soft.gz", "wt") as f:
    for _, r in meta.iterrows():
        a = age_of[r.donor_id]
        f.write(f"^SAMPLE = {r.gsm}\n!Sample_characteristics_ch1 = sample_id: {r.sample_id}\n"
                f"!Sample_characteristics_ch1 = age: {a}-{a + 9}\n"
                f"!Sample_characteristics_ch1 = smoker_status: {rng.choice(['Never', 'Current', 'Former'])}\n")

# --- sites and betas -------------------------------------------------------------------------------
chroms, per_chrom, per_block = [1, 2, 3, 4, 5, 6], 2, 400
var_rows, probes, block_of = [], [], []
b = 0
for c in chroms:
    for k in range(per_chrom):
        start = k * 10_000_000
        for j in range(per_block):
            var_rows.append((c - 1, start + 1000 + j * 1000))
            probes.append(f"cg{len(probes):08d}")
            block_of.append(b)
        b += 1
n_sites, n_blocks = len(probes), b
probes = np.array(probes)
is_mqtl = rng.random(n_sites) < 0.3
dosage = rng.binomial(2, 0.4, size=(len(donors), n_sites))  # same in every tissue of a donor
donor_idx = meta.donor_id.map({dn: i for i, dn in enumerate(donors)}).to_numpy()
tissue_prof = {t: rng.normal(0, 1.0, n_sites) for t in ("Lung", "Colon - Transverse", "Whole Blood")}
age_slope = rng.normal(0, 0.02, n_sites)
plate_off = {p: rng.normal(0, 0.15, n_sites) for p in plates}
age = meta.donor_id.map(age_of).to_numpy(dtype=float)
M = np.stack([tissue_prof[t] for t in meta.tissue]) + age[:, None] * age_slope[None, :] \
    + np.stack([plate_off[p] for p in meta.plate]) + rng.normal(0, 0.3, (n, n_sites))
M += 0.8 * (dosage[donor_idx] - 0.8) * is_mqtl[None, :]
X = (1 / (1 + np.exp(-M))).astype(np.float32)
X[rng.random(X.shape) < 0.002] = np.nan
np.save(proc / "X_shape.npy", np.array([n, n_sites]))
X.tofile(proc / "X.mmap")
np.array(var_rows, dtype=np.int32).tofile(proc / "var.mmap")
np.save(proc / "obs_names.npy", meta.sample_id.to_numpy(dtype=object), allow_pickle=True)
np.save(proc / "species.npy", np.array("homo_sapiens", dtype=object), allow_pickle=True)
np.save(proc / "probe_ids.npy", probes)

# --- mQTL tables -----------------------------------------------------------------------------------
for stem in ("Lung", "ColonTransverse", "WholeBlood"):
    q = np.where(is_mqtl, rng.uniform(0, 0.01, n_sites), rng.uniform(0.1, 1, n_sites))
    pd.DataFrame({"cpg_id": probes, "slope": np.where(is_mqtl, 0.8, 0.0) + rng.normal(0, 0.05, n_sites),
                  "qval": q}).to_csv(egtex / "mqtl" / f"{stem}.regular.perm.fdr.txt", sep="\t", index=False)
    (egtex / f"mqtl_sites_{stem}.txt").write_text("\n".join(probes[is_mqtl]) + "\n")
(egtex / "mqtl_sites_any.txt").write_text("\n".join(probes[is_mqtl]) + "\n")
for stem in ("Lung", "ColonTransverse", "WholeBlood"):
    cov = pd.DataFrame(rng.normal(size=(25, len(donors))), columns=donors,
                       index=[f"PC{i}" for i in range(1, 6)] + [f"PEER{i}" for i in range(1, 21)])
    cov.to_csv(egtex / "metadata" / f"{stem}.covariates.txt", sep="\t")

# --- stage 02 stand-in -----------------------------------------------------------------------------
d = 16
Xf = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
Z = np.zeros((n, n_blocks, d), dtype=np.float32)
W = np.zeros((n_sites, d), dtype=np.float32)
block_of = np.array(block_of)
for bb in range(n_blocks):
    cols = np.flatnonzero(block_of == bb)
    proj = rng.normal(0, 1 / np.sqrt(len(cols)), (len(cols), d))
    Z[:, bb] = (Xf[:, cols] - Xf[:, cols].mean(0)) @ proj + rng.normal(0, 0.5, (n, d))  # lossy, like CpGPT
    W[cols] = proj
np.save(enc / "Z.npy", Z)
np.save(enc / "C.npy", rng.normal(size=(n_blocks, d)).astype(np.float32))
blocks = np.array([{"chrom": str(chroms[bb // per_chrom]), "start": (bb % per_chrom) * 10_000_000,
                    "end": (bb % per_chrom + 1) * 10_000_000, "n_sites": per_block} for bb in range(n_blocks)], dtype=object)
np.save(enc / "blocks.npy", blocks, allow_pickle=True)
np.save(enc / "obs_names.npy", meta.sample_id.to_numpy(dtype=object), allow_pickle=True)
mm = np.memmap(enc / "W.mmap", dtype=np.float16, mode="w+", shape=W.shape)
mm[:] = W.astype(np.float16)
mm.flush()
np.save(enc / "W_index.npy", np.column_stack([np.arange(n_sites), block_of]).astype(np.int64))
np.save(enc / "site_probe_ids.npy", probes)
np.save(enc / "site_embedding_rows.npy", np.arange(n_sites))
(enc / "encode_meta.json").write_text(json.dumps({
    "model": "fixture", "d": d, "n_samples": n, "n_blocks": n_blocks, "window_bp": 10_000_000,
    "dataset_dir": str(proc.resolve()), "sample_idx": list(range(n)), "W_shape": list(W.shape), "W_dtype": "float16"}))
print(f"fixture: {n} samples, {len(donors)} donors ({n_both} lung+colon), {n_sites} sites, {n_blocks} blocks -> {root}")
