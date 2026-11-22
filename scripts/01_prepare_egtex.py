"""Stage 1: eGTEx betas (probes x samples, .txt.gz) -> arrow (samples x probes) -> CpGPTDataSaver memmaps.

  python scripts/01_prepare_egtex.py [--egtex-dir data/egtex] [--processed-dir data/processed] [--model small]
  smoke subset: --n-samples 50 --min-lung-colon-pairs 5 --arrow data/smoke/egtex.arrow --processed-dir data/smoke/processed

Follows tutorials/quick_setup.ipynb: keep the probes in the model vocab, write a feather file with
the sample id as first column, then CpGPTDataSaver.process_files(prober, embedder).
"""
import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cpgpt_jepa import finish  # noqa: E402
from cpgpt_jepa.blocks import embedding_rows, find_dataset_dir, open_processed  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--egtex-dir", default="data/egtex")
p.add_argument("--processed-dir", default="data/processed")
p.add_argument("--model", default="small", help="CpGPT model whose vocab filters the probes")
p.add_argument("--chunk-rows", type=int, default=20000)
p.add_argument("--arrow", default=None, help="arrow file to write (default <egtex-dir>/arrow/egtex.arrow)")
p.add_argument("--n-samples", type=int, default=None, help="keep a random subset of this many samples")
p.add_argument("--min-lung-colon-pairs", type=int, default=0,
               help="with --n-samples: include at least this many donors with both lung and colon")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

egtex = Path(args.egtex_dir)
betas_path = egtex / "geo" / "GSE213478_betas.txt.gz"
arrow_path = Path(args.arrow) if args.arrow else egtex / "arrow" / "egtex.arrow"
t0 = time.time()

# --- metadata ---------------------------------------------------------------------------------
print("=== scripts/build_metadata.py ===", flush=True)
subprocess.run([sys.executable, "scripts/build_metadata.py", str(egtex)], check=True)
meta = pd.read_csv(egtex / "metadata.csv", dtype=str)

# --- align beta columns to metadata sample_id --------------------------------------------------
# The beta header is comma-separated and names samples "<donor_id>-<tissue site>-<sample_id>",
# e.g. GTEX-11EM3-1326-SM-GW24C for sample_id SM-GW24C of donor GTEX-11EM3.
with gzip.open(betas_path, "rt") as f:
    header = [c.strip('"') for c in f.readline().rstrip("\n").split(",")]
beta_cols = header[1:]
donor_of = dict(zip(meta.sample_id, meta.donor_id))
col_of = {}  # sample_id -> beta column
for col in beta_cols:
    parts = col.split("-")
    hits = [s for s in ("-".join(parts[i:]) for i in range(len(parts))) if s in donor_of]
    if len(hits) == 1 and (col == hits[0] or col.startswith(donor_of[hits[0]] + "-")):
        col_of.setdefault(hits[0], []).append(col)
matched = [s for s in meta.sample_id if len(col_of.get(s, [])) == 1]
print("\n=== sample alignment ===")
print(f"beta header: {len(beta_cols)} sample columns; first 5: {beta_cols[:5]}")
print(f"samples matched to metadata sample_id (suffix match, donor prefix agrees): "
      f"{len(matched)}/{len(meta)}  (exact name matches: {sum(c in donor_of for c in beta_cols)})")
if len(matched) < 987:
    print("first 5 header names:", header[:5])
    finish(False, f"only {len(matched)}/987 beta columns matched metadata sample_id")
meta = meta[meta.sample_id.isin(matched)].reset_index(drop=True)
if args.n_samples is not None and args.n_samples < len(meta):
    # subset: every sample of `min_lung_colon_pairs` random lung+colon donors' lung and colon, then random fill
    rng = np.random.default_rng(args.seed)
    per_donor = meta.groupby("donor_id").tissue.apply(set)
    both_donors = sorted(d for d, ts in per_donor.items() if {"Lung", "Colon - Transverse"} <= ts)
    pick = rng.choice(both_donors, size=min(args.min_lung_colon_pairs, len(both_donors)), replace=False)
    keep = set(meta.index[meta.donor_id.isin(pick) & meta.tissue.isin(["Lung", "Colon - Transverse"])])
    rest = [i for i in meta.index if i not in keep]
    keep |= set(rng.choice(rest, size=max(0, args.n_samples - len(keep)), replace=False).tolist())
    meta = meta.loc[sorted(keep)].reset_index(drop=True)
    print(f"subset: {len(meta)} samples, {len(pick)} lung+colon donors forced in (seed {args.seed})")
ordered_cols = [col_of[s][0] for s in meta.sample_id]

# --- model + dependencies (notebook settings) --------------------------------------------------
from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver  # noqa: E402
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder  # noqa: E402
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber  # noqa: E402
from cpgpt.downloads import resolve_cached_dependencies, resolve_cached_model  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

resources = resolve_cached_model(args.model)
deps_dir = str(resolve_cached_dependencies("human"))
config = OmegaConf.load(resources.config_path)
vocab = set(json.load(open(resources.vocab_path))["input"])

# --- stream betas -> arrow ---------------------------------------------------------------------
n_probes_file = None
if arrow_path.exists():
    print(f"\n{arrow_path} exists; reusing it")
    probes = [c for c in pa.ipc.open_file(arrow_path).schema.names if c not in ("sample_id", "species")]
else:
    print(f"\n=== streaming {betas_path} ===", flush=True)
    reader = pd.read_csv(
        betas_path, chunksize=args.chunk_rows, index_col=0, usecols=[0, *sorted(header.index(c) for c in ordered_cols)],  # positions: the probe-id column is unnamed
        dtype=dict.fromkeys(ordered_cols, np.float32),
    )
    chunks, probes, n_probes_file = [], [], 0
    for i, chunk in enumerate(reader):
        n_probes_file += len(chunk)
        keep = chunk.index.isin(vocab)
        chunks.append(chunk.loc[keep, ordered_cols].to_numpy(dtype=np.float32))
        probes.extend(chunk.index[keep].astype(str))
        if i % 5 == 0:
            print(f"  {n_probes_file:>7d} probes read, {len(probes):>7d} kept  ({time.time() - t0:.0f}s)", flush=True)
    betas = np.concatenate(chunks)  # probes x samples, columns in metadata order
    del chunks
    if len(set(probes)) != len(probes):
        finish(False, "duplicate probe ids in the beta file")
    betas = np.ascontiguousarray(betas.T)  # samples x probes
    print(f"read {n_probes_file} probes; {len(probes)} in the {args.model} vocab; matrix {betas.shape}")

    arrays = [pa.array(meta.sample_id.to_numpy()), pa.array(["homo_sapiens"] * len(meta))]
    arrays += [pa.array(betas[:, j]) for j in range(betas.shape[1])]
    arrow_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = arrow_path.with_suffix(".arrow.tmp")
    feather.write_feather(pa.Table.from_arrays(arrays, names=["sample_id", "species", *probes]), tmp)
    tmp.rename(arrow_path)
    del arrays, betas
    print(f"wrote {arrow_path} ({arrow_path.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s)", flush=True)

# --- CpGPTDataSaver ----------------------------------------------------------------------------
print("\n=== CpGPTDataSaver.process_files ===", flush=True)
embedder = DNALLMEmbedder(dependencies_dir=deps_dir)
prober = IlluminaMethylationProber(dependencies_dir=deps_dir, embedder=embedder)
CpGPTDataSaver(data_paths=str(arrow_path), processed_dir=args.processed_dir).process_files(prober, embedder)

dataset_dir = find_dataset_dir(args.processed_dir)
X, var, obs_names, species = open_processed(dataset_dir)

# Probe id of every X column. The datasaver stores only (chrom, position) per site
# (var_names.npy is ["chrom", "position"]), so repeat its column filter to keep the probe ids.
illumina = prober.illumina_metadata_dict[species]
chrom_vocab = embedder.ensembl_metadata_dict[species]["vocab"]
kept = [(p, illumina[p]) for p in probes if p in illumina and illumina[p].split(":")[0] in chrom_vocab]
probe_ids = np.array([p for p, _ in kept])
expected_var = np.array(
    [[chrom_vocab[loc.replace("chr", "").split(":")[0]], int(loc.split(":")[1])] for _, loc in kept],
    dtype=np.int32,
)
if expected_var.shape != var.shape or not np.array_equal(expected_var, np.asarray(var)):
    finish(False, "probe ids could not be aligned to the datasaver's var.mmap")
if list(obs_names) != list(meta.sample_id):
    finish(False, "obs_names.npy does not match the metadata sample order")
np.save(dataset_dir / "probe_ids.npy", probe_ids)

rows = embedding_rows(embedder, species, var, config.data.dna_llm, config.data.dna_context_len)
n_nan = sum(int(np.isnan(X[i : i + 64]).sum()) for i in range(0, X.shape[0], 64))
reverse_vocab = embedder.ensembl_metadata_dict[species]["reverse_vocab"]
per_chrom = pd.Series([reverse_vocab[int(c)] for c in var[:, 0]]).value_counts()
order = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
per_chrom = per_chrom.reindex([c for c in order if c in per_chrom.index] + sorted(set(per_chrom.index) - set(order)))
per_donor = meta.groupby("donor_id").tissue.apply(set)
n_both = sum({"Lung", "Colon - Transverse"} <= s for s in per_donor)

print("\n=== STAGE 1 REPORT ===")
print(f"samples matched to metadata : {len(matched)}/987")
print(f"samples written             : {len(meta)}")
print(f"probes in beta file         : {n_probes_file if n_probes_file is not None else '(arrow reused; not re-read)'}")
print(f"probes in arrow             : {len(probes)}  (those in the {args.model} model vocab, as in the notebook)")
print(f"probes retained by CpGPT    : {X.shape[1]}  (mapped to a genomic location by the datasaver)")
print(f"  with a DNA embedding      : {int((rows >= 0).sum())}  ({config.data.dna_llm}, {config.data.dna_context_len} bp)")
print(f"X.mmap shape                : {X.shape}  ({dataset_dir}/X.mmap)")
print(f"NaN fraction                : {n_nan / X.size:.6f}")
print(f"unique genomic locations    : {len(np.unique(np.asarray(var), axis=0))}")
print("sites per chromosome        :")
print("  " + "  ".join(f"{c}:{n}" for c, n in per_chrom.items()))
print(f"lung-colon donors           : {n_both}  (expect 153)")
print(f"probe ids of the X columns  : {dataset_dir}/probe_ids.npy")
print(f"elapsed                     : {time.time() - t0:.0f}s")
finish(True)
