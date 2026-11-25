"""Stage 2: frozen-CpGPT embeddings of genomic blocks.

  python scripts/02_encode_blocks.py [--processed-dir data/processed] [--out outputs/encodings] [--model small]
                                     [--window-mb 5|10|20] [--include-x] [--smoke]

Writes Z.npy (samples x blocks x d), C.npy (blocks x d), W.mmap + W_index.npy (per-site decoder
vectors), blocks.npy, obs_names.npy, site_probe_ids.npy, site_embedding_rows.npy, Z_full.npy, encode_meta.json.
Frozen model, no grad; fp16 autocast on CUDA, fp32 on CPU (CPU only with --smoke or --allow-cpu).
--smoke: 20 samples (lung+colon donors first) x 5 blocks on 5 different chromosomes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cpgpt_jepa import finish  # noqa: E402
from cpgpt_jepa.blocks import (  # noqa: E402
    AUTOSOMES,
    autocast,
    build_blocks,
    embedding_rows,
    encode_samples,
    encode_sequence,
    find_dataset_dir,
    open_dna_embeddings,
    open_processed,
    pad_inputs,
    pearson,
)

p = argparse.ArgumentParser()
p.add_argument("--processed-dir", default="data/processed")
p.add_argument("--out", default="outputs/encodings")
p.add_argument("--metadata", default="data/egtex/metadata.csv")
p.add_argument("--model", default="small")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--min-sites", type=int, default=500)
p.add_argument("--window-mb", type=int, default=None,
               help="block size; default 10 Mb if stage 1 retained >= 600k probes, else 20 Mb")
p.add_argument("--include-x", action="store_true", help="also encode chrX blocks (positive control, eval only)")
p.add_argument("--full-sites", type=int, default=10000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--probe-mask", default="data/annotations/EPIC.hg38.mask.tsv.gz",
               help="Zhou et al. 2017 EPIC hg38 mask (InfiniumAnnotationV1); probes with --mask-column TRUE are "
                    "excluded from every block and from Z_full. 'none' keeps all probes (the secondary, "
                    "unmasked analysis in docs/PREREG.md section 1)")
p.add_argument("--mask-column", default="M_general")
p.add_argument("--smoke", action="store_true", help="20 samples x 5 blocks, CPU allowed")
p.add_argument("--allow-cpu", action="store_true")
p.add_argument("--bundle", default=None,
               help="encode with a model bundle (<bundle>/config.yaml + model.ckpt) from the Hugging Face cache, "
                    "e.g. perceiver_large, instead of the released --model")
p.add_argument("--bundle-repo", default="dnam-lab/am16489-bundles")
p.add_argument("--cpgpt-src", default=None,
               help="put this CpGPT checkout first on sys.path (the code the bundle was made with)")
args = p.parse_args()
if args.cpgpt_src:
    sys.path.insert(0, str(Path(args.cpgpt_src).resolve()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu" and not (args.smoke or args.allow_cpu):
    finish(False, "CUDA is not available in this job")
print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}; torch {torch.__version__}", flush=True)

from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder  # noqa: E402
from cpgpt.downloads import resolve_cached_dependencies, resolve_cached_model  # noqa: E402
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer  # noqa: E402

t0 = time.time()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# --- frozen model ------------------------------------------------------------------------------
deps_dir = str(resolve_cached_dependencies("human"))
if args.bundle:
    from huggingface_hub import hf_hub_download  # noqa: E402

    # local_files_only: the bundle is downloaded once beforehand, so jobs need no token
    config_path, ckpt_path = (
        hf_hub_download(args.bundle_repo, f"{args.bundle}/{f}", local_files_only=True)
        for f in ("config.yaml", "model.ckpt")
    )
    model_name = f"{args.bundle_repo}/{args.bundle}"
else:
    resources = resolve_cached_model(args.model)
    config_path, ckpt_path = resources.config_path, resources.checkpoint_path
    model_name = args.model
import cpgpt  # noqa: E402

print(f"cpgpt code: {Path(cpgpt.__file__).parent}", flush=True)
inferencer = CpGPTInferencer(dependencies_dir=deps_dir, data_dir="data")
config = inferencer.load_cpgpt_config(config_path)
model = inferencer.load_cpgpt_model(config, model_ckpt_path=ckpt_path, strict_load=True)
model.to(device).eval().requires_grad_(False)
net = model.net
d = net.d_embedding
# Perceiver bundles: the sample embedding is Linear(mean of the latents); the decoder reads the latents
# and neighbouring sites, not z, so there is no per-site linear decoder. W comes from a post-hoc ridge
# fit z_B -> block betas on train donors instead (as stage 09 does for LeWM).
perceiver = getattr(net, "backbone", "transformer") == "perceiver"
n_params = sum(t.numel() for t in net.parameters())
print(f"model {model_name}: {n_params / 1e6:.1f}M parameters, d_embedding {d}, "
      f"backbone {'perceiver' if perceiver else 'transformer'}", flush=True)
if config.model.training.get("binarize_input", False):
    print("WARNING: this model was trained with binarize_input; betas are passed unbinarised here")

# --- data --------------------------------------------------------------------------------------
dataset_dir = find_dataset_dir(args.processed_dir)
X_mm, var_mm, obs_names, species = open_processed(dataset_dir)
var = np.array(var_mm)
probe_ids = np.load(dataset_dir / "probe_ids.npy")
sample_idx = np.arange(X_mm.shape[0])
if args.smoke:
    meta = pd.read_csv(args.metadata, dtype=str).set_index("sample_id").loc[list(obs_names)]
    per_donor = meta.groupby("donor_id").tissue.apply(set)
    both = {dn for dn, ts in per_donor.items() if {"Lung", "Colon - Transverse"} <= ts}
    first = np.flatnonzero(meta.donor_id.isin(both) & meta.tissue.isin(["Lung", "Colon - Transverse"]).to_numpy())
    rest = np.setdiff1d(sample_idx, first)
    sample_idx = np.sort(np.concatenate([first, rest])[:20])
    print(f"smoke: {len(sample_idx)} samples ({len(first)} lung/colon samples of {len(both)} paired donors)")
X = np.array(X_mm[sample_idx])  # samples x sites in RAM: block columns are scattered across the file
obs_names = obs_names[sample_idx]
n_samples, n_sites = X.shape

embedder = DNALLMEmbedder(dependencies_dir=deps_dir)
dna_llm, context_len = config.data.dna_llm, config.data.dna_context_len
rows = embedding_rows(embedder, species, var, dna_llm, context_len)
reverse_vocab = embedder.ensembl_metadata_dict[species]["reverse_vocab"]
chrom_names = {int(c): str(reverse_vocab[int(c)]) for c in np.unique(var[:, 0])}
print(f"X {X.shape}; {int((rows >= 0).sum())} sites with a DNA embedding ({time.time() - t0:.0f}s)", flush=True)
usable = rows >= 0
n_masked = 0
if args.probe_mask.lower() != "none":
    mask_tab = pd.read_csv(args.probe_mask, sep="\t", usecols=["Probe_ID", args.mask_column])
    masked_ids = set(mask_tab.loc[mask_tab[args.mask_column].astype(str).str.upper() == "TRUE", "Probe_ID"])
    is_masked = np.isin(probe_ids, sorted(masked_ids))
    n_masked = int((usable & is_masked).sum())
    usable &= ~is_masked
    print(f"probe mask {args.probe_mask} ({args.mask_column}): {len(masked_ids)} probes listed, "
          f"{n_masked} of the embedded sites removed, {int(usable.sum())} remain", flush=True)
else:
    print("probe mask: none (all embedded sites kept)", flush=True)

# --- blocks ------------------------------------------------------------------------------------
window_bp = (args.window_mb or (10 if n_sites >= 600_000 else 20)) * 1_000_000
chromosomes = AUTOSOMES + (["X"] if args.include_x else [])
blocks = build_blocks(var, chrom_names, usable, window_bp, args.min_sites, chromosomes)
if args.smoke:
    # 5 blocks on 5 different chromosomes (one of them X with --include-x), so sources exist
    picked, seen = [], set()
    for b in blocks:
        if b["chrom"] not in seen and (b["chrom"] != "X" or args.include_x):
            picked.append(b)
            seen.add(b["chrom"])
    x_blocks = [b for b in picked if b["chrom"] == "X"]
    picked = [b for b in picked if b["chrom"] != "X"][: 5 - len(x_blocks[:1])] + x_blocks[:1]
    blocks = [dict(b, block_id=i) for i, b in enumerate(picked)]
n_blocks = len(blocks)
sizes = np.array([b["n_sites"] for b in blocks])
np.save(out / "blocks.npy", np.array(blocks, dtype=object), allow_pickle=True)
np.save(out / "obs_names.npy", obs_names, allow_pickle=True)
n_x = sum(b["chrom"] == "X" for b in blocks)
print(f"{n_blocks} blocks of {window_bp // 1_000_000} Mb ({n_x} on chrX), {sizes.min()}-{sizes.max()} sites", flush=True)

rng = np.random.default_rng(args.seed)
n_full = min(args.full_sites if not args.smoke else 2000, int(usable.sum()))
full_cols = rng.choice(np.flatnonzero(usable), size=n_full, replace=False)
full_cols = full_cols[np.lexsort((var[full_cols, 1], var[full_cols, 0]))]  # by chromosome, then position

# DNA embeddings of the sites used, gathered once into RAM (sorted reads from the memmap)
needed = np.unique(np.concatenate([rows[b["site_cols"]] for b in blocks] + [rows[full_cols]]))
dna_mm = open_dna_embeddings(embedder, species, dna_llm, context_len)
dna = np.asarray(dna_mm[needed])
local = {int(r): i for i, r in enumerate(needed)}
def dna_of(cols: np.ndarray) -> np.ndarray:
    return dna[[local[int(r)] for r in rows[cols]]]
print(f"gathered {len(needed)} DNA embeddings ({time.time() - t0:.0f}s)", flush=True)

n_total = int(sizes.sum())
Z = open_memmap(out / "Z.npy", mode="w+", dtype=np.float32, shape=(n_samples, n_blocks, d))
C = np.zeros((n_blocks, d), dtype=np.float32)
W = np.memmap(out / "W.mmap", dtype=np.float16, mode="w+", shape=(n_total, d))
W_index = np.zeros((n_total, 2), dtype=np.int64)  # (X column, block_id) of each W row

check_blocks = set(rng.choice(n_blocks, size=min(3, n_blocks), replace=False).tolist())
checks, seconds, offset = [], [], 0
for b in blocks:
    tb = time.time()
    cols = b["site_cols"]
    seq = encode_sequence(net, dna_of(cols), device)  # (1, n_sites, d), shared across samples
    C[b["block_id"]] = seq[0].float().mean(0).cpu().numpy()
    if not perceiver:  # Perceiver: W is the post-hoc ridge decoder, filled after this loop
        with torch.no_grad(), autocast(device):
            w = net.meth_decoder(seq)[0]
        W[offset : offset + len(cols)] = w.float().cpu().numpy().astype(np.float16)
    W_index[offset : offset + len(cols), 0] = cols
    W_index[offset : offset + len(cols), 1] = b["block_id"]
    offset += len(cols)

    meth = X[:, cols]
    z = encode_samples(net, meth, seq, var[cols], device, args.batch_size)
    Z[:, b["block_id"]] = z.cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds.append(time.time() - tb)

    if b["block_id"] in check_blocks:
        # decode path: query_methylation(z, seq_emb) against the observed betas
        pred = []
        with torch.no_grad(), autocast(device):
            if perceiver:
                # the Perceiver decodes from its context (latents + neighbouring observed sites), not from z
                ch = torch.from_numpy(np.ascontiguousarray(var[cols, 0])).to(device).unsqueeze(0)
                ps = torch.from_numpy(np.ascontiguousarray(var[cols, 1])).to(device).unsqueeze(0)
                for i in range(0, n_samples, 16):
                    mb = torch.from_numpy(meth[i : i + 16]).to(device)
                    n_b = mb.size(0)
                    inputs = pad_inputs(mb, seq.expand(n_b, -1, -1), ch.expand(n_b, -1), ps.expand(n_b, -1))
                    ctx = net.encode_context(**inputs)
                    pred.append(net.query_methylation(
                        ctx["sample_embedding"], inputs["sequence_embeddings"], m_or_beta="beta", context=ctx,
                        query_chroms=inputs["chroms"], query_positions=inputs["positions"],
                    )[0].float().cpu())
            else:
                for i in range(0, n_samples, 64):
                    zi = z[i : i + 64].to(seq.dtype)
                    pred.append(net.query_methylation(zi, seq.expand(len(zi), -1, -1), m_or_beta="beta")[0].float().cpu())
        pred = torch.cat(pred).numpy()
        ok = ~np.isnan(meth)
        r_all = pearson(pred[ok], meth[ok])
        # same, with each site's mean over samples removed (what the site identity alone cannot explain)
        mean_obs = np.nanmean(meth, axis=0, keepdims=True)
        mean_pred = np.nanmean(np.where(ok, pred, np.nan), axis=0, keepdims=True)
        r_centred = pearson((pred - mean_pred)[ok], (meth - mean_obs)[ok])
        checks.append((b["block_id"], b["chrom"], b["start"], b["n_sites"], r_all, r_centred))
    if b["block_id"] % 25 == 0 or args.smoke:
        print(f"  block {b['block_id']:>3d}/{n_blocks} chr{b['chrom']}:{b['start'] // 1_000_000}Mb "
              f"{len(cols)} sites  {seconds[-1]:.1f}s", flush=True)

Z.flush()
decoder_r2 = None
if perceiver:
    # Post-hoc linear decoder z_B -> block betas, fit on the train donors of the stage 03 split
    # (make_splits, split seed 0), so stages 04/07 have a w_i per site for the mQTL tests.
    from cpgpt_jepa.beta_space import ridge_decoder  # noqa: E402
    from cpgpt_jepa.predictor import make_splits  # noqa: E402

    meta_s = pd.read_csv(args.metadata, dtype=str).set_index("sample_id").loc[list(obs_names)]
    chrom_b = np.array([b["chrom"] for b in blocks])
    splits = make_splits(meta_s.donor_id.to_numpy(), meta_s.tissue.to_numpy(), n_blocks, seed=0,
                         eval_only_blocks=np.flatnonzero(chrom_b == "X"))
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    W_dec, decoder_r2 = ridge_decoder(np.asarray(Z), X[:, W_index[:, 0]], offsets,
                                      np.array(splits["train_samples"]), np.array(splits["test_samples"]))
    W[:] = W_dec.astype(np.float16)
    print(f"post-hoc ridge decoder z_B -> betas: mean test R^2 over blocks {decoder_r2:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
W.flush()
np.save(out / "C.npy", C)
np.save(out / "W_index.npy", W_index)
np.save(out / "site_probe_ids.npy", probe_ids[W_index[:, 0]])
np.save(out / "site_embedding_rows.npy", rows[W_index[:, 0]])  # DNA-embedding / annotation row per W row
t_blocks = time.time() - t0

# --- Z_full: the standard CpGPT embedding from one fixed random draw of sites over the array ----
seq = encode_sequence(net, dna_of(full_cols), device)
z_full = encode_samples(net, X[:, full_cols], seq, var[full_cols], device, min(args.batch_size, 4))
np.save(out / "Z_full.npy", z_full.cpu().numpy())
np.save(out / "Z_full_sites.npy", full_cols)

Z_arr = np.load(out / "Z.npy", mmap_mode="r")
finite = bool(np.isfinite(Z_arr).all() and np.isfinite(C).all() and torch.isfinite(z_full).all())
meta = {
    "model": model_name, "d": d, "n_params": n_params, "n_samples": n_samples, "n_blocks": n_blocks,
    "n_x_blocks": n_x, "window_bp": window_bp, "min_sites": args.min_sites, "n_sites_total": n_total,
    "W_shape": [n_total, d], "W_dtype": "float16", "dataset_dir": str(dataset_dir),
    "sample_idx": sample_idx.tolist(), "dna_llm": dna_llm, "dna_context_len": context_len,
    "seed": args.seed, "smoke": args.smoke, "device": device.type,
    "probe_mask": args.probe_mask, "mask_column": args.mask_column if args.probe_mask.lower() != "none" else None,
    "n_masked_sites": n_masked,
    "model_name": model_name, "cpgpt_src": str(Path(cpgpt.__file__).parent),
    "backbone": "perceiver" if perceiver else "transformer",
    "W_kind": "post-hoc ridge decoder: z_B -> block betas, train donors" if perceiver else "meth_decoder(seq)",
    "ridge_test_r2": decoder_r2,
}
(out / "encode_meta.json").write_text(json.dumps(meta, indent=2))

print("\n=== STAGE 2 REPORT ===")
print(f"model                : {model_name} ({n_params / 1e6:.1f}M parameters, d = {d}), {device.type}")
print(f"window               : {window_bp // 1_000_000} Mb  (stage 1 retained {n_sites} probes)")
print(f"n_blocks             : {n_blocks}  ({n_blocks - n_x} autosomal, {n_x} chrX; >= {args.min_sites} sites)")
print(f"sites in blocks      : {n_total}")
print(f"median sites/block   : {int(np.median(sizes))}  (min {sizes.min()}, max {sizes.max()})")
print(f"seconds/block        : {np.mean(seconds):.2f}  ({n_samples} samples; total {t_blocks:.0f}s)")
print(f"Z.npy                : {Z_arr.shape}   C.npy: {C.shape}   W.mmap: {(n_total, d)} fp16   Z_full.npy: {tuple(z_full.shape)}")
print(f"all finite           : {finite}")
if decoder_r2 is not None:
    print(f"ridge decoder R^2    : {decoder_r2:.3f}  (z_B -> block betas, test donors; W for the mQTL tests)")
print("decode check (Pearson r, query_methylation(z_B, seq_emb) vs observed betas):")
for bid, chrom, start, n, r_all, r_centred in checks:
    print(f"  block {bid:>3d} chr{chrom}:{start // 1_000_000}Mb  {n} sites   r = {r_all:.3f}   "
          f"(site means removed: r = {r_centred:.3f})")
print(f"elapsed              : {time.time() - t0:.0f}s")

if not finite:
    finish(False, "non-finite values in the encodings")
bad = [f"block {c[0]} r={c[4]:.3f}" for c in checks if not c[4] >= 0.3]
if bad:
    finish(False, "encode path does not match the model's decode path: " + ", ".join(bad))
finish(True)
