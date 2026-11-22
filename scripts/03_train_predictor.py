"""Stage 3: train the latent block predictors and save their residuals.

  python scripts/03_train_predictor.py [--enc outputs/encodings] [--out outputs/predictor] [--metadata data/egtex/metadata.csv]
                                       [--kinds mlp,linear,block_mean] [--struct profile.npz --tag mlp_struct] [--smoke]

Writes <out>/splits.json, <out>/standardise.npz (mu, sd), <out>/metrics.json and, per kind,
<out>/<kind>/{R.npy, P.npy, metrics.json, model.pt}. R and P are in standardised space.
chrX blocks (stage 2 --include-x) are never training targets or sources; they get residuals for evaluation.
--struct: a per-block profile (npz with key "profile", blocks x k) appended to c_B; outputs go to <out>/<tag>.
--beta-space: the same predictors and residuals on per-block beta vectors PCA-reduced to d (fit on the train
samples) instead of CpGPT embeddings: encodings <enc>_beta (built from <enc> if absent), outputs <out>_beta.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cpgpt_jepa import finish  # noqa: E402
from cpgpt_jepa.predictor import (  # noqa: E402
    KINDS,
    fixed_sources,
    make_splits,
    predict,
    standardise_descriptors,
    train_predictor,
)

p = argparse.ArgumentParser()
p.add_argument("--enc", "--encodings", dest="enc", default="outputs/encodings")
p.add_argument("--out", default="outputs/predictor")
p.add_argument("--metadata", default="data/egtex/metadata.csv")
p.add_argument("--kinds", default=",".join(KINDS))
p.add_argument("--struct", default=None, help="npz with a per-block 'profile' appended to c_B")
p.add_argument("--tag", default=None, help="output name for a single --kinds entry (default: the kind)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--pairs-per-sample", type=int, default=64)
p.add_argument("--batch-size", type=int, default=1024)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--k-sources", type=int, default=8)
p.add_argument("--beta-space", action="store_true", help="control: PCA of each block's betas instead of CpGPT")
p.add_argument("--smoke", action="store_true", help="CPU allowed; the mlp-vs-block_mean check only warns")
p.add_argument("--allow-cpu", action="store_true")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu" and not (args.smoke or args.allow_cpu):
    finish(False, "CUDA is not available in this job")
kinds = args.kinds.split(",")
if args.tag and len(kinds) != 1:
    finish(False, "--tag needs exactly one --kinds entry")
t0 = time.time()
enc, out = Path(args.enc), Path(args.out)
if args.beta_space:
    from cpgpt_jepa.beta_space import build_beta_encodings

    beta_enc, out = Path(f"{enc}_beta"), Path(f"{out}_beta")
    if not (beta_enc / "Z.npy").exists():
        build_beta_encodings(enc, beta_enc, args.metadata, seed=args.seed, log=lambda m: print(m, flush=True))
    enc = beta_enc
out.mkdir(parents=True, exist_ok=True)

# --- data + splits -----------------------------------------------------------------------------
Z = np.load(enc / "Z.npy")
C = np.load(enc / "C.npy")
blocks = np.load(enc / "blocks.npy", allow_pickle=True)  # written by stage 2 (object array of dicts)
obs_names = np.load(enc / "obs_names.npy", allow_pickle=True)  # written by stage 2
meta = pd.read_csv(args.metadata, dtype=str).set_index("sample_id").loc[list(obs_names)]
n_samples, n_blocks, d = Z.shape
chrom_names = np.array([b["chrom"] for b in blocks])
block_chrom = pd.factorize(chrom_names)[0]
x_blocks = np.flatnonzero(chrom_names == "X")
auto_blocks = np.flatnonzero(chrom_names != "X")

splits = make_splits(meta.donor_id.to_numpy(), meta.tissue.to_numpy(), n_blocks, seed=args.seed,
                     eval_only_blocks=x_blocks)
sources = fixed_sources(block_chrom, args.k_sources, args.seed, pool=auto_blocks)  # (n_blocks, K)
splits["k_sources"] = int(sources.shape[1])
splits["fixed_sources"] = sources.tolist()
(out / "splits.json").write_text(json.dumps(splits))
train_s, test_s = np.array(splits["train_samples"]), np.array(splits["test_samples"])
train_b, held_b = np.array(splits["train_blocks"]), np.array(splits["heldout_blocks"])
print(f"donors: {len(splits['train_donors'])} train / {len(splits['test_donors'])} test "
      f"({len(splits['lung_colon_donors'])} lung+colon, {splits['n_filled_test_donors']} random fill)")
print(f"samples: {len(train_s)} train / {len(test_s)} test;  blocks: {len(train_b)} train / {len(held_b)} held out"
      f" / {len(x_blocks)} chrX (eval only);  K = {sources.shape[1]} sources; device {device.type}")

# standardise Z per dimension with train-donor statistics (autosomal blocks)
z_train = Z[train_s][:, auto_blocks].reshape(-1, d)
mu, sd = z_train.mean(0), z_train.std(0)
if not (sd > 0).all():
    finish(False, "an embedding dimension has zero variance over the train samples")
np.savez(out / "standardise.npz", mu=mu, sd=sd)
Zs = torch.from_numpy((Z - mu) / sd).float().to(device)
profile = np.load(args.struct)["profile"] if args.struct else None
Cn = torch.from_numpy(standardise_descriptors(C, profile)).to(device)
test_s_t = torch.from_numpy(test_s).to(device)


def target_variance(block_idx: np.ndarray) -> tuple[float, float]:
    """Per-dimension variance of the test-donor targets, averaged over dimensions: over samples
    and blocks together (total), and over samples within each block (within)."""
    z = Zs[test_s_t][:, torch.from_numpy(block_idx).to(device)]
    return float(z.reshape(-1, d).var(0).mean()), float(z.var(0).mean())


metrics_path = out / "metrics.json"
all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
for kind in kinds:
    name = args.tag or kind
    model, train_log = train_predictor(
        kind, Zs, Cn, train_s, train_b, block_chrom, auto_blocks, epochs=args.epochs,
        pairs_per_sample=args.pairs_per_sample, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed, log=lambda m: print(m, flush=True))
    P, pair_mse = predict(model, Zs, Cn, sources)
    R = Zs - P
    kind_dir = out / name
    kind_dir.mkdir(exist_ok=True)
    np.save(kind_dir / "R.npy", R.cpu().numpy())
    np.save(kind_dir / "P.npy", P.cpu().numpy())
    torch.save(model.state_dict(), kind_dir / "model.pt")

    metrics = {"kind": name, "base_kind": kind, "struct": args.struct, "d_c": int(Cn.size(1)),
               "train_mse_per_epoch": train_log}
    for split_name, block_idx in (("train_blocks", train_b), ("heldout_blocks", held_b), ("x_blocks", x_blocks)):
        if len(block_idx) == 0:
            continue
        bi = torch.from_numpy(block_idx).to(device)
        var_total, var_within = target_variance(block_idx)
        metrics[split_name] = {
            "test_mse": float(pair_mse[test_s_t][:, bi].mean()),  # per (sample, source, target) pair
            "test_mse_k_averaged": float((R[test_s_t][:, bi] ** 2).mean()),  # of the K-source mean prediction
            "target_var_total": var_total,
            "target_var_within_block": var_within,
        }
    (kind_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    all_metrics[name] = metrics
metrics_path.write_text(json.dumps(all_metrics, indent=2))

print("\n=== STAGE 3 REPORT ===")
print(f"donors  : {len(splits['train_donors'])} train / {len(splits['test_donors'])} test "
      f"(all {len(splits['lung_colon_donors'])} lung+colon donors are test; {splits['n_filled_test_donors']} random fill)")
print(f"samples : {len(train_s)} train / {len(test_s)} test     blocks: {len(train_b)} train / {len(held_b)} held out"
      f" / {len(x_blocks)} chrX     d = {d}")
print(f"Test-donor MSE in standardised space (per source-target pair; K-avg = error of the mean over "
      f"K={sources.shape[1]} sources).")
print("MSE / var reads as 1 - R^2: var_tot is over samples and blocks, var_within over samples inside a block.")
print(f"{'':<12s}|{'train blocks':^36s} |{'held-out blocks':^36s}")
print(f"{'kind':<12s}|{'MSE':>8s}{'K-avg':>8s}{'var_tot':>9s}{'var_within':>11s} |{'MSE':>8s}{'K-avg':>8s}{'var_tot':>9s}{'var_within':>11s}")
for name, m_all in all_metrics.items():
    row = f"{name:<12s}|"
    for split_name in ("train_blocks", "heldout_blocks"):
        m = m_all.get(split_name)
        row += (f"{m['test_mse']:>8.4f}{m['test_mse_k_averaged']:>8.4f}{m['target_var_total']:>9.4f}"
                f"{m['target_var_within_block']:>11.4f}" if m else f"{'-':>36s}")
        row += " |" if split_name == "train_blocks" else ""
    print(row)
print(f"elapsed: {time.time() - t0:.0f}s")

if not all(np.isfinite(m["train_blocks"]["test_mse"]) for m in all_metrics.values()):
    finish(False, "non-finite test MSE")
if "mlp" in all_metrics and "block_mean" in all_metrics:
    mlp, base = (all_metrics[k]["train_blocks"]["test_mse"] for k in ("mlp", "block_mean"))
    if not mlp < base:
        msg = f"mlp test MSE on train blocks ({mlp:.4f}) >= block_mean's ({base:.4f}): the predictor learned nothing"
        if not args.smoke:
            finish(False, msg)
        print(f"WARNING (smoke, not a failure): {msg}")
finish(True)
