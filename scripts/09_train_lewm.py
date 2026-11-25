"""Phase B / 09: train the end-to-end block encoder (src/cpgpt_jepa/lewm.py), then evaluate it like Phase A.

  python scripts/09_train_lewm.py [--lam 0.1 | --lams 0.01,0.1,1 (picked by SLURM_ARRAY_TASK_ID)]
                                  [--encodings outputs/encodings] [--out-root outputs/lewm] [--smoke]

Uses the frozen run's block table, c_B (C.npy) and samples. Training pairs follow Phase A: train samples, target
B a train block, source A an autosomal block on another chromosome, 64 fresh pairs per sample per epoch.
Writes outputs/lewm/lam<lambda>/ in the encodings layout (Z.npy = g per sample and block, C.npy, blocks.npy,
obs_names.npy, encode_meta.json, and W.mmap from a post-hoc ridge decoder g -> block betas fit on train
samples), then runs, on those encodings:
  03 (same splits, same residual construction) -> <run>/predictor, then 04, 05, 06, 07 -> <run>/experiments/,
plus linear probes on g (block-mean latent) for sex, age, tissue and genotype PC1-5, next to the same probes
on the frozen CpGPT embeddings.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cpgpt_jepa import finish  # noqa: E402
from cpgpt_jepa.blocks import autocast, open_processed  # noqa: E402
from cpgpt_jepa.beta_space import ridge_decoder, write_site_vectors  # noqa: E402
from cpgpt_jepa.covariates import genotype_pcs  # noqa: E402
from cpgpt_jepa.evaluation import fmt, load_metadata, write_outputs  # noqa: E402
from cpgpt_jepa.lewm import BlockData, BlockEncoder, LeWMPredictor, sigreg  # noqa: E402
from cpgpt_jepa.predictor import (  # noqa: E402
    make_splits,
    sample_sources,
    standardise_descriptors,
)

p = argparse.ArgumentParser()
p.add_argument("--encodings", "--enc", dest="enc", default="outputs/encodings")
p.add_argument("--egtex-dir", default="data/egtex")
p.add_argument("--metadata", default="data/egtex/metadata.csv")
p.add_argument("--out-root", default="outputs/lewm")
p.add_argument("--lam", type=float, default=None, help="default: LeWM's 0.1, or --lams[SLURM_ARRAY_TASK_ID] in an array job")
p.add_argument("--lams", default="0.01,0.1,1", help="sweep values; SLURM_ARRAY_TASK_ID picks one")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--pairs-per-sample", type=int, default=64)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--max-sites-train", type=int, default=1024, help="sites subsampled per block per training step")
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--d", type=int, default=256)
p.add_argument("--n-directions", type=int, default=1024, help="SIGReg projections M (LeWM default 1024)")
p.add_argument("--sigreg-mode", choices=["global", "within"], default="global",
               help="global: SIGReg on z_A and z_B over the batch (LeWM; lets samples collapse inside a block). "
                    "within: batches of --blocks-per-batch target blocks x (batch-size / blocks-per-batch) samples, "
                    "SIGReg on z_B minus its block's batch mean; z_A keeps the global term. Run dir gets a _within suffix")
p.add_argument("--blocks-per-batch", type=int, default=4, help="target blocks per batch in --sigreg-mode within")
p.add_argument("--encode-batch", type=int, default=8)
p.add_argument("--max-sites-encode", type=int, default=None,
               help="sites per block when encoding after training (default: all). Full attention over a whole "
                    "10 Mb block (~5k sites) needs ~6 GB per layer at batch 8 in fp32, so --smoke on CPU caps it")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--skip-eval", action="store_true", help="train and encode only")
p.add_argument("--smoke", action="store_true", help="2 epochs, few pairs, CPU allowed")
args = p.parse_args()

lams = [float(x) for x in args.lams.split(",")]
if args.lam is not None:
    lam = args.lam
elif "SLURM_ARRAY_TASK_ID" in os.environ:
    lam = lams[int(os.environ["SLURM_ARRAY_TASK_ID"])]
else:
    lam = 0.1  # LeWM default
if args.smoke:
    args.epochs, args.pairs_per_sample, args.max_sites_train = 2, 4, 128
    args.max_sites_encode = args.max_sites_encode or 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu" and not args.smoke:
    finish(False, "CUDA is not available in this job")
dtype = torch.float16 if device.type == "cuda" else torch.float32
t0 = time.time()
within = args.sigreg_mode == "within"
if within and (args.batch_size % args.blocks_per_batch or args.batch_size // args.blocks_per_batch < 2):
    finish(False, f"--batch-size {args.batch_size} must be a multiple of --blocks-per-batch {args.blocks_per_batch}"
                  " with at least 2 samples per block")
run_dir = Path(args.out_root) / (f"lam{lam:g}" + ("_within" if within else ""))
run_dir.mkdir(parents=True, exist_ok=True)
log = lambda m: print(m, flush=True)  # noqa: E731
log(f"lambda {lam}; device {device.type}; run dir {run_dir}")

# --- data from the frozen run ------------------------------------------------------------------
enc = Path(args.enc)
enc_meta = json.loads((enc / "encode_meta.json").read_text())
blocks = np.load(enc / "blocks.npy", allow_pickle=True)  # written by stage 2 (object array of dicts)
obs_names = np.load(enc / "obs_names.npy", allow_pickle=True)  # written by stage 2
C = np.load(enc / "C.npy")
W_index = np.load(enc / "W_index.npy")  # (X column, block) per site, blocks contiguous in order
site_rows = np.load(enc / "site_embedding_rows.npy")
offsets = np.searchsorted(W_index[:, 1], np.arange(len(blocks) + 1))
X_mm, _, _, _ = open_processed(Path(enc_meta["dataset_dir"]))
sample_rows = np.array(enc_meta.get("sample_idx", np.arange(X_mm.shape[0])))
betas = np.asarray(X_mm[sample_rows])[:, W_index[:, 0]]

from cpgpt.downloads import resolve_cached_dependencies  # noqa: E402

nt_path = (resolve_cached_dependencies("human") / "dna_embeddings" / "homo_sapiens" / enc_meta["dna_llm"]
           / f"{enc_meta['dna_context_len']}bp_dna_embeddings.mmap")
nt_mm = np.memmap(nt_path, dtype="float32", mode="r").reshape(-1, 1024)
order = np.argsort(site_rows)
nt = np.empty((len(site_rows), nt_mm.shape[1]), dtype=np.float32)
nt[order] = nt_mm[site_rows[order]]
data = BlockData(blocks, offsets, nt, betas, device, dtype)
del nt  # betas (numpy) is kept for the post-hoc decoder
log(f"{len(obs_names)} samples, {len(blocks)} blocks, {len(site_rows)} sites ({time.time() - t0:.0f}s)")

meta = pd.read_csv(args.metadata, dtype=str).set_index("sample_id").loc[list(obs_names)]
chrom_names = np.array([b["chrom"] for b in blocks])
codes = pd.factorize(chrom_names)[0]
x_blocks, auto = np.flatnonzero(chrom_names == "X"), np.flatnonzero(chrom_names != "X")
splits = make_splits(meta.donor_id.to_numpy(), meta.tissue.to_numpy(), len(blocks), seed=args.seed,
                     eval_only_blocks=x_blocks)  # identical to stage 3's splits (same inputs and seed)
train_s, train_b = np.array(splits["train_samples"]), np.array(splits["train_blocks"])

# --- train g and f end to end ------------------------------------------------------------------
torch.manual_seed(args.seed)
gen = torch.Generator(device=device).manual_seed(args.seed)
g = BlockEncoder(nt_mm.shape[1], args.d).to(device)
Cn = torch.from_numpy(standardise_descriptors(C)).to(device)
f = LeWMPredictor(args.d, Cn.size(1)).to(device)
opt = torch.optim.AdamW(list(g.parameters()) + list(f.parameters()), lr=args.lr, weight_decay=args.weight_decay)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
train_s_t, train_b_t = torch.as_tensor(train_s, device=device), torch.as_tensor(train_b, device=device)
codes_t, pool_t = torch.as_tensor(codes, device=device), torch.as_tensor(auto, device=device)
k_blk = args.blocks_per_batch
m_smp = min(args.batch_size // k_blk, len(train_s))  # capped for --smoke-sized data


def grouped_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """--sigreg-mode within: k_blk random train target blocks, m_smp distinct random train samples for each."""
    tb = train_b_t[torch.randint(len(train_b), (k_blk,), generator=gen, device=device)].repeat_interleave(m_smp)
    sb = torch.cat([train_s_t[torch.randperm(len(train_s), generator=gen, device=device)[:m_smp]]
                    for _ in range(k_blk)])
    return sb, sample_sources(tb, codes_t, pool_t, gen), tb


history, step = [], 0
for epoch in range(args.epochs):
    g.train(), f.train()
    n_ex = len(train_s) * args.pairs_per_sample
    if not within:
        s = train_s_t.repeat_interleave(args.pairs_per_sample)
        tgt = train_b_t[torch.randint(len(train_b), (n_ex,), generator=gen, device=device)]
        src = sample_sources(tgt, codes_t, pool_t, gen)
        perm = torch.randperm(n_ex, generator=gen, device=device)
    tot_pred = tot_reg = tot_wvar = 0.0
    for i in range(0, n_ex, args.batch_size):
        if within:
            s_i, src_i, tgt_i = grouped_batch()
        else:
            idx = perm[i : i + args.batch_size]
            if len(idx) < 2:  # BatchNorm in the projectors needs at least 2 examples
                continue
            s_i, src_i, tgt_i = s[idx], src[idx], tgt[idx]
        with autocast(device):
            z_a = g(*data.batch(s_i, src_i, args.max_sites_train, gen))
            z_b = g(*data.batch(s_i, tgt_i, args.max_sites_train, gen))
            pred_loss = torch.nn.functional.mse_loss(f(z_a, Cn[tgt_i], tgt_i).float(), z_b.float())
        if within:
            # z_B minus its block's batch mean: the regulariser now spreads samples inside a block
            zb = z_b.float().view(k_blk, m_smp, -1)
            zb_dev = (zb - zb.mean(1, keepdim=True)).reshape(len(s_i), -1)
            reg = 0.5 * (sigreg(z_a.float(), args.n_directions, step) + sigreg(zb_dev, args.n_directions, step))
            wvar = zb.var(1).mean().item()
        else:
            # SIGReg on the encoder embeddings: z_A and z_B separately, averaged (LeWM applies it per time step)
            reg = 0.5 * (sigreg(z_a.float(), args.n_directions, step) + sigreg(z_b.float(), args.n_directions, step))
            wvar = float("nan")
        loss = pred_loss + lam * reg
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot_pred += pred_loss.item() * len(s_i)
        tot_reg += reg.item() * len(s_i)
        tot_wvar += wvar * len(s_i)
        step += 1
    history.append({"epoch": epoch + 1, "pred_mse": tot_pred / n_ex, "sigreg": tot_reg / n_ex,
                    "within_block_var_zb": tot_wvar / n_ex})
    log(f"epoch {epoch + 1:>2d}/{args.epochs}  pred MSE {history[-1]['pred_mse']:.4f}  SIGReg {history[-1]['sigreg']:.4f}"
        + (f"  within-block var(z_B) {history[-1]['within_block_var_zb']:.4f}" if within else "")
        + f"  ({time.time() - t0:.0f}s)")
torch.save({"g": g.state_dict(), "f": f.state_dict(), "lam": lam, "args": vars(args)}, run_dir / "lewm.pt")

# --- encode every (sample, block) with all its sites -------------------------------------------
g.eval()
Z = np.zeros((len(obs_names), len(blocks), args.d), dtype=np.float32)
enc_gen = torch.Generator(device=device).manual_seed(args.seed)  # fixed subsample when --max-sites-encode is set
with torch.no_grad(), autocast(device):
    for b in range(len(blocks)):
        for i in range(0, len(obs_names), args.encode_batch):
            sm = torch.arange(i, min(i + args.encode_batch, len(obs_names)), device=device)
            Z[sm.cpu().numpy(), b] = g(*data.batch(sm, torch.full_like(sm, b), args.max_sites_encode,
                                                    enc_gen)).float().cpu().numpy()
np.save(run_dir / "Z.npy", Z)
np.save(run_dir / "C.npy", C)
np.save(run_dir / "blocks.npy", blocks, allow_pickle=True)
np.save(run_dir / "obs_names.npy", obs_names, allow_pickle=True)
(run_dir / "encode_meta.json").write_text(json.dumps({
    **{k: enc_meta[k] for k in ("window_bp", "dataset_dir", "dna_llm", "dna_context_len", "n_x_blocks") if k in enc_meta},
    "model": "lewm", "lam": lam, "sigreg_mode": args.sigreg_mode, "blocks_per_batch": k_blk if within else None,
    "d": args.d, "n_samples": len(obs_names), "n_blocks": len(blocks),
    "sample_idx": sample_rows.tolist(), "frozen_encodings": str(enc), "history": history, "smoke": args.smoke}, indent=2))
finite = bool(np.isfinite(Z).all())
log(f"encoded Z {Z.shape}, finite {finite} ({time.time() - t0:.0f}s)")
if not finite:
    finish(False, "non-finite LeWM embeddings")

# --- post-hoc linear decoder g -> block betas (train samples), so 04/07 have w_i for the mQTL test ----
W_dec, decoder_r2 = ridge_decoder(Z, betas, offsets, train_s, np.array(splits["test_samples"]))
write_site_vectors(run_dir, W_dec, enc, "post-hoc ridge decoder: raw g -> block betas, fit on train samples")
log(f"post-hoc decoder: mean test R^2 over blocks {decoder_r2:.3f} ({time.time() - t0:.0f}s)")

# --- linear probes on the block-mean latent, next to frozen CpGPT ------------------------------
from sklearn.linear_model import LogisticRegression, RidgeCV  # noqa: E402

full_meta = load_metadata(args.egtex_dir, obs_names)
test_s = np.array(splits["test_samples"])
targets = {"sex": ("class", full_meta.sex.to_numpy()), "tissue": ("class", full_meta.tissue.to_numpy()),
           "age": ("reg", full_meta.age_years.to_numpy(dtype=float))}
pcs = genotype_pcs(args.egtex_dir, full_meta.donor_id.to_numpy())
for k in range(pcs.shape[1]):
    targets[f"genotype PC{k + 1}"] = ("reg", pcs[:, k])
probes = {}
for rep_name, rep in (("LeWM g", Z), ("frozen CpGPT z", np.load(enc / "Z.npy"))):
    feats = rep[:, auto].mean(1)
    feats = (feats - feats[train_s].mean(0)) / (feats[train_s].std(0) + 1e-8)
    probes[rep_name] = {}
    for t_name, (task, y) in targets.items():
        ok = pd.notna(y) if task == "class" else np.isfinite(y)
        tr, te = train_s[ok[train_s]], test_s[ok[test_s]]
        if len(tr) < 3 or len(te) < 1 or (task == "class" and len(set(y[tr])) < 2):
            probes[rep_name][t_name] = float("nan")
            continue
        if task == "class":
            clf = LogisticRegression(max_iter=2000).fit(feats[tr], y[tr])
            probes[rep_name][t_name] = float((clf.predict(feats[te]) == y[te]).mean())
        else:
            reg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(feats[tr], y[tr])
            pred = reg.predict(feats[te])
            probes[rep_name][t_name] = float(1 - ((y[te] - pred) ** 2).sum() / (((y[te] - y[te].mean()) ** 2).sum() + 1e-12))
write_outputs(run_dir / "probes", {"lam": lam, "metric": "accuracy for sex/tissue, R^2 for age and PCs (test donors)",
                                   "probes": probes, "history": history},
              [{"representation": r, **v} for r, v in probes.items()])

# --- Phase A evaluation on these encodings -----------------------------------------------------
eval_status = {}
if not args.skip_eval:
    env = {k: v for k, v in os.environ.items() if k != "SANITY_FILE"}
    pred_dir, exp_dir = run_dir / "predictor", run_dir / "experiments"
    smoke = ["--smoke"] if args.smoke else []
    steps = [
        ("03", ["scripts/03_train_predictor.py", "--enc", str(run_dir), "--out", str(pred_dir), "--metadata", args.metadata]
         + (["--allow-cpu"] if device.type == "cpu" else []) + smoke),
        ("04", ["scripts/04_evaluate.py", "--encodings", str(run_dir), "--pred", str(pred_dir), "--egtex-dir", args.egtex_dir] + smoke),
        ("05", ["scripts/05_ablations.py", "--encodings", str(run_dir), "--pred", str(pred_dir), "--egtex-dir", args.egtex_dir,
                "--out", str(exp_dir / "ablations"), "--size-encodings", str(run_dir) + "_{mb}mb"] + smoke),
        ("06", ["scripts/06_covariates.py", "--encodings", str(run_dir), "--pred", str(pred_dir), "--egtex-dir", args.egtex_dir,
                "--out", str(exp_dir / "covariates")] + smoke),
        ("07", ["scripts/07_mqtl_deep.py", "--encodings", str(run_dir), "--pred", str(pred_dir), "--egtex-dir", args.egtex_dir,
                "--out", str(exp_dir / "mqtl_deep")] + smoke),
    ]
    for name, cmd in steps:
        log(f"\n----- {name}: {' '.join(cmd)}")
        rc = subprocess.run([sys.executable, *cmd], cwd=REPO, env=env).returncode
        eval_status[name] = "ok" if rc == 0 else f"exit code {rc}"

print(f"\n=== 09 LeWM (lambda = {lam}, SIGReg {args.sigreg_mode}) ===")
print(f"training: {args.epochs} epochs; last pred MSE {history[-1]['pred_mse']:.4f}, SIGReg {history[-1]['sigreg']:.4f}")
print(f"post-hoc decoder (g -> block betas): mean test R^2 {decoder_r2:.3f}")
print("linear probes on the block-mean latent (test donors): accuracy for sex/tissue, R^2 for age / genotype PCs")
names = list(targets)
print(f"{'representation':<18s}" + "".join(f"{n[:12]:>13s}" for n in names))
for r, v in probes.items():
    print(f"{r:<18s}" + "".join(f"{fmt(v[n]):>13s}" for n in names))
if eval_status:
    print("Phase A evaluation on these encodings: " + ", ".join(f"{k} {v}" for k, v in eval_status.items()))
print(f"elapsed: {time.time() - t0:.0f}s")
failed = [k for k, v in eval_status.items() if v != "ok"]
if failed:
    finish(False, f"evaluation step(s) failed on the LeWM encodings: {', '.join(failed)}")
finish(True)
