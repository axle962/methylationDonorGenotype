"""Step 5: is CpGPT's leftover error organised by 3D genome structure?

H3 showed the model's attention ignores contacts. That leaves two possibilities,
and they lead to opposite decisions:

  - the model's remaining error IS structured by 3D  -> headroom, a prior could
    help, proceed to H1
  - the remaining error is NOT structured by 3D      -> low ceiling, adding the
    prior cannot help much, write up as a negative result

This measures that directly. Hold out half the sites in a window, reconstruct
them, and take per-site residuals across many samples. If two sites that are
3D-proximal tend to be wrong *together*, after controlling for linear distance,
the model is missing something that folding explains.

Usage:
    python experiments/threed/h3b_residual_structure.py --n-samples 100
"""

import argparse
import json
import sys
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
from cpgpt.model.cpgpt_module import m_to_beta
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer

from h3_attention_vs_contacts import (  # noqa: E402
    fetch_contacts, observed_over_expected, stratified_spearman,
)

ROOT = Path(__file__).resolve().parents[2]


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gse", default="GSE40279")
    ap.add_argument("--chrom", default="1")
    ap.add_argument("--start", type=int, default=30_000_000)
    ap.add_argument("--end", type=int, default=50_000_000)
    ap.add_argument("--resolution", type=int, default=25_000)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--max-sites", type=int, default=3000)
    ap.add_argument("--min-sd", type=float, default=0.0,
                   help="keep only sites whose true beta varies at least this much "
                        "across samples; below it, co-variation is mostly noise")
    ap.add_argument("--tag", default=None,
                   help="run directory suffix; defaults to gse_chrom, which would "
                        "collide between two windows on the same chromosome")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    res = download_cpgpt(model="small", species="human")
    deps = str(res.dependencies_path)
    emb = DNALLMEmbedder(dependencies_dir=deps)
    prober = IlluminaMethylationProber(dependencies_dir=deps, embedder=emb)

    df = pd.read_feather(ROOT / f"data/geo/processed/{args.gse}_betas.arrow").set_index("GSM_ID")
    vocab = set(json.load(open(res.vocab_path))["input"])
    df = df.loc[:, df.columns.isin(vocab)]
    if len(df) > args.n_samples:
        idx = np.linspace(0, len(df) - 1, args.n_samples).round().astype(int)
        df = df.iloc[np.unique(idx)]

    probes = list(df.columns)
    locs = prober.locate_probes(probes, "homo_sapiens")
    keep, pos = [], []
    for probe, loc in zip(probes, locs):
        c, p = loc.split(":")
        if c == args.chrom and args.start <= int(p) < args.end:
            keep.append(probe)
            pos.append(int(p))
    o = np.argsort(pos)
    keep = [keep[i] for i in o]
    pos = np.array([pos[i] for i in o])
    if len(keep) > args.max_sites:
        sel = np.linspace(0, len(keep) - 1, args.max_sites).round().astype(int)
        keep = [keep[i] for i in sel]
        pos = pos[sel]

    pos_by_probe = dict(zip(keep, pos))
    window = df[keep].dropna(axis=1, how="any")
    keep = list(window.columns)
    pos = np.array([pos_by_probe[k] for k in keep])
    log(f"[data] {window.shape[0]} samples x {window.shape[1]} complete sites in "
        f"chr{args.chrom}:{args.start:,}-{args.end:,}")

    # Half the sites become targets; the rest stay as input context.
    n = len(keep)
    perm = rng.permutation(n)
    tgt_idx = np.sort(perm[: n // 2])
    targets = [keep[i] for i in tgt_idx]
    tgt_pos = pos[tgt_idx]
    truth = window[targets].to_numpy(dtype="float32")

    masked = window.copy()
    masked[targets] = np.nan
    log(f"[split] {len(targets)} target sites held out, {n - len(targets)} kept as input")

    tag = args.tag or f"{args.gse}_chr{args.chrom}"
    run = ROOT / f"data/3d/h3b_{tag}"
    (run / "raw").mkdir(parents=True, exist_ok=True)
    masked.reset_index().to_feather(run / "raw/b.arrow")
    CpGPTDataSaver(data_paths=str(run / "raw/b.arrow"),
                   processed_dir=str(run / "proc")).process_files(prober, emb)

    inf = CpGPTInferencer(dependencies_dir=str(ROOT / "dependencies"), data_dir=str(ROOT / "data"))
    cfg = inf.load_cpgpt_config(str(res.config_path))
    model = inf.load_cpgpt_model(cfg, model_ckpt_path=str(res.checkpoint_path), strict_load=True)
    dm = CpGPTDataModule(
        predict_dir=str(run / "proc"), dependencies_dir=deps, batch_size=1, num_workers=0,
        max_length=max(5000, n + 10), dna_llm=cfg.data.dna_llm,
        dna_context_len=cfg.data.dna_context_len,
        sorting_strategy=cfg.data.sorting_strategy, pin_memory=False)

    locations = prober.locate_probes(targets, "homo_sapiens")
    trainer = CpGPTTrainer(accelerator="cpu", devices=1, enable_progress_bar=False)
    log(f"[predict] reconstructing {len(targets)} sites for {len(masked)} samples")
    pred = m_to_beta(trainer.predict(
        model=model, datamodule=dm, predict_mode="reconstruct",
        genomic_locations=locations, species="homo_sapiens",
        return_keys=["pred_meth"])["pred_meth"]).float().numpy()

    resid = pred - truth
    mae = np.abs(resid).mean()
    log(f"[predict] beta MAE on held-out window sites: {mae:.4f}")

    np.savez_compressed(run / "residuals.npz", resid=resid, truth=truth,
                        pred=pred, pos=tgt_pos)

    # How much does the model's prediction actually move between samples? If it
    # returns near the site mean for everyone, residual co-variation is just the
    # data's own co-variation, and at invariant sites that is pure noise.
    log(f"[diag] per-site SD across samples: truth {truth.std(axis=0).mean():.4f}, "
        f"pred {pred.std(axis=0).mean():.4f}, resid {resid.std(axis=0).mean():.4f}")

    # Remove each site's mean bias, so what is left is sample-to-sample variation.
    resid_c = resid - resid.mean(axis=0, keepdims=True)
    sd = resid_c.std(axis=0)
    good = sd > 1e-6
    resid_c, tgt_pos_g = resid_c[:, good], tgt_pos[good]
    log(f"[resid] {good.sum()} sites with variable residuals")

    if args.min_sd > 0:
        keep_var = truth[:, good].std(axis=0) >= args.min_sd
        log(f"[filter] {keep_var.sum()} of {good.sum()} sites have truth SD >= "
            f"{args.min_sd}; restricting to those")
        if keep_var.sum() < 50:
            log("[filter] too few variable sites; keeping all")
        else:
            resid_c, tgt_pos_g = resid_c[:, keep_var], tgt_pos_g[keep_var]

    corr = np.corrcoef(resid_c.T)
    iu = np.triu_indices(corr.shape[0], k=1)
    rc = corr[iu]
    dist = np.abs(tgt_pos_g[iu[0]] - tgt_pos_g[iu[1]]).astype(float)

    mat = fetch_contacts(args.chrom, args.start, args.end, args.resolution)
    oe = observed_over_expected(mat)
    bins = ((tgt_pos_g - args.start) // args.resolution).astype(int)
    ok = (bins >= 0) & (bins < oe.shape[0])
    pair_ok = ok[iu[0]] & ok[iu[1]]
    contact = np.full(len(rc), np.nan)
    contact[pair_ok] = oe[bins[iu[0]][pair_ok], bins[iu[1]][pair_ok]]

    ann_path = ROOT / "data/3d/annotations/pool5.mmap"
    same_comp = None
    if ann_path.exists():
        meta = json.loads(ann_path.with_suffix(".json").read_text())
        ann = np.memmap(ann_path, dtype="float32", mode="r", shape=tuple(meta["shape"]))
        index = emb.ensembl_metadata_dict["homo_sapiens"][
            "nucleotide-transformer-v2-500m-multi-species"][2001]
        rows = np.array([index[f"{args.chrom}:{p}"] for p in tgt_pos_g])
        eig = np.asarray(ann[rows, 0])
        same_comp = (np.sign(eig[iu[0]]) == np.sign(eig[iu[1]])).astype(float)

    log("")
    log("=" * 62)
    log("RESIDUAL STRUCTURE TEST")
    log("=" * 62)

    rho_dist = spearmanr(-np.log10(dist), rc, nan_policy="omit").statistic
    log(f"positive control, residual corr vs linear proximity : rho {rho_dist:+.4f}")

    rho_raw = spearmanr(rc[pair_ok], contact[pair_ok], nan_policy="omit").statistic
    rho_ctrl, npairs = stratified_spearman(rc, contact, dist)
    log(f"residual corr vs contact, raw                       : rho {rho_raw:+.4f}")
    log(f"residual corr vs contact, distance-controlled        : rho {rho_ctrl:+.4f} "
        f"({npairs:,} pairs)")

    if same_comp is not None:
        rho_comp, ncp = stratified_spearman(rc, same_comp, dist)
        a = rc[(same_comp == 1) & np.isfinite(rc)]
        b = rc[(same_comp == 0) & np.isfinite(rc)]
        log(f"residual corr vs same-compartment, dist-controlled   : rho {rho_comp:+.4f}")
        log(f"  mean residual corr, same compartment  : {a.mean():+.4f} (n={len(a):,})")
        log(f"  mean residual corr, diff compartment  : {b.mean():+.4f} (n={len(b):,})")

    out = ROOT / "results/h3b_residual_structure.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({
        "metric": ["mae", "rho_distance_control", "rho_contact_raw",
                   "rho_contact_dist_controlled"],
        "value": [mae, rho_dist, rho_raw, rho_ctrl],
    }).to_csv(out, index=False)
    log(f"\n[write] {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
