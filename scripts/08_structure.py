"""Phase A / 08: does 3D genome structure explain what the predictor shares between blocks? (test donors)

  python scripts/08_structure.py [--encodings outputs/encodings] [--pred outputs/predictor] [--out outputs/experiments/structure]
                                 [--cell-type IMR-90] [--hic-url URL] [--n-shifts 200]

1. Per-block A-compartment fraction (experiments/threed/build_annotations.py track, built if absent), Hi-C
   inter-chromosomal contact (hic-straw stream, default the GM12878 file used in h3_attention_vs_contacts.py),
   and gene density (build_annotations.locus_density).
2. Pair-error matrix E[a, b] of the trained mlp on test donors (autosomal blocks, different chromosomes).
3. Partial Spearman of log E vs compartment similarity and vs log Hi-C contact, controlling gene density;
   circular-shift null for the compartment term from make_shift_control.py (--n-shifts shifts).
4. mlp_struct: 03 with the per-block compartment profile appended to c_B, then donor matching vs mlp.
Needs pyBigWig (only if the track must be built) and hic-straw; without them a full run fails with the reason,
and --smoke uses a synthetic track and skips Hi-C (both reported).
"""
import argparse
import importlib.util
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
from cpgpt_jepa.evaluation import donor_matching, fmt, load_run, write_outputs  # noqa: E402
from cpgpt_jepa.predictor import load_predictor, standardise_descriptors  # noqa: E402
from cpgpt_jepa.structure import (  # noqa: E402
    ANNOT_DIR,
    block_profile,
    build_annotation,
    interchrom_contacts,
    load_annotation,
    locus_density_rows,
    partial_spearman,
    shifted_a_fractions,
    synthetic_annotation,
)

p = argparse.ArgumentParser()
p.add_argument("--encodings", "--enc", dest="enc", default="outputs/encodings")
p.add_argument("--pred", default="outputs/predictor")
p.add_argument("--egtex-dir", default="data/egtex")
p.add_argument("--metadata", default="data/egtex/metadata.csv")
p.add_argument("--out", default="outputs/experiments/structure")
p.add_argument("--kind", default="mlp")
p.add_argument("--cell-type", default="IMR-90", help="ENCODE biosample for the compartment track")
p.add_argument("--annotation", default=None, help="annotation name under data/3d/annotations (default: cell type)")
p.add_argument("--hic-url", default="https://www.encodeproject.org/files/ENCFF916GWS/@@download/ENCFF916GWS.hic",
               help="default: GM12878 intact Hi-C, as in experiments/threed/h3_attention_vs_contacts.py")
p.add_argument("--hic-resolution", type=int, default=2_500_000)
p.add_argument("--n-shifts", type=int, default=200)
p.add_argument("--max-samples", type=int, default=300, help="test samples used for the pair-error matrix")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
if args.smoke:
    args.n_shifts = min(args.n_shifts, 5)
t0 = time.time()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
log = lambda m: print(m, flush=True)  # noqa: E731

run = load_run(args.enc, args.pred, args.egtex_dir)
if not (run.pred / args.kind / "model.pt").exists():
    finish(False, f"no trained {args.kind} under {run.pred}")
n_samples, n_blocks, d = run.Z.shape
result, notes = {"kind": args.kind}, []
has_hicstraw = importlib.util.find_spec("hicstraw") is not None
if not has_hicstraw and not args.smoke:
    finish(False, "hic-straw is not installed in the uv env (needed for the Hi-C contacts; ask before adding it)")

# --- 1. per-block annotations ------------------------------------------------------------------
site_rows = np.load(run.enc / "site_embedding_rows.npy")
name = args.annotation or args.cell_type.replace(" ", "_")
annot_dir = ANNOT_DIR
loaded = load_annotation(name)
if loaded is None:
    try:
        loaded = build_annotation(args.cell_type, name, log)
    except Exception as e:  # noqa: BLE001
        if not args.smoke:
            finish(False, f"compartment track {name} could not be built: {type(e).__name__}: {e}")
        name, annot_dir = "smoke_synthetic", out / "annotations"
        loaded = synthetic_annotation(name, annot_dir, args.seed)
        notes.append(f"SMOKE: compartment track replaced by a synthetic one ({type(e).__name__}: {e})")
annot, annot_meta = loaded
dens_rows = locus_density_rows()
prof = block_profile(annot, site_rows, run.W_block, n_blocks, dens_rows)
result["annotation"] = {"name": name, "meta": annot_meta,
                        "mean_coverage": float(np.nanmean(prof["coverage"])),
                        "mean_a_fraction": float(np.nanmean(prof["a_fraction"]))}
log(f"[1] annotation {name}: coverage {np.nanmean(prof['coverage']):.3f}, "
    f"A-fraction {np.nanmean(prof['a_fraction']):.3f} ({time.time() - t0:.0f}s)")

auto = run.auto_blocks
blocks_auto = [run.blocks[b] for b in auto]
contact = None
if has_hicstraw:
    try:
        contact = interchrom_contacts(blocks_auto, args.hic_url, args.hic_resolution, log=log)
    except Exception as e:  # noqa: BLE001
        if not args.smoke:
            finish(False, f"Hi-C contacts could not be streamed: {type(e).__name__}: {e}")
        notes.append(f"SMOKE: Hi-C skipped ({type(e).__name__}: {e})")
else:
    notes.append("SMOKE: Hi-C skipped (hic-straw not installed)")
pd.DataFrame({"block_id": auto, "chrom": [b["chrom"] for b in blocks_auto], "start": [b["start"] for b in blocks_auto],
              **{k: v[auto] for k, v in prof.items()}}).to_csv(out / "block_annotations.csv", index=False)

# --- 2. pair-error matrix ----------------------------------------------------------------------
device = torch.device("cpu")
Cn = torch.from_numpy(standardise_descriptors(run.C))
model = load_predictor(run.pred / args.kind, args.kind, d, Cn.size(1), device)
rng = np.random.default_rng(args.seed)
samples = run.test_s if len(run.test_s) <= args.max_samples else np.sort(rng.choice(run.test_s, args.max_samples, replace=False))
Zs = torch.from_numpy(run.Zs[samples]).float()
codes = pd.factorize(run.block_chrom)[0]
E = np.full((n_blocks, n_blocks), np.nan)
with torch.no_grad():
    for b in auto:
        srcs = auto[codes[auto] != codes[b]]
        pred = model(Zs[:, srcs], Cn[b].expand(len(samples), len(srcs), -1),
                     torch.full((len(samples), len(srcs)), int(b)))
        E[srcs, b] = ((pred - Zs[:, b : b + 1]) ** 2).mean((0, 2)).numpy()
np.save(out / "pair_error.npy", E)
log(f"[2] pair-error matrix over {len(auto)} blocks, {len(samples)} test samples ({time.time() - t0:.0f}s)")

# --- 3. partial Spearman + circular-shift null -------------------------------------------------
ia, ib = np.triu_indices(len(auto), k=1)
a_idx, b_idx = auto[ia], auto[ib]
diff = codes[a_idx] != codes[b_idx]
a_idx, b_idx, ia, ib = a_idx[diff], b_idx[diff], ia[diff], ib[diff]
logE = np.log((E[a_idx, b_idx] + E[b_idx, a_idx]) / 2)
dens = prof["density"]
controls = np.column_stack([dens[a_idx] + dens[b_idx], np.abs(dens[a_idx] - dens[b_idx])])
af = prof["a_fraction"]
comp_sim = -np.abs(af[a_idx] - af[b_idx])
rho_comp = partial_spearman(logE, comp_sim, controls)
rho_hic = float("nan")
if contact is not None:
    c = contact[ia, ib]
    eps = np.nanmin(c[c > 0]) / 2 if np.any(c > 0) else 1e-12
    rho_hic = partial_spearman(logE, np.log(c + eps), controls)
log(f"[3] partial Spearman: compartment {rho_comp:.3f}, Hi-C {rho_hic:.3f}; shifting ({args.n_shifts})")
null = shifted_a_fractions(name, annot_dir, range(args.n_shifts), site_rows, run.W_block, n_blocks,
                           out / "shift_work")
null_rho = np.array([partial_spearman(logE, -np.abs(s[a_idx] - s[b_idx]), controls) for s in null])
p_null = float((1 + np.sum(np.abs(null_rho) >= abs(rho_comp))) / (1 + np.isfinite(null_rho).sum()))
result["pair_error"] = {"n_pairs": int(len(logE)), "n_samples": int(len(samples)),
                        "partial_spearman_logE_vs_compartment_similarity": rho_comp,
                        "partial_spearman_logE_vs_log_hic": rho_hic,
                        "controls": "gene density (locus density): sum and |difference| over the pair",
                        "shift_null": {"n": int(np.isfinite(null_rho).sum()), "p_two_sided": p_null,
                                       "null_mean": float(np.nanmean(null_rho)),
                                       "null_q025_q975": [float(np.nanquantile(null_rho, 0.025)), float(np.nanquantile(null_rho, 0.975))]
                                       if np.isfinite(null_rho).any() else [float("nan")] * 2}}
pd.DataFrame({"block_a": a_idx, "block_b": b_idx, "log_error": logE, "compartment_similarity": comp_sim,
              "log_hic": np.log(contact[ia, ib] + 1e-12) if contact is not None else np.nan,
              "density_sum": controls[:, 0], "density_absdiff": controls[:, 1]}).to_csv(out / "pairs.csv", index=False)
np.save(out / "shift_null_rho.npy", null_rho)
log(f"[3] shift null done ({time.time() - t0:.0f}s)")

# --- 4. mlp_struct -----------------------------------------------------------------------------
profile = np.column_stack([np.nan_to_num(prof["a_fraction"], nan=np.nanmean(prof["a_fraction"])),
                           np.nan_to_num(prof["mean_value"], nan=0.0)])
np.savez(out / "struct_profile.npz", profile=profile)
cmd = [sys.executable, "scripts/03_train_predictor.py", "--enc", str(run.enc), "--out", str(run.pred),
       "--metadata", args.metadata, "--kinds", "mlp", "--struct", str(out / "struct_profile.npz"),
       "--tag", "mlp_struct", "--allow-cpu"] + (["--smoke"] if args.smoke else [])
env = {k: v for k, v in os.environ.items() if k != "SANITY_FILE"}
log("[4] training mlp_struct: " + " ".join(cmd))
subprocess.run(cmd, check=True, cwd=REPO, env=env)
metrics = json.loads((run.pred / "metrics.json").read_text())
result["mlp_struct"] = {}
for kind in ("mlp", "mlp_struct"):
    result["mlp_struct"][kind] = {"donor_matching": donor_matching(run, run.residual(kind)),
                                  "test_mse_train_blocks": metrics[kind]["train_blocks"]["test_mse"],
                                  "test_mse_heldout_blocks": metrics[kind].get("heldout_blocks", {}).get("test_mse")}
result["notes"] = notes
rows = [{"predictor": k, "l2c_top1": v["donor_matching"]["lung_to_colon"]["top1"],
         "l2c_mrr": v["donor_matching"]["lung_to_colon"]["mrr"], "c2l_top1": v["donor_matching"]["colon_to_lung"]["top1"],
         "c2l_mrr": v["donor_matching"]["colon_to_lung"]["mrr"], "test_mse_train_blocks": v["test_mse_train_blocks"]}
        for k, v in result["mlp_struct"].items()]
write_outputs(out, result, rows)

# --- report ------------------------------------------------------------------------------------
pe = result["pair_error"]
print("\n=== 08 STRUCTURE (test donors) ===")
for n in notes:
    print(n)
print(f"annotation: {name}  (coverage {result['annotation']['mean_coverage']:.3f}, mean A-fraction "
      f"{result['annotation']['mean_a_fraction']:.3f});  Hi-C: {'yes' if contact is not None else 'no'}")
print(f"pair-error matrix: {pe['n_pairs']} inter-chromosomal block pairs, {pe['n_samples']} test samples")
print(f"partial Spearman (control: gene density)  log E vs compartment similarity: {fmt(rho_comp)}"
      f"   shift null ({pe['shift_null']['n']}): mean {fmt(pe['shift_null']['null_mean'])}, "
      f"95% [{fmt(pe['shift_null']['null_q025_q975'][0])}, {fmt(pe['shift_null']['null_q025_q975'][1])}], p {fmt(p_null)}")
print(f"partial Spearman (control: gene density)  log E vs log Hi-C contact:        {fmt(rho_hic)}")
print(f"{'predictor':<12s}|{'l->c top1':>10s}{'MRR':>7s} |{'c->l top1':>10s}{'MRR':>7s} |{'test MSE (train blocks)':>24s}")
for r in rows:
    print(f"{r['predictor']:<12s}|{fmt(r['l2c_top1']):>10s}{fmt(r['l2c_mrr']):>7s} |{fmt(r['c2l_top1']):>10s}"
          f"{fmt(r['c2l_mrr']):>7s} |{fmt(r['test_mse_train_blocks'], '.4f'):>24s}")
print(f"elapsed: {time.time() - t0:.0f}s")
finish(True)
