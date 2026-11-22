"""Phase A / 08: does 3D genome structure explain what the predictor shares between blocks? (test donors)

  python scripts/08_structure.py [--encodings outputs/encodings] [--pred outputs/predictor] [--out outputs/experiments/structure]
                                 [--tracks "Lung=lower lobe of left lung;Colon - Transverse=transverse colon"]
                                 [--hic-url URL] [--n-shifts 200] [--train-seed 0]

docs/PREREG.md section 5 requires a tissue-matched ENCODE compartment track per tested tissue, so every
compartment analysis is done per tissue: that tissue's test samples with that tissue's track.
1. Per tissue, per-block A-compartment fraction from the ENCODE track of the matched biosample
   (experiments/threed/build_annotations.py, built if absent); inter-chromosomal Hi-C contact (hic-straw stream,
   default the GM12878 file used in h3_attention_vs_contacts.py; not tissue-matched, recorded as such) and gene
   density (build_annotations.locus_density).
2. Per tissue, pair-error matrix E[a, b] of the trained predictor on that tissue's test samples (autosomal blocks,
   different chromosomes; one-source error, a joint model gets each source as a set of one).
3. Per tissue, partial Spearman of log E vs compartment similarity and vs log Hi-C contact, controlling gene
   density; circular-shift null of that tissue's track from make_shift_control.py (--n-shifts shifts).
4. mlp_struct: 03 with the compartment profiles of all tissue tracks appended to c_B (c_B is per block, the
   same for every sample, so each tissue's track enters as its own channels), then donor matching vs the predictor.
Needs pyBigWig (only if a track must be built) and hic-straw; without them a full run fails with the reason,
and --smoke uses synthetic tracks and skips Hi-C (both reported).
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
from cpgpt_jepa.predictor import is_joint, load_predictor, standardise_descriptors  # noqa: E402
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

DEFAULT_TRACKS = "Lung=lower lobe of left lung;Colon - Transverse=transverse colon"
p = argparse.ArgumentParser()
p.add_argument("--encodings", "--enc", dest="enc", default="outputs/encodings")
p.add_argument("--pred", default="outputs/predictor")
p.add_argument("--egtex-dir", default="data/egtex")
p.add_argument("--metadata", default="data/egtex/metadata.csv")
p.add_argument("--out", default="outputs/experiments/structure")
p.add_argument("--kind", default="mlp_joint", help="predictor kind (primary: mlp_joint, docs/PREREG.md)")
p.add_argument("--tracks", default=DEFAULT_TRACKS,
               help="'<metadata tissue>=<ENCODE biosample term name>' pairs separated by ';' (one track per tissue)")
p.add_argument("--hic-url", default="https://www.encodeproject.org/files/ENCFF916GWS/@@download/ENCFF916GWS.hic",
               help="default: GM12878 intact Hi-C, as in experiments/threed/h3_attention_vs_contacts.py")
p.add_argument("--hic-cell-type", default="GM12878", help="recorded with the Hi-C result")
p.add_argument("--hic-resolution", type=int, default=2_500_000)
p.add_argument("--n-shifts", type=int, default=200)
p.add_argument("--max-samples", type=int, default=300, help="test samples per tissue used for the pair-error matrix")
p.add_argument("--seed", type=int, default=0, help="sample subset and synthetic smoke tracks")
p.add_argument("--train-seed", type=int, default=0, help="stage 03 --seed for the struct predictor (match the run's seed)")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
if args.smoke:
    args.n_shifts = min(args.n_shifts, 5)
t0 = time.time()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
log = lambda m: print(m, flush=True)  # noqa: E731
tracks = dict(kv.split("=", 1) for kv in args.tracks.split(";") if kv.strip())
if len(set(tracks.values())) < len(tracks):
    finish(False, "the same ENCODE biosample is given for two tissues (docs/PREREG.md section 5 forbids this)")

run = load_run(args.enc, args.pred, args.egtex_dir)
if not (run.pred / args.kind / "model.pt").exists():
    finish(False, f"no trained {args.kind} under {run.pred}")
n_samples, n_blocks, d = run.Z.shape
result = {"kind": args.kind, "train_seed": args.train_seed, "struct_tracks": tracks, "pair_error_by_tissue": {}}
notes = []
has_hicstraw = importlib.util.find_spec("hicstraw") is not None
if not has_hicstraw and not args.smoke:
    finish(False, "hic-straw is not installed in the uv env (needed for the Hi-C contacts; ask before adding it)")

# --- 1. per-tissue annotations -----------------------------------------------------------------
site_rows = np.load(run.enc / "site_embedding_rows.npy")
dens_rows = locus_density_rows()
prof, annot_name, annot_dir_of = {}, {}, {}
result["annotation"] = {}
for tissue, cell in tracks.items():
    name, annot_dir = cell.replace(" ", "_").replace("/", "_"), ANNOT_DIR
    loaded = load_annotation(name)
    if loaded is None:
        try:
            loaded = build_annotation(cell, name, log)
        except Exception as e:  # noqa: BLE001
            if not args.smoke:
                finish(False, f"compartment track for {tissue} ({cell}) could not be built: {type(e).__name__}: {e}")
            name, annot_dir = f"smoke_synthetic_{len(prof)}", out / "annotations"
            loaded = synthetic_annotation(name, annot_dir, args.seed + len(prof))
            notes.append(f"SMOKE: {tissue} track replaced by a synthetic one ({type(e).__name__}: {e})")
    annot, annot_meta = loaded
    prof[tissue] = block_profile(annot, site_rows, run.W_block, n_blocks, dens_rows)
    annot_name[tissue], annot_dir_of[tissue] = name, annot_dir
    result["annotation"][tissue] = {"cell_type": cell, "name": name, "meta": annot_meta,
                                    "mean_coverage": float(np.nanmean(prof[tissue]["coverage"])),
                                    "mean_a_fraction": float(np.nanmean(prof[tissue]["a_fraction"]))}
    log(f"[1] {tissue}: track {name} ({cell}), coverage {np.nanmean(prof[tissue]['coverage']):.3f}, "
        f"A-fraction {np.nanmean(prof[tissue]['a_fraction']):.3f} ({time.time() - t0:.0f}s)")

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
result["hic"] = {"url": args.hic_url, "cell_type": args.hic_cell_type, "tissue_matched": False,
                 "available": contact is not None}
cols = {"block_id": auto, "chrom": [b["chrom"] for b in blocks_auto], "start": [b["start"] for b in blocks_auto]}
for tissue, pr in prof.items():
    cols.update({f"{k}__{annot_name[tissue]}": v[auto] for k, v in pr.items()})
pd.DataFrame(cols).to_csv(out / "block_annotations.csv", index=False)

# --- 2-3. per-tissue pair-error matrix, partial Spearman, shift null ---------------------------
device = torch.device("cpu")
Cn = torch.from_numpy(standardise_descriptors(run.C))
model = load_predictor(run.pred / args.kind, args.kind, d, Cn.size(1), device)
codes = pd.factorize(run.block_chrom)[0]
ia, ib = np.triu_indices(len(auto), k=1)
a_idx, b_idx = auto[ia], auto[ib]
diff = codes[a_idx] != codes[b_idx]
a_idx, b_idx, ia, ib = a_idx[diff], b_idx[diff], ia[diff], ib[diff]
log_hic = None
if contact is not None:
    c = contact[ia, ib]
    eps = np.nanmin(c[c > 0]) / 2 if np.any(c > 0) else 1e-12
    log_hic = np.log(c + eps)

for tissue in tracks:
    t_samples = run.test_s[run.tissue[run.test_s] == tissue]
    entry = {"cell_type": tracks[tissue], "track": annot_name[tissue], "n_test_samples": int(len(t_samples))}
    if len(t_samples) < 2:
        entry["skipped"] = f"{len(t_samples)} test samples of {tissue}"
        result["pair_error_by_tissue"][tissue] = entry
        continue
    rng = np.random.default_rng(args.seed)
    samples = t_samples if len(t_samples) <= args.max_samples else np.sort(rng.choice(t_samples, args.max_samples, replace=False))
    Zs = torch.from_numpy(run.Zs[samples]).float()
    E = np.full((n_blocks, n_blocks), np.nan)
    with torch.no_grad():
        for b in auto:
            srcs = auto[codes[auto] != codes[b]]
            z_a = Zs[:, srcs].unsqueeze(-2) if is_joint(model) else Zs[:, srcs]
            pred = model(z_a, Cn[b].expand(len(samples), len(srcs), -1),
                         torch.full((len(samples), len(srcs)), int(b)))
            E[srcs, b] = ((pred - Zs[:, b : b + 1]) ** 2).mean((0, 2)).numpy()
    tag = annot_name[tissue]
    np.save(out / f"pair_error__{tag}.npy", E)
    logE = np.log((E[a_idx, b_idx] + E[b_idx, a_idx]) / 2)
    pr = prof[tissue]
    dens = pr["density"]
    controls = np.column_stack([dens[a_idx] + dens[b_idx], np.abs(dens[a_idx] - dens[b_idx])])
    af = pr["a_fraction"]
    comp_sim = -np.abs(af[a_idx] - af[b_idx])
    rho_comp = partial_spearman(logE, comp_sim, controls)
    rho_hic = partial_spearman(logE, log_hic, controls) if log_hic is not None else float("nan")
    log(f"[3] {tissue}: partial Spearman compartment {rho_comp:.3f}, Hi-C {rho_hic:.3f}; shifting ({args.n_shifts})")
    null = shifted_a_fractions(annot_name[tissue], annot_dir_of[tissue], range(args.n_shifts), site_rows, run.W_block,
                               n_blocks, out / f"shift_work__{tag}")
    null_rho = np.array([partial_spearman(logE, -np.abs(s[a_idx] - s[b_idx]), controls) for s in null])
    ok = np.isfinite(null_rho)
    p_null = float((1 + np.sum(np.abs(null_rho[ok]) >= abs(rho_comp))) / (1 + ok.sum()))
    entry.update({"n_pairs": int(len(logE)), "n_samples": int(len(samples)),
                  "partial_spearman_logE_vs_compartment_similarity": rho_comp,
                  "partial_spearman_logE_vs_log_hic": rho_hic, "hic_cell_type": args.hic_cell_type,
                  "controls": "gene density (locus density): sum and |difference| over the pair",
                  "shift_null": {"n": int(ok.sum()), "p_two_sided": p_null, "null_mean": float(np.nanmean(null_rho)),
                                 "null_q025_q975": [float(np.nanquantile(null_rho, 0.025)), float(np.nanquantile(null_rho, 0.975))]
                                 if ok.any() else [float("nan")] * 2}})
    result["pair_error_by_tissue"][tissue] = entry
    pd.DataFrame({"block_a": a_idx, "block_b": b_idx, "log_error": logE, "compartment_similarity": comp_sim,
                  "log_hic": log_hic if log_hic is not None else np.nan,
                  "density_sum": controls[:, 0], "density_absdiff": controls[:, 1]}).to_csv(out / f"pairs__{tag}.csv", index=False)
    np.save(out / f"shift_null_rho__{tag}.npy", null_rho)
    log(f"[3] {tissue}: shift null done ({time.time() - t0:.0f}s)")

# --- 4. mlp_struct -----------------------------------------------------------------------------
profile = np.column_stack([col for t in tracks for col in (
    np.nan_to_num(prof[t]["a_fraction"], nan=np.nanmean(prof[t]["a_fraction"])),
    np.nan_to_num(prof[t]["mean_value"], nan=0.0))])
np.savez(out / "struct_profile.npz", profile=profile, channels=np.array(
    [f"{k}__{annot_name[t]}" for t in tracks for k in ("a_fraction", "mean_value")]))
cmd = [sys.executable, "scripts/03_train_predictor.py", "--enc", str(run.enc), "--out", str(run.pred),
       "--metadata", args.metadata, "--kinds", args.kind, "--struct", str(out / "struct_profile.npz"),
       "--tag", f"{args.kind}_struct", "--seed", str(args.train_seed), "--allow-cpu"] + (["--smoke"] if args.smoke else [])
env = {k: v for k, v in os.environ.items() if k != "SANITY_FILE"}
log(f"[4] training {args.kind}_struct: " + " ".join(cmd))
subprocess.run(cmd, check=True, cwd=REPO, env=env)
metrics = json.loads((run.pred / "metrics.json").read_text())
result["mlp_struct"] = {}
for kind in (args.kind, f"{args.kind}_struct"):
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
print("\n=== 08 STRUCTURE (test donors) ===")
for n in notes:
    print(n)
print(f"Hi-C: {'yes' if contact is not None else 'no'} ({args.hic_cell_type}, not tissue-matched)")
for tissue, pe in result["pair_error_by_tissue"].items():
    a = result["annotation"][tissue]
    print(f"{tissue}: track {pe['track']} ({pe['cell_type']}; coverage {a['mean_coverage']:.3f}, A-fraction {a['mean_a_fraction']:.3f})")
    if "skipped" in pe:
        print(f"   skipped: {pe['skipped']}")
        continue
    sn = pe["shift_null"]
    print(f"   {pe['n_pairs']} inter-chromosomal block pairs, {pe['n_samples']} test samples")
    print(f"   partial Spearman (control: gene density)  log E vs compartment similarity: "
          f"{fmt(pe['partial_spearman_logE_vs_compartment_similarity'])}   shift null ({sn['n']}): mean {fmt(sn['null_mean'])}, "
          f"95% [{fmt(sn['null_q025_q975'][0])}, {fmt(sn['null_q025_q975'][1])}], p {fmt(sn['p_two_sided'])}")
    print(f"   partial Spearman (control: gene density)  log E vs log Hi-C contact:        "
          f"{fmt(pe['partial_spearman_logE_vs_log_hic'])}")
print(f"{'predictor':<18s}|{'l->c top1':>10s}{'MRR':>7s} |{'c->l top1':>10s}{'MRR':>7s} |{'test MSE (train blocks)':>24s}")
for r in rows:
    print(f"{r['predictor']:<18s}|{fmt(r['l2c_top1']):>10s}{fmt(r['l2c_mrr']):>7s} |{fmt(r['c2l_top1']):>10s}"
          f"{fmt(r['c2l_mrr']):>7s} |{fmt(r['test_mse_train_blocks'], '.4f'):>24s}")
print(f"elapsed: {time.time() - t0:.0f}s")
finish(True)
