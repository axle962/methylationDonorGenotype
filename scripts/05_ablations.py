"""Phase A / 05: ablations of the residual (test donors only).

  python scripts/05_ablations.py [--encodings outputs/encodings] [--pred outputs/predictor] [--out outputs/experiments/ablations]

1. K source blocks in {1, 4, 8, 32}: residual from the trained mlp, donor matching for each K.
2. Block size in {5, 10, 20} Mb: re-partitioned encodings from outputs/encodings_<N>mb (slurm/02b_encode_sizes.sbatch);
   sizes whose encodings are absent are skipped and reported. A fresh mlp is trained per size (same splits).
3. Donor matching for every ordered tissue pair with >= --min-shared shared test donors (top-1 matrix).
4. Per-donor residual averaged over the donor's other tissues, leave-one-tissue-out retrieval.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cpgpt_jepa import finish  # noqa: E402
from cpgpt_jepa.evaluation import (  # noqa: E402
    donor_matching,
    fmt,
    load_run,
    rank_of_true,
    ranks_to_metrics,
    retrieval,
    tissue_centred,
    write_outputs,
)
from cpgpt_jepa.predictor import (  # noqa: E402
    fixed_sources,
    load_predictor,
    make_splits,
    predict,
    standardise_descriptors,
    train_predictor,
)

p = argparse.ArgumentParser()
p.add_argument("--encodings", "--enc", dest="enc", default="outputs/encodings")
p.add_argument("--pred", default="outputs/predictor")
p.add_argument("--egtex-dir", default="data/egtex")
p.add_argument("--out", default="outputs/experiments/ablations")
p.add_argument("--kind", default="mlp_joint", help="predictor kind (primary: mlp_joint, docs/PREREG.md)")
p.add_argument("--ks", default="1,4,8,32")
p.add_argument("--sizes-mb", default="5,10,20")
p.add_argument("--size-encodings", default="outputs/encodings_{mb}mb", help="pattern for the other block sizes")
p.add_argument("--min-shared", type=int, default=20)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
if args.smoke:
    args.min_shared = min(args.min_shared, 2)
t0 = time.time()
device = torch.device("cpu")

run = load_run(args.enc, args.pred, args.egtex_dir)
if not (run.pred / args.kind / "model.pt").exists():
    finish(False, f"no trained {args.kind} under {run.pred}")
n_samples, n_blocks, d = run.Z.shape
block_codes = pd.factorize(run.block_chrom)[0]
result, rows = {"kind": args.kind}, []
Zs_t = torch.from_numpy(run.Zs).float()
Cn = torch.from_numpy(standardise_descriptors(run.C))
model = load_predictor(run.pred / args.kind, args.kind, d, Cn.size(1), device)
R_main = run.residual(args.kind)


def dm_entry(section: str, label: str, m: dict, **extra) -> dict:
    a, b = m["lung_to_colon"], m["colon_to_lung"]
    return {"section": section, "setting": label, **extra,
            "l2c_top1": a["top1"], "l2c_top5": a["top5"], "l2c_mrr": a["mrr"],
            "c2l_top1": b["top1"], "c2l_top5": b["top5"], "c2l_mrr": b["mrr"], "chance": a["chance"]}


# --- 1. number of source blocks ----------------------------------------------------------------
result["k_sources"] = {}
for k in (int(x) for x in args.ks.split(",")):
    sources = fixed_sources(block_codes, k, args.seed, pool=run.auto_blocks)
    P, _ = predict(model, Zs_t, Cn, sources)
    m = donor_matching(run, (Zs_t - P).numpy())
    result["k_sources"][str(k)] = {"k_used": int(sources.shape[1]), "donor_matching": m}
    rows.append(dm_entry("k_sources", f"K={k}", m, k_used=int(sources.shape[1])))
    print(f"[K={k}] done (K used {sources.shape[1]}; {time.time() - t0:.0f}s)", flush=True)

# --- 2. block size -----------------------------------------------------------------------------
result["block_size"] = {}
main_mb = run.enc_meta["window_bp"] // 1_000_000
for mb in (int(x) for x in args.sizes_mb.split(",")):
    if mb == main_mb:
        m = donor_matching(run, R_main)
        result["block_size"][f"{mb}Mb"] = {"source": str(run.enc), "n_blocks": n_blocks, "donor_matching": m}
        rows.append(dm_entry("block_size", f"{mb}Mb", m, n_blocks=n_blocks))
        continue
    enc_dir = Path(args.size_encodings.format(mb=mb))
    if not (enc_dir / "Z.npy").exists():
        result["block_size"][f"{mb}Mb"] = {"skipped": f"no encodings at {enc_dir} (run slurm/02b_encode_sizes.sbatch)"}
        print(f"[{mb} Mb] skipped: no encodings at {enc_dir}", flush=True)
        continue
    obs = np.load(enc_dir / "obs_names.npy", allow_pickle=True)  # written by stage 2
    if list(obs) != list(run.obs_names):
        result["block_size"][f"{mb}Mb"] = {"skipped": f"{enc_dir} has different samples"}
        continue
    Z_b = np.load(enc_dir / "Z.npy")
    blocks_b = np.load(enc_dir / "blocks.npy", allow_pickle=True)  # written by stage 2
    chrom_b = np.array([b["chrom"] for b in blocks_b])
    auto_b, x_b = np.flatnonzero(chrom_b != "X"), np.flatnonzero(chrom_b == "X")
    codes_b = pd.factorize(chrom_b)[0]
    sp = make_splits(run.donor, run.tissue, len(blocks_b), seed=args.seed, eval_only_blocks=x_b)
    tr_s = np.array(sp["train_samples"])
    z_tr = Z_b[tr_s][:, auto_b].reshape(-1, Z_b.shape[2])
    Zs_b = torch.from_numpy((Z_b - z_tr.mean(0)) / z_tr.std(0)).float()
    Cn_b = torch.from_numpy(standardise_descriptors(np.load(enc_dir / "C.npy")))
    model_b, _ = train_predictor(args.kind, Zs_b, Cn_b, tr_s, np.array(sp["train_blocks"]), codes_b, auto_b,
                                 seed=args.seed, k_sources=8, log=lambda msg: None)
    P_b, _ = predict(model_b, Zs_b, Cn_b, fixed_sources(codes_b, 8, args.seed, pool=auto_b))
    R_b = (Zs_b - P_b).numpy()[:, auto_b]
    m = donor_matching(run, R_b, blocks=np.arange(R_b.shape[1]))
    result["block_size"][f"{mb}Mb"] = {"source": str(enc_dir), "n_blocks": len(blocks_b), "donor_matching": m}
    rows.append(dm_entry("block_size", f"{mb}Mb", m, n_blocks=len(blocks_b)))
    print(f"[{mb} Mb] {len(blocks_b)} blocks ({time.time() - t0:.0f}s)", flush=True)

# --- 3. all tissue pairs -----------------------------------------------------------------------
test = run.test_s
reps = {"residual R": R_main[:, run.auto_blocks].reshape(n_samples, -1),
        "Z - tissue mean": tissue_centred(run, run.Zs)[:, run.auto_blocks].reshape(n_samples, -1)}
tissues = sorted(set(run.tissue[test]))
pair_rows, result["tissue_pairs"] = [], {}
for t1 in tissues:
    for t2 in tissues:
        if t1 == t2:
            continue
        q = test[run.tissue[test] == t1]
        g = test[run.tissue[test] == t2]
        shared = set(run.donor[q]) & set(run.donor[g])
        if len(shared) < args.min_shared:
            continue
        q = q[np.isin(run.donor[q], sorted(shared))]
        entry = {name: retrieval(rep, run.donor, q, g) for name, rep in reps.items()}
        result["tissue_pairs"][f"{t1} -> {t2}"] = {"n_shared_donors": len(shared), **entry}
        pair_rows.append({"query": t1, "gallery": t2, "n_shared": len(shared),
                          **{f"{name} top1": entry[name]["top1"] for name in reps},
                          **{f"{name} mrr": entry[name]["mrr"] for name in reps}, "chance": entry["residual R"]["chance"]})

# --- 4. per-donor average, leave one tissue out ------------------------------------------------
result["donor_average_lot"] = {}
for name, rep in reps.items():
    ranks, n_gal = [], []
    for t in tissues:
        query = test[run.tissue[test] == t]
        others = test[run.tissue[test] != t]
        gal_donors = np.array(sorted(set(run.donor[others])))
        if len(gal_donors) == 0:
            continue
        gallery = np.stack([rep[others[run.donor[others] == dn]].mean(0) for dn in gal_donors])
        query = query[np.isin(run.donor[query], gal_donors)]
        if len(query) == 0:
            continue
        qv = rep[query] / (np.linalg.norm(rep[query], axis=1, keepdims=True) + 1e-12)
        gv = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
        ranks.append(rank_of_true(qv @ gv.T, run.donor[query], gal_donors))
        n_gal += [len(gal_donors)] * len(query)
    if ranks:
        rk = np.concatenate(ranks)
        m = ranks_to_metrics(rk, int(np.round(np.mean(n_gal))))
        m["chance"] = float(np.mean(1 / np.array(n_gal)))
        result["donor_average_lot"][name] = m
        rows.append({"section": "donor_average_lot", "setting": name, "l2c_top1": m["top1"], "l2c_top5": m["top5"],
                     "l2c_mrr": m["mrr"], "chance": m["chance"]})

out = Path(args.out)
write_outputs(out, result, rows)
if pair_rows:
    pd.DataFrame(pair_rows).to_csv(out / "tissue_pairs.csv", index=False)
    for name in reps:
        mat = pd.DataFrame(pair_rows).pivot(index="query", columns="gallery", values=f"{name} top1")
        mat.to_csv(out / f"tissue_pairs_top1_{name.replace(' ', '_')}.csv")

# --- report ------------------------------------------------------------------------------------
print("\n=== 05 ABLATIONS (test donors) ===")
print(f"{'section':<12s}{'setting':<18s}|{'top1':>7s}{'top5':>7s}{'MRR':>7s} |{'top1':>7s}{'top5':>7s}{'MRR':>7s} |{'chance':>8s}")
print(f"{'':<30s}|{'lung -> colon':^21s} |{'colon -> lung':^21s} |")
for r in rows:
    if r["section"] == "donor_average_lot":
        continue
    print(f"{r['section']:<12s}{r['setting']:<18s}|{fmt(r['l2c_top1']):>7s}{fmt(r['l2c_top5']):>7s}{fmt(r['l2c_mrr']):>7s} |"
          f"{fmt(r['c2l_top1']):>7s}{fmt(r['c2l_top5']):>7s}{fmt(r['c2l_mrr']):>7s} |{fmt(r['chance'], '.4f'):>8s}")
for mb, v in result["block_size"].items():
    if "skipped" in v:
        print(f"{'block_size':<12s}{mb:<18s}| skipped: {v['skipped']}")
print("\nDonor-average leave-one-tissue-out retrieval (query sample vs donors' mean over their other tissues):")
for name, m in result["donor_average_lot"].items():
    print(f"  {name:<18s} top1 {fmt(m['top1'])}  top5 {fmt(m['top5'])}  MRR {fmt(m['mrr'])}  chance {fmt(m['chance'], '.4f')}  (n={m['n_query']})")
print(f"\nTissue pairs with >= {args.min_shared} shared test donors: {len(pair_rows)} (top-1 matrices in {out}/tissue_pairs_top1_*.csv)")
for r in pair_rows:
    print(f"  {r['query'][:22]:<22s} -> {r['gallery'][:22]:<22s} n={r['n_shared']:>3d}  R top1 {fmt(r['residual R top1'])}"
          f"  Z-tissue top1 {fmt(r['Z - tissue mean top1'])}  chance {fmt(r['chance'], '.3f')}")
print(f"elapsed: {time.time() - t0:.0f}s")
finish(True)
