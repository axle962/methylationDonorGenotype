"""Phase A / 06: regress known covariates out of the residual and repeat donor matching (test donors).

  python scripts/06_covariates.py [--encodings outputs/encodings] [--pred outputs/predictor] [--out outputs/experiments/covariates]

Per block (all blocks and dimensions in one multi-output least-squares fit over the test samples):
  M1 tissue + sex + age + smoking   M2 + plate   M3 + sentrix_id   M4 = M3, then PEER1-20 within each tissue
PEER factors are per tissue (data/egtex/metadata/<Tissue>.covariates.txt, aligned by donor), so M4 fits them
within tissue; a tissue with too few test samples for 20 PEERs keeps its M3 values and is reported.
Genotype PCs are deliberately NOT regressed: they are the donor-genotype signal being tested.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cpgpt_jepa import finish  # noqa: E402
from cpgpt_jepa.covariates import numeric, one_hot, peer_factors, residualise  # noqa: E402
from cpgpt_jepa.evaluation import donor_matching, fmt, load_run, tissue_centred, write_outputs  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--encodings", "--enc", dest="enc", default="outputs/encodings")
p.add_argument("--pred", default="outputs/predictor")
p.add_argument("--egtex-dir", default="data/egtex")
p.add_argument("--out", default="outputs/experiments/covariates")
p.add_argument("--kind", default="mlp_joint", help="predictor kind (primary: mlp_joint, docs/PREREG.md)")
p.add_argument("--n-perm", type=int, default=10_000, help="permutations vs chance for rung 3 (smoke: 200)")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
if args.smoke:
    args.n_perm = min(args.n_perm, 200)
t0 = time.time()

run = load_run(args.enc, args.pred, args.egtex_dir)
if not (run.pred / args.kind / "R.npy").exists():
    finish(False, f"no residuals for {args.kind} under {run.pred}")
test = run.test_s
meta = run.meta.iloc[test]
n_samples, n_blocks, d = run.Z.shape

covs = {
    "tissue": one_hot(meta.tissue.to_numpy()),
    "sex": numeric(meta.sex.to_numpy()),
    "age": numeric(meta.age_years.to_numpy()),
    "smoking": one_hot(meta.smoker_status.fillna("NA").to_numpy()),  # docs/PREREG.md rung 3
    "plate": one_hot(meta.plate.to_numpy()),
    "sentrix_id": one_hot(meta.sentrix_id.to_numpy()),
}
models = {"M1 tissue+sex+age+smoking": ["tissue", "sex", "age", "smoking"],
          "M2 +plate": ["tissue", "sex", "age", "smoking", "plate"],
          "M3 +sentrix_id": ["tissue", "sex", "age", "smoking", "plate", "sentrix_id"]}
print(f"test samples {len(test)}; columns: " + ", ".join(f"{k} {v.shape[1]}" for k, v in covs.items()))


def peer_within_tissue(Y: np.ndarray, n_peer: int = 20) -> tuple[np.ndarray, dict]:
    """Within each tissue, regress Y on that tissue's PEER factors (donors without PEERs set to the
    tissue mean of each factor). Tissues with fewer than 2 * (n_peer + 1) test samples are skipped."""
    out, status = Y.copy(), {}
    for t in sorted(set(meta.tissue)):
        idx = np.flatnonzero(meta.tissue.to_numpy() == t)
        peers = peer_factors(args.egtex_dir, t, meta.donor_id.to_numpy()[idx], n_peer)
        have = np.isfinite(peers).all(axis=1)
        if len(idx) < 2 * (peers.shape[1] + 1) or have.sum() == 0:
            status[t] = f"skipped ({len(idx)} test samples, {int(have.sum())} with PEERs)"
            continue
        col_mean = np.nanmean(peers, axis=0)
        peers = np.where(np.isfinite(peers), peers, col_mean)
        out[idx] = residualise(Y[idx], peers)
        status[t] = f"fitted ({len(idx)} test samples, {int(have.sum())} with PEERs, {peers.shape[1]} factors)"
    return out, status


def clean_and_match(rep: np.ndarray) -> tuple[dict, dict]:
    """rep: samples x blocks x d. Donor matching before and after each covariate model."""
    Y = rep[test][:, run.auto_blocks].reshape(len(test), -1).astype(np.float64)
    full = np.zeros((n_samples, Y.shape[1]))
    res = {"before": donor_matching(run, rep, n_perm=args.n_perm)}
    cleaned = Y
    for name, cols in models.items():
        cleaned = residualise(Y, np.column_stack([covs[c] for c in cols]))
        full[test] = cleaned
        res[name] = donor_matching(run, full.copy())
    cleaned, peer_status = peer_within_tissue(cleaned)
    full[test] = cleaned
    # M4 = every covariate of docs/PREREG.md rung 3; it carries the permutation test against chance
    res["M4 +PEER within tissue"] = donor_matching(run, full.copy(), n_perm=args.n_perm)
    return res, peer_status


result, rows = {"kind": args.kind, "covariate_columns": {k: int(v.shape[1]) for k, v in covs.items()}}, []
for rep_name, rep in (("residual R", run.residual(args.kind)), ("Z - tissue mean", tissue_centred(run, run.Zs))):
    res, peer_status = clean_and_match(rep)
    result[rep_name] = res
    result["peer_status"] = peer_status
    for model, m in res.items():
        a, b = m["lung_to_colon"], m["colon_to_lung"]
        rows.append({"representation": rep_name, "model": model, "l2c_top1": a["top1"], "l2c_mrr": a["mrr"],
                     "c2l_top1": b["top1"], "c2l_mrr": b["mrr"], "chance": a["chance"]})
    print(f"[{rep_name}] done ({time.time() - t0:.0f}s)", flush=True)
write_outputs(Path(args.out), result, rows)

print("\n=== 06 COVARIATES (test donors; donor matching before / after regressing covariates) ===")
print(f"{'representation':<18s}{'model':<26s}|{'top1':>7s}{'MRR':>7s} |{'top1':>7s}{'MRR':>7s} |{'chance':>8s}")
print(f"{'':<44s}|{'lung->colon':^14s} |{'colon->lung':^14s} |")
for r in rows:
    print(f"{r['representation']:<18s}{r['model']:<26s}|{fmt(r['l2c_top1']):>7s}{fmt(r['l2c_mrr']):>7s} |"
          f"{fmt(r['c2l_top1']):>7s}{fmt(r['c2l_mrr']):>7s} |{fmt(r['chance'], '.4f'):>8s}")
print("PEER fit per tissue: " + "; ".join(f"{t}: {s}" for t, s in result["peer_status"].items()))
print("Genotype PCs were not regressed (they carry the donor signal under test).")
print(f"elapsed: {time.time() - t0:.0f}s")
finish(True)
