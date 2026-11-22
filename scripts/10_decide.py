"""Phase A / 10: apply docs/PREREG.md mechanically to the pipeline outputs.

  python scripts/10_decide.py [--outputs outputs] [--seeds 0,1,2] [--kind mlp_joint] [--smoke]

Reads, for each predictor seed s (seed 0 -> <outputs>/predictor, else <outputs>/predictor_seed<s>):
  stage 04  <pred>/<kind>/evaluation.json   (donor matching, inference, mQTL enrichment)
  stage 06  <exp>/covariates/result.json    (rung 3)
  stage 07  <exp>/mqtl_deep/result.json     (rung 4 negative set, rung 5)
  stage 08  <exp>/structure/result.json     (rung 6, 3D test)
where <exp> is <outputs>/experiments for seed 0 and <outputs>/experiments_seed<s> otherwise; plus the beta-space
evaluation (<pred>_beta) and the unmasked run (<outputs>/predictor_unmasked, seed 0) for the leakage guard.

Each rung is PASS, FAIL or NOT EVALUATED (a needed output is missing), with the numbers behind it. Where the
PREREG says so (every criterion), statistics are averaged over the seeds that have the output: top-1, p-values,
odds ratios, Spearman rho and bootstrap CI bounds are each the mean over seeds; per-seed values are kept too.
Secondary p-values are Holm-corrected within each rung (the lung->colon permutation p of rung 2 is the primary
endpoint and is not corrected). Writes <outputs>/decision.json and prints a table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cpgpt_jepa import finish  # noqa: E402
from cpgpt_jepa.evaluation import paired_bootstrap  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--outputs", default="outputs")
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--kind", default="mlp_joint", help="primary predictor (docs/PREREG.md section 1)")
p.add_argument("--n-boot", type=int, default=10_000)
p.add_argument("--out", default=None, help="default <outputs>/decision.json")
p.add_argument("--smoke", action="store_true", help="outputs/smoke, seed 0; missing outputs are expected")
args = p.parse_args()
if args.smoke:
    args.outputs = "outputs/smoke" if args.outputs == "outputs" else args.outputs
    args.seeds, args.n_boot = "0", min(args.n_boot, 200)
root = Path(args.outputs)
seeds = [int(s) for s in args.seeds.split(",")]
K = args.kind
DIRS = ("lung_to_colon", "colon_to_lung")
TOP1_MIN = 5 / 153  # rung 2 chance criterion, docs/PREREG.md section 4
PASS, FAIL, NE = "PASS", "FAIL", "NOT EVALUATED"


# --- loading -------------------------------------------------------------------------------------
def pred_dir(s: int) -> Path:
    return root / ("predictor" if s == 0 else f"predictor_seed{s}")


def exp_dir(s: int) -> Path:
    return root / ("experiments" if s == 0 else f"experiments_seed{s}")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


ev = {s: load(pred_dir(s) / K / "evaluation.json") for s in seeds}
ev_beta = {s: load(Path(f"{pred_dir(s)}_beta") / K / "evaluation.json") for s in seeds}
cov = {s: load(exp_dir(s) / "covariates" / "result.json") for s in seeds}
mq = {s: load(exp_dir(s) / "mqtl_deep" / "result.json") for s in seeds}
st = {s: load(exp_dir(s) / "structure" / "result.json") for s in seeds}
ev_unmasked = load(root / "predictor_unmasked" / K / "evaluation.json")
inputs = {"stage04": {s: v is not None for s, v in ev.items()},
          "stage04_beta": {s: v is not None for s, v in ev_beta.items()},
          "stage06": {s: v is not None for s, v in cov.items()},
          "stage07": {s: v is not None for s, v in mq.items()},
          "stage08": {s: v is not None for s, v in st.items()},
          "unmasked_stage04": ev_unmasked is not None}


def get(d: dict | None, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def per_seed(src: dict, *keys) -> dict:
    """{seed: value} for the seeds whose output has the value (numbers or [lo, hi] lists)."""
    return {s: v for s, d in src.items() if (v := get(d, *keys)) is not None and not isinstance(v, str)}


def mean(vals: dict):
    """Mean over seeds; works for scalars and for [lo, hi] lists. None if no seed has it."""
    if not vals:
        return None
    arr = np.array(list(vals.values()), dtype=float)
    m = arr.mean(0)
    return m.tolist() if m.ndim else float(m)


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for {name: p} (NaN/None are left out)."""
    items = sorted(((k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        running = max(running, min(1.0, (m - i) * v))
        out[k] = running
    return out


def rung(status: str, criteria: dict, numbers: dict, note: str = "") -> dict:
    return {"status": status, "criteria": criteria, "numbers": numbers, **({"note": note} if note else {})}


def verdict(checks: dict) -> str:
    if any(v is None for v in checks.values()):
        return NE
    return PASS if all(checks.values()) else FAIL


def missing(what: str) -> dict:
    return rung(NE, {}, {}, f"missing output: {what}")


decision = {"outputs": str(root), "seeds": seeds, "kind": K, "inputs_found": inputs, "rungs": {}}
R = decision["rungs"]

# --- rung 1: predictor non-trivial ---------------------------------------------------------------
r1 = per_seed(ev, "inference", "rung1 heldout MSE vs block_mean")
if not r1:
    R["1"] = missing("stage 04 'rung1 heldout MSE vs block_mean' (needs block_mean in stage 03)")
else:
    rel = mean({s: v["rel_reduction"] for s, v in r1.items()})
    ci = mean({s: v["ci95"] for s, v in r1.items()})
    checks = {"rel_reduction >= 0.10": rel >= 0.10, "CI of MSE difference excludes 0 (upper < 0)": ci[1] < 0}
    R["1"] = rung(verdict(checks), checks,
                  {"rel_reduction_mean": rel, "diff_ci95_mean": ci, "mse_kind_mean": mean({s: v["mse_a"] for s, v in r1.items()}),
                   "mse_block_mean_mean": mean({s: v["mse_b"] for s, v in r1.items()}),
                   "per_seed": {s: {k: v[k] for k in ("rel_reduction", "diff", "ci95")} for s, v in r1.items()}})


# --- rung 2: residual identifies the donor -------------------------------------------------------
def matching_numbers(src: dict, rep: str) -> dict:
    out = {}
    for d in DIRS:
        out[d] = {"top1": mean(per_seed(src, "donor_matching", rep, d, "top1")),
                  "perm_p": mean(per_seed(src, "donor_matching", rep, d, "perm_p_top1")),
                  "n_query": mean(per_seed(src, "donor_matching", rep, d, "n_query")),
                  "n_gallery": mean(per_seed(src, "donor_matching", rep, d, "n_gallery")),
                  "top1_per_seed": per_seed(src, "donor_matching", rep, d, "top1")}
    return out


if not any(ev.values()):
    R["2"] = missing("stage 04 evaluation.json")
else:
    m = matching_numbers(ev, "residual R")
    vs_tc = {d: {"diff": mean(per_seed(ev, "inference", "R vs Z - tissue mean", d, "diff")),
                 "ci95": mean(per_seed(ev, "inference", "R vs Z - tissue mean", d, "ci95"))} for d in DIRS}
    adj = holm({"colon_to_lung": m["colon_to_lung"]["perm_p"]})  # lung_to_colon is primary: uncorrected
    p_use = {"lung_to_colon": m["lung_to_colon"]["perm_p"], "colon_to_lung": adj.get("colon_to_lung")}
    checks = {}
    for d in DIRS:
        checks[f"{d}: top1 >= 5/153"] = None if m[d]["top1"] is None else m[d]["top1"] >= TOP1_MIN
        checks[f"{d}: perm p < 0.01"] = None if p_use[d] is None else p_use[d] < 0.01
        ci = vs_tc[d]["ci95"]
        checks[f"{d}: above Z - tissue mean (bootstrap CI lower > 0)"] = None if ci is None else ci[0] > 0
    R["2"] = rung(verdict(checks), checks, {"residual_R": m, "perm_p_used": p_use, "R_minus_Z_tissue_centred": vs_tc,
                                            "Z_tissue_centred": matching_numbers(ev, "Z - tissue mean"),
                                            "Z_full": matching_numbers(ev, "Z_full")})

# --- rung 3: survives covariate removal ----------------------------------------------------------
if not any(cov.values()):
    R["3"] = missing("stage 06 result.json")
else:
    FINAL = "M4 +PEER within tissue"
    nums, checks = {}, {}
    pvals = {d: mean(per_seed(cov, "residual R", FINAL, d, "perm_p_top1")) for d in DIRS}
    adj = holm(pvals)
    for d in DIRS:
        before = mean(per_seed(cov, "residual R", "before", d, "top1"))
        after = mean(per_seed(cov, "residual R", FINAL, d, "top1"))
        frac = after / before if (before and after is not None) else None
        nums[d] = {"top1_before": before, "top1_after": after, "fraction_kept": frac,
                   "perm_p_after": pvals[d], "perm_p_after_holm": adj.get(d)}
        checks[f"{d}: top1 after >= 5/153"] = None if after is None else after >= TOP1_MIN
        checks[f"{d}: perm p after (Holm) < 0.01"] = None if adj.get(d) is None else adj[d] < 0.01
        checks[f"{d}: keeps >= 50% of pre-regression top1"] = None if frac is None else frac >= 0.5
    covs = get(next(v for v in cov.values() if v), "covariate_columns")
    R["3"] = rung(verdict(checks), checks, {**nums, "covariate_model": FINAL, "covariate_columns": covs},
                  "" if covs and "smoking" in covs else "WARNING: smoking is not among the stage 06 covariates")

# --- rung 4: genetic in the specific sense -------------------------------------------------------
KEY = "any list / all test samples"
e4 = {s: v for s, v in per_seed(ev, "mqtl", KEY, "variance").items()}
NEG, POS7 = "negative (age/tissue, not mQTL) / all test", "any tissue / all test"
neg = per_seed(mq, "enrichment", NEG)
pos7 = per_seed(mq, "enrichment", POS7)
if not e4:
    R["4"] = missing("stage 04 mQTL enrichment ('any list / all test samples')")
else:
    pct = mean({s: v["mean_pct_set"] for s, v in e4.items()})
    pct_other = mean({s: v["mean_pct_other"] for s, v in e4.items()})
    mw = mean({s: v["mannwhitney_p"] for s, v in e4.items()})
    orr = mean({s: v["top5_odds_ratio"] for s, v in e4.items()})
    fp = mean({s: v["top5_fisher_p"] for s, v in e4.items()})
    or_ci = mean({s: v["top5_or_ci95"] for s, v in e4.items() if "top5_or_ci95" in v})
    adj = holm({"mannwhitney": mw, "fisher": fp})
    checks = {"mQTL mean percentile > 0.5": pct > 0.5, "Mann-Whitney p (Holm) < 0.01": adj.get("mannwhitney", 1) < 0.01,
              "top-5% OR > 1": orr > 1, "Fisher p (Holm) < 0.01": adj.get("fisher", 1) < 0.01}
    nums = {"mqtl_set": {"mean_pct_set": pct, "mean_pct_other": pct_other, "mannwhitney_p": mw, "fisher_p": fp,
                         "holm": adj, "odds_ratio": orr, "or_ci95": or_ci,
                         "background": get(next(iter(per_seed(ev, "mqtl", KEY).values()), None), "background"),
                         "per_seed": e4}}
    if not neg or not pos7 or not all("top5_log_or" in v for v in list(neg.values()) + list(pos7.values())):
        checks["negative set not enriched"] = None
        nums["negative_set"] = "missing: stage 07 enrichment with odds-ratio CIs"
    else:
        n_ci = mean({s: v["top5_or_ci95"] for s, v in neg.items()})
        # log OR(negative) - log OR(mQTL set), same seeds, same score (stage 07); Wald CI from the two SEs
        sd = sorted(set(neg) & set(pos7))
        dl = mean({s: neg[s]["top5_log_or"] - pos7[s]["top5_log_or"] for s in sd})
        dse = mean({s: float(np.hypot(neg[s]["top5_log_or_se"], pos7[s]["top5_log_or_se"])) for s in sd})
        diff_ci = [dl - 1.96 * dse, dl + 1.96 * dse]
        includes_1 = n_ci[0] <= 1 <= n_ci[1]
        below = diff_ci[1] < 0
        checks["negative set not enriched (OR CI includes 1, or OR significantly below the mQTL set's)"] = includes_1 or below
        nums["negative_set"] = {"odds_ratio": mean({s: v["top5_odds_ratio"] for s, v in neg.items()}), "or_ci95": n_ci,
                                "mean_pct_set": mean({s: v["mean_pct_set"] for s, v in neg.items()}),
                                "or_ci_includes_1": includes_1, "log_or_minus_mqtl_log_or": dl,
                                "diff_ci95": diff_ci, "significantly_below_mqtl": below,
                                "mqtl_set_stage07_odds_ratio": mean({s: v["top5_odds_ratio"] for s, v in pos7.items()}),
                                "seeds": sd}
    R["4"] = rung(verdict(checks), checks, nums)

# --- rung 5: structured where it should be -------------------------------------------------------
sp = per_seed(mq, "block_contribution", "spearman_contribution_vs_density")
if not sp:
    R["5"] = missing("stage 07 block_contribution")
else:
    rho = mean({s: v["rho"] for s, v in sp.items()})
    pv = mean({s: v["p"] for s, v in sp.items()})
    checks = {"Spearman rho > 0": rho > 0, "p < 0.05": pv < 0.05}
    R["5"] = rung(verdict(checks), checks, {"rho": rho, "p": pv, "n_blocks": mean(per_seed(mq, "block_contribution", "n_blocks")),
                                            "per_seed": sp})

# --- rung 6: not 3D structure --------------------------------------------------------------------
SK = f"{K}_struct"
if not per_seed(st, "mlp_struct", SK):
    R["6"] = missing(f"stage 08 mlp_struct ({SK})")
else:
    nums, checks = {}, {}
    for d in DIRS:
        base = mean(per_seed(st, "mlp_struct", K, "donor_matching", d, "top1"))
        struct = mean(per_seed(st, "mlp_struct", SK, "donor_matching", d, "top1"))
        frac = struct / base if base else None
        nums[d] = {"top1_" + K: base, "top1_" + SK: struct, "fraction_kept": frac}
        checks[f"{d}: struct top1 >= 80% of {K}"] = None if (base is None or struct is None) else struct >= 0.8 * base
    note = ""
    if any(nums[d]["top1_" + K] == 0 for d in DIRS):
        note = f"{K} top-1 is 0 in a direction, so 'keeps >= 80%' holds trivially there"
    R["6"] = rung(verdict(checks), checks, {**nums, "struct_tracks": per_seed(st, "struct_tracks")}, note)

# --- beta-space decision rule --------------------------------------------------------------------
bs = {d: {"diff": mean(per_seed(ev, "inference", "R vs beta-space R", d, "diff")),
          "ci95": mean(per_seed(ev, "inference", "R vs beta-space R", d, "ci95"))} for d in DIRS}
if any(v["ci95"] is None for v in bs.values()):
    decision["beta_space"] = {"status": NE, "note": "missing 'R vs beta-space R' in stage 04 (run 03/04 --beta-space first)"}
else:
    cpgpt_better = {d: bs[d]["ci95"][0] > 0 for d in DIRS}
    decision["beta_space"] = {
        "status": "ABOUT CpGPT" if all(cpgpt_better.values()) else "ABOUT METHYLATION DATA",
        "rule": "beta-space top-1 significantly below CpGPT's (CI of CpGPT - beta lower > 0) in both directions, "
                "else the finding is about methylation data, not CpGPT",
        "R_minus_beta_R": bs, "cpgpt_significantly_better": cpgpt_better,
        "beta_space_residual": matching_numbers(ev_beta, "residual R"),
        "raw_betas": matching_numbers(ev, "raw betas (same sites)"),
        "raw_betas_tissue_centred": matching_numbers(ev, "raw betas - tissue mean")}

# --- 3D test (section 5) -------------------------------------------------------------------------
def three_d(entry: dict) -> dict:
    rho = entry.get("partial_spearman_logE_vs_compartment_similarity")
    q = get(entry, "shift_null", "null_q025_q975")
    return {"rho": rho, "null_q025_q975": q, "p_two_sided": get(entry, "shift_null", "p_two_sided"),
            "rho_hic": entry.get("partial_spearman_logE_vs_log_hic"), "cell_type": entry.get("cell_type")}


by_t = {s: d.get("pair_error_by_tissue") for s, d in st.items() if d and d.get("pair_error_by_tissue")}
if not by_t:
    decision["three_d"] = {"status": NE, "note": "missing stage 08 pair_error_by_tissue (a tissue-matched track per "
                                                 "tested tissue is required; one track for all tissues is an analysis error)"}
else:
    tissues = sorted(set().union(*(set(v) for v in by_t.values())))
    out3 = {}
    for t in tissues:
        ent = {s: three_d(v[t]) for s, v in by_t.items() if t in v and "partial_spearman_logE_vs_compartment_similarity" in v[t]}
        if not ent:
            out3[t] = {"status": NE, "note": next((v[t].get("skipped") for v in by_t.values() if t in v), "no result")}
            continue
        rho = mean({s: e["rho"] for s, e in ent.items()})
        q = mean({s: e["null_q025_q975"] for s, e in ent.items()})
        real = rho < q[0] or rho > q[1]
        size = "negligible" if abs(rho) < 0.1 else "non-negligible"
        status = (("DETECTABLE BUT NEGLIGIBLE" if size == "negligible" else "REAL") if real else "NOT DETECTED")
        out3[t] = {"status": status, "rho": rho, "null_q025_q975_mean": q,
                   "p_two_sided": mean({s: e["p_two_sided"] for s, e in ent.items()}),
                   "expected_direction_rho_negative": rho < 0, "rho_hic": mean({s: e["rho_hic"] for s, e in ent.items()
                                                                               if e["rho_hic"] is not None}),
                   "cell_type": next(iter(ent.values()))["cell_type"], "per_seed": ent}
    decision["three_d"] = {"by_tissue": out3,
                           "rule": "effect real if observed rho is outside the [2.5, 97.5] percentiles of the 200-shift "
                                   "null; |rho| < 0.1 reported as detectable but negligible"}

# --- leakage guard (section 6) -------------------------------------------------------------------
if ev_unmasked is None or ev[seeds[0]] is None:
    decision["leakage"] = {"status": NE, "note": "missing masked (seed 0) or unmasked stage 04 evaluation"}
else:
    masked0 = ev[seeds[0]]["donor_matching"]["residual R"]
    unm = ev_unmasked["donor_matching"]["residual R"]
    cmp_ = {d: paired_bootstrap(unm[d], masked0[d], n_boot=args.n_boot) for d in DIRS}
    excess = {d: cmp_[d]["ci95"][0] > 0 for d in DIRS}
    decision["leakage"] = {
        "status": "UNMASKED EXCESS" if any(excess.values()) else "NO SIGNIFICANT EXCESS",
        "rule": "unmasked minus masked top-1 (same donors, paired bootstrap, seed 0); a significant excess is "
                "attributed to SNP-affected probes, not to CpGPT. 'Much higher' is read as: CI lower bound > 0.",
        "masked_top1": {d: masked0[d]["top1"] for d in DIRS}, "unmasked_top1": {d: unm[d]["top1"] for d in DIRS},
        "unmasked_minus_masked": cmp_, "significant_excess": excess}

# --- interpretation (section 4, fixed in advance) ------------------------------------------------
st_ = {k: v["status"] for k, v in R.items()}
if st_.get("2") == FAIL:
    interp = "Rung 2 fails: frozen CpGPT block residuals do not identify donors (negative result)."
elif all(st_.get(k) == PASS for k in "123"):
    interp = "Rungs 1-3 pass: a donor-specific subspace exists."
    if st_.get("4") == PASS and st_.get("5") == PASS:
        interp += " Rungs 4-5 pass as well: it is genetic."
else:
    interp = "The ladder is incomplete: " + ", ".join(f"rung {k} {v}" for k, v in st_.items() if v != PASS) + "."
decision["interpretation"] = interp

out_path = Path(args.out) if args.out else root / "decision.json"
out_path.write_text(json.dumps(decision, indent=2, default=float))


# --- table ---------------------------------------------------------------------------------------
def f(x, spec=".3f"):
    return "NA" if x is None else (f"[{x[0]:{spec}}, {x[1]:{spec}}]" if isinstance(x, list) else f"{x:{spec}}")


print(f"\n=== 10 DECISION (docs/PREREG.md; kind {K}; seeds {seeds}; outputs {root}) ===")
print(f"inputs found: {json.dumps(inputs)}")
for k in "123456":
    r = R[k]
    print(f"\nRung {k}: {r['status']}" + (f"  ({r['note']})" if r.get("note") else ""))
    for c, ok in r["criteria"].items():
        print(f"   [{'x' if ok else ' ' if ok is not None else '?'}] {c}")
n = R["1"]["numbers"]
if n:
    print(f"   rung 1: rel. MSE reduction {f(n['rel_reduction_mean'])}, diff CI {f(n['diff_ci95_mean'], '.4f')}")
n = R["2"]["numbers"]
if n:
    for d in DIRS:
        print(f"   rung 2 {d}: top1 {f(n['residual_R'][d]['top1'])} (min {TOP1_MIN:.3f}), perm p {f(n['perm_p_used'][d], '.4f')}, "
              f"R - (Z - tissue mean) {f(n['R_minus_Z_tissue_centred'][d]['diff'])} CI {f(n['R_minus_Z_tissue_centred'][d]['ci95'])}")
n = R["3"]["numbers"]
if n:
    for d in DIRS:
        print(f"   rung 3 {d}: top1 {f(n[d]['top1_before'])} -> {f(n[d]['top1_after'])} (kept {f(n[d]['fraction_kept'])}), "
              f"perm p (Holm) {f(n[d]['perm_p_after_holm'], '.4f')}")
n = R["4"]["numbers"]
if n:
    q = n["mqtl_set"]
    print(f"   rung 4 mQTL: pct {f(q['mean_pct_set'], '.4f')} vs {f(q['mean_pct_other'], '.4f')}, MW p {f(q['mannwhitney_p'], '.2e')}, "
          f"OR {f(q['odds_ratio'])} CI {f(q['or_ci95'])}, Fisher p {f(q['fisher_p'], '.2e')}; Holm {json.dumps(q['holm'])}")
    if isinstance(n["negative_set"], dict):
        g = n["negative_set"]
        print(f"   rung 4 negative set: OR {f(g['odds_ratio'])} CI {f(g['or_ci95'])}; log OR minus mQTL's "
              f"{f(g['log_or_minus_mqtl_log_or'])} CI {f(g['diff_ci95'])}")
n = R["5"]["numbers"]
if n:
    print(f"   rung 5: Spearman rho {f(n['rho'])}, p {f(n['p'], '.2e')} over {f(n['n_blocks'], '.0f')} blocks")
n = R["6"]["numbers"]
if n:
    for d in DIRS:
        print(f"   rung 6 {d}: {K} {f(n[d]['top1_' + K])}, {K}_struct {f(n[d]['top1_' + K + '_struct'])} (kept {f(n[d]['fraction_kept'])})")
b = decision["beta_space"]
print(f"\nBeta-space rule: {b['status']}")
if "R_minus_beta_R" in b:
    for d in DIRS:
        print(f"   {d}: CpGPT R - beta R top1 {f(b['R_minus_beta_R'][d]['diff'])} CI {f(b['R_minus_beta_R'][d]['ci95'])}; "
              f"beta R top1 {f(b['beta_space_residual'][d]['top1'])}; raw betas {f(b['raw_betas'][d]['top1'])}, "
              f"tissue-centred {f(b['raw_betas_tissue_centred'][d]['top1'])}")
t = decision["three_d"]
print("\n3D test (section 5): " + (t.get("note", "") if "by_tissue" not in t else ""))
for tissue, v in t.get("by_tissue", {}).items():
    print(f"   {tissue:<20s} {v['status']:<26s} " + (f"cell type {v['cell_type']}, rho {f(v['rho'])}, null 95% {f(v['null_q025_q975_mean'])}, "
                                                  f"p {f(v['p_two_sided'], '.3f')}, Hi-C rho {f(v['rho_hic'])}" if "rho" in v else v.get("note", "")))
lk = decision["leakage"]
print(f"\nLeakage guard (section 6): {lk['status']}" + (f"  ({lk['note']})" if "note" in lk else ""))
if "masked_top1" in lk:
    for d in DIRS:
        c = lk["unmasked_minus_masked"][d]
        print(f"   {d}: masked {f(lk['masked_top1'][d])}, unmasked {f(lk['unmasked_top1'][d])}, diff {f(c['diff'])} CI {f(c['ci95'])}")
print(f"\nInterpretation: {interp}")
print(f"wrote {out_path}")

if args.smoke:
    finish(R["2"]["status"] != NE, "rung 2 could not be evaluated on the smoke outputs")
finish(all(v["status"] != NE for v in R.values()) and decision["beta_space"]["status"] != NE,
       "some rungs or the beta-space rule could not be evaluated: " + ", ".join(k for k, v in R.items() if v["status"] == NE))
