"""Replicate the residual-structure result across independent genomic windows.

Entry 07 found a compartment-scale effect on the model's residuals, but only in
one variance cut, from 238 sites, in a single 20 Mb window on chromosome 1. That
is one draw. If the effect is real it should reappear across windows on other
chromosomes; if it was the window we happened to pick, the gaps will scatter
around zero.

Each window is an independent run: hold out half its sites, reconstruct them,
correlate residuals, and compare 3D-proximal against 3D-distant pairs within
matched distance bands.

Usage:
    python experiments/threed/h3d_replication.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]

# One 20 Mb window per chromosome, avoiding centromeres and picking a mix of
# gene-rich (17, 19) and gene-poor (4, 8) chromosomes.
WINDOWS = [
    ("1", 30_000_000, 50_000_000),
    ("2", 50_000_000, 70_000_000),
    ("4", 60_000_000, 80_000_000),
    ("6", 30_000_000, 50_000_000),
    ("8", 40_000_000, 60_000_000),
    ("11", 20_000_000, 40_000_000),
    ("17", 30_000_000, 50_000_000),
    ("19", 10_000_000, 30_000_000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gse", default="GSE40279")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--max-sites", type=int, default=3000)
    ap.add_argument("--skip-predict", action="store_true",
                    help="reuse cached residuals if the windows were already run")
    ap.add_argument("--annotation", default="pool5",
                    help="compartment track set under data/3d/annotations; pool5 is "
                         "what entries 07-08 used")
    args = ap.parse_args()

    from h3c_residual_3d_contrast import analyse

    all_rows = []
    for chrom, start, end in WINDOWS:
        tag = f"{args.gse}_chr{chrom}_{start // 1_000_000}"
        run_dir = ROOT / f"data/3d/h3b_{tag}"
        if not (run_dir / "residuals.npz").exists() and not args.skip_predict:
            print(f"\n=== window chr{chrom}:{start//1_000_000}-{end//1_000_000} Mb ===",
                  flush=True)
            cmd = [sys.executable, str(Path(__file__).parent / "h3b_residual_structure.py"),
                   "--gse", args.gse, "--chrom", chrom, "--start", str(start),
                   "--end", str(end), "--n-samples", str(args.n_samples),
                   "--max-sites", str(args.max_sites), "--tag", tag]
            r = subprocess.run(cmd, capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if line.startswith(("[data]", "[split]", "[predict]", "[diag]")):
                    print("  " + line, flush=True)
            if r.returncode != 0:
                print(f"  FAILED: {r.stdout[-400:]}{r.stderr[-400:]}", flush=True)
                continue
        if not (run_dir / "residuals.npz").exists():
            continue
        try:
            rows = analyse(run_dir, chrom, start, end, quiet=True,
                           annotation=args.annotation)
        except Exception as e:  # noqa: BLE001
            print(f"  analysis failed for chr{chrom}: {e}", flush=True)
            continue
        for row in rows:
            print(f"  chr{chrom} SD>={row['min_sd']}: "
                  f"compartment {row['comp_gap']:+.4f} "
                  f"[{row['comp_lo']:+.4f},{row['comp_hi']:+.4f}]   "
                  f"contact {row['contact_gap']:+.4f} "
                  f"[{row['contact_lo']:+.4f},{row['contact_hi']:+.4f}]", flush=True)
        all_rows += rows

    if not all_rows:
        raise SystemExit("no windows completed")

    df = pd.DataFrame(all_rows)
    suffix = "" if args.annotation == "pool5" else f"_{args.annotation}"
    out = ROOT / f"results/h3d_replication{suffix}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    print("\n" + "=" * 74)
    print("REPLICATION ACROSS WINDOWS")
    print("=" * 74)
    for min_sd, sub in df.groupby("min_sd"):
        n = len(sub)
        comp_pos = (sub.comp_gap > 0).sum()
        comp_sig = (sub.comp_lo > 0).sum()
        cont_pos = (sub.contact_gap > 0).sum()
        cont_sig = (sub.contact_lo > 0).sum()
        # Equal-weight mean across windows, with a t-style interval over windows.
        m, sd = sub.comp_gap.mean(), sub.comp_gap.std(ddof=1)
        se = sd / np.sqrt(n) if n > 1 else np.nan
        mc, sdc = sub.contact_gap.mean(), sub.contact_gap.std(ddof=1)
        sec = sdc / np.sqrt(n) if n > 1 else np.nan
        print(f"\nsites varying >= {min_sd}   ({n} windows)")
        print(f"  compartment gap: mean {m:+.4f}  95% CI [{m-1.96*se:+.4f}, {m+1.96*se:+.4f}]"
              f"   positive in {comp_pos}/{n}, individually significant in {comp_sig}/{n}")
        print(f"  contact gap    : mean {mc:+.4f}  95% CI [{mc-1.96*sec:+.4f}, {mc+1.96*sec:+.4f}]"
              f"   positive in {cont_pos}/{n}, individually significant in {cont_sig}/{n}")
        verdict = ("REPLICATES" if (m - 1.96 * se) > 0 and comp_pos >= 0.75 * n
                   else "DOES NOT REPLICATE")
        print(f"  compartment effect: {verdict}")
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
