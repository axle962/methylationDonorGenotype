"""Build the control arm for H1: a real compartment track, circularly shifted.

A gain from the compartment channel could just mean "any smoothly varying extra
input helps", which would say nothing about 3D structure. The control has to
match the real track in everything except its alignment to the genome.

A circular shift does that exactly. Each chromosome's track is slid by a random
offset (20-80% of the chromosome), wrapping at the end, and read back at every
locus. Spatial autocorrelation is preserved because it is the same signal; only
which loci it lands on changes (nearest-locus lookup in sparse regions makes it
slightly smoother: 0.85 vs 0.78 at 100 kb for CD14). Loci are dense (one per
~2.5 kb on average), so each shifted position takes the value of the nearest
locus in the source track.

A shift also breaks the track's enrichment for A at array loci (73% -> ~44%),
and that enrichment is really gene density: at loci the compartment value
correlates r ~ 0.5-0.7 with local locus density. So a win over the shift control
could mean "the model was told gene density", not "told 3D structure". The
second control, `--kind density`, is exactly that non-3D channel: log locus
count per 100 kb, standardised. Compartment has to beat both.

Usage:
    python experiments/threed/make_shift_control.py --source CD14_mono --seed 0
    python experiments/threed/make_shift_control.py --kind density
    python experiments/threed/make_shift_control.py --kind zeros

`--kind zeros` is the H1 baseline: same widened model, annotation channels all
zero, so the arms differ only in information, not architecture or training.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_annotations import ANNOT_DIR, load_index, locus_density


def autocorr(values, positions, lag):
    """Correlation between loci and the nearest locus `lag` bp downstream."""
    j = np.searchsorted(positions, positions + lag)
    ok = j < len(positions)
    return np.corrcoef(values[ok], values[j[ok]])[0, 1]


def density_control() -> None:
    index, _ = load_index()
    keys = list(index)
    rows = np.fromiter((index[k] for k in keys), dtype=np.int64, count=len(keys))
    loci = [(k.split(":")[0], int(k.split(":")[1])) for k in keys]
    n_rows = int(rows.max()) + 1
    _, dens = locus_density(loci, n_rows, rows)
    out = np.zeros((n_rows, 2), dtype="float32")
    out[:, 0] = (dens - dens.mean()) / dens.std()
    out[:, 1] = 1.0
    name = "density"
    mm = np.memmap(ANNOT_DIR / f"{name}.mmap", dtype="float32", mode="w+",
                   shape=out.shape)
    mm[:] = out
    mm.flush()
    (ANNOT_DIR / f"{name}.json").write_text(json.dumps(
        {"name": name, "kind": "control", "shape": list(out.shape),
         "channels": ["log_locus_density_100kb_z", "covered"], "dtype": "float32"},
        indent=2))
    print(f"[control] density: log loci per 100 kb, z-scored; "
          f"range [{out[:, 0].min():.2f}, {out[:, 0].max():.2f}]")
    print(f"[write] {ANNOT_DIR / (name + '.mmap')}")


def zeros_control() -> None:
    index, _ = load_index()
    n_rows = max(index.values()) + 1
    name = "zeros"
    mm = np.memmap(ANNOT_DIR / f"{name}.mmap", dtype="float32", mode="w+",
                   shape=(n_rows, 2))
    mm[:] = 0.0
    mm.flush()
    (ANNOT_DIR / f"{name}.json").write_text(json.dumps(
        {"name": name, "kind": "baseline", "shape": [n_rows, 2],
         "channels": ["zero", "zero"], "dtype": "float32"}, indent=2))
    print(f"[write] {ANNOT_DIR / (name + '.mmap')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["shift", "density", "zeros"], default="shift")
    ap.add_argument("--source", help="annotation set to shift")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.kind == "density":
        return density_control()
    if args.kind == "zeros":
        return zeros_control()
    if not args.source:
        ap.error("--source is required for --kind shift")

    meta = json.loads((ANNOT_DIR / f"{args.source}.json").read_text())
    src = np.memmap(ANNOT_DIR / f"{args.source}.mmap", dtype="float32", mode="r",
                    shape=tuple(meta["shape"]))
    out = np.zeros_like(np.asarray(src))

    index, _ = load_index()
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for key, row in index.items():
        c, p = key.split(":")
        by_chrom.setdefault(c, []).append((int(p), row))

    rng = np.random.default_rng(args.seed)
    offsets, checks = {}, []
    for c in sorted(by_chrom):
        pr = np.array(sorted(by_chrom[c]))
        pos, rows = pr[:, 0], pr[:, 1]
        length = int(pos.max()) + 1
        off = int(rng.uniform(0.2, 0.8) * length)
        offsets[c] = off
        # Locus at position x takes the source value found at (x + off) mod length.
        target = (pos + off) % length
        j = np.clip(np.searchsorted(pos, target), 1, len(pos) - 1)
        nearest = np.where(target - pos[j - 1] <= pos[j] - target, j - 1, j)
        out[rows] = src[rows[nearest]]

        cov = src[rows, 1] > 0
        if cov.sum() > 1000:
            a, b = src[rows, 0], out[rows, 0]
            checks.append((c, np.corrcoef(a[cov], b[cov])[0, 1],
                           autocorr(a, pos, 100_000), autocorr(b, pos, 100_000)))

    name = f"{args.source}_shift{args.seed}"
    mm = np.memmap(ANNOT_DIR / f"{name}.mmap", dtype="float32", mode="w+",
                   shape=out.shape)
    mm[:] = out
    mm.flush()
    (ANNOT_DIR / f"{name}.json").write_text(json.dumps(
        {**meta, "name": name, "control_of": args.source, "seed": args.seed,
         "shift_offsets_bp": offsets}, indent=2))

    r_real = np.array([x[1] for x in checks])
    ac_src = np.array([x[2] for x in checks])
    ac_ctl = np.array([x[3] for x in checks])
    cov_src = np.asarray(src[:, 1]) > 0
    cov_ctl = out[:, 1] > 0
    print(f"[control] {name}")
    print(f"  correlation with the real track, per chromosome: "
          f"median {np.median(r_real):+.3f}, range [{r_real.min():+.3f}, {r_real.max():+.3f}]"
          f"  (should be near 0)")
    print(f"  autocorrelation at 100 kb: source {np.median(ac_src):.3f}, "
          f"control {np.median(ac_ctl):.3f}  (should match)")
    print(f"  A-fraction: source {np.mean(src[cov_src, 0] > 0)*100:.1f}%, "
          f"control {np.mean(out[cov_ctl, 0] > 0)*100:.1f}%")
    print(f"[write] {ANNOT_DIR / (name + '.mmap')}")


if __name__ == "__main__":
    main()
