"""Block-level 3D-structure annotations built on experiments/threed (build_annotations.py, make_shift_control.py,
and the Hi-C stream helper in h3_attention_vs_contacts.py)."""

import contextlib
import functools
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cpgpt_jepa import REPO

THREED = REPO / "experiments" / "threed"
ANNOT_DIR = REPO / "data" / "3d" / "annotations"
HIC_CACHE = REPO / "data" / "3d" / "hic_cache"


def _threed():
    if str(THREED) not in sys.path:
        sys.path.insert(0, str(THREED))
    import build_annotations
    import make_shift_control
    return build_annotations, make_shift_control


@functools.cache
def locus_index() -> tuple[dict, str]:
    """build_annotations.load_index(), loaded once: {"chrom:pos": DNA-embedding row}, dependencies dir."""
    build_annotations, _ = _threed()
    return build_annotations.load_index()


def locus_density_rows() -> np.ndarray:
    """build_annotations.locus_density: log count of array loci in each locus's 100 kb bin, per row
    (the gene-density stand-in used by the threed tooling)."""
    build_annotations, _ = _threed()
    index, _ = locus_index()
    keys = list(index)
    rows = np.fromiter((index[k] for k in keys), dtype=np.int64, count=len(keys))
    loci = [(k.split(":")[0], int(k.split(":")[1])) for k in keys]
    _, dens = build_annotations.locus_density(loci, int(rows.max()) + 1, rows)
    return dens


def load_annotation(name: str, annot_dir: Path = ANNOT_DIR) -> tuple[np.ndarray, dict] | None:
    meta_path = annot_dir / f"{name}.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    mm = np.memmap(annot_dir / f"{name}.mmap", dtype="float32", mode="r", shape=tuple(meta["shape"]))
    return np.asarray(mm), meta


def build_annotation(cell_type: str, name: str, log=print) -> tuple[np.ndarray, dict]:
    """Run experiments/threed/build_annotations.py (needs pyBigWig and network access to ENCODE)."""
    import pyBigWig  # noqa: F401  (fail early with ImportError if missing)

    log(f"[annot] building {name} with build_annotations.py --cell-type {cell_type!r}")
    subprocess.run([sys.executable, str(THREED / "build_annotations.py"), "--cell-type", cell_type, "--name", name],
                   check=True, cwd=REPO)
    loaded = load_annotation(name)
    if loaded is None:
        raise FileNotFoundError(f"build_annotations.py did not write {ANNOT_DIR / name}.json")
    return loaded


def synthetic_annotation(name: str, annot_dir: Path, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Smoke-test stand-in with the build_annotations layout (channels: value, covered): a smooth random
    signal along each chromosome. Never used outside --smoke."""
    index, _ = locus_index()
    rng = np.random.default_rng(seed)
    n_rows = max(index.values()) + 1
    out = np.zeros((n_rows, 2), dtype="float32")
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for key, row in index.items():
        c, pos = key.split(":")
        by_chrom.setdefault(c, []).append((int(pos), row))
    for c, items in by_chrom.items():
        pr = np.array(sorted(items))
        knots = rng.standard_normal(int(pr[-1, 0] // 2_000_000) + 2)
        out[pr[:, 1], 0] = np.interp(pr[:, 0] / 2_000_000, np.arange(len(knots)), knots)
        out[pr[:, 1], 1] = 1.0
    annot_dir.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(annot_dir / f"{name}.mmap", dtype="float32", mode="w+", shape=out.shape)
    mm[:] = out
    mm.flush()
    meta = {"name": name, "kind": "synthetic smoke-test track", "shape": list(out.shape),
            "channels": ["synthetic_scaled", "synthetic_covered"], "dtype": "float32"}
    (annot_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return out, meta


def block_profile(annot: np.ndarray, site_rows: np.ndarray, site_block: np.ndarray, n_blocks: int,
                  dens_rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per block: A-compartment fraction (covered sites with value > 0), mean value, coverage, and the
    mean locus density of its sites."""
    out = {k: np.full(n_blocks, np.nan) for k in ("a_fraction", "mean_value", "coverage", "density")}
    for b in range(n_blocks):
        r = site_rows[site_block == b]
        if len(r) == 0:
            continue
        cov = annot[r, 1] > 0
        out["coverage"][b] = cov.mean()
        if cov.any():
            out["a_fraction"][b] = (annot[r[cov], 0] > 0).mean()
            out["mean_value"][b] = annot[r[cov], 0].mean()
        if dens_rows is not None:
            out["density"][b] = dens_rows[r].mean()
    return out


def shifted_a_fractions(source: str, source_dir: Path, seeds: range, site_rows: np.ndarray, site_block: np.ndarray,
                        n_blocks: int, work_dir: Path) -> np.ndarray:
    """Circular-shift null: make_shift_control.py's shift of the source track, one per seed, reduced to the
    per-block A-fraction (n_seeds x n_blocks). Runs the script's main() in-process with the locus index
    loaded once; shifted tracks are written to work_dir and deleted after reading."""
    _, msc = _threed()
    work_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("json", "mmap"):
        link = work_dir / f"{source}.{ext}"
        if not link.exists():
            link.symlink_to((source_dir / f"{source}.{ext}").resolve())
    msc.ANNOT_DIR = work_dir
    msc.load_index = locus_index
    out = np.empty((len(seeds), n_blocks))
    argv = sys.argv
    try:
        for i, seed in enumerate(seeds):
            sys.argv = ["make_shift_control.py", "--kind", "shift", "--source", source, "--seed", str(seed)]
            with contextlib.redirect_stdout(io.StringIO()):
                msc.main()
            name = f"{source}_shift{seed}"
            shifted, _ = load_annotation(name, work_dir)
            out[i] = block_profile(shifted, site_rows, site_block, n_blocks)["a_fraction"]
            (work_dir / f"{name}.mmap").unlink()
            (work_dir / f"{name}.json").unlink()
    finally:
        sys.argv = argv
        shutil.rmtree(work_dir, ignore_errors=True)
    return out


def interchrom_contacts(blocks, url: str, resolution: int = 2_500_000, norm: str = "VC", log=print,
                        retries: int = 3) -> np.ndarray:
    """Mean inter-chromosomal Hi-C contact per block pair (NaN within a chromosome), streamed with hic-straw
    from a .hic URL and cached per chromosome pair under data/3d/hic_cache."""
    import hicstraw

    n = len(blocks)
    contact = np.full((n, n), np.nan)
    chroms = sorted({b["chrom"] for b in blocks}, key=lambda c: (len(c), c))
    tag = url.rstrip("/").split("/")[-1].replace(".hic", "")
    hic = None
    for i, ca in enumerate(chroms):
        for cb in chroms[i + 1 :]:
            cache = HIC_CACHE / f"inter_{tag}_chr{ca}_chr{cb}_{resolution}_{norm}.npz"
            if cache.exists():
                rec = np.load(cache)
                bx, by, counts = rec["bx"], rec["by"], rec["counts"]
            else:
                for attempt in range(1, retries + 1):
                    try:
                        hic = hic or hicstraw.HiCFile(url)
                        lengths = {c.name: c.length for c in hic.getChromosomes()}
                        mzd = hic.getMatrixZoomData(f"chr{ca}", f"chr{cb}", "observed", norm, "BP", resolution)
                        recs = mzd.getRecords(0, lengths[f"chr{ca}"], 0, lengths[f"chr{cb}"])
                        break
                    except Exception as e:  # noqa: BLE001
                        log(f"[hic] chr{ca}-chr{cb} attempt {attempt}/{retries} failed: {e}")
                        if attempt == retries:
                            raise
                        time.sleep(10 * attempt)
                        hic = None
                bx = np.array([r.binX for r in recs], dtype=np.int64)
                by = np.array([r.binY for r in recs], dtype=np.int64)
                counts = np.array([r.counts for r in recs], dtype=np.float64)
                HIC_CACHE.mkdir(parents=True, exist_ok=True)
                np.savez(cache, bx=bx, by=by, counts=counts)
            ok = np.isfinite(counts)
            bx, by, counts = bx[ok], by[ok], counts[ok]
            ia = [k for k, b in enumerate(blocks) if b["chrom"] == ca]
            ib = [k for k, b in enumerate(blocks) if b["chrom"] == cb]
            for a in ia:
                in_a = (bx >= blocks[a]["start"]) & (bx < blocks[a]["end"])
                for b in ib:
                    sel = in_a & (by >= blocks[b]["start"]) & (by < blocks[b]["end"])
                    n_bins = ((blocks[a]["end"] - blocks[a]["start"]) // resolution) * \
                             ((blocks[b]["end"] - blocks[b]["start"]) // resolution)
                    contact[a, b] = contact[b, a] = counts[sel].sum() / max(n_bins, 1)
        log(f"[hic] chr{ca} done")
    return contact


def partial_spearman(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    """Spearman correlation of y and x after regressing the ranks of both on the ranks of the controls."""
    from scipy.stats import rankdata

    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(controls).all(axis=1)
    if ok.sum() < 5:
        return float("nan")
    ry, rx = rankdata(y[ok]), rankdata(x[ok])
    rc = np.column_stack([np.ones(ok.sum())] + [rankdata(c) for c in controls[ok].T])
    res_y = ry - rc @ np.linalg.lstsq(rc, ry, rcond=None)[0]
    res_x = rx - rc @ np.linalg.lstsq(rc, rx, rcond=None)[0]
    return float(np.corrcoef(res_y, res_x)[0, 1])
