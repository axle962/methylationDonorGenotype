"""Build a per-locus 3D-structure annotation memmap aligned to CpGPT's DNA embeddings.

CpGPT caches one 1024-d DNA embedding per locus in a memmap, indexed by
`ensembl_metadata_dict[species][dna_llm][context_len]`, a dict of "chrom:pos" ->
row. This writes a second memmap with the same row order and k annotation
channels, so the two can simply be concatenated at the model's input adapter.

Tracks come from ENCODE as GRCh38-native bigWigs, which avoids the hg19 liftOver
that silently shifts every coordinate.

Usage:
    python experiments/threed/build_annotations.py --cell-type GM12878
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
TRACK_DIR = ROOT / "data/3d/tracks"
ANNOT_DIR = ROOT / "data/3d/annotations"
ENCODE_FILE = "https://www.encodeproject.org/files/{acc}/@@download/{acc}.bigWig"
ENCODE_SEARCH = (
    "https://www.encodeproject.org/search/?type=File&output_type={output_type}"
    "&assembly=GRCh38&status=released&biosample_ontology.term_name={cell}"
    "&limit=all&format=json"
)


def encode_lookup(cell: str, output_type: str) -> list[dict]:
    url = ENCODE_SEARCH.format(cell=urllib.parse.quote(cell),
                               output_type=urllib.parse.quote(output_type))
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    files = []
    for f in data.get("@graph", []):
        # Aggregate series pool several cell types into one track (the one first
        # used here as "GM12878" was GM12878 + IMR-90 + MCF 10A + HUVEC + HMEC).
        # A pooled file lists every contributing biosample, so keep only files
        # from exactly the requested cell type.
        bio = f.get("biosample_ontology")
        if not isinstance(bio, dict) or bio.get("term_name") != cell:
            continue
        files.append({"accession": f["accession"], "size": f.get("file_size", 0),
                      "biosample": cell, "output_type": output_type,
                      "dataset": f.get("dataset", ""),
                      "parent": (f.get("derived_from") or [""])[0]})
    # Tracks come in pairs (5 kb and 10 kb bins of one eigenvector, same parent);
    # order so the first two are one such pair.
    files.sort(key=lambda f: (f["dataset"], f["parent"], -f["size"]))
    return files


def download_track(accession: str) -> Path:
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    dest = TRACK_DIR / f"{accession}.bigWig"
    if dest.exists():
        return dest
    url = ENCODE_FILE.format(acc=accession)
    print(f"  downloading {accession}")
    urllib.request.urlretrieve(url, dest)
    return dest


def load_index(species: str = "homo_sapiens"):
    from cpgpt import download_cpgpt
    from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder

    res = download_cpgpt(model="small", species="human")
    embedder = DNALLMEmbedder(dependencies_dir=str(res.dependencies_path))
    meta = embedder.ensembl_metadata_dict[species]
    index = meta["nucleotide-transformer-v2-500m-multi-species"][2001]
    return index, str(res.dependencies_path)


def sample_track(path: Path, loci: list[tuple[str, int]], n_rows: int,
                 rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, covered) for every locus row, reading each chromosome once."""
    import pyBigWig

    bw = pyBigWig.open(str(path))
    chrom_sizes = bw.chroms()
    values = np.zeros(n_rows, dtype="float32")
    covered = np.zeros(n_rows, dtype=bool)

    by_chrom: dict[str, list[int]] = {}
    for i, (chrom, _pos) in enumerate(loci):
        by_chrom.setdefault(chrom, []).append(i)

    for chrom, idxs in by_chrom.items():
        # CpGPT uses Ensembl naming ("1"), ENCODE bigWigs use UCSC ("chr1").
        key = f"chr{chrom}" if f"chr{chrom}" in chrom_sizes else chrom
        if key not in chrom_sizes:
            continue
        intervals = bw.intervals(key)
        if not intervals:
            continue
        bin_size = intervals[0][1] - intervals[0][0]
        n_bins = chrom_sizes[key] // bin_size + 1
        track = np.full(n_bins, np.nan, dtype="float32")
        for start, end, val in intervals:
            track[start // bin_size] = val
        for i in idxs:
            b = loci[i][1] // bin_size
            # Exact zeros are bins ENCODE did not compute (e.g. chromosome ends).
            if 0 <= b < n_bins and not np.isnan(track[b]) and track[b] != 0:
                values[rows[i]] = track[b]
                covered[rows[i]] = True
    bw.close()
    return values, covered


DENSITY_BIN = 100_000


def locus_density(loci, n_rows, rows):
    """Per row: chromosome, and log count of loci sharing its 100 kb bin.

    Array probes cluster in gene-rich open chromatin, so locus density is a free
    stand-in for gene density, the usual reference for orienting eigenvectors.
    """
    chrom = np.empty(n_rows, dtype=object)
    bins = np.zeros(n_rows, dtype=np.int64)
    for i, (c, pos) in enumerate(loci):
        chrom[rows[i]] = c
        bins[rows[i]] = pos // DENSITY_BIN
    dens = np.zeros(n_rows, dtype="float32")
    for c in set(chrom):
        idx = np.flatnonzero(chrom == c)
        _, inv, counts = np.unique(bins[idx], return_inverse=True, return_counts=True)
        dens[idx] = np.log1p(counts[inv])
    return chrom, dens


def orient(values, covered, chrom, dens):
    """Flip each chromosome so the eigenvector correlates positively with density.

    ENCODE does not orient its eigenvectors consistently: the sign flips between
    files and between chromosomes of one file. Same-vs-different comparisons
    within a chromosome do not care, but a model reading the value does.
    Returns the oriented values and the per-chromosome |r| as a quality score.
    """
    out = values.copy()
    quality = {}
    for c in sorted(set(chrom[covered])):
        m = covered & (chrom == c)
        if m.sum() < 100:
            continue
        r = np.corrcoef(values[m], dens[m])[0, 1]
        if r < 0:
            out[m] = -values[m]
        quality[c] = abs(r)
    return out, quality


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-type", default="GM12878")
    ap.add_argument("--output-types", default="genome compartments",
                    help="comma-separated ENCODE output_type values")
    ap.add_argument("--max-tracks", type=int, default=2,
                    help="tracks per output type; replicates get averaged")
    ap.add_argument("--name", default=None, help="annotation set name")
    ap.add_argument("--dataset", default=None,
                    help="restrict to one ENCODE experiment accession, e.g. ENCSR538MYU "
                         "(HL-60/S4 differentiated toward granulocytes with retinoic acid)")
    args = ap.parse_args()

    name = args.name or args.cell_type.replace(" ", "_")
    print(f"[index] loading CpGPT locus index")
    index, _deps = load_index()
    n_rows = max(index.values()) + 1
    keys = list(index.keys())
    rows = np.fromiter((index[k] for k in keys), dtype=np.int64, count=len(keys))
    loci = [(k.split(":")[0], int(k.split(":")[1])) for k in keys]
    print(f"[index] {len(keys)} loci, {n_rows} memmap rows")
    chrom_of, dens = locus_density(loci, n_rows, rows)

    channels: list[str] = []
    used: list[str] = []
    mats: list[np.ndarray] = []
    for output_type in args.output_types.split(","):
        output_type = output_type.strip()
        found = encode_lookup(args.cell_type, output_type)
        if args.dataset:
            found = [f for f in found if args.dataset in f["dataset"]]
        if not found:
            print(f"[encode] no '{output_type}' for {args.cell_type}")
            continue
        # Score every candidate eigenvector and keep the best parent computation.
        # Some ENCODE eigenvectors track something other than compartments on
        # several chromosomes (|r| with density near zero), so first-found is
        # not safe.
        scored = []
        for f in found:
            v, c = sample_track(download_track(f["accession"]), loci, n_rows, rows)
            v, q = orient(v, c, chrom_of, dens)
            scored.append((f, v, c, q, float(np.median(list(q.values())))))
            print(f"  {f['accession']} {f['dataset']:<34} median |r| {scored[-1][4]:.2f}"
                  f"  worst {min(q, key=q.get)} {min(q.values()):.2f}")
        # Score each parent computation (its 5 kb + 10 kb pair) by its weakest
        # member, preferring pairs with no chromosome below |r| 0.3.
        groups: dict[str, list] = {}
        for t in scored:
            groups.setdefault(t[0]["parent"], []).append(t)
        def group_score(ts):
            worst = min(min(t[3].values()) for t in ts)
            return (worst >= 0.3, min(t[4] for t in ts))
        best_parent = max(groups, key=lambda k: group_score(groups[k]))
        chosen = [t for t in scored if t[0]["parent"] == best_parent][: args.max_tracks]
        picked = [t[0] for t in chosen]
        stack, cover = [t[1] for t in chosen], [t[2] for t in chosen]
        weak = {k: round(float(x), 2) for t in chosen for k, x in t[3].items() if x < 0.3}
        print(f"[encode] {output_type}: {len(found)} files, using "
              f"{[f['accession'] for f in picked]}"
              + (f"; weak chromosomes {weak}" if weak else ""))
        used += [f"{f['accession']} ({f['dataset']})" for f in picked]
        for j in range(1, len(stack)):
            both = cover[0] & cover[j]
            r = np.corrcoef(stack[0][both], stack[j][both])[0, 1]
            print(f"  {picked[j]['accession']} vs {picked[0]['accession']}: r = {r:+.3f}")
            if r < 0.5:
                raise SystemExit(f"tracks disagree (r = {r:+.3f}); refusing to average")
        vals = np.nanmean(np.stack(stack), axis=0)
        cov = np.any(np.stack(cover), axis=0)
        # Scale by sd but do NOT subtract the mean: A vs B compartment is defined
        # by the SIGN of the eigenvector, so re-centring would move the A/B
        # boundary and silently corrupt any same-compartment comparison.
        # Uncovered rows stay at 0, which the covered flag distinguishes from a
        # genuine boundary value.
        sd = vals[cov].std()
        z = np.zeros_like(vals)
        z[cov] = vals[cov] / (sd if sd > 0 else 1.0)
        mats.append(z)
        channels.append(f"{output_type.replace(' ', '_')}_scaled")
        mats.append(cov.astype("float32"))
        channels.append(f"{output_type.replace(' ', '_')}_covered")
        print(f"  coverage {cov.mean()*100:.1f}% of loci, sd {sd:.4g}, "
              f"A-fraction at these loci {np.mean(vals[cov] > 0)*100:.1f}% "
              f"(oriented per chromosome, sign(channel) == A/B)")

    if not mats:
        raise SystemExit("no tracks built")

    annot = np.stack(mats, axis=1).astype("float32")
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    out = ANNOT_DIR / f"{name}.mmap"
    mm = np.memmap(out, dtype="float32", mode="w+", shape=annot.shape)
    mm[:] = annot
    mm.flush()
    meta = {"name": name, "cell_type": args.cell_type, "dataset": args.dataset,
            "tracks": used, "shape": list(annot.shape),
            "channels": channels, "dtype": "float32"}
    (ANNOT_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"[write] {out}  shape {annot.shape}  channels {channels}")


if __name__ == "__main__":
    main()
