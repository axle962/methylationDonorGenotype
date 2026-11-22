"""Build metadata.csv and significant-mQTL CpG lists from the eGTEx downloads.
   python scripts/build_metadata.py [DATA_DIR]   (default ./data/egtex)"""
import gzip, sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "./data/egtex")

# GEO SOFT -> metadata.csv
rows, cur = [], {}
with gzip.open(root / "geo" / "GSE213478_family.soft.gz", "rt") as f:
    for line in f:
        if line.startswith("^SAMPLE"):
            if cur: rows.append(cur)
            cur = {"gsm": line.split("=")[1].strip()}
        elif line.startswith("!Sample_title"):
            cur["sample_id"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_characteristics_ch1"):
            k, v = line.split("=", 1)[1].strip().split(":", 1)
            cur[k.strip()] = v.strip()
if cur: rows.append(cur)
meta = pd.DataFrame(rows).rename(columns={"collaborator_participant_id": "donor_id", "Sex": "sex", "sample_plate": "plate"})
meta = meta[["sample_id", "gsm", "donor_id", "tissue", "sex", "plate", "sentrix_id"]]
meta.to_csv(root / "metadata.csv", index=False)

print(f"samples: {len(meta)}   donors: {meta.donor_id.nunique()}")
print(meta.tissue.value_counts().to_string())
per_donor = meta.groupby("donor_id").tissue.apply(set)
print(f"\ndonors with both Lung and Colon - Transverse: {sum({'Lung','Colon - Transverse'} <= s for s in per_donor)}")
print("tissues per donor:", pd.Series([len(s) for s in per_donor]).value_counts().sort_index().to_dict())

# beta header check
betas = root / "geo" / "GSE213478_betas.txt.gz"
if betas.exists():
    with gzip.open(betas, "rt") as f:
        cols = [c.strip('"') for c in f.readline().rstrip("\n").split("\t")]
        first = f.readline().split("\t")
    hit = sum(c in set(meta.sample_id) for c in cols)
    print(f"\nbeta file: {len(cols)} header fields; first probe = {first[0]!r}; {hit}/{len(meta)} columns match metadata sample_id")
    if hit < len(meta): print("  -> header naming differs; first 5 header fields:", cols[:5])
else:
    print("\n(beta file missing; skipped header check)")

# significant mQTL CpGs per tissue (q < 0.05) + union
union = set()
for p in sorted((root / "mqtl").glob("*.regular.perm.fdr.txt")):
    tissue = p.name.split(".")[0]
    df = pd.read_csv(p, sep="\t")
    sig = df.loc[df["qval"] < 0.05, "cpg_id"].astype(str)
    (root / f"mqtl_sites_{tissue}.txt").write_text("\n".join(sig) + "\n")
    union |= set(sig)
    print(f"{tissue:22s} tested {len(df):7d}   significant {len(sig):7d}")
(root / "mqtl_sites_any.txt").write_text("\n".join(sorted(union)) + "\n")
print(f"union of significant mQTL CpGs: {len(union)}")
