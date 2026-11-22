"""Genomic blocks over the CpGPTDataSaver memmaps and the frozen-CpGPT encode path for one block.

The encode path mirrors CpGPTLitModule._predict_forward_step with every observed site as input:
NaN mask -> NaN to 0 -> drop each sample's NaN sites and pad (as _pad_input_data) -> net.encode_sample.
CpGPTDataset is not used because it subsamples sites randomly.
"""

import contextlib
from pathlib import Path

import numpy as np
import torch

AUTOSOMES = [str(i) for i in range(1, 23)]


def autocast(device: torch.device) -> contextlib.AbstractContextManager:
    """fp16 autocast on CUDA (the trainer's 16-mixed precision); plain fp32 on CPU."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def find_dataset_dir(processed_dir: str | Path) -> Path:
    """The single dataset folder CpGPTDataSaver wrote under processed_dir."""
    dirs = sorted(p.parent for p in Path(processed_dir).glob("*/X.mmap"))
    if len(dirs) != 1:
        msg = f"expected exactly one processed dataset under {processed_dir}, found {len(dirs)}"
        raise FileNotFoundError(msg)
    return dirs[0]


def open_processed(dataset_dir: Path) -> tuple[np.memmap, np.memmap, np.ndarray, str]:
    """Open X.mmap (samples x sites), var.mmap (chrom index, position), obs_names, species.

    allow_pickle: these .npy files are written by CpGPTDataSaver in stage 1 (object arrays).
    """
    n_obs, n_sites = (int(v) for v in np.load(dataset_dir / "X_shape.npy", allow_pickle=True))
    X = np.memmap(dataset_dir / "X.mmap", dtype="float32", mode="r", shape=(n_obs, n_sites))
    var = np.memmap(dataset_dir / "var.mmap", dtype="int32", mode="r", shape=(n_sites, 2))
    obs_names = np.load(dataset_dir / "obs_names.npy", allow_pickle=True)
    species = str(np.load(dataset_dir / "species.npy", allow_pickle=True))
    return X, var, obs_names, species


def embedding_rows(
    embedder, species: str, var: np.ndarray, dna_llm: str, dna_context_len: int
) -> np.ndarray:
    """Row of each site in the DNA-embedding memmap, looked up by "chrom:pos" as
    CpGPTDataset._get_dna_embeddings does; -1 where the site has no embedding."""
    reverse_vocab = embedder.ensembl_metadata_dict[species]["reverse_vocab"]
    index = embedder.ensembl_metadata_dict[species][dna_llm][dna_context_len]
    return np.array(
        [index.get(f"{reverse_vocab[int(chrom)]}:{int(pos)}", -1) for chrom, pos in var],
        dtype=np.int64,
    )


def open_dna_embeddings(embedder, species: str, dna_llm: str, dna_context_len: int) -> np.memmap:
    """The DNA-embedding memmap, with the path and shape CpGPTDataset._get_dna_embeddings uses."""
    index = embedder.ensembl_metadata_dict[species][dna_llm][dna_context_len]
    path = (
        Path(embedder.dependencies_dir)
        / "dna_embeddings"
        / species
        / dna_llm
        / f"{dna_context_len}bp_dna_embeddings.mmap"
    )
    size = embedder.llm_embedding_size_dict[dna_llm]
    return np.memmap(path, dtype="float32", mode="r", shape=(len(index), size))


def build_blocks(
    var: np.ndarray,
    chrom_names: dict[int, str],
    usable: np.ndarray,
    window_bp: int,
    min_sites: int = 500,
    chromosomes: list[str] = AUTOSOMES,
) -> list[dict]:
    """Partition the sites of `chromosomes` (default the autosomes) into fixed windows; drop windows
    with fewer than min_sites.

    Site column indices are sorted by position (the model's sorted_chromosome order).
    """
    chroms, positions = np.asarray(var[:, 0]), np.asarray(var[:, 1])
    name_to_index = {name: idx for idx, name in chrom_names.items()}
    blocks = []
    for name in chromosomes:
        if name not in name_to_index:
            continue
        cols = np.flatnonzero((chroms == name_to_index[name]) & usable)
        cols = cols[np.argsort(positions[cols], kind="stable")]
        window = positions[cols] // window_bp
        for w in np.unique(window):
            site_cols = cols[window == w]
            if len(site_cols) < min_sites:
                continue
            blocks.append(
                {
                    "block_id": len(blocks),
                    "chrom": name,
                    "start": int(w) * window_bp,
                    "end": (int(w) + 1) * window_bp,
                    "n_sites": len(site_cols),
                    "site_cols": site_cols.astype(np.int64),
                }
            )
    return blocks


def pad_inputs(
    meth: torch.Tensor,
    sequence_embeddings: torch.Tensor,
    chroms: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """net.encode_sample inputs built as _get_meth_and_na_mask + _pad_input_data build them,
    with all observed sites selected as input (the input mask is every site minus NaNs)."""
    mask_na = torch.isnan(meth)
    meth = torch.nan_to_num(meth, nan=0.0)
    tensors = {
        "meth": meth,
        "sequence_embeddings": sequence_embeddings,
        "chroms": chroms,
        "positions": positions,
        "mask_na": mask_na,
    }
    if not mask_na.any():
        return tensors
    padding_values = {
        "meth": 0.0,
        "sequence_embeddings": 0.0,
        "chroms": -1,
        "positions": -1,
        "mask_na": True,
    }
    keep = ~mask_na
    return {
        key: torch.nn.utils.rnn.pad_sequence(
            [value[i, keep[i]] for i in range(meth.size(0))],
            batch_first=True,
            padding_value=padding_values[key],
        )
        for key, value in tensors.items()
    }


@torch.no_grad()
def encode_sequence(net, dna_embeddings: np.ndarray, device: torch.device) -> torch.Tensor:
    """net.encode_sequence for one set of sites, shared across samples: (1, n_sites, d)."""
    dna = torch.from_numpy(np.ascontiguousarray(dna_embeddings)).to(device).unsqueeze(0)
    with autocast(device):
        return net.encode_sequence(dna)


@torch.no_grad()
def encode_samples(
    net,
    meth: np.ndarray,
    sequence_embeddings: torch.Tensor,
    var_sites: np.ndarray,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Sample embeddings (n_samples, d) from one site set. meth is samples x sites with NaNs."""
    chroms = torch.from_numpy(np.ascontiguousarray(var_sites[:, 0])).to(device).unsqueeze(0)
    positions = torch.from_numpy(np.ascontiguousarray(var_sites[:, 1])).to(device).unsqueeze(0)
    out, start = [], 0
    while start < len(meth):
        batch = torch.from_numpy(meth[start : start + batch_size]).to(device)
        n = batch.size(0)
        try:
            with autocast(device):
                inputs = pad_inputs(
                    batch,
                    sequence_embeddings.expand(n, -1, -1),
                    chroms.expand(n, -1),
                    positions.expand(n, -1),
                )
                z = net.encode_sample(**inputs)
        except torch.OutOfMemoryError:
            if batch_size == 1 or device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(f"  CUDA OOM at {meth.shape[1]} sites; batch size -> {batch_size}")
            continue
        out.append(z.float())
        start += n
    return torch.cat(out)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
