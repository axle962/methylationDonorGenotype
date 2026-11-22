"""Latent block predictors f(z_A, c_B) -> z_B, their splits, training loop and residuals."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Joint kinds see all K source blocks at once (one prediction per target, so the residual is exactly
# orthogonal to what f can compute from those sources); pairwise kinds average K one-source predictions.
# Joint kinds come first so that base_kind matches "mlp_joint_struct" to "mlp_joint", not "mlp".
KINDS = ("mlp_joint", "linear_joint", "mlp", "linear", "block_mean")
PRIMARY_KIND = "mlp_joint"  # docs/PREREG.md section 1


class MLPPredictor(nn.Module):
    """concat(z_A, c_B) -> 3-layer MLP (hidden 1024, GELU, dropout 0.1)."""

    def __init__(self, d: int, d_c: int | None = None, hidden: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        d_c = d if d_c is None else d_c
        self.net = nn.Sequential(
            nn.Linear(d + d_c, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )

    def forward(self, z_a: torch.Tensor, c_b: torch.Tensor, block_b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_a, c_b], dim=-1))


class LinearPredictor(nn.Module):
    """concat(z_A, c_B) -> one linear layer."""

    def __init__(self, d: int, d_c: int | None = None) -> None:
        super().__init__()
        self.net = nn.Linear(d + (d if d_c is None else d_c), d)

    def forward(self, z_a: torch.Tensor, c_b: torch.Tensor, block_b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_a, c_b], dim=-1))


class JointMLPPredictor(nn.Module):
    """Deep-sets predictor over K source blocks: mean_k phi(z_Ak), concatenated with c_B, then a 3-layer
    MLP (hidden 1024, GELU, dropout 0.1). Invariant to source order and usable with any K."""

    joint = True

    def __init__(self, d: int, d_c: int | None = None, d_set: int = 512, hidden: int = 1024,
                 dropout: float = 0.1) -> None:
        super().__init__()
        d_c = d if d_c is None else d_c
        self.phi = nn.Sequential(nn.Linear(d, d_set), nn.GELU(), nn.Linear(d_set, d_set))
        self.rho = nn.Sequential(
            nn.Linear(d_set + d_c, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )

    def forward(self, z_a: torch.Tensor, c_b: torch.Tensor, block_b: torch.Tensor) -> torch.Tensor:
        """z_a: (..., K, d); c_b: (..., d_c) -> (..., d)."""
        return self.rho(torch.cat([self.phi(z_a).mean(-2), c_b], dim=-1))


class JointLinearPredictor(nn.Module):
    """Linear baseline for the joint kinds: concat(mean_k z_Ak, c_B) -> one linear layer."""

    joint = True

    def __init__(self, d: int, d_c: int | None = None) -> None:
        super().__init__()
        self.net = nn.Linear(d + (d if d_c is None else d_c), d)

    def forward(self, z_a: torch.Tensor, c_b: torch.Tensor, block_b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_a.mean(-2), c_b], dim=-1))


def is_joint(model: nn.Module) -> bool:
    return bool(getattr(model, "joint", False))


class BlockMeanPredictor(nn.Module):
    """Ignores z_A: the train-sample mean of z_B. A block that is never a training target has no
    training targets, so it gets the mean over the train blocks instead."""

    def __init__(self, z_train: torch.Tensor, train_blocks: torch.Tensor) -> None:
        super().__init__()
        table = z_train.mean(0)  # (n_blocks, d)
        not_train = torch.ones(table.size(0), dtype=torch.bool, device=table.device)
        not_train[train_blocks] = False
        table[not_train] = table[train_blocks].mean(0)
        self.register_buffer("table", table)

    def forward(self, z_a: torch.Tensor, c_b: torch.Tensor, block_b: torch.Tensor) -> torch.Tensor:
        return self.table[block_b]


def base_kind(kind: str) -> str:
    """'mlp_struct' -> 'mlp'."""
    return next(k for k in KINDS if kind == k or kind.startswith(k + "_"))


def make_predictor(kind: str, d: int, d_c: int, z_train: torch.Tensor, train_blocks: torch.Tensor) -> nn.Module:
    kind = base_kind(kind)
    if kind == "mlp_joint":
        return JointMLPPredictor(d, d_c)
    if kind == "linear_joint":
        return JointLinearPredictor(d, d_c)
    if kind == "mlp":
        return MLPPredictor(d, d_c)
    if kind == "linear":
        return LinearPredictor(d, d_c)
    return BlockMeanPredictor(z_train, train_blocks)


def make_splits(donor_ids: np.ndarray, tissues: np.ndarray, n_blocks: int, seed: int = 0,
                test_frac: float = 0.25, heldout_block_frac: float = 0.15,
                eval_only_blocks: np.ndarray | None = None) -> dict:
    """Donor-disjoint split: every donor with both Lung and Colon - Transverse is a test donor, then
    random donors fill the test set up to test_frac of donors. heldout_block_frac of the autosomal
    blocks are never used as a training target; eval_only_blocks (chrX) are never targets or sources."""
    rng = np.random.default_rng(seed)
    donors = sorted(set(donor_ids))
    tissues_of = {dn: set(tissues[donor_ids == dn]) for dn in donors}
    both = [dn for dn in donors if {"Lung", "Colon - Transverse"} <= tissues_of[dn]]
    test = set(both)
    n_fill = max(0, round(test_frac * len(donors)) - len(test))
    rest = [dn for dn in donors if dn not in test]
    test |= set(rng.choice(rest, size=min(n_fill, len(rest)), replace=False).tolist())
    is_test = np.isin(donor_ids, sorted(test))
    eval_only = np.array([] if eval_only_blocks is None else eval_only_blocks, dtype=int)
    candidates = np.setdiff1d(np.arange(n_blocks), eval_only)
    heldout = np.sort(rng.choice(candidates, size=round(heldout_block_frac * len(candidates)), replace=False))
    return {
        "seed": seed,
        "test_donors": sorted(test),
        "train_donors": [dn for dn in donors if dn not in test],
        "lung_colon_donors": both,
        "n_filled_test_donors": n_fill,
        "test_samples": np.flatnonzero(is_test).tolist(),
        "train_samples": np.flatnonzero(~is_test).tolist(),
        "heldout_blocks": heldout.tolist(),
        "eval_only_blocks": eval_only.tolist(),
        "train_blocks": np.setdiff1d(candidates, heldout).tolist(),
    }


def sample_sources(target_blocks: torch.Tensor, block_chrom: torch.Tensor, pool: torch.Tensor,
                   generator: torch.Generator) -> torch.Tensor:
    """One random source block per target from `pool`, on a different chromosome (rejection sampling)."""
    src = pool[torch.randint(len(pool), target_blocks.shape, generator=generator, device=pool.device)]
    while True:
        same = block_chrom[src] == block_chrom[target_blocks]
        if not same.any():
            return src
        src[same] = pool[torch.randint(len(pool), (int(same.sum()),), generator=generator, device=pool.device)]


def fixed_sources(block_chrom: np.ndarray, k: int, seed: int, pool: np.ndarray | None = None) -> np.ndarray:
    """k fixed random source blocks from `pool` (default all) on other chromosomes, per target block.
    k is capped at the number of candidates."""
    rng = np.random.default_rng(seed)
    pool = np.arange(len(block_chrom)) if pool is None else np.asarray(pool)
    cands = [pool[block_chrom[pool] != block_chrom[b]] for b in range(len(block_chrom))]
    k = min(k, min(len(c) for c in cands))
    return np.stack([rng.choice(c, size=k, replace=False) for c in cands])


def standardise_descriptors(C: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """Block descriptors scaled per dimension over blocks; an optional per-block profile is appended."""
    out = (C - C.mean(0)) / (C.std(0) + 1e-8)
    if extra is not None:
        extra = (extra - extra.mean(0)) / (extra.std(0) + 1e-8)
        out = np.concatenate([out, extra], axis=1)
    return out.astype(np.float32)


def train_predictor(kind: str, Zs: torch.Tensor, Cn: torch.Tensor, train_s: np.ndarray, train_b: np.ndarray,
                    block_chrom: np.ndarray, source_pool: np.ndarray, *, epochs: int = 20,
                    pairs_per_sample: int = 64, batch_size: int = 1024, lr: float = 3e-4,
                    weight_decay: float = 1e-4, seed: int = 0, k_sources: int = 8,
                    log: Callable[[str], None] = print) -> tuple[nn.Module, list[float]]:
    """Train f on (sample s, source A, target B): chrom(A) != chrom(B), B a train block, s a train
    sample, `pairs_per_sample` fresh pairs per sample per epoch, MSE in standardised space.
    Joint kinds draw a fresh set of `k_sources` sources per pair (random sets, so the deep-sets model
    learns to pool any K; evaluation uses the fixed sources from fixed_sources)."""
    device = Zs.device
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    train_s_t = torch.as_tensor(train_s, device=device)
    train_b_t = torch.as_tensor(train_b, device=device)
    chrom_t = torch.as_tensor(block_chrom, device=device)
    pool_t = torch.as_tensor(source_pool, device=device)
    model = make_predictor(kind, Zs.size(-1), Cn.size(-1), Zs[train_s_t], train_b_t).to(device)
    train_log: list[float] = []
    if base_kind(kind) == "block_mean":
        return model, train_log
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_examples = len(train_s) * pairs_per_sample
    joint = is_joint(model)
    for epoch in range(epochs):
        model.train()
        s = train_s_t.repeat_interleave(pairs_per_sample)
        tgt = train_b_t[torch.randint(len(train_b), (n_examples,), generator=gen, device=device)]
        if joint:  # (n_examples, K) sources, each on a different chromosome from its target
            src = sample_sources(tgt[:, None].expand(-1, k_sources).contiguous(), chrom_t, pool_t, gen)
        else:
            src = sample_sources(tgt, chrom_t, pool_t, gen)
        perm = torch.randperm(n_examples, generator=gen, device=device)
        total = 0.0
        for i in range(0, n_examples, batch_size):
            idx = perm[i : i + batch_size]
            z_a = Zs[s[idx, None], src[idx]] if joint else Zs[s[idx], src[idx]]
            pred = model(z_a, Cn[tgt[idx]], tgt[idx])
            loss = nn.functional.mse_loss(pred, Zs[s[idx], tgt[idx]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        train_log.append(total / n_examples)
        log(f"[{kind}] epoch {epoch + 1:>2d}/{epochs}  train MSE {train_log[-1]:.4f}")
    return model, train_log


@torch.no_grad()
def predict(model: nn.Module, Zs: torch.Tensor, Cn: torch.Tensor, sources: np.ndarray,
            targets: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """For every sample and target block: P = f over that block's fixed sources, and the squared error
    averaged over dimensions (and, for pairwise kinds, over sources). Pairwise kinds average K
    one-source predictions; joint kinds make one prediction from all K. Returns (P, pair_mse)."""
    model.eval()
    n, n_blocks, d = Zs.shape
    targets = np.arange(n_blocks) if targets is None else targets
    src = torch.as_tensor(sources, device=Zs.device)
    k = src.size(1)
    P = torch.zeros_like(Zs)
    pair_mse = torch.full((n, n_blocks), float("nan"), device=Zs.device)
    joint = is_joint(model)
    for b in targets:
        if joint:
            pred = model(Zs[:, src[b]], Cn[b].expand(n, -1), torch.full((n,), int(b), device=Zs.device))
            P[:, b] = pred
            pair_mse[:, b] = ((pred - Zs[:, b]) ** 2).mean(1)
            continue
        pred = model(Zs[:, src[b]], Cn[b].expand(n, k, -1), torch.full((n, k), int(b), device=Zs.device))
        P[:, b] = pred.mean(1)
        pair_mse[:, b] = ((pred - Zs[:, b : b + 1]) ** 2).mean((1, 2))
    return P, pair_mse


def load_predictor(kind_dir: Path, kind: str, d: int, d_c: int, device: torch.device) -> nn.Module:
    """Rebuild a trained predictor from <kind_dir>/model.pt."""
    state = torch.load(kind_dir / "model.pt", map_location=device, weights_only=True)
    if base_kind(kind) == "block_mean":
        model = BlockMeanPredictor.__new__(BlockMeanPredictor)
        nn.Module.__init__(model)
        model.register_buffer("table", state["table"])
    else:
        model = make_predictor(kind, d, d_c, None, None)
    model.load_state_dict(state)
    return model.to(device).eval()
