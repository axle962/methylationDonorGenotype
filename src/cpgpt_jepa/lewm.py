"""Phase B: an end-to-end block encoder trained in the LeWorldModel recipe (arXiv 2603.19312).

g reads the sites of one block. Each site token is CpGPT's frozen Nucleotide-Transformer sequence embedding
concatenated with its beta value; a 4-layer set transformer (self-attention without positional encoding,
d = 256) is mean-pooled over the observed sites and passed through a projector (1-layer MLP + BatchNorm;
LeWM adds it because a final LayerNorm keeps SIGReg from being optimised). The predictor is the Phase A
f(z_A, c_B) followed by a projector of the same kind, as in LeWM. Loss = prediction MSE + lambda * SIGReg on
the encoder embeddings (LeWM eq.: L_pred + lambda SIGReg(Z), default lambda = 0.1, M = 1024 directions); SIGReg
is applied to z_A and z_B separately and averaged, the analogue of LeWM's per-time-step application.
Gradients flow through every term: no EMA target, no stop-gradient (LeWM: "We do not employ stop-gradient,
exponential moving averages, or additional stabilization heuristics").

SIGReg (LeJEPA, Balestriero & LeCun 2025, arXiv 2511.08544) tests the embedding distribution against an isotropic Gaussian
along random 1-D projections with the Epps-Pulley statistic: for each unit direction a, with projections
p_i = a . z_i, T(a) = N * integral w(t) |phi_N(t) - exp(-t^2/2)|^2 dt, where phi_N is the empirical
characteristic function of the p_i and w(t) = exp(-t^2/2); evaluated by the trapezoid rule on 17 points in
[-5, 5] and averaged over M directions redrawn every step (seeded by the step). This follows LeJEPA's reference
code listing; LeJEPA's text writes w(t) = exp(-t^2/sigma^2) with sigma = 1 and LeWM's text describes knots in
[0.2, 4] with w(t) = exp(-t^2 / (2 sigma^2)); both papers report the results insensitive to the knots.
"""

import numpy as np
import torch
from torch import nn


def sigreg(z: torch.Tensor, n_directions: int = 1024, step: int = 0, t_max: float = 5.0, n_points: int = 17) -> torch.Tensor:
    """Sketched isotropic-Gaussian regulariser: mean Epps-Pulley statistic over random projections of z (N x d)."""
    gen = torch.Generator(device=z.device).manual_seed(int(step))
    a = torch.randn(z.size(1), n_directions, generator=gen, device=z.device, dtype=z.dtype)
    a = a / a.norm(dim=0, keepdim=True)
    t = torch.linspace(-t_max, t_max, n_points, device=z.device, dtype=z.dtype)
    gauss_cf = torch.exp(-0.5 * t**2)  # characteristic function of N(0, 1); also the weight w(t)
    xt = (z @ a).unsqueeze(-1) * t  # N x directions x points
    cos_m, sin_m = torch.cos(xt).mean(0), torch.sin(xt).mean(0)
    err = ((cos_m - gauss_cf) ** 2 + sin_m**2) * gauss_cf
    return (torch.trapezoid(err, t, dim=-1) * z.size(0)).mean()


class Projector(nn.Module):
    """LeWM's projector: a 1-layer MLP with Batch Normalization (applied over all leading dimensions)."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm1d(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        return self.bn(self.linear(x.reshape(-1, shape[-1]))).reshape(*shape[:-1], -1)


class BlockEncoder(nn.Module):
    """Set transformer over a block's sites: [NT embedding, beta] -> d, 4 self-attention layers, mean pool,
    projector."""

    def __init__(self, d_seq: int = 1024, d: int = 256, n_layers: int = 4, n_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.inp = nn.Linear(d_seq + 1, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.proj = Projector(d, d)

    def forward(self, seq: torch.Tensor, beta: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """seq: B x L x d_seq, beta: B x L (NaN = missing), pad: B x L bool (True = no site). -> B x d"""
        mask = pad | torch.isnan(beta)
        x = self.inp(torch.cat([seq, torch.nan_to_num(beta, nan=0.0).unsqueeze(-1).to(seq.dtype)], dim=-1))
        x = self.norm(self.encoder(x, src_key_padding_mask=mask))
        keep = (~mask).unsqueeze(-1).to(x.dtype)
        return self.proj((x * keep).sum(1) / keep.sum(1).clamp(min=1.0))


class LeWMPredictor(nn.Module):
    """Phase A predictor f(z_A, c_B) followed by a projector of the encoder's kind (LeWM)."""

    def __init__(self, d: int, d_c: int) -> None:
        super().__init__()
        from cpgpt_jepa.predictor import MLPPredictor

        self.f = MLPPredictor(d, d_c)
        self.proj = Projector(d, d)

    def forward(self, z_a: torch.Tensor, c_b: torch.Tensor, block_b: torch.Tensor) -> torch.Tensor:
        return self.proj(self.f(z_a, c_b, block_b))


class BlockData:
    """Sites of every block, their frozen NT embeddings and the samples' betas, on one device."""

    def __init__(self, blocks, site_offsets: np.ndarray, nt: np.ndarray, betas: np.ndarray, device: torch.device,
                 dtype: torch.dtype) -> None:
        self.blocks = blocks
        self.offsets = site_offsets  # block b's sites are rows offsets[b]:offsets[b + 1] of nt / betas columns
        self.nt = torch.from_numpy(nt).to(device=device, dtype=dtype)  # n_sites_total x d_seq
        self.betas = torch.from_numpy(betas).to(device=device, dtype=dtype)  # n_samples x n_sites_total
        self.device = device

    def batch(self, samples: torch.Tensor, blocks: torch.Tensor, max_sites: int | None = None,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Padded (seq, beta, pad) for (sample, block) pairs; a random subset of max_sites sites per block if set."""
        cols = []
        for b in blocks.tolist():
            c = torch.arange(int(self.offsets[b]), int(self.offsets[b + 1]), device=self.device)
            if max_sites is not None and len(c) > max_sites:
                c = c[torch.randperm(len(c), generator=generator, device=self.device)[:max_sites]]
            cols.append(c)
        length = max(len(c) for c in cols)
        idx = torch.zeros(len(cols), length, dtype=torch.long, device=self.device)
        pad = torch.ones(len(cols), length, dtype=torch.bool, device=self.device)
        for i, c in enumerate(cols):
            idx[i, : len(c)] = c
            pad[i, : len(c)] = False
        seq = self.nt[idx]
        beta = self.betas[samples.unsqueeze(1), idx]
        return seq, beta, pad
