"""Widen a CpGPT checkpoint's DNA input adapter to accept extra annotation channels.

`dna_encoder.input_adapter` is the only place the DNA embedding enters the model,
a single (d_embedding, d_dna_embedding) linear layer. Padding its weight with
zeros leaves the widened model numerically identical to the original, so an
annotation arm starts exactly at baseline and whatever it learns in those
channels measures how much the model actually wants the prior.

Note that `train.py` calls `load_state_dict(..., strict=cfg.strict_load)`, and
strict=False still raises on shape mismatch, so the padding has to happen here
rather than via a config flag.

Usage:
    python experiments/threed/widen_checkpoint.py --k 2 --out data/3d/small_k2.ckpt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
KEY = "dna_encoder.input_adapter.weight"


def widen(state: dict, k: int) -> dict:
    out = dict(state)
    matches = [key for key in state if key.endswith(KEY)]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {KEY}, found {matches}")
    key = matches[0]
    w = state[key]
    out[key] = torch.cat([w, torch.zeros(w.shape[0], k, dtype=w.dtype)], dim=1)
    print(f"  {key}: {tuple(w.shape)} -> {tuple(out[key].shape)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--k", type=int, required=True, help="annotation channels to add")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true", default=True)
    args = ap.parse_args()

    from cpgpt import download_cpgpt

    res = download_cpgpt(model=args.model, species="human")
    ckpt = torch.load(res.checkpoint_path, weights_only=False, map_location="cpu")
    print(f"[load] {res.checkpoint_path}")

    ckpt["state_dict"] = widen(ckpt["state_dict"], args.k)
    if "hyper_parameters" in ckpt:
        hp = ckpt["hyper_parameters"]
        net = hp.get("net")
        if hasattr(net, "d_dna_embedding"):
            net.d_dna_embedding = net.d_dna_embedding + args.k
        elif isinstance(net, dict) and "d_dna_embedding" in net:
            net["d_dna_embedding"] += args.k

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"[write] {out}")

    if args.verify:
        import yaml

        from cpgpt.model.components.model import CpGPT

        cfg = {k: v for k, v in
               yaml.safe_load(open(res.config_path))["model"]["net"].items()
               if k != "_target_"}
        base_sd = {k.replace("net._orig_mod.", ""): v
                   for k, v in torch.load(res.checkpoint_path, weights_only=False,
                                          map_location="cpu")["state_dict"].items()
                   if k.startswith("net.")}
        wide_sd = {k.replace("net._orig_mod.", ""): v
                   for k, v in ckpt["state_dict"].items() if k.startswith("net.")}

        base = CpGPT(**cfg)
        base.load_state_dict(base_sd, strict=True)
        base.eval()
        wcfg = dict(cfg)
        wcfg["d_dna_embedding"] = cfg["d_dna_embedding"] + args.k
        wide = CpGPT(**wcfg)
        wide.load_state_dict(wide_sd, strict=True)
        wide.eval()

        torch.manual_seed(0)
        x = torch.randn(2, 32, cfg["d_dna_embedding"])
        ann = torch.randn(2, 32, args.k) * 5
        with torch.no_grad():
            a = base.encode_sequence(x)
            b = wide.encode_sequence(torch.cat([x, ann], dim=-1))
        diff = (a - b).abs().max().item()
        print(f"[verify] max abs diff vs original: {diff:.2e}")
        if diff > 1e-6:
            raise SystemExit("widened checkpoint is NOT equivalent at init")
        print("[verify] OK: widened model starts identical to the pretrained one")


if __name__ == "__main__":
    main()
