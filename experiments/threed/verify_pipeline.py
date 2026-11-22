"""End-to-end check of the 3D annotation arm.

Runs the baseline model and the widened model (fed real GM12878 compartment
annotations) over the same samples. With zero-initialised annotation weights the
two must agree exactly; any difference means the plumbing is altering the input
the model already had, which would confound every later comparison.
"""

import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from cpgpt import download_cpgpt
from cpgpt.data.components.cpgpt_datasaver import CpGPTDataSaver
from cpgpt.data.components.dna_llm_embedder import DNALLMEmbedder
from cpgpt.data.components.illumina_methylation_prober import IlluminaMethylationProber
from cpgpt.data.cpgpt_datamodule import CpGPTDataModule
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
from cpgpt.trainer.cpgpt_trainer import CpGPTTrainer

ROOT = Path(__file__).resolve().parents[2]
ANNOT = ROOT / "data/3d/annotations/pool5.mmap"
WIDE = ROOT / "data/3d/small_k2.ckpt"
CTX = 2000


def build(cfg_path, ckpt_path, k, inferencer):
    cfg = inferencer.load_cpgpt_config(str(cfg_path))
    if k:
        cfg.model.net.d_dna_embedding = cfg.model.net.d_dna_embedding + k
    return cfg, inferencer.load_cpgpt_model(cfg, model_ckpt_path=str(ckpt_path),
                                            strict_load=True)


def main() -> None:
    res = download_cpgpt(model="small", species="human")
    deps = str(res.dependencies_path)
    k = json.loads(Path(str(ANNOT).replace(".mmap", ".json")).read_text())["shape"][1]
    print(f"[annot] {ANNOT.name}, {k} channels")

    df = pd.read_feather(ROOT / "data/geo/processed/GSE182215_betas.arrow").set_index("GSM_ID")
    vocab = set(json.load(open(res.vocab_path))["input"])
    df = df.loc[:, df.columns.isin(vocab)].iloc[:4]

    run = ROOT / "data/3d/verify"
    (run / "raw").mkdir(parents=True, exist_ok=True)
    arrow = run / "raw/b.arrow"
    df.reset_index().to_feather(arrow)
    emb = DNALLMEmbedder(dependencies_dir=deps)
    prober = IlluminaMethylationProber(dependencies_dir=deps, embedder=emb)
    CpGPTDataSaver(data_paths=str(arrow), processed_dir=str(run / "proc")).process_files(
        prober, emb)

    inf = CpGPTInferencer(dependencies_dir=str(ROOT / "dependencies"), data_dir=str(ROOT / "data"))

    def dm(annotations):
        return CpGPTDataModule(
            predict_dir=str(run / "proc"), dependencies_dir=deps, batch_size=1,
            num_workers=0, max_length=CTX, dna_llm="nucleotide-transformer-v2-500m-multi-species",
            dna_context_len=2001, sorting_strategy="sorted_chromosome", pin_memory=False,
            annotations_path=annotations)

    base_cfg, base_model = build(res.config_path, res.checkpoint_path, 0, inf)
    wide_cfg, wide_model = build(res.config_path, WIDE, k, inf)

    trainer = CpGPTTrainer(accelerator="cpu", devices=1, enable_progress_bar=False)
    a = trainer.predict(model=base_model, datamodule=dm(None),
                        predict_mode="forward", return_keys=["sample_embedding"])
    b = trainer.predict(model=wide_model, datamodule=dm(str(ANNOT)),
                        predict_mode="forward", return_keys=["sample_embedding"])

    ea, eb = a["sample_embedding"].float(), b["sample_embedding"].float()
    diff = (ea - eb).abs().max().item()
    print(f"[baseline] embedding {tuple(ea.shape)} norm {ea.norm(dim=-1).mean():.4f}")
    print(f"[widened ] embedding {tuple(eb.shape)} norm {eb.norm(dim=-1).mean():.4f}")
    print(f"[verify  ] max abs diff: {diff:.3e}")
    print("PASS: annotation arm starts identical to baseline" if diff < 1e-4
          else "FAIL: plumbing changed the model's input")

    ann = np.memmap(ANNOT, dtype="float32", mode="r",
                    shape=tuple(json.loads(Path(str(ANNOT).replace('.mmap','.json')).read_text())["shape"]))
    print(f"[annot   ] channel 0 mean {ann[:, 0].mean():.4f} sd {ann[:, 0].std():.4f}, "
          f"coverage {ann[:, 1].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
