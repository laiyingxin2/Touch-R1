# Touch-R1 Training and Inference Code

This directory contains the original Touch-R1 research source snapshots for visual pretraining, supervised fine-tuning (SFT), GRPO, and inference. The project website remains in the repository's separate `docs/` directory.

## Installation

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-research.txt
```

Choose a PyTorch build compatible with your CUDA environment. The recorded dependency versions are in `research/provenance/environment_versions.json`.

Obtain the backbone, adapters, and data separately:

- Backbone: `Qwen/Qwen2.5-VL-7B-Instruct`.
- Touch-R1 model repository: <https://huggingface.co/laiyingxin/Touch-R1>.
- TouchReason dataset repository: <https://huggingface.co/datasets/laiyingxin/TouchReason>.

## Setup Before Running

These are historical research implementations, not a standalone portable training package. Before using the worker commands below:

1. Work on a copy of `research/source/` and update the original `/mnt/workspace/lyx` project references, backbone paths, media paths, and feature-cache locations for your machine.
2. Prepare the required episode metadata, SQLite runtime views, splits, and per-stage plan/runtime JSON files. Model weights, original media, databases, and caches are not included in this Git repository.
3. Configure the distributed launcher for the chosen worker, including its rank, world size, rendezvous, and device visibility. Several workers assume one visible GPU per process and are not interchangeable with an unconfigured multi-GPU `torchrun` command.
4. Keep the Stage 1 visual adapter and the corresponding SFT parent consistent when continuing GRPO or loading a checkpoint.

Full training and GPU inference have not been validated in a clean environment by this source upload. Original cloud submission/controller tooling is not included.

## Training

The following are the original worker entry points, to run only after the setup above and with the required launcher environment:

```bash
python research/source/touchreason_touchr1_three_stage_20261024/stage1.py
python research/source/touchreason_v12_best_sft_20261025/fast_worker.py
python research/source/touchr1_grpo_accelerated_20261028/grpo_worker.py
```

They implement Stage 1 future tactile-token prediction, SFT, and GRPO, respectively. Public-data continual SFT is implemented in `research/source/touchr1_continual_dual_20261028/train_job.py`, which takes a prepared job JSON path as its positional argument.

See [the source map](research/README.md) for shared components and additional entry points.

## Inference

After relocating the source paths and preparing a checkpoint:

```bash
python research/source/touchreason_touchr1_three_stage_20261024/infer.py \
  --checkpoint /path/to/Touch-R1 \
  --inputs /path/to/normalized_inputs.jsonl \
  --output /path/to/predictions.jsonl
```

Inputs must use the normalized, answer-free JSONL schema consumed by the archived model/data modules; arbitrary chat JSON is not supported by this entry point. Each output record contains `id`, `response`, and `token_ids`.

The original loader restores language LoRA and tactile adaptation from `adaptation.pt`, and loads `vision_adapter.pt` separately. Loading only `adapter_model.safetensors` does not restore all trained components.

## Inspection Utilities

These utilities can be inspected without loading the GPU model:

```bash
python scripts/checkpoint_info.py /path/to/Touch-R1
python scripts/dataset_preview.py --help
python scripts/score_touchreason.py --help
```

`research/provenance/source_manifest.json` records the original paths, file sizes, and SHA-256 hashes of the source snapshots. See [license status](LICENSE_STATUS.md) before redistributing code, weights, or data.
