# Touch-R1

Training and inference code for **Touch-R1: Reinforcing Touch Reasoning in MLLMs**.

[Project Page](https://laiyingxin2.github.io/Touch-R1/) · [Paper](https://arxiv.org/abs/2605.27154) · [Model](https://huggingface.co/laiyingxin/Touch-R1) · [Dataset](https://huggingface.co/datasets/laiyingxin/TouchReason)

## Installation

```bash
git clone https://github.com/laiyingxin2/Touch-R1.git
cd Touch-R1/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-research.txt
```

Use a PyTorch build compatible with your CUDA environment. The backbone is [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

Download the model weights and TouchReason data from the links above. Update local paths in `research/source/` for the backbone, shared modules, media, and feature caches.

## Training

| Stage | Objective | Entry point |
| --- | --- | --- |
| 1. Visual pretraining | Future tactile-token prediction | [stage1.py](research/source/touchreason_touchr1_three_stage_20261024/stage1.py) |
| 2. Supervised fine-tuning | Tactile instruction tuning | [fast_worker.py](research/source/touchreason_v12_best_sft_20261025/fast_worker.py) |
| 3. GRPO | Reinforcement learning for touch reasoning | [grpo_worker.py](research/source/touchr1_grpo_accelerated_20261028/grpo_worker.py) |

Prepare episode metadata, SQLite runtime views, data splits, and stage-specific plan/runtime JSON files before training. Configure each worker's rank, world size, rendezvous, GPU visibility, and memory limits for your hardware.

GRPO starts from an SFT checkpoint paired with its Stage 1 visual adapter. See the [source map](research/README.md) for data preparation, continual SFT, and evaluation modules.

## Inference

Use a checkpoint directory containing `adaptation.pt` and its matching `vision_adapter.pt`:

```bash
python research/source/touchreason_touchr1_three_stage_20261024/infer.py \
  --checkpoint /path/to/Touch-R1 \
  --inputs /path/to/normalized_inputs.jsonl \
  --output /path/to/predictions.jsonl
```

Inputs follow the normalized, answer-free JSONL format in [prepare_test_inputs.py](research/source/touchreason_touchr1_three_stage_20261024/prepare_test_inputs.py). Each output record contains `id`, `response`, and `token_ids`.

## Utilities

| Script | Purpose |
| --- | --- |
| [checkpoint_info.py](scripts/checkpoint_info.py) | Inspect checkpoint metadata |
| [dataset_preview.py](scripts/dataset_preview.py) | Preview dataset records and model inputs |
| [score_touchreason.py](scripts/score_touchreason.py) | Score predictions with the local TouchReason parser |

See [license status](LICENSE_STATUS.md) for code, model, and data terms.
