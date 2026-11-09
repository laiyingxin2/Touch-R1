# Training and Inference Source Map

All paths below are relative to `source/`. These snapshots preserve the original implementation; installation and machine-specific setup are documented in [the code guide](../README.md).

| Component | Directory | Entry points |
| --- | --- | --- |
| Public tactile data preparation | `public_tactile_qa_20261022` | `organize_datasets.py`, `clean_datasets.py`, `prepare_qa.py` |
| Shared backbone and PEFT runtime | `public_tactile_touchr1_peft_20261023` | `runtime.py`, `method_runtime.py` |
| Stage 1 visual training | `touchreason_touchr1_three_stage_20261024` | `stage1.py`, `vision.py`, `efficient_attention.py` |
| Tactile model and objectives | `touchreason_touchr1_three_stage_20261024` | `model.py`, `language.py`, `rewarding.py` |
| SFT | `touchreason_v12_best_sft_20261025` | `worker.py`, `fast_worker.py`, `experiment.py` |
| Continual SFT | `touchr1_continual_dual_20261028` | `train_job.py`, `common.py` |
| Fixed-reference GRPO | `touchr1_public_sft_grpo_16a800_20261028` | `grpo_worker.py`, `grpo_ops.py` |
| Accelerated GRPO | `touchr1_grpo_accelerated_20261028` | `grpo_worker.py`, feature-cache and batching helpers |
| Inference | `touchreason_touchr1_three_stage_20261024` | `infer.py` |
| Batch generation and scoring | `touchr1_best_full_eval_20261102` | `run_eval.py` |

Supporting data preparation, scoring, and generation modules are retained alongside the training code. Source hashes are recorded in `provenance/source_manifest.json`; this upload does not include experimental reports, model tensors, raw media, credentials, or cloud orchestration tooling.
