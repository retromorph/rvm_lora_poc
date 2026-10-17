# Literature and implementation references

Use these to understand methods and verify API compatibility. Pin **exact Git commits** for external source used in code; comply with licenses. Do not assume an upstream benchmark configuration is equivalent to our frozen pilot.

## Theory

1. M. E. Tipping & A. C. Faul, *Fast Marginal Likelihood Maximisation for Sparse Bayesian Models* (AISTATS 2003): https://proceedings.mlr.press/r4/tipping03a.html — foundational sequential evidence maximization, addition and deletion. Our rank-one formula is not novel.
2. *Learning Adapter Rank via Symmetry Breaking* (2025/2026): https://arxiv.org/abs/2506.22809 — relevant rank-wise Bayesian adaptation prior art; do not claim that ARD-rank LoRA by itself is new.
3. AdaLoRA, *Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*: https://arxiv.org/abs/2303.10512 — original rank-budget context.

## Strong allocation baselines

4. EVA, *Parameter Efficient Fine-tuning via Explained Variance Adaptation* (NeurIPS 2025): https://proceedings.neurips.cc/paper_files/paper/2025/hash/41d33bd41fd44bd9dba0e092047cf213-Abstract-Conference.html
   - Authors' GitHub: https://github.com/ml-jku/EVA
   - PEFT reference & examples: https://github.com/huggingface/peft/blob/main/examples/eva_finetuning/README.md
5. DPLoRA / OPLoRA (ACL Findings 2026): https://github.com/cjpark-ai/DPLoRA — **OPLoRA = initial ILP allocation only**, while DPLoRA includes progressive pruning. Contains RoBERTa/GLUE implementation. Check this code's assumptions before using the source.
6. Hugging Face PEFT LoRA config, `rank_pattern`, `target_modules`, `use_rslora`, EVA: https://huggingface.co/docs/peft/package_reference/lora
7. Hugging Face Transformers GLUE sequence-classification example: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py — currently uses `nyu-mll/glue`; pin exact dataset revision if possible.
8. RoBERTa model card: https://huggingface.co/FacebookAI/roberta-base

## Engineering / secure operations

9. `uv` + PyTorch CUDA installation: https://docs.astral.sh/uv/guides/integration/pytorch/
10. MLflow tracking server: https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/
11. MLflow network security and allowed hosts: https://mlflow.org/docs/latest/self-hosting/security/network/
12. MLflow CLI: https://mlflow.org/docs/latest/api_reference/cli.html

## Principle

External code may be consulted or incorporated, but the final report needs to say **exactly** which repo + SHA + method variant was used. In particular, PEFT EVA may bring its own initialization benefit, and a hand-ported OPLoRA-style ILP is not identical to an official reproduction unless checked.
