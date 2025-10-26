# Reinforcement Learning for Fine-Tuning Large Language Models

Welcome to the repository on **RL-based fine-tuning of LLMs**. This project explores methods for aligning large language models with human preferences, improving instruction following, and optimizing task-specific performance.

---

## Why RL-Based Fine-Tuning?

Large Language Models (LLMs) trained on massive text corpora often produce outputs that are plausible but not always aligned with user intent or desired behavior.  
Reinforcement Learning (RL) enables fine-tuning LLMs by optimizing for **specific objectives**, **human feedback**, or **pairwise preferences**, allowing models to generate more reliable and controllable responses.

---

## Methods Overview

This repository covers the following key RL-based fine-tuning methods:

| Method | Focus | Highlights |
|--------|-------|-----------|
| [PPO](./methods/ppo.md) | Proximal Policy Optimization | Balances exploration and stability in policy updates. |
| [DPO](./methods/dpo.md) | Direct Preference Optimization | Aligns model outputs with human preferences without a full RL loop. |
| [DRPO](./methods/decoupled_rpo.md) | Decoupled Reward Policy Optimization | Separates reward modeling from policy updates for improved stability. |
| [GRPO](./methods/guided_rpo.md) | Guided Relative Policy Optimization | Incorporates pairwise comparisons or structured guidance during training. |
| [Direct Reward Policy Optimization (DRPO-Offline)](./methods/drpo.md) | Offline Reward Optimization | Uses pre-collected reward data to guide policy updates without live RL interaction. |
| [Guided RPO](./methods/guided_rpo.md) | Guided Reward Policy Optimization | Combines structured guidance with pairwise preference learning for improved alignment. |

> Each method page contains detailed explanations, mathematical formulations, and references for further reading.

---

## Advanced Topics

In addition to the main methods, the repository also explores advanced topics related to RL fine-tuning:

- [KL Penalty in Policy Optimization](./advanced_topics/kl_penalty.md) – Techniques to stabilize learning by penalizing deviations from reference policies.  
- [Reward Hacking](./advanced_topics/rewards_hacking.md) – Understanding how misaligned rewards can lead to unintended behaviors.  
- [DeepSeek RL Details](./advanced_topics/deepseek_rl_finetuning.md) – In-depth insights into the DeepSeek approach to LLM alignment.

---