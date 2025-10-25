# 🧮 DeepSeek-R1: Reinforcement Learning for Fine-Tuning LLMs  
This document provides a detailed overview of the DeepSeek-R1 strategy for fine-tuning and preference-tuning large language models (LLMs). It covers the RL methods, distinctions from traditional approaches, the optimization algorithm (GRPO) summary, multi-stage training pipeline, reward design, model distillation, and additional technical detail.

---

### 1. Overview  
DeepSeek-R1 introduces a novel approach to improving reasoning capabilities and general instruction-following in large language models using reinforcement learning (RL). Rather than relying solely on large human-annotated supervised fine-tuning datasets and learned reward models, DeepSeek emphasises verifiable tasks (particularly reasoning/maths/code), multi-stage pipelines, and knowledge distillation to smaller models.

#### 1.1 Key Variants  
- **DeepSeek-R1-Zero**: RL-only variant without initial supervised fine-tuning (SFT). Uses verifiable reasoning tasks (e.g., math, code, logic) with automatically-computable reward signals.  
- **DeepSeek-R1**: Multi-stage pipeline starting with a “cold-start” SFT, followed by reasoning-oriented RL, generation of an SFT dataset from high-quality RL outputs, further SFT fine-tuning, and then a second RL stage for broader instruction-following.

---

### 2. The GRPO Algorithm: Overview
This section gives a summary of the core optimisation algorithm used in DeepSeek-R1, namely Grouped Relative Policy Optimization (GRPO).  
**For full mathematical details and derivation see the dedicated [GRPO Algorithm](../methods/grouped_relative_po_deepseek.md) .**

#### 2.1 Intuition  
- Instead of using a value network (critic) as in PPO, GRPO works by sampling multiple candidate outputs for each prompt and comparing their performance **within the group** (via ranking).  
- Encourages responses that outperform peers in the same group, emphasising **relative** improvement rather than absolute reward magnitudes.  
- Offers more stable training for LLMs at scale by avoiding the complexity and instability of critic/value training.

#### 2.2 Key Features  
- Group-based candidate sampling (group size $G$, e.g., 8–16).  
- Compute advantage  
  $$
    A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}
  $$  
  where $r_i$ is the reward of candidate $o_i$.  
- Use PPO-style clipped ratio of new policy vs old policy for each candidate, regularised by a KL-penalty to a reference policy.  
- Regularisation ensures the policy does not drift too far from the initial/frozen “reference” model.  
- No explicit value (critic) function required — critical for large-scale LLM fine-tuning.

---

### 3. Reward Design  
DeepSeek splits reward design into two main domains: reasoning-oriented tasks, and general instruction-following tasks.

#### 3.1 Reasoning-Oriented Tasks  
- **Correctness**: Verified automatically (e.g., solvers verify math answers, compilers/tests verify code solutions).  
- **Chain-of-Thought (CoT) / Format Enforcement**: Encourage structured reasoning, for example via tags or designated reasoning segments.  
- **Language Consistency / Style**: Penalise language mixing (e.g., mixing English & Mandarin) or incoherent formatting.  
- **Weighted Sum**: The overall reward may combine correctness + readability/style metrics.

#### 3.2 General Instruction-Following Tasks  
- Use of preference models or mixtures of rule-based checks (for helpfulness/harmlessness/style) and/or learned reward models for broader tasks.  
- After the reasoning-oriented RL stage, DeepSeek shifts to include these broader tasks so that the model becomes more instruction-capable beyond strictly verifiable reasoning domains.

---

### 4. Multi-Stage Training Pipeline  
Here is the step-by-step overview of the training strategy employed by DeepSeek-R1.

| Stage | Description |
|-------|-------------|
| **Stage 1: Cold-Start SFT** | A small curated dataset of chain-of-thought reasoning examples is used to bootstrap the model. Helps stabilise initial policy before heavy RL. |
| **Stage 2: Reasoning-Oriented RL** | Using GRPO on verifiable reasoning tasks (math, code, logic) to drive emergent reasoning capability, e.g., using only RL (DeepSeek-R1-Zero) or RL after SFT (DeepSeek-R1). |
| **Stage 3: Rejection Sampling → SFT Dataset** | Filter RL-generated outputs by quality/readability; this creates a high-quality dataset for SFT fine-tuning. This helps address issues such as language mixing or readability observed in R1-Zero. |
| **Stage 4: Second RL Stage (General Instruction-Following)** | Expand prompt coverage to include broad instructions and incorporate general reward signals (helpfulness, style, harmlessness). Encourages the model to generalise beyond reasoning tasks. |
| **Stage 5: Distillation to Smaller Models** | Use the high-capability RL-trained model as a **teacher**, generate reasoning-rich data, then fine-tune smaller student models via SFT on that data (rather than performing full RL on smaller models). |

#### 4.1 Pipeline Highlights / Additional Details  
- The reasoning-only RL variant (R1-Zero) demonstrates that emergent reasoning can arise via RL alone (no SFT) but suffers readability/language consistency issues.  
- For R1 proper, the cold-start SFT acts as a “kick-start” to the policy, improving readability and general language handling before RL.  
- Distillation step: Available models labelled e.g. DeepSeek-R1-Distill-Qwen-7B, 1.5B, 8B, 14B, 32B, 70B based on Qwen or Llama series.  
- According to public sources, R1 achieved reasoning performance comparable to OpenAI o1-1217 model on reasoning/multitask benchmarks.  

---

### 5. Distinctive Features Compared to Traditional Methods  
| Feature | Conventional RLHF / SFT + RL | DeepSeek-R1 Strategy |
|---------|-----------------------------|------------------------|
| **Initial SFT** | Often uses large human-annotated dataset | R1-Zero: none; R1: small cold-start SFT |
| **Reward Source** | Learned reward model (often from human preferences) | Reasoning tasks: rule-based correctness + ranking; General tasks: mixture |
| **Policy Optimisation** | PPO (with value network / critic, learned rewards) | GRPO (group ranking + clipped ratio + KL penalty) |
| **Domain Focus** | Broad instruction-following from the start | Emphasis on reasoning first → then general instructions |
| **Post-RL Dataset Generation** | Sometimes limited | RL outputs → filtered → SFT dataset then distillation |
| **Distillation to Smaller Models** | Optional / less emphasised | Explicit large → dataset → smaller models path |
| **Emergence of Reasoning** | Often via SFT + RL; may require large data/human-labelled | Demonstrated via RL alone (R1-Zero), then refined by SFT + RL |

---

### 6. Technical & Training Details (Additional Depth)  
Here are deeper details and notes from publicly available sources.

#### 6.1 Model Sizes and Releases  
- The R1 paper describes two main models: R1-Zero and R1, both built on a 37B-activated parameter MoE architecture with total 671B parameters.  
- Distilled variants: 1.5B, 7B, 8B, 14B, 32B, 70B parameters based on Qwen2.5 and Llama3 series.

#### 6.2 Reward Function & Sampling Details  
- For reasoning tasks, the reward function is largely rule-based: checking final answers for correctness, format tags for reasoning sections, penalising language mixing.  
- Some commentary notes that weaker signals like process rewards (how many reasoning steps) were less effective than outcome rewards (correct answer) in this context.  
- In sampling / generation for distillation: e.g., for R1-Distill models, generation settings included temperature 0.6, top-p 0.95, 64 responses per query used for pass@1 estimation.

#### 6.3 Training and Distillation Strategy  
- The distillation process uses the high-capability teacher model to generate large “reasoning-rich” datasets. Student models are then fine-tuned via SFT (not full RL) to inherit reasoning patterns.  
- Some sources indicate the smaller models may still underperform the teacher, but give much better cost/efficiency trade-offs.

#### 6.4 Observed Strengths & Weaknesses  
- Strength: Emergent reasoning capability via RL, high reasoning benchmark performance.  
- Weaknesses noted: The R1-Zero model had issues with readability and language mixing (because of skipping SFT).  
- Distilled models, while efficient, may have some drop in reasoning performance compared to full model.

---

### 7. Summary Table  
| Component                     | Role                                         | Example / Notes                                      |
|------------------------------|----------------------------------------------|------------------------------------------------------|
| **Policy Model (LLM)**       | Learns improved policy via RL                 | DeepSeek-R1, DeepSeek-R1-Zero                        |
| **Reference Model**          | Provides KL regularisation baseline           | Frozen SFT model (in GRPO)                           |
| **Reward Function**          | Scores responses                              | Correctness, readability, chain-of-thought format    |
| **Group Size (G)**           | Sampling granularity for GRPO                 | 8–16 outputs per prompt (typical)                    |
| **Advantage \(A_i\)**        | Relative performance metric within group      | Normalisation: \((r_i-\text{mean})/\text{std}\)     |
| **Objective (GRPO)**         | PPO-style surrogate + KL penalty             | See GRPO doc for full derivation                     |
| **Training Pipeline**        | Multi-stage (Cold-SFT → RL → SFT → RL → Distill) | Reasoning first, then broad instruction             |
| **Distillation**             | Transfer reasoning to smaller models          | Student models 1.5B–70B params                       |
| **Goal**                     | Efficient reasoning/instruction-fine-tuning  | Stable RL fine-tuning for large LLMs                |

---

### 8. Advantages & Limitations  
#### Advantages  
- Emergent reasoning via RL (especially R1-Zero) without relying only on large human-annotated SFT datasets.  
- Efficient multi-stage training strategy combining SFT, RL, filtering, and distillation.  
- Verifiable reward signals (correctness + format) reduce noise and training instability.  
- Distillation path enables smaller, deployable models with reasoning capability — good for cost/production trade-offs.

#### Limitations  
- Transparency of full dataset, hyperparameters, and training cost is limited in public domain.  
- General instruction-following beyond reasoning may still lag (especially in R1-Zero or smaller distilled variants).  
- Because smaller models are SFT-only post-distillation, they may not fully retain RL-derived benefits.  
- Effective reward design and especially filtering of RL outputs remain key; if RL outputs are of low quality, filtering becomes a bottleneck.

---

