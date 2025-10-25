# 🧩 Guided Reward Policy Optimization (GRPO) — Structured RL for LLM Fine-Tuning

### 1. Overview

**Guided Reward Policy Optimization (GRPO)** is a reinforcement learning-based algorithm for fine-tuning **Large Language Models (LLMs)** using **structured, guided rewards**. It extends preference-based or reward-based methods like DRPO by incorporating richer feedback signals — human, LLM-generated, or heuristic — to improve instruction-following, safety, and multi-step reasoning.

GRPO combines the stability of offline policy optimization with **explicit guidance**, helping models learn efficiently while avoiding reward hacking and instability issues common in standard RLHF.

---

### 2. The Big Picture: From PPO/DRPO to GRPO

| Stage               | PPO                        | DRPO                  | GRPO                                                      |
| ------------------- | -------------------------- | --------------------- | --------------------------------------------------------- |
| **Reward**          | Scalar, learned from RM    | Pairwise preferences  | Guided / structured (scalar, stepwise, multi-dimensional) |
| **RL Loop**         | Online, iterative          | Offline               | Offline or online                                         |
| **KL Penalty**      | Manual, global             | ❌ Not used            | ✅ Adaptive, local                                         |
| **Complexity**      | High (policy + value + RM) | Moderate              | Higher (requires guided reward design)                    |
| **Data Efficiency** | Moderate                   | Uses preference pairs | Efficient with structured feedback                        |

GRPO can be interpreted as **“DRPO with a smarter teacher”**, providing stepwise or hierarchical guidance rather than only pairwise comparisons.

---

### 3. Intuitive Understanding

Imagine training an assistant:

* **PPO:** Assistant writes an answer → scored by a reward model → RL updates policy.
* **DRPO:** Assistant sees preferred vs. rejected responses → updates policy offline.
* **GRPO:** Assistant receives **guided signals at each reasoning step**, including safety checks, intermediate correctness, or multi-objective rewards, then updates its policy while staying close to the reference model.

Thus, GRPO provides **structured supervision**, improving sample efficiency and safety without requiring purely online scalar reward exploration.

---

### 4. Training Data and Setup

Each GRPO training example consists of $(x, y, R_{\text{guided}})$

where:

* $x$ = prompt or input query
* $y$ = model response
* $R_{\text{guided}}$ = guided reward, potentially multi-dimensional or stepwise (human, LLM, heuristics)

The model learns to **maximize expected guided reward**, while staying close to a **reference policy** $\pi_{\text{ref}}$ (usually the pretrained SFT model).

---

### 5. GRPO Formulation

!!! example "📘 Mathematical Formulation"

    #### 5.1. Objective Function

    GRPO extends offline policy optimization with structured reward guidance and KL regularization:

    $$
    L_{\mathrm{GRPO}}(\theta)
    = -\mathbb{E}_{x \sim D} \Big[ R_{\text{guided}}(x, \pi_\theta(x)) - \beta \text{KL}(\pi_\theta(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x)) \Big]
    $$

    where:

    * $\pi_\theta$ = trainable LLM policy
    * $\pi_{\text{ref}}$ = frozen reference model
    * $R_{\text{guided}}$ = structured reward signal (scalar, vector, or stepwise)
    * $\beta$ = KL regularization coefficient controlling faithfulness to base model

    This objective encourages maximizing **expected guided reward** while preventing drift from the reference model.
    
    ---
    
    #### 5.2. Intuition

    * The **guided reward** can be seen as a rich, informative signal replacing sparse scalar rewards.
    * The **KL term** serves as an implicit regularizer to prevent catastrophic forgetting or reward hacking.
    * GRPO generalizes DRPO: if (R_{\text{guided}}) is constructed from pairwise preference comparisons, GRPO reduces to DRPO.

    ---

    #### 5.3. Implementation Notes / Best Practices

    * **Reference model frozen:** Ensure $\pi_{\text{ref}}$ does not receive gradients.
    * **Reward shaping:** Use stepwise or multi-objective rewards for complex reasoning tasks.
    * **Numerical stability:** Normalize rewards and scale KL penalties to avoid instability.
    * **KL regularization:** Monitor divergence from $\pi_{\text{ref}}$ to prevent drift.
    * **Batching:** Aggregate rewards over multiple steps when using multi-step guidance._
    * **Reward flexibility:** Can combine human labels, LLM feedback, or programmatic heuristics.
    * **Monitoring:** Track reward improvements and KL drift separately for stable training.

    ---

    #### 5.4. Key Takeaways

    * GRPO generalizes DRPO by incorporating **structured, guided rewards**.
    * KL regularization ensures alignment with base model while learning complex behaviors.
    * Efficient for multi-step reasoning, safety-critical tasks, and long-horizon instructions.
    * Provides a flexible, expressive framework for offline or online RLHF-style optimization.

    ---

### 6. Implementation Example (Pseudocode)

```python
log_probs = model.logprobs(batch_inputs, batch_outputs)
log_probs_ref = ref_model.logprobs(batch_inputs, batch_outputs)

# Compute KL divergence
kl = (log_probs_ref - log_probs).mean(dim=-1)

# GRPO loss
loss = -(guided_rewards - beta * kl).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 7. Example Use Cases

1. **Multi-step reasoning tasks:** math, coding, planning
2. **Safety-critical tasks:** medical, legal, regulated instructions
3. **Long-horizon tasks:** document summarization, multi-turn dialogue
4. **Complex instruction-following:** multi-step workflows, simulations

---

### 8. Comparison with PPO and DRPO

| Aspect                 | PPO               | DRPO                  | GRPO                                |
| ---------------------- | ----------------- | --------------------- | ----------------------------------- |
| **Reward Model**       | Learned scalar RM | ❌ Not needed          | Optional or explicit guided signals |
| **RL Loop**            | Yes (online)      | ❌ Offline             | Offline or online                   |
| **KL Penalty**         | Manual, global    | ❌ Implicit            | ✅ Adaptive, local                   |
| **Training Stability** | Sensitive         | More stable           | Stable with guidance                |
| **Data Efficiency**    | Moderate          | Uses preference pairs | Efficient with structured guidance  |
| **Complexity**         | High              | Moderate              | Higher due to guided reward design  |

---

### 9. Limitations and Challenges

* High-quality guided rewards required — noisy or biased signals can harm performance.
* More computationally expensive than DRPO.
* Designing stepwise or structured rewards requires careful planning.
* Less standardized tooling compared to PPO or DRPO.

---

### 10. Summary Table

| Component              | Role                                             | Example                  |
| ---------------------- | ------------------------------------------------ | ------------------------ |
| **Policy Model (LLM)** | Learns from guided rewards                       | `GPT-3`, `Llama-2`       |
| **Reference Model**    | Provides baseline probabilities                  | SFT model                |
| **GRPO Objective**     | Maximizes guided reward + KL penalty             | Loss formula above       |
| **β Parameter**        | Controls proximity to reference                  | Tuning hyperparameter    |
| **Goal**               | Align behavior efficiently with complex feedback | Stable, guided alignment |
