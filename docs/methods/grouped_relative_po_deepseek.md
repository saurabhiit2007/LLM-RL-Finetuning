# 🧮 Grouped Relative Policy Optimization (GRPO) — Reinforcement Learning for Efficient LLM Alignment

### 1. Overview

**Grouped Relative Policy Optimization (GRPO)** is a reinforcement learning algorithm introduced in the **DeepSeek** series (DeepSeekMath, DeepSeek-R1) to fine-tune **Large Language Models (LLMs)** efficiently on reasoning-intensive tasks.  
Unlike traditional PPO, which requires a **critic (value network)**, GRPO eliminates the critic and computes **relative advantages within groups** of sampled outputs.  
This approach reduces computational cost and stabilizes training, making it well-suited for large-scale language model alignment.

---

### 2. The Big Picture: From PPO to GRPO

Traditional **RLHF** pipelines (using PPO) require a policy model, a reward model, and a value function. GRPO simplifies this process by using **group-wise relative advantages** instead of an explicit value estimator.

| Stage | PPO-Based RLHF | GRPO-Based Alignment |
|-------|----------------|----------------------|
| 1️⃣ SFT | Train base LLM on human demonstrations | ✅ Same |
| 2️⃣ RM  | Train reward or value model | ❌ Removed (uses reward function directly) |
| 3️⃣ RL  | Fine-tune using PPO updates | ✅ Fine-tune using group-based GRPO objective |

This design significantly reduces training instability and memory usage while preserving the benefits of policy-gradient fine-tuning.

---

### 3. Intuitive Understanding

For each prompt, GRPO samples **G** candidate responses from the old policy, evaluates each response using a reward function, and compares them within the group.  
The model then updates its policy to favor responses that outperform others in the same group — a *relative* rather than *absolute* improvement process.

Intuitively:

* PPO optimizes each response using absolute advantages from a critic.
* GRPO optimizes by ranking multiple sampled responses and pushing the policy toward higher-ranked ones.

This allows GRPO to focus on *comparative improvement* while maintaining diversity and avoiding overfitting to noisy rewards.

---

### 4. Training Data and Setup

Each GRPO training example includes:

* **Prompt**: \( q \)
* **Group of outputs**: \( \{o_1, o_2, \dots, o_G\} \) sampled from the old policy \( \pi_{\text{old}} \)
* **Reward values**: \( r_i = r(q, o_i) \) from a scoring or reward function

The policy model \( \pi_\theta \) is optimized to assign higher probabilities to outputs with higher *relative* rewards, regularized by a KL penalty with respect to a frozen **reference policy** \( \pi_{\text{ref}} \).

---

### 5. GRPO Formulation

!!! example "📘 Mathematical Formulation"

    #### 5.1. Objective Function

    GRPO generalizes the PPO objective using group-wise normalization:

    $$
    J_{\mathrm{GRPO}}(\theta)
    = \mathbb{E}_{q, \{o_i\}} \left[
    \frac{1}{G} \sum_{i=1}^G
    \min \Big(
      \frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)} A_i,\,
      \text{clip}\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i
    \Big)
    - \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta \| \pi_{\text{ref}}\big)
    \right]
    $$

    where:

    * \( \pi_{\text{old}} \): policy before update  
    * \( A_i \): normalized advantage within the group  
    * \( \epsilon \): PPO clipping coefficient  
    * \( \beta \): KL regularization coefficient  
    * \( \pi_{\text{ref}} \): frozen reference model

    ---
    #### 5.2. Grouped Advantage

    The *relative* advantage \(A_i\) is computed within each group:

    $$
    A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}
    $$

    where \(r_i\) is the reward for output \(o_i\).  
    This ensures that updates depend on *relative* performance rather than absolute reward magnitude.

    ---
    #### 5.3. KL Regularization

    The KL term ensures that the updated policy remains close to the reference model:

    $$
    D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}}) =
    \frac{\pi_{\text{ref}}(o_i|q)}{\pi_\theta(o_i|q)} -
    \log \frac{\pi_{\text{ref}}(o_i|q)}{\pi_\theta(o_i|q)} - 1
    $$

    ---
    #### 5.4. Intuition

    * **Group-normalized advantages** remove the need for a critic.  
    * **KL regularization** replaces the explicit PPO penalty term.  
    * **Clipping** prevents large, unstable policy updates.  
    * **Efficiency**: GRPO avoids computing value baselines, making it highly scalable for LLMs.

    ---
    #### 5.5. Implementation Details

    * **Group size (G)** — Typically 8–16 samples per prompt.  
    * **β (beta)** — 0.001–0.01 to control KL regularization.  
    * **ε (epsilon)** — Clipping coefficient, often 0.1–0.2.  
    * **Reference policy** — Frozen SFT model to anchor learning.  
    * **Reward function** — Task-specific (e.g., correctness, coherence, reasoning completeness).  
    * **Advantage normalization** — Essential for stable updates; normalize per group.

---

### 6. Implementation Example (Pseudocode)

```python
for prompt in dataset:
    outputs = [policy_old.generate(prompt) for _ in range(G)]
    rewards = [reward_fn(prompt, o) for o in outputs]
    mean_r, std_r = np.mean(rewards), np.std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]

    logp_old = [policy_old.logprob(prompt, o) for o in outputs]
    logp_new = [policy.logprob(prompt, o) for o in outputs]

    ratios = [torch.exp(lp_new - lp_old) for lp_new, lp_old in zip(logp_new, logp_old)]
    surr = [torch.min(r * A, torch.clamp(r, 1-ε, 1+ε) * A)
            for r, A in zip(ratios, advantages)]

    loss_policy = -torch.mean(torch.stack(surr))
    kl_loss = β * compute_KL(policy, ref_policy, prompt, outputs)
    loss = loss_policy + kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```


### 7. Why GRPO Instead of PPO?

| Aspect                 | PPO                                    | GRPO                                   |
|------------------------|----------------------------------------|----------------------------------------|
| **Critic / Value Net** | Required                               | ❌ Removed                             |
| **Advantage Computation** | From value estimates (GAE)           | Group-normalized rewards               |
| **KL Regularization**  | Explicit or adaptive penalty           | Included via reference policy          |
| **Training Stability** | Sensitive to critic/value bias         | More stable and memory-efficient       |
| **Data Efficiency**    | Uses single rollout per update         | Leverages multiple outputs per prompt  |
| **Compute Cost**       | High (policy + value models)           | Low (policy-only)                      |
| **Suitability**        | General RL tasks                       | LLM fine-tuning with verifiable rewards|

---

### 8. Limitations and Challenges

#### 📉 1. Group Reward Homogeneity
If all responses in a group have similar rewards, normalized advantages vanish, yielding weak gradients.

#### 🔄 2. Reward Function Quality
GRPO still relies on reward signal design; noisy or biased rewards can misguide optimization.

#### ⚖️ 3. KL Coefficient Sensitivity
If β is too small, the model may drift from the base; too large, and updates stall.

#### 💡 4. Group Size Tradeoff
Larger groups improve ranking precision but increase compute cost.

#### 🎭 5. Limited Exploration
As with PPO, GRPO may struggle to explore novel or diverse outputs if rewards are narrow.

---

### 9. Summary Table

| Component              | Role                                        | Example                                |
|------------------------|---------------------------------------------|----------------------------------------|
| **Policy Model (LLM)** | Learns improved policy via group comparison | DeepSeek-R1, DeepSeekMath              |
| **Reference Model**    | Provides KL regularization baseline         | Frozen SFT model                       |
| **Reward Function**    | Scores responses                            | Correctness, Coherence, Style, etc.    |
| **Group Size (G)**     | Defines sampling granularity                | 8–16 outputs                           |
| **Advantage \(A_i\)**  | Relative performance metric                 | Normalized per group                   |
| **Objective**          | PPO-like surrogate + KL penalty             | Eq. (5.1) above                        |
| **Goal**               | Efficient RL fine-tuning for LLMs           | Stable, critic-free optimization       |

---
