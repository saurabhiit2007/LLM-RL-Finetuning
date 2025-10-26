# 🧠 Decoupled Reward Policy Optimization (DRPO)

### 1. Overview
Decoupled Reward Policy Optimization (DRPO) is a reinforcement learning algorithm designed to enhance the reasoning capabilities of Large Language Models (LLMs).  
It addresses limitations in traditional reinforcement learning methods by **decoupling the length-based learning signal** of correct rollouts from incorrect ones, preventing penalization of correct but lengthy reasoning processes.  
This is particularly beneficial for tasks requiring complex reasoning and long-form outputs.

---

### 2. DRPO Pipeline
The DRPO framework follows the typical RLHF pipeline with key modifications:

1. **Supervised Fine-Tuning (SFT)**  
    - Train a base LLM on high-quality human demonstration data (prompt–response pairs).

2. **Reward Model (RM) Training**  
    - Train a model to assign scalar rewards to outputs based on human preferences.

3. **Reinforcement Learning (DRPO)**  
    - Fine-tune the policy (SFT model) to maximize predicted rewards from the RM, with decoupled reasoning length consideration.

> 💡 **Intuition:** DRPO ensures that correct but lengthy responses are appropriately rewarded, allowing the model to generate more complex and accurate outputs.

---

### 3. Why DRPO Instead of Traditional Methods?
- Traditional RL methods can **penalize correct but long reasoning processes** due to group-relative advantage functions.  
- DRPO **decouples the length-based learning signal** of correct rollouts from incorrect ones, ensuring valid reasoning is reinforced.  

---

### 4. DRPO Key Concepts

#### 4.1 Components

| Component             | Description                                           |
|-----------------------|-------------------------------------------------------|
| **Policy Model (πₜ)** | The trainable LLM generating responses.             |
| **Reward Model (Rₘ)** | Evaluates outputs, providing scalar rewards.         |
| **Reference Model (πₜₒₗᵈ)** | Snapshot of policy before update, for stability.  |
| **Value Function (Vₜ)** | Estimates expected reward for a given prompt.      |
| **Advantage (Aₜ)**     | Measures how much better an action is than expected: `Aₜ = Rₘ - Vₜ(sₜ)` |

#### 4.2 Intuition
- Generates outputs → reward model evaluates → advantage guides update.  
- Decoupled objective ensures correct but lengthy reasoning is rewarded.

---

### 5. DRPO Objective Function

!!! example "📘 DRPO Mathematical Formulation" 

    #### 5.1 Probability Ratio
    Measures how much the new policy’s likelihood of an action changes compared to the old policy:
      
    $$
    rₜ(πₜ) = \frac{πₜ(aₜ | sₜ)}{πₜₒₗᵈ(aₜ | sₜ)}
    $$

    #### 5.2 Decoupled Surrogate Loss
    Ensures stable updates while rewarding correct reasoning lengths:

    $$
    L^{DRPO}(πₜ) = \mathbb{E}[ \min(rₜ(πₜ) Aₜ, \text{clip}(rₜ(πₜ), 1 - ε, 1 + ε) Aₜ)]
    $$

    Where:

    - $A_t$: Advantage function — how much better an action is than expected.  
    - $\epsilon$: Clipping threshold (typically 0.1–0.3).  

---

### 6. Value Function, Advantage, and Reward Computation

!!! example "📗 Supporting Mathematical Components"
    #### 6.1 Cumulative Reward (Return)
    $$
    Rₜ = \sum_{k=0}^{∞} γ^k rₜ₊ₖ
    $$

    - \(rₜ\): reward received at time \(t\) (from reward model).  
    - \(γ\): discount factor (typically 0.95–0.99).  

    In RLHF, responses are often short, so sequence-level rewards are used.

    #### 6.2 Value Function
    $$
    Vₜ(sₜ) ≈ \mathbb{E}[Rₜ | sₜ]
    $$

    Value loss:

    $$
    L^{value}(πₜ) = \frac{1}{2} (Vₜ(sₜ) - Rₜ)^2
    $$

    #### 6.3 Advantage Function
    $$
    Aₜ = Rₜ - Vₜ(sₜ)
    $$

    Optionally, **Generalized Advantage Estimation (GAE)**:

    $$
    Aₜ = \sum_{l=0}^{∞} (γλ)^l δₜ₊ₗ
    $$

    Where:

    - \(δₜ = rₜ + γ Vₜ(sₜ₊₁) − Vₜ(sₜ)\)  
    - \(λ\) = GAE smoothing factor (0.9–0.97)

    #### 6.4 Entropy Bonus
    $$
    L^{entropy}(πₜ) = - \sum_a πₜ(a|sₜ) \log πₜ(a|sₜ)
    $$

    Promotes exploration and diversity in responses.

    #### 6.5 Combined DRPO Loss
    $$
    L_{total}(πₜ) = -L^{DRPO}(πₜ) + c₁ \cdot L^{value}(πₜ) - c₂ \cdot H[πₜ]
    $$

    Where $H[πₜ]$ is the entropy term; $c₁, c₂$ control loss weighting.

---

### 7. Iterative DRPO Update
1. Generate response with policy model.  
2. Compute reward using reward model.  
3. Compute log probabilities (old vs new policy).  
4. Estimate value using value head.  
5. Compute advantage.  
6. Update policy using **clipped surrogate loss**.  
7. Update value function.  
8. Apply entropy bonus.  
9. Update reference model.

> ✅ DRPO updates only when new behavior is better and within a controlled region, ensuring valid reasoning lengths are rewarded.

---

### 8. Implementation Example (Pseudocode)

```python

for prompt in prompts:
    response = policy_model.generate(prompt)
    reward = reward_model(prompt, response)
    
    logprobs_old = ref_model.logprobs(prompt, response)
    logprobs_new = policy_model.logprobs(prompt, response)
    
    value = value_head(prompt)
    
    advantage = reward - value  # sequence-level
    
    ratio = torch.exp(logprobs_new - logprobs_old)
    
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
    policy_loss = -torch.mean(torch.min(ratio * advantage, clipped_ratio * advantage))
    
    value_loss = 0.5 * (value - reward) ** 2
    
    entropy = -torch.sum(torch.exp(logprobs_new) * logprobs_new)
    entropy_coeff = 0.01
    
    total_loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    ref_model.load_state_dict(policy_model.state_dict())

```

### 9. Limitations and Challenges

1. **KL Divergence Sensitivity**: Requires KL penalty to prevent divergence.  
2. **High Training Cost**: Multiple models increase compute.  
3. **Reward Hacking**: LLM may over-optimize for reward model.  
4. **Sparse/Noisy Rewards**: One reward per sequence can hinder credit assignment.  
5. **Credit Assignment**: Per-token updates with sequence-level rewards may be ambiguous.  
6. **Exploration vs Alignment**: Balancing safe exploration is challenging.  
7. **Implementation Complexity**: Multiple models and hyperparameter tuning required.

---

### 10. Summary

| Component             | Role                                  | Example |
|-----------------------|--------------------------------------|---------|
| **Policy Model (LLM)** | Generates responses                   | GPT-3, LLaMA-2 |
| **Reward Model**       | Scores outputs based on preferences  | Fine-tuned classifier |
| **DRPO Algorithm**     | Updates policy with decoupled reward | Training loop |
| **KL Penalty**         | Prevents over-deviation              | Regularization |
| **Goal**               | Align LLM behavior with human intent | Helpful, safe, accurate reasoning |
