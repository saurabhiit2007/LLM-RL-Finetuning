
# 🧠 PPO and Reward Models in LLM Training

### 1. Overview
Proximal Policy Optimization (PPO) is a reinforcement learning algorithm widely used in **fine-tuning Large Language Models (LLMs)** under the Reinforcement Learning from Human Feedback (RLHF) framework. It helps bridge the gap between **human preferences** and **LLM outputs** by optimizing the model’s responses to align with what humans find helpful, safe, or relevant.

---

### 2. RLHF Pipeline
RLHF typically consists of three stages:

1. **Supervised Fine-Tuning (SFT)**
    - Train a base LLM on high-quality human demonstration data (prompt–response pairs).

2. **Reward Model (RM) Training**
    - Train a model to assign **scalar rewards** to outputs based on human preferences.

3. **Reinforcement Learning (PPO)**
    - Fine-tune the policy (SFT model) to maximize predicted rewards from the RM.

> 💡 **Intuition:** PPO teaches the LLM to generate preferred responses indirectly, using the reward model as scalable feedback.

---

### 3. Why PPO instead of Direct Human Feedback?
Direct human labeling for all outputs is **impractical and noisy**. PPO helps by:

- **Scaling feedback:** Reward models generalize human preferences to unseen outputs.
- **Credit assignment:** Uses value function and advantage to propagate sequence-level rewards to tokens.
- **Stable updates:** Ensures the model does not deviate too far from its original behavior.

---

### 4. PPO Key Concepts

#### 4.1 Components

| Component | Description |
|-----------|-------------|
| **Policy Model (π_θ)** | The trainable LLM generating responses. |
| **Reward Model (R_ϕ)** | Evaluates outputs, providing scalar rewards. |
| **Reference Model (π_θ_old)** | Snapshot of policy before update, used for stable updates. |
| **Value Function (V_θ)** | Estimates expected reward for a given prompt. |
| **Advantage (A_t)** | Measures how much better an action is than expected: `A = R - V_θ(s)` |

#### 4.2 Intuition
PPO adjusts the LLM to improve rewards **without drastic changes**:

- Generates outputs → reward model evaluates → advantage guides update.
- The **clipped objective** prevents extreme updates and maintains stability.

---


### 5. PPO Objective Function

The **Proximal Policy Optimization (PPO)** algorithm optimizes a policy model \(π_θ\) while constraining how much it can diverge from a reference (old) policy \(π_{θ_{old}}\).

!!! example "📘 PPO Mathematical Formulation"
    #### 5.1. Probability Ratio

    $$
    r_t(\theta) = \frac{π_θ(a_t | s_t)}{π_{θ_{old}}(a_t | s_t)}
    $$

    The ratio measures how much the new policy’s likelihood of an action changes compared to the old policy.
    This ratio quantifies the magnitude and direction of policy change for each sampled token or action.
    ---

    ### 5.2. Clipped PPO Objective

    The clipped surrogate loss ensures stable updates by penalizing large deviations in \(r_t(\theta)\):

    $$
    L^{PPO}(\theta) = \mathbb{E}_t \left[\min\left(r_t(\theta) A_t,\ \text{clip}(r_t(\theta),\ 1-\epsilon,\ 1+\epsilon)\ A_t\right)\right]
    $$

    Where:

    - \(A_t\): **Advantage function** — how much better an action is than expected.  
    - \(ε\): **Clipping threshold** (typically 0.1–0.3).  
    - The `min` operation limits large, destabilizing updates.

> For computing \(A_t\) (advantage), \(V_θ(s_t)\) (value), and rewards, refer to the next section on *Value Function and Reward Computation.*

---

### 6. Value Function, Advantage, and Reward Computation

The PPO algorithm relies on several auxiliary components — **value function**, **advantage estimation**, and **entropy regularization** — that ensure stable and meaningful policy updates.

!!! example "📗 Supporting Mathematical Components"

    #### 6.1. Cumulative Reward (Return)

    The **cumulative reward** (or *return*) represents the total discounted reward starting from time \(t\):

    $$
    R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
    $$

    - \(r_t\): reward received at time \(t\) (from the reward model in RLHF).  
    - \(\gamma\): discount factor (typically 0.95–0.99).  

    In RLHF, this is often simplified since responses are short (e.g., one reward per sequence).

    ??? info "Reward Simplification in RLHF"
        In **Reinforcement Learning from Human Feedback (RLHF)** — especially in **language model fine-tuning** — the setup is simplified because responses are short and discrete.

        - A **prompt** acts as the state \( s \).  
        - The **model's response** (a sequence of tokens) is treated as the action \( a \).  
        - A **reward model (RM)** assigns **a single scalar reward** \( r(s, a) \) for the *entire sequence*, not per token.
    
        Therefore:
        $$
        R = r(s, a)
        $$
    
        The **advantage** and **value function** are computed at the sequence level rather than stepwise.  
        This eliminates the need to sum discounted rewards across timesteps — simplifying PPO training in RLHF.  
        The loss functions remain structurally similar but are applied to **sequence-level rewards**.

    ---

    #### 6.2. Value Function

    The **value function** estimates the expected return given a state (or prompt context):

    $$
    V_\theta(s_t) \approx \mathbb{E}[R_t \mid s_t]
    $$

    The **value loss** penalizes inaccurate predictions:

    $$
    L^{value}(\theta) = \frac{1}{2} \left(V_θ(s_t) - R_t\right)^2
    $$

    This helps the model learn accurate value estimates for better advantage computation.

    ??? info "Value Function in Practice"
        In practice, the **value function** is implemented as a **learned neural network head** attached to the policy model.  
        During training:

        1. The environment (or reward model, in RLHF) provides rewards \( r_t \) for each step or sequence.  
        2. The **cumulative discounted reward** \( R_t = \sum_k \gamma^k r_{t+k} \) is computed for each state \( s_t \).  
        3. The network learns to predict \( V_θ(s_t) \) such that it **matches the observed return** \( R_t \).

        There are two common approaches:

        - **Monte Carlo estimate:** directly use full episode returns \( R_t \) (common in RLHF since responses are short).  
        - **Bootstrapped estimate:** use \( r_t + \gamma V_θ(s_{t+1}) \) to reduce variance (used in continuous RL environments).

        Over time, the model minimizes:
        $$
        L^{value}(\theta) = \frac{1}{2} (V_θ(s_t) - R_t)^2
        $$
        making \( V_θ(s_t) \) a reliable **baseline** for computing the advantage:
        $$
        A_t = R_t - V_θ(s_t)
        $$

    ---

    #### 6.3. Advantage Function

    The **advantage** quantifies how much better an action \(a_t\) was compared to the expected baseline:
   
    $$
    A_t = R_t - V_θ(s_t)
    $$
   
    In practice, PPO often uses **Generalized Advantage Estimation (GAE)** for smoother and lower-variance estimates:
   
    $$
    A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
    $$
   
    where  
    \(\delta_t = r_t + \gamma V_θ(s_{t+1}) - V_θ(s_t)\),  
    and \(\lambda\) is the *GAE smoothing factor* (typically 0.9–0.97).
   
    ??? info "Advantage in Practice for LLMs"
        In **LLM fine-tuning with PPO**, the **advantage** is typically computed at the **sequence level** rather than per-token, since the reward model provides a **single scalar reward** for the entire generated response.

        #### 🧩 Practical Computation Steps
        1. For each prompt \(s\), the model generates a sequence \(a = (a_1, a_2, ..., a_T)\).  
        2. The **reward model** provides a scalar reward \(r(s, a)\) for the whole sequence.  
        3. The **value head** of the policy predicts \(V_θ(s)\), estimating the expected reward before generation.  
        4. The **advantage** is then computed as:
           $$
           A = r(s, a) - V_θ(s)
           $$
           representing how much better or worse the actual outcome was compared to the model’s expectation.
   
        #### 🧮 When Token-Level Advantages Are Used
        - Some PPO implementations for LLMs compute **token-level advantages** to better attribute credit across the generated sequence.  
        - This is achieved by assigning the same scalar reward to all tokens in a sequence and using GAE to smooth the signal:
          $$
          A_t = \text{GAE}(r_t, V_θ(s_t))
          $$
        - This provides more stable gradients and allows finer control during backpropagation.
   
        #### ⚖️ Summary
        - **Sequence-level PPO (common in RLHF):**  
          \(A = r(s, a) - V_θ(s)\)  
          → simpler and effective when rewards are sparse (one per output).
        - **Token-level PPO (advanced setups):**  
          Uses GAE to propagate reward information across tokens, reducing variance in updates.
   
        Overall, the advantage serves as the **direction and strength** of the policy gradient update — guiding PPO to reinforce actions that outperform the model’s baseline expectations.
        
    #### 6.4. Entropy Bonus (Exploration Term)

    The **entropy loss** encourages the policy to explore rather than prematurely converge:

    $$
    L^{entropy}(\theta) = - \sum_a π_θ(a|s_t) \log π_θ(a|s_t)
    $$

    Higher entropy = more exploration and diversity in generated responses.

    ---

    #### 6.5. Combined PPO Loss

    The full training objective combines all three components — PPO loss, value loss, and entropy term:

    $$
    L_{total}(\theta) = -L^{PPO}(\theta) + c_1 \cdot L^{value}(\theta) - c_2 \cdot H[π_θ]
    $$

    Where:

    - \(H[π_θ]\): entropy term promoting exploration.  
    - \(c_1, c_2\): coefficients controlling relative weighting of the losses.  

    This total loss balances **policy improvement**, **value estimation accuracy**, and **exploration**.

### 7. Iterative PPO Update
1. Generate response with policy model.
2. Compute reward using reward model.
3. Compute log probabilities (old vs new policy).
4. Estimate value using value head.
5. Compute advantage.
6. Update policy using **clipped surrogate loss**.
7. Update value function.
8. Apply entropy bonus.
9. Update reference model for next iteration.

> ✅ **Intuition:** PPO updates only when new behavior is better and within a controlled region.

---

### 8. Pseudo-code example of PPO in Practice

```python

for prompt in prompts:
    # 1. Generate response
    response = policy_model.generate(prompt)

    # 2. Compute reward from reward model (sequence-level reward)
    reward = reward_model(prompt, response)

    # 3. Compute log probabilities from old and new policies
    logprobs_old = ref_model.logprobs(prompt, response)
    logprobs_new = policy_model.logprobs(prompt, response)

    # 4. Compute value estimate from value head
    value = value_head(prompt)  # V_theta(s)

    # 5. Compute advantage
    advantage = reward - value  # sequence-level advantage
    # optionally: use GAE for token-level advantages

    # 6. Compute PPO ratio and clipped surrogate loss
    ratio = torch.exp(logprobs_new - logprobs_old)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.mean(torch.min(ratio * advantage, clipped_ratio * advantage))

    # 7. Compute value loss
    value_loss = 0.5 * (value - reward) ** 2

    # 8. Compute entropy bonus for exploration
    entropy = -torch.sum(torch.exp(logprobs_new) * logprobs_new)
    entropy_coeff = 0.01  # example weight

    # 9. Combine losses
    total_loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy

    # 10. Backpropagate and update model
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 11. Update reference model for next iteration
    ref_model.load_state_dict(policy_model.state_dict())

```


### 9. Limitations and Challenges of PPO in LLM Training

#### 🧩 1. KL Divergence Sensitivity
PPO adds a **KL penalty** to prevent the model from drifting too far:
$$
L = L^{PPO} - \beta D_{KL}(\pi_{\theta} || \pi_{\text{ref}})
$$

    - **Too small β:** model diverges.
    - **Too large β:** slow learning.
    - Adaptive KL control helps adjust automatically.

---

#### ⏳ 2. High Training Cost
- Multiple models (policy, reference, reward, value) increase compute.
- Fine-tuning large LLMs can require **thousands of GPU-hours**.

---

#### ⚠️ 3. Reward Hacking
- LLM may over-optimize for the reward model instead of true human preference.
- Can result in overly polite, verbose, or misleading responses.

---

#### 🧮 4. Sparse or Noisy Rewards
- **Sparse:** One reward per sequence makes credit assignment harder.
- **Noisy:** Subjective or inconsistent human preferences can lead to unstable updates.
> 💡 Sparse/noisy rewards increase variance and slow learning.

---

#### 🔁 5. Credit Assignment
- Per-token updates but per-sequence rewards create ambiguity about which tokens contributed most.

---

#### ⚖️ 6. Exploration vs Alignment
- Encouraging exploration may generate unsafe outputs.
- Balancing diversity and alignment is challenging.

---

#### 🔍 7. Implementation Complexity
- Multiple models and careful hyperparameter tuning required.
- Can be unstable if any component is suboptimal.


### 10. Summary

| Component | Role | Example |
|------------|------|----------|
| **Policy Model (LLM)** | Generates responses to prompts | `GPT-3`, `Llama-2` |
| **Reward Model** | Scores how much humans like the output | Fine-tuned classifier |
| **PPO Algorithm** | Updates the policy using rewards | Training loop |
| **KL Penalty** | Prevents over-deviation from base model | Regularization |
| **Goal** | Align LLM behavior with human intent | Helpful, safe, and truthful answers |

---
