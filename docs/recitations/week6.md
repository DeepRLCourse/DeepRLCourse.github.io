# Week 6: Multi-Armed Bandits

## 1. Definition of the Problem and Non-Associativity Property

The **multi-armed bandit (MAB) problem** is a fundamental **reinforcement learning** setting that models the process of **decision-making under uncertainty**. It represents a scenario where an agent repeatedly selects from a set of possible actions (called "arms"), each yielding an unknown and potentially different reward.

### **Formal Definition**
- Let **\( k \)** be the number of available actions (or "arms").
- Each action \( a \) is associated with an **unknown expected reward** \( q_*(a) \), defined as:

  $$
  q_*(a) = \mathbb{E}[R | A = a]
  $$

  where \( R \) is the stochastic reward received when selecting action \( A \).

- The goal is to **maximize cumulative reward** over a sequence of time steps \( T \), defined as:

  $$
  G = \sum_{t=1}^{T} R_t.
  $$

- However, the agent **does not know** \( q_*(a) \) and must **estimate it over time** while balancing **exploration** (gathering new information) and **exploitation** (choosing the best-known action).

### **Non-Associativity Property**
- Unlike **full reinforcement learning problems**, **multi-armed bandits are non-associative**.
- This means that **the best action does not depend on the state**; the optimal action is the same for all time steps.
- Formally, we do **not** consider state transitions: The bandit setting **only involves selecting actions** and receiving rewards, without any long-term effects from past decisions.

---

## 2. Action-Value Methods and Types

To solve the MAB problem, we need a way to estimate action values. The **action-value function** \( Q_t(a) \) estimates the expected reward of choosing action \( a \) at time step \( t \):

$$
Q_t(a) \approx q_*(a).
$$

We define **sample-average estimation** of \( Q_t(a) \) as:

$$
Q_t(a) = \frac{1}{N_t(a)} \sum_{i=1}^{N_t(a)} R_i
$$

where:
- \( N_t(a) \) is the number of times action \( a \) has been selected.
- \( R_i \) is the reward received when selecting \( a \).

### **Incremental Update Rule for Efficient Computation**
Instead of storing all past rewards, we can update \( Q_t(a) \) incrementally:

$$
Q_{t+1}(a) = Q_t(a) + \frac{1}{N_t(a)} (R_t - Q_t(a)).
$$

### **Constant Step-Size Update (For Nonstationary Problems)**
When dealing with **changing reward distributions**, we use a **constant step-size** \( \alpha \):

$$
Q_{t+1}(a) = Q_t(a) + \alpha (R_t - Q_t(a)).
$$

where \( \alpha \) determines **how much weight** is given to recent rewards.

- If \( \alpha = \frac{1}{N_t(a)} \), this becomes **sample-average estimation**.
- If \( \alpha \) is **constant**, this results in **exponentially weighted averaging**, useful for **nonstationary problems**.

---

## 3. Exploration-Exploitation Dilemma and Uncertainty

A **key challenge** in MAB problems is balancing:

- **Exploration**: Selecting **less-tried actions** to improve our estimates of \( q_*(a) \).
- **Exploitation**: Selecting **the best-known action** to maximize immediate reward.

The **uncertainty** in action-value estimates drives the need for exploration:

$$
P\left( | Q_t(a) - q_*(a) | \geq \sqrt{\frac{\ln t}{2N_t(a)}} \right) \leq \frac{1}{t^2}.
$$

- If an action is **selected many times**, \( N_t(a) \) increases and uncertainty **shrinks**.
- If an action is **rarely selected**, uncertainty remains **high**.

An optimal strategy must explore **enough** to **reduce uncertainty** but not **waste too much time** on bad actions.

---

## 4. Exploration in Bandits

### **4.1. \(\epsilon\)-Greedy Exploration**
One of the simplest strategies for exploration is **\(\epsilon\)-greedy**, where:
- With probability **\(1 - \epsilon\)**, we **exploit** by selecting the **best-known action**.
- With probability **\(\epsilon\)**, we **explore** by selecting a **random action**.

$$
A_t =
\begin{cases}
\arg\max_a Q_t(a), & \text{with probability } 1-\epsilon \\
\text{random } a, & \text{with probability } \epsilon.
\end{cases}
$$

### **4.2. Optimistic Initial Values**
Another exploration method is **optimistic initialization**:

$$
Q_1(a) = 5, \quad \forall a.
$$

### **4.3. Upper Confidence Bound (UCB) Exploration**
**UCB** improves on \(\epsilon\)-greedy by considering **uncertainty** in action values.

#### **UCB Formula:**
$$
A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right].
$$

where:
- \( Q_t(a) \) is the estimated action value.
- \( \sqrt{\frac{\ln t}{N_t(a)}} \) is the **confidence bound**.
- \( c \) controls exploration.

---

## **5. Summary**
| **Method** | **Exploration Type** | **Pros** | **Cons** |
|------------|----------------|--------|--------|
| **\(\epsilon\)-greedy** | Random exploration | Simple, ensures some exploration | Wastes time on bad actions |
| **Optimistic Initial Values** | Encourages early exploration | Systematic, fast learning | Not good for nonstationary problems |
| **UCB** | Focuses on uncertain actions | Logarithmic regret, better than \(\epsilon\)-greedy | More complex, struggles with large state spaces |

---

