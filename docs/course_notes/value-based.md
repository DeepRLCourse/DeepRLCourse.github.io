---
comments: True
description: This page provides detailed notes on value-based methods in reinforcement learning, including Bellman equations, dynamic programming, Monte Carlo methods, and temporal difference learning.
---

# Week 2: Value-based Methods
## 1. Bellman Equations and Value Functions


### 1.1. State Value Function $V(s)$

#### **Definition:**
The **state value function** $V^\pi(s)$ measures the expected return when an agent starts in state $s$ and follows a policy $\pi$. It provides a scalar value for each state that reflects the desirability of that state under the given policy. Formally, it is defined as:

$$
V^\pi(s) = \mathbb{E} \left[ G_t \mid s_t = s \right]
$$

Where $G_t$ represents the return (total reward) from time step $t$ onwards:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
$$

- $R_t$ is the reward received at time step $t$.
- $\gamma$ is the discount factor ($0 \leq \gamma \leq 1$), controlling how much future rewards are valued compared to immediate rewards.

#### **Bellman Expectation Equation for $V^\pi(s)$:**
The Bellman Expectation Equation for the state value function expresses the value of a state $s$ in terms of the expected immediate reward and the discounted value of the next state. It is written as:

$$
V^\pi(s) = \mathbb{E} \left[ R_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t = s \right]
$$

Using the transition probabilities of the environment, this can be expanded as:

$$
V^\pi(s) = \sum_{s'} P(s'|s, \pi(s)) \left[ R(s, \pi(s), s') + \gamma V^\pi(s') \right]
$$

Where:
- $P(s'|s, \pi(s))$ is the probability of transitioning from state $s$ to state $s'$ when following action $\pi(s)$.
- $R(s, \pi(s), s')$ is the reward for transitioning from state $s$ to $s'$ under action $\pi(s)$.

This equation allows for the iterative computation of state values in a model-based setting.

#### Bellman Optimality Equation for $V^*(s)$:
The **Bellman Optimality Equation** for $V^*(s)$ expresses the optimal state value function. It is given by:

$$
V^*(s) = \max_a \mathbb{E} \left[ R_{t+1} + \gamma V^\*(S_{t+1}) \mid s_t = s, a_t = a \right]
$$

This shows that the optimal value at each state is the immediate reward plus the discounted maximum expected value from the next state, where the next action is chosen optimally.

---

### 1.2. Action Value Function $Q(s, a)$


#### **Definition:**
The **action value function** $Q^\pi(s, a)$ represents the expected return when an agent starts in state $s$, takes action $a$, and then follows policy $\pi$:

$$
Q^\pi(s, a) = \mathbb{E} \left[ G_t \mid s_t = s, a_t = a \right]
$$

Where $G_t$ is the return starting at time $t$.

#### Bellman Expectation Equation for $Q^\pi(s, a)$:
The Bellman Expectation Equation for the action value function is similar to the one for the state value function but includes both the action and the subsequent states and actions. It is given by:

$$
Q^\pi(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma Q^\pi(s_{t+1}, a_{t+1}) \mid s_t = s, a_t = a \right]
$$

Expanding this into a sum over possible next states, we get:

$$
Q^\pi(s, a) = \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]
$$

Where:
- $P(s'|s, a)$ is the transition probability from state $s$ to state $s'$ under action $a$.
- $\pi(a'|s')$ is the probability of taking action $a'$ in state $s'$ under policy $\pi$.


#### Bellman Optimality Equation for $Q^*(s, a)$:
The **Bellman Optimality Equation** for $Q^*(s, a)$ expresses the optimal action value function. It is given by:

$$
Q^*(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} Q^\*(s_{t+1}, a') \mid s_t = s, a_t = a \right]
$$

This shows that the optimal action value at each state-action pair is the immediate reward plus the discounted maximum expected value from the next state, where the next action is chosen optimally.

---
### 2. Dynamic Programming

Dynamic Programming (DP) is a powerful technique used to solve reinforcement learning problems where the environment is fully known (i.e., the model is available). DP algorithms compute the optimal policy and value functions by iteratively updating estimates based on a model of the environment. 

### 2.1. Value Iteration

##### Bellman Optimality Equation:
The Bellman Optimality Equation for the value function is:

$$
V_{k+1}(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V_k(s') \right]
$$

Where:
- $\max_a$ selects the action $a$ that maximizes the expected return from state $s$.

##### Value Iteration Algorithm:
1. **Initialize** the value function $V_0(s)$ arbitrarily.
2. **Repeat** until convergence:
    - For each state $s$, update the value function:

    $$
    V_{k+1}(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V_k(s') \right]
    $$

3. Once the value function converges, the optimal policy $\pi^*(s)$ can be derived by selecting the action that maximizes the expected return:

$$
\pi^*(s) = \arg\max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^*(s') \right]
$$

##### Convergence:
Value Iteration is guaranteed to converge to the optimal value function and policy. The number of iterations required depends on the problem's dynamics, but it typically converges faster than Policy Iteration in terms of the number of iterations, though it may require more computation per iteration.

---

### 2.2. Policy Evaluation

Policy Evaluation calculates the state value function $V^\pi(s)$ for a given policy $\pi$ by iteratively updating the value function using the Bellman Expectation Equation:

$$
V^\pi(s) = R(s, \pi(s)) + \gamma \sum_{s'} P(s'|s, \pi(s)) V^\pi(s')
$$

This process is repeated until $V^\pi(s)$ converges to a fixed point for all s.


---
### 2.3. Policy Improvement

Policy Improvement refines a policy $\pi$ by making it greedy with respect to the current value function:

$$
\pi'(s) = \arg\max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^\pi(s') \right]
$$

It is proven that the new policy’s value function is at least as good as the previous one:

$$
V^{\pi'}(s) \geq V^\pi(s), \quad \forall s.
$$

By repeating policy evaluation and improvement, the policy converges to the optimal one.

1. **Single-Step Improvement:**  
    - Modify the policy at **only** $t = 0$, keeping the rest unchanged.  
    - This new policy achieves a higher or equal value:  

    $$
    V^{\pi_{(k+1)}^{(1)}}(s) \geq V^{\pi_k}(s), \quad \forall s.
    $$

2. **Extending to Multiple Steps:**  
    - Modify the policy at **$t = 0$ and $t = 1$**, keeping the rest unchanged.  
    - Again, the value function improves: 

    $$
    V^{\pi_{(k+1)}^{(2)}}(s) \geq V^{\pi_{(k+1)}^{(1)}}(s) \geq V^{\pi_k}(s).
    $$


3. **Repeating for All Steps:**  
    - After applying this to all time steps, the final policy matches the fully improved one:  

    $$
    \pi_{(k+1)}^{(\infty)}(s) = \pi_{k+1}(s).
    $$
     
    - This ensures:  

    $$
    V^{\pi_{k+1}}(s) \geq V^{\pi_k}(s), \quad \forall s.
    $$

The value function **never decreases** with each update.  



---
### 2.4. Policy Iteration

Policy Iteration alternates between **policy evaluation** and **policy improvement** to compute the optimal policy.

1. **Initialize policy $\pi_0$** randomly.
2. **Policy Evaluation:** Compute the value function $V^{\pi_k}(s)$ for the current policy $\pi_k$ using the Bellman Expectation Equation.
3. **Policy Improvement:** Update the policy $\pi_{k+1}(s)$ by making it greedy with respect to the current value function:

$$ 
\pi_{k+1}(s) = \arg\max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^{\pi_k}(s') \right] 
$$

4. Repeat the above steps until the policy converges (i.e., $\pi_k = \pi_{k+1}$).

##### Convergence:
Each policy update ensures that the value function **does not decrease**.

Since there are only a **finite number of deterministic policies** in a finite Markov Decision Process (MDP), the sequence of improving policies must eventually reach an policy $\pi^*$, where further improvement do not change it.

The value function of the fixed point $\pi^*$ satisfy the **Bellman Optimality Equation**:

$$
V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^*(s') \right]
$$

This confirms that the final policy $\pi^*$ is optimal.

---
### 2.5. Comparison of Policy Iteration and Value Iteration

Policy Iteration and Value Iteration are two dynamic programming methods for finding the optimal policy in an MDP. Both rely on iterative updates but differ in **efficiency** and **computation**.

| Feature               | Policy Iteration               | Value Iteration                |
|-----------------------|--------------------------------|--------------------------------|
| **Update Method**     | Alternates between policy evaluation and improvement | Updates value function directly |
| **Computational Cost Per Iteration** | $O(\|S\|^3)$ (solving linear equations) | $O(\|S\| \cdot \|A\|)$ (maximization over actions) |
| **Number of Iterations** | Fewer iterations, but each is expensive | More iterations, but each is cheaper |
| **Best for**          | Small state spaces, deterministic transitions | Large state spaces, stochastic transitions |

**Policy Iteration** explicitly computes the value function for a given policy, requiring **solving a system of equations**. Each iteration is **computationally expensive** but results in a significant improvement, leading to **faster convergence** in terms of iterations.

**Value Iteration** avoids solving a system of equations by updating the value function **incrementally**. Each iteration is computationally **cheaper**, but because the value function is updated gradually, **more iterations** are needed for convergence.

Thus, **Policy Iteration takes fewer iterations but is computationally heavy per step, while Value Iteration takes more iterations but is computationally lighter per step**.


[Watch on YouTube](https://www.youtube.com/watch?v=sJIFUTITfBc)

---

## 3. Monte Carlo Methods


### 3.1. Planning vs. Learning in RL

Reinforcement learning can be approached in two ways: **planning** and **learning**. The main difference is that **planning relies on a model of the environment**, while **learning uses real-world interactions** to improve decision-making.

##### Planning (Model-Based RL)
- Uses a model to predict state transitions and rewards.
- The agent can simulate future actions without interacting with the environment.
- Examples: **Dynamic Programming (DP), Monte Carlo Tree Search (MCTS).**

##### Learning (Model-Free RL)
- No access to a model; the agent learns by interacting with the environment.
- The agent updates value estimates based on observed rewards.
- Examples: **Monte Carlo, Temporal Difference (TD), Q-Learning.**

Planning is efficient when a reliable model is available, but learning is necessary when the model is unknown or too complex to compute.

Monte Carlo methods fall under **Model-Free RL**, where the agent improves through experience. The following sections introduce how **Monte Carlo Sampling** is used to estimate value functions without needing a model.

---

### 3.2. Introduction to Monte Carlo 

Monte Carlo methods use **random sampling** to estimate numerical results, especially when direct computation is infeasible or the underlying distribution is unknown. These methods are widely applied in **physics, finance, optimization, and reinforcement learning**.

Monte Carlo estimates an expectation:

$$
I = \mathbb{E}[f(X)] = \int f(x) p(x) dx
$$

using **sample averaging**:

$$
\hat{I}_N = \frac{1}{N} \sum_{i=1}^{N} f(x_i),
$$

where $ x_i $ are **independent** samples drawn from $ p(x) $. The **Law of Large Numbers (LLN)** ensures that as $N \to \infty$:

$$
\hat{I}_N \to I.
$$

This guarantees **convergence**, but the **speed of convergence** depends on the variance of the samples.


Monte Carlo estimates become more accurate as $N$ increases, but **independent samples** are crucial for unbiased estimation.

By the **Central Limit Theorem (CLT)**, for large $N$, the Monte Carlo estimate follows a normal distribution:

$$
\hat{I}_N \approx \mathcal{N} \left(I, \frac{\sigma^2}{N} \right).
$$

This shows that the **variance decreases at a rate of** $O(1/N)$, meaning that as the number of independent samples increases, the estimate becomes more stable. However, this reduction is slow, requiring a large number of samples to achieve high precision.


---
#### Example: Estimating $\pi$

Monte Carlo methods can estimate **$\pi$** by randomly sampling points and analyzing their distribution relative to a known geometric shape.

##### Steps:
1. Generate $N$ random points $(x, y)$ where $x, y \sim U(-1,1)$, meaning they are uniformly sampled in the square $[-1,1] \times [-1,1]$.
2. Define an **indicator function** $I(x, y)$ that takes the value:

$$
I(x, y) =
\begin{cases}
1, & \text{if } x^2 + y^2 \leq 1 \quad \text{(inside the circle)} \\
0, & \text{otherwise}.
\end{cases}
$$

   Since each point is either inside or outside the circle, the variable $I(x, y)$ follows a **Bernoulli distribution** with probability $p = \frac{\pi}{4}$.

3. Compute the proportion of points inside the circle. The expectation of $I(x, y)$ gives:

$$
\mathbb{E}[I] = P(I = 1) = \frac{\pi}{4}.
$$

   By the **Law of Large Numbers (LLN)**, the sample mean of $I(x, y)$ over $N$ points converges to this expected value:

$$
\frac{\text{Points inside the circle}}{\text{Total points}} \approx \frac{\pi}{4}.
$$

<center> 
<img src="\assets\images\course_notes\value-based\pi-estimation.png"
    alt="pi estimation with monte carlo"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>

---

#### Example: Integration

Monte Carlo methods can also estimate definite integrals using random sampling. Given an integral:

$$
I = \int_a^b f(x) dx,
$$

we approximate it using Monte Carlo sampling:

$$
\hat{I}_N = \frac{b-a}{N} \sum_{i=1}^{N} f(x_i),
$$

where $x_i$ are sampled uniformly from $[a, b]$. By the **LLN**, as $N \to \infty$, the estimate $\hat{I}_N$ converges to the true integral.

<center> 
<img src="\assets\images\course_notes\value-based\integral.png"
    alt="integration with monte carlo"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>

[Source](https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/monte-carlo-methods-in-practice/monte-carlo-integration.html)


[Watch on YouTube](https://www.youtube.com/watch?v=7TqhmX92P6U&t=173s)

---
### 3.3. Monte Carlo Prediction

In reinforcement learning, an **episode** is a sequence of states, actions, and rewards that starts from an **initial state** and ends in a **terminal state**. Each episode represents a **complete trajectory** of the agent’s interaction with the environment.


The **return** for a time step $t$ in an episode is the **cumulative discounted reward**:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
$$

#### Estimating $V^\pi(s)$
Monte Carlo methods estimate the **state value function** $V^\pi(s)$ by **averaging the returns** observed after visiting state $s$ in multiple episodes.

The estimate of $V^\pi(s)$ is:

$$
V^\pi(s) = \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i
$$

where:
- $N(s)$ is the number of times state $s$ has been visited.
- $G_i$ is the return observed from the $i$-th visit to state $s$.

Since Monte Carlo methods rely entirely on **sampled episodes**, they do **not require knowledge of transition probabilities or rewards** so they learn directly from experience.

---

### 3.4. Monte Carlo Control

In **Monte Carlo Control**, the goal is to improve the policy $\pi$ by optimizing it based on the action-value function $Q^\pi(s, a)$. 
#### **Algorithm:**
1. **Generate Episodes**: Generate episodes by interacting with the environment under the current policy $\pi$.
2. **Compute Returns**: For each state-action pair $(s, a)$ in the episode, compute the return $G_t$ from that time step onward.
3. **Update Action-Value Function**: For each state-action pair, update the action-value function as:
   
$$ 
Q^\pi(s, a) = \frac{1}{N(s, a)} \sum_{i=1}^{N(s, a)} G_i $$

   Where $N(s, a)$ is the number of times the state-action pair $(s, a)$ has been visited.
4. **Policy Improvement**: After updating the action-value function, improve the policy by selecting the action that maximizes $Q^\pi(s, a)$ for each state $s$:

$$ 
\pi'(s) = \arg\max_a Q^\pi(s, a) 
$$

This method is used to optimize the policy iteratively, improving it by making the policy greedy with respect to the current action-value function.

---

### 3.5. First-Visit vs. Every-Visit Monte Carlo

There are two main variations of Monte Carlo methods for estimating value functions: **First-Visit Monte Carlo** and **Every-Visit Monte Carlo**.

#### First-Visit Monte Carlo:
In **First-Visit Monte Carlo**, the return for a state is only updated the first time it is visited in an episode. This approach helps avoid over-counting and ensures that the value estimate for each state is updated only once per episode.

##### Algorithm:
1. Initialize $N(s) = 0$ and $G(s) = 0$ for all states.
2. For each episode, visit each state $s$ for the first time, and when it is first visited, add the return $G_t$ to $G(s)$ and increment $N(s)$.
3. After the episode, update the value estimate for each state as:  

$$
V^\pi(s) = \frac{G(s)}{N(s)}
$$

#### Every-Visit Monte Carlo:
In **Every-Visit Monte Carlo**, the return for each state is updated every time it is visited in an episode. This approach uses all occurrences of a state to update its value function, which can sometimes lead to more stable estimates.

##### Algorithm:
1. Initialize $N(s) = 0$ and $G(s) = 0$ for all states.
2. For each episode, visit each state $s$, and every time it is visited, add the return $G_t$ to $G(s)$ and increment $N(s)$.
3. After the episode, update the value estimate for each state as:

$$
V^\pi(s) = \frac{G(s)}{N(s)}
$$

#### Comparison:
First-Visit Monte Carlo updates the value function only the first time a state is encountered in an episode, ensuring an **unbiased** estimate but using **fewer samples**, which can result in **higher variance** and **slower** learning. In contrast, Every-Visit Monte Carlo updates the value function on all occurrences of a state within an episode, **reducing variance** and improving **sample efficiency** by utilizing more data. Although it may introduce **bias**, it often converges faster, making it more practical in many applications.

---
### 3.6. Incremental Monte Carlo Policy  

In addition to First-Visit and Every-Visit Monte Carlo, an alternative approach is the **Incremental Monte Carlo Policy**, which updates the value function **incrementally after each visit** instead of computing an average over all episodes. This method is **more memory-efficient** and allows real-time updates without storing past returns.

Given the return $G_{i,t}$ observed for state $s$ at time $t$ in episode $i$, we update the value function as:

$$
V^\pi(s) = V^\pi(s) \frac{N(s) - 1}{N(s)} + \frac{G_{i,t}}{N(s)}
$$

which can be rewritten as:

$$
V^\pi(s) = V^\pi(s) + \frac{1}{N(s)} (G_{i,t} - V^\pi(s))
$$

- The update formula behaves like a **running average**, gradually incorporating new information.

This approach ensures **smooth updates**, avoids storing all past returns, and is **more computationally efficient**, especially in long episodes or large state spaces.

---
## 4. Temporal Difference (TD) Learning

Temporal Difference (TD) Learning is a method for estimating value functions in reinforcement learning. 

### 4.1. TD Prediction

TD Learning updates the value function using the Bellman equation. It differs from Monte Carlo methods in that it updates after each step rather than waiting for the entire episode to finish. The general TD update rule for state-value function $V^\pi(s_t)$ is:

$$
V^\pi(s_t) = V^\pi(s_t) + \alpha \left[ R_{t+1} + \gamma V^\pi(s_{t+1}) - V^\pi(s_t) \right]
$$


This approach is called **bootstrapping** since it estimates future rewards based on the current value function rather than waiting for the full return.

Just like for state-value functions, we can extend TD Learning to action-value functions (Q-values). The TD update rule for $Q(s_t, a_t)$ is:

$$
Q(s_t, a_t) = Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
$$

This allows TD learning to be applied to **control** tasks, where the agent needs to improve its policy while learning. 

### 4.3. On-Policy vs. Off-Policy TD Learning

TD methods can be used for both **on-policy** and **off-policy** learning:

- **SARSA (On-Policy TD Control)**: Updates the action-value function based on the agent’s actual policy.
- **Q-Learning (Off-Policy TD Control)**: Updates based on the optimal action, regardless of the agent’s current policy.

#### SARSA Algorithm (On-Policy)

In SARSA, the agent chooses the next action $a_{t+1}$ according to its current policy and updates the Q-value based on the immediate reward and the Q-value for the next state-action pair.

```pseudo
Initialize Q(s, a) arbitrarily, for all s ∈ S, a ∈ A(s), and Q(terminal-state, ·) = 0
Repeat (for each episode):
    Initialize S
    Choose A from S using policy derived from Q (e.g., ɛ-greedy)
    Repeat (for each step of episode):
         Take action A, observe R and next state S'
         Choose A' from S' using policy derived from Q (e.g., ɛ-greedy)

         Update Q-value:
         Q(S, A) ← Q(S, A) + α[R + γ * Q(S', A') - Q(S, A)]
        
         Set S ← S', A ← A'
        
    until S is terminal
```
<!--  
![alt text](IMG_20250223_163420.png) -->

#### Q-Learning Algorithm (Off-Policy)
Q-learning is an off-policy algorithm that learns the best action-value function, no matter what behavior policy the agent used to gather the data.
 

```pseudo
Initialize Q(s, a) arbitrarily, for all s ∈ S, a ∈ A(s), and Q(terminal-state, ·) = 0
Repeat (for each episode):
    Initialize S
   
    Repeat (for each step of episode):
         Choose A from S using policy derived from Q (e.g., ɛ-greedy)
    
         Take action A, observe R and next state S'
        
         Update Q-value:
         Q(S, A) ← Q(S, A) + α[R + γ * max_a Q(S', A') - Q(S, A)]
        
         Set S ← S'
        
    until S is terminal
```
 
<!-- 
![alt text](IMG_20250223_163452-1.png) -->

### 4.4. Exploitation vs Exploration

#### Balancing Exploration and Exploitation

In reinforcement learning, an agent needs to balance **exploration** (trying new actions) and **exploitation** (using known actions that give good rewards). To do this, we use an **$\epsilon$-greedy policy**:

$$
\pi(a_t | s_t) = 
\begin{cases} 
\arg\max_a Q(s_t, a) & \text{with probability } 1 - \epsilon \\
\text{random action} & \text{with probability } \epsilon
\end{cases}
$$


At the start of learning, **$\epsilon$** is high to encourage exploration. As the agent learns more about the environment, **$\epsilon$** decreases, allowing the agent to focus more on exploiting the best actions it has learned. This process is called **epsilon decay**.

Common ways to decay **$\epsilon$** include:

- **Linear Decay**: 

$$
\epsilon_t = \frac{1}{t}
$$

  where $t$ is the time step.

- **Exponential Decay**:
  
$$
\epsilon_t = \epsilon_0 \cdot \text{decay_rate}^t
$$


[Watch on YouTube](https://www.youtube.com/watch?v=0iqz4tcKN58)

## 5. Summary of Key Concepts and Methods

In reinforcement learning, various methods are used to estimate value functions and find optimal policies. These methods can be broadly categorized into **Model-Based** and **Model-Free** learning, as well as **On-Policy** and **Off-Policy** learning. Below is a concise summary of these key concepts and a comparison of different approaches.

### 5.1. Model-Based vs. Model-Free Learning

#### Model-Based Learning:
- **Definition**: The agent uses a model of the environment to predict future states and rewards. The model allows the agent to simulate actions and outcomes.
- **Example**: **Dynamic Programming (DP)** relies on a complete model of the environment.
- **Advantages**: Efficient when the model is available and provides exact solutions when the environment is known.

#### Model-Free Learning:
- **Definition**: The agent learns directly from interactions with the environment by estimating value functions based on observed rewards, without needing a model.
- **Examples**: **Monte Carlo (MC)**, **Temporal Difference (TD)**.
- **Advantages**: More flexible, applicable when the model is unknown or too complex to compute.

### 5.2. On-Policy vs. Off-Policy Learning

#### On-Policy Learning:
- **Definition**: The agent learns about and improves the policy it is currently following. The policy that generates the data is the same as the one being evaluated and improved.
- **Example**: **SARSA** updates based on actions taken under the current policy.
- **Advantages**: Simpler and guarantees that the agent learns from its own actions.

#### Off-Policy Learning:
- **Definition**: The agent learns about an optimal policy while following a different behavior policy. The target policy is updated while the agent explores using a behavior policy.
- **Example**: **Q-Learning** updates based on the optimal action, independent of the behavior policy.
- **Advantages**: More flexible, allows for learning from past experiences and using different exploration strategies.

### 5.3. Comparison

| Feature                        | **TD (Temporal Difference)**          | **Monte Carlo (MC)**                | **Dynamic Programming (DP)**          |
|---------------------------------|---------------------------------------|-------------------------------------|---------------------------------------|
| **Model Requirement**           | No model required (model-free)       | No model required (model-free)      | Requires a full model of the environment |
| **Learning Method**             | Updates based on current estimates (bootstrapping) | Updates after complete episode (no bootstrapping) | Updates based on exact model (transition probabilities) |
| **Update Frequency**            | After each step                      | After each episode                  | After each step or full sweep over states |
| **Efficiency**                  | More sample efficient (incremental learning) | Less efficient (requires full episodes) | Very efficient, but needs a model      |
| **Convergence**                 | Converges with sufficient exploration | Converges with sufficient exploration | Converges to optimal policy with a known model |
| **Suitability**                 | Works well in ongoing tasks          | Works well for episodic tasks       | Works well in fully known environments |

The choice of method depends on the environment, the availability of a model, and the trade-off between exploration and exploitation.

### References
- Sutton, R.S., & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- [monte-carlo for integration](https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/monte-carlo-methods-in-practice/monte-carlo-integration.html?url=mathematics-physics-for-computer-graphics/monte-carlo-methods-in-practice/monte-carlo-integration)



<!-- ## Author(s)

<div class="grid cards" markdown>
-   ![Instructor Avatar](/assets/images/staff/Ghazal-Hosseini.jpg){align=left width="150"}
    <span class="description">
        <p>**Ghazal Hosseini**</p>
        <p>Teaching Assistant</p>
        <p>[ghazaldesu@gmail.com](mailto:ghazaldesu@gmail.com)</p>
        <p>
        [:fontawesome-brands-linkedin-in:](https://www.linkedin.com/in/ghazal-hosseini-mighan-8b911823a){:target="_blank"}
        </p>
    </span>
</div> -->
