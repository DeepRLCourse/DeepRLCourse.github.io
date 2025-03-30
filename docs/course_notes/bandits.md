# Multi-Armed Bandit Problem

## Introduction

The **multi-armed bandit (MAB)** problem represents one of the simplest yet profoundly insightful frameworks for analyzing the fundamental dilemma known as the **exploration-exploitation tradeoff** in decision-making under uncertainty. This tradeoff arises naturally whenever an agent faces multiple choices whose outcomes are uncertain, requiring it to continually balance between **exploring** unknown actions to discover their potential rewards and **exploiting** known actions to maximize immediate returns. The elegance and simplicity of the MAB setup enable rigorous theoretical analysis while maintaining relevance to numerous practical scenarios.

## Formal Problem Statement

Formally, a multi-armed bandit problem can be modeled as a simplified form of a Markov Decision Process (MDP) characterized solely by an **action set** and **reward functions**, without state dynamics. Specifically, the bandit setup is represented by the tuple $(\mathcal{A}, \mathcal{R})$, where:

- **Action Set:** $\mathcal{A}$ is a finite set of discrete actions, often referred to as "bandit arms," indexed by $a = 1, 2, \dots, k$. Here, $k$ denotes the total number of available actions.

- **Reward Distributions:** Each action $a \in \mathcal{A}$ is associated with a distinct probability distribution over rewards, denoted by $\mathcal{R}^a$. Formally, the reward obtained from action $a$ at time step $t$, represented as $R_t$, is sampled from this distribution:

$$
R_t \sim \mathcal{R}^{A_t}, \quad \text{where } A_t \in \mathcal{A}
$$

This means the reward for choosing action $a$ is a random variable with a specific but unknown probability distribution.

- **Objective:** The goal of the agent in this setting is explicitly to maximize the **cumulative reward** collected over a finite horizon of $T$ steps:

$$
G_T = \sum_{t=1}^{T} R_t
$$

#### Action-Value Functions (Q-values)

To formally analyze and optimize decisions in the multi-armed bandit problem, we define an essential concept known as the **action-value** or **Q-value** of an action. The Q-value of an action represents its expected or average reward:

$$
q(a) = \mathbb{E}[R \mid A = a] = \int_{-\infty}^{\infty} r \cdot \mathcal{R}^{a}(r) \,dr
$$

In simpler terms, the action-value $q(a)$ captures the average reward the agent can expect if it repeatedly selects action $a$. Estimating these action-values accurately is central to solving bandit problems, as optimal actions will naturally correspond to those with higher Q-values.

#### Optimal Action and Optimal Value

Within the multi-armed bandit framework, there always exists at least one optimal action, denoted by $a^\star$, that maximizes the expected reward. The corresponding maximum Q-value, known as the **optimal value**, is defined as:

$$
v_\star = q(a^\star) = \max_{a \in \mathcal{A}} q(a)
$$

Identifying the optimal action is the primary challenge, as the agent initially lacks knowledge about the reward distributions and must learn through interaction.

#### Exploration vs. Exploitation: Core Difficulty

The fundamental difficulty faced by agents in the MAB scenario arises precisely from the lack of initial knowledge about the underlying reward distributions. The agent must simultaneously accomplish two conflicting tasks:

- **Exploration:** By choosing less-understood or infrequently selected arms, the agent gathers crucial information about their reward structures. Exploration can yield long-term benefits by identifying potentially superior actions.

- **Exploitation:** By selecting the actions known to yield high rewards, the agent maximizes immediate returns. Excessive exploitation, however, risks prematurely converging to suboptimal actions due to inadequate exploration.

Balancing these two aspects to maximize cumulative reward over time forms the crux of solving any bandit problem effectively.

### Non-Associativity Property

One unique characteristic of the multi-armed bandit setting, which significantly simplifies its theoretical analysis compared to general MDPs, is the property of **non-associativity**. Formally:

- Non-associativity means the optimal action does **not depend on any notion of "state" or previous actions**. In other words, the bandit problem does not include state transitions—each action choice is independent of any past or future decision.

- Therefore, the optimal action $a^\star$ remains constant for all time steps, unaffected by previously selected actions. Mathematically, no state-based transition probabilities or value functions conditioned on states are necessary, making the bandit problem a purely action-oriented optimization scenario.

This non-associativity greatly simplifies both theoretical and practical treatment, allowing researchers to isolate the core exploration-exploitation dynamics from more complex temporal or state-dependent phenomena.

#### Real-World Applications

Despite its apparent simplicity, the multi-armed bandit framework finds extensive applications across diverse fields, where efficient decision-making under uncertainty directly influences outcomes. Some key areas include:

- **Medical Trials:** Clinical research often faces the challenge of testing multiple treatments while minimizing patient risk. MAB strategies help researchers adaptively assign treatments, effectively balancing learning (exploring treatment efficacy) and optimizing patient outcomes (exploiting effective treatments).

- **Online Advertising:** Digital platforms utilize MAB algorithms to dynamically select advertisements that maximize user engagement and revenue. By continuously balancing exploration of new ads and exploitation of proven performers, businesses optimize long-term profits.

- **Recommendation Systems:** Platforms like streaming services or e-commerce websites employ MAB methods to personalize content delivery. Adaptive recommendation algorithms efficiently learn user preferences by experimenting with various content while maintaining user satisfaction.

- **Financial Investment:** Asset allocation and portfolio management tasks naturally map onto bandit problems, where investment decisions must balance immediate financial returns against uncertainty about future asset performance. Using MAB-based decision frameworks, investors systematically explore financial instruments to identify strategies that yield superior long-term returns.

In all these applications, the fundamental logic of balancing exploration and exploitation captured by the multi-armed bandit problem remains central to achieving optimal performance under uncertainty.










## Action-Value Methods and Types

To effectively approach and solve the Multi-Armed Bandit (MAB) problem, we require a method for accurately estimating the value associated with each action. This value, commonly referred to as the **action-value function**, denoted by $Q_t(a)$, represents the estimated expected reward of choosing a particular action $a$ at time step $t$. Formally, the goal is for $Q_t(a)$ to approximate the true expected reward $q_*(a)$, as closely as possible:

$$
Q_t(a) \approx q_*(a).
$$

In practice, the exact values $q_*(a)$ are unknown and must be estimated through experience.

#### Sample-Average Estimation

A straightforward approach for estimating the action-value is known as **sample-average estimation**. Under this method, the value of an action $a$ is estimated by averaging all the observed rewards obtained from selecting action $a$ up to time step $t$. The sample-average estimator is formally defined as:

$$
Q_t(a) = \frac{1}{N_t(a)} \sum_{i=1}^{N_t(a)} R_i,
$$

where:
- $N_t(a)$ is the total number of times action $a$ has been selected up to time step $t$.
- $R_i$ is the reward received at the $i^{th}$ time action $a$ was selected.

##### Intuition

This method relies on the Law of Large Numbers, where averaging a large number of observations converges to the true expected reward. Initially, the estimates are inaccurate due to limited observations, but as the action is repeatedly selected, the estimate $Q_t(a)$ increasingly stabilizes and converges towards the true mean reward $q_*(a)$.

#### Incremental Update Rule for Efficient Computation

While computing the action-value through sample-average estimation, it would be computationally inefficient and memory-intensive to store and sum all previous rewards each time a new reward is obtained. Instead, an efficient, incremental update rule can be derived, allowing the estimate $Q_t(a)$ to be updated using only the previously calculated estimate and the most recent reward.

This incremental rule is given by:

$$
Q_{t+1}(a) = Q_t(a) + \frac{1}{N_t(a)} \left(R_t - Q_t(a)\right).
$$

##### Derivation of the Incremental Update Rule

Starting from the definition of the sample-average estimate at the next time step $t+1$, we have:

$$
Q_{t+1}(a) = \frac{1}{N_{t+1}(a)} \sum_{i=1}^{N_{t+1}(a)} R_i.
$$

Breaking this down into the previous $N_t(a)$ rewards plus the most recent reward $R_t$, we have:

$$
Q_{t+1}(a) = \frac{1}{N_{t+1}(a)} \left(\sum_{i=1}^{N_t(a)} R_i + R_t\right).
$$

We already have from the previous step:

$$
Q_t(a) = \frac{1}{N_t(a)} \sum_{i=1}^{N_t(a)} R_i \quad \Rightarrow \quad \sum_{i=1}^{N_t(a)} R_i = N_t(a)Q_t(a).
$$

Substituting this into the equation above gives:

$$
Q_{t+1}(a) = \frac{1}{N_{t+1}(a)} \left(N_t(a)Q_t(a) + R_t\right).
$$

Recognizing that $N_{t+1}(a) = N_t(a) + 1$, we can rewrite this as:

$$
Q_{t+1}(a) = Q_t(a) + \frac{1}{N_t(a) + 1}\left(R_t - Q_t(a)\right),
$$

which is precisely the incremental update rule. This formulation clearly demonstrates that updating action-value estimates does not require retaining all historical rewards—only the current estimate, $Q_t(a)$, and the most recent observation, $R_t$, are needed.

#### Constant Step-Size Update for Nonstationary Problems

The previously discussed sample-average estimation assumes the reward distributions are stationary (constant over time). However, many practical problems involve nonstationary environments, where the true action values can change over time. To handle such scenarios, we introduce a modified update rule that uses a **constant step-size parameter** $\alpha$ instead of the diminishing factor $\frac{1}{N_t(a)}$:

$$
Q_{t+1}(a) = Q_t(a) + \alpha(R_t - Q_t(a)),
$$

where $0 < \alpha \leq 1$ determines how much emphasis is placed on recent rewards.

- If $\alpha = \frac{1}{N_t(a)}$, this formulation reverts back to the sample-average method.
- If $\alpha$ is constant and fixed, recent rewards have greater influence, making the estimates more responsive to changes in the environment.

##### Exponential Weighted Averaging

When employing a constant step-size, the estimate effectively becomes an exponentially weighted average of past rewards, giving exponentially decreasing weights to older observations. This becomes clear by expanding the incremental update recursively:

$$
Q_{t+1}(a) = (1 - \alpha)Q_t(a) + \alpha R_t
$$

Continuing recursively for additional steps, we have:

$$
Q_{t+2}(a) = (1 - \alpha)^2 Q_t(a) + \alpha(1 - \alpha) R_t + \alpha R_{t+1}.
$$

Generalizing this recursive expansion, the influence of the initial estimate $Q_0(a)$ decreases exponentially, and we have the general form:

$$
Q_t(a) = (1 - \alpha)^t Q_0(a) + \sum_{i=0}^{t-1} \alpha(1 - \alpha)^i R_{t-i}.
$$

This explicitly illustrates the exponential weighting mechanism: recent rewards (closer to the current time $t$) exert a higher influence on the current estimate, while older rewards have their influence gradually diminished by a factor of $(1 - \alpha)$ per time step.

This exponential weighting characteristic makes the constant step-size update particularly well-suited for dynamic, nonstationary environments, where quickly adapting to changes in action-value distributions is critical.






## Regret: Measuring Suboptimality

In reinforcement learning, particularly in the context of multi-armed bandit problems, an important concept used to evaluate and analyze the performance of different action-selection strategies is called **regret**. Regret quantitatively captures the difference between the reward obtained by the actions actually selected by the agent and the reward that would have been obtained if the optimal action had always been selected. In other words, regret measures how much an agent "misses out" by not always choosing the best possible action.

#### Formal Definition of Instantaneous Regret

We first define the instantaneous regret at any timestep $t$. Suppose an agent selects action $A_t$ at timestep $t$, and let $q(a)$ represent the true expected value (or expected reward) associated with any action $a$. Furthermore, let us denote $v_\star$ as the expected reward of the optimal action, i.e., the action with the highest expected value:

$$
v_\star = \max_{a \in \mathcal{A}} q(a)
$$

Then, the instantaneous regret $I_t$ at timestep $t$ can be formally defined as the difference in expected rewards between the optimal action and the chosen action:

$$
I_t = \mathbb{E}[v_\star - q(A_t)]
$$

This expression captures the expected immediate loss of reward incurred by choosing suboptimal action $A_t$ rather than the optimal action $a^\star$. Clearly, if the chosen action at timestep $t$ is the optimal one, the instantaneous regret at that timestep is zero. If a suboptimal action is selected, the instantaneous regret will be positive, indicating an opportunity loss.

#### Total Regret

Building upon the instantaneous regret, we define **total regret** as the accumulated regret over multiple timesteps within an episode or experiment. Formally, the total regret over a horizon of $t$ steps is defined as:

$$
L_t = \mathbb{E}\left[\sum_{\tau=1}^t \left(v_\star - q(A_\tau)\right)\right]
$$

Here, the summation aggregates the instantaneous regrets across all actions selected from timestep 1 up to timestep $t$. The goal of an agent is typically to minimize this total regret. Equivalently, minimizing regret translates directly into maximizing cumulative reward, as minimizing the difference from optimal performance implies maximizing total rewards earned.

#### Action Counts and Gap Definition

To better analyze regret, it's beneficial to consider how often specific actions are selected. We define $N_t(a)$ as the expected number of times action $a$ is chosen by the agent up to timestep $t$. Another useful definition is the **gap** $\Delta_a$, which explicitly measures the quality difference between action $a$ and the optimal action:

$$
\Delta_a = v_\star - q(a)
$$

This gap is always non-negative and directly indicates how much less rewarding action $a$ is compared to the best action available. By definition, the gap is zero for the optimal action.

Using these notions, we rewrite the total regret in a more insightful way, emphasizing the contribution of each action individually:

$$
\begin{aligned}
  L_t &= \mathbb{E}\left[\sum_{\tau=1}^t \left(v_\star - q(A_\tau)\right)\right] \\
      &= \sum_{a \in \mathcal{A}} \mathbb{E}[N_t(a)]\Delta_a
\end{aligned}
$$

Here, we see explicitly how regret accumulates through repeatedly selecting suboptimal actions, weighted by how many times each of these suboptimal actions is chosen.

#### Intuition and Behavior of Regret

Examining the behavior of regret can reveal critical insights about action selection strategies. For instance, consider a purely greedy strategy—always selecting the action currently believed to yield the highest immediate reward based on past observations. Such a greedy strategy can potentially result in linear regret because it may prematurely commit or "lock" onto a suboptimal action due to initial misleading rewards. Once locked, it repeatedly incurs a fixed positive regret at every timestep, causing regret to increase linearly over time.

To address this issue, one common heuristic is called **optimistic initialization**, where the estimated Q-values for all actions are initialized optimistically (higher than their true values). Mathematically, we typically update the estimated value for each action as the empirical average reward:

$$
Q(a) = \frac{1}{N_t(a)}\sum_{\tau=1}^t \mathbf{1}(A_\tau = a)R_\tau
$$

This optimistic initialization encourages exploration by giving each action an initially favorable estimate, which makes it more likely for the agent to try every action at least once, thus reducing the risk of prematurely locking onto a suboptimal choice.

#### Regret Bounds and Their Significance

Theoretical studies in reinforcement learning and multi-armed bandit problems provide important guarantees in the form of **regret bounds**. Simple algorithms like the $\varepsilon$-greedy method, which randomly explore with a small fixed probability $\varepsilon$, generally incur linear regret. However, strategies that carefully decay the exploration probability $\varepsilon$ over time can achieve much more favorable performance, attaining logarithmic regret. Specifically, these algorithms incrementally become more confident and thus reduce unnecessary exploration, resulting in a logarithmic growth of regret over time.

Importantly, there exists a fundamental theoretical lower bound on the achievable regret by any algorithm. This lower bound is logarithmic in nature and is formally expressed as follows:

$$
\lim_{t\rightarrow\infty} L_t \geq \log t \sum_{a\,|\,\Delta_a>0} \frac{\Delta_a}{\text{KL}(\mathcal{R}^a \|\| \mathcal{R}^{a^\star})}
$$

In this equation, $\text{KL}(\mathcal{R}^a \|\| \mathcal{R}^{a^\star})$ denotes the Kullback-Leibler divergence between the reward distribution $\mathcal{R}^a$ of suboptimal action $a$ and the reward distribution $\mathcal{R}^{a^\star}$ of the optimal action. This divergence measures the statistical distinguishability between these two distributions. A higher divergence implies it is easier to distinguish suboptimal actions from optimal actions, which reduces the achievable regret.

Thus, the lower bound reveals two crucial insights:
- Regret grows at least logarithmically with time.
- The bound depends positively on the gaps (larger gaps yield greater minimum regret) and inversely on the distinguishability of action distributions.

Understanding these foundational properties of regret provides the conceptual tools and mathematical frameworks necessary to design, analyze, and compare reinforcement learning strategies effectively.
