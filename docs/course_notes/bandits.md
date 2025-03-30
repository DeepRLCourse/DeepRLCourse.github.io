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

### Concept of Regret

In sequential decision-making, especially within reinforcement learning and multi-armed bandit frameworks, a central concept is the **regret**. Regret quantifies the notion of lost opportunity incurred by choosing suboptimal actions over optimal ones. Intuitively, regret measures how much better the agent could have performed had it always selected the best possible action available, denoted by $a^\star$. Formally, we define the instantaneous regret at iteration $t$ as the expected difference between the reward from the optimal action and the reward received from the chosen action $A_t$:

$$
I_t = \mathbb{E}[v_\star - q(A_t)],
$$

where $v_\star$ represents the expected reward of the optimal action $a^\star$, and $q(A_t)$ represents the expected reward from the action actually taken at step $t$.

### Total Regret

To evaluate the performance of an agent over a sequence of decisions, we typically consider the cumulative effect of these instantaneous regrets. The **total regret** over a horizon of $t$ steps is thus:

$$
L_t = \mathbb{E}\left[\sum_{\tau=1}^{t} (v_\star - q(A_\tau))\right].
$$

Minimizing total regret is directly equivalent to maximizing cumulative reward, making regret a natural performance metric for learning algorithms in reinforcement learning contexts.

### 3. Regret, Gap, and Action Counts

To analyze regret in greater detail, we introduce two important concepts:

- The **action-count** $N_t(a)$, which denotes the expected number of times an action $a$ has been selected up to iteration $t$.
- The **gap** $\Delta_a$, defined as the difference between the optimal action's expected value and the expected value of action $a$:

$$
\Delta_a = v_\star - q(a).
$$

Given these definitions, the total regret $L_t$ can also be expressed in terms of the gaps and action counts. Specifically, by decomposing the regret according to how often each suboptimal action is chosen, we have:

$$
\begin{aligned}
L_t &= \mathbb{E}\left[\sum_{\tau=1}^{t} (v_\star - q(A_\tau))\right] \\
&= \sum_{a \in \mathcal{A}} \mathbb{E}[N_t(a)](v_\star - q(a)) \\
&= \sum_{a \in \mathcal{A}} \mathbb{E}[N_t(a)] \Delta_a.
\end{aligned}
$$

Thus, the problem of regret minimization reduces to minimizing the expected count of suboptimal actions chosen, particularly those with large gaps.

### Regret Dynamics and Algorithmic Insights

An important insight about regret is how it grows as a function of time $t$ under various strategies. For instance, a purely greedy algorithm—one that selects actions solely based on current value estimates—will exhibit linear regret. This linear growth occurs because the algorithm might prematurely "lock in" on a suboptimal action indefinitely, accruing constant regret at each step.

One powerful mitigation strategy is known as **optimistic initialization**, where we deliberately overestimate initial action values. Formally, the action-value estimates $Q(a)$ are updated using an averaging process:

$$
Q(a) = \frac{1}{N_t(a)} \sum_{\tau=1}^{t} \mathbf{1}(A_\tau = a) R_\tau.
$$

This optimistic approach incentivizes initial exploration, reducing the chance of permanently settling on a suboptimal action, thereby improving long-term regret performance.

### Lower Bound on Regret (Lai-Robbins Bound)

An essential theoretical result by Lai and Robbins (1985) provides a fundamental lower bound on achievable regret growth for any "consistent" algorithm—that is, any algorithm whose regret grows sublinearly for all problem instances. Formally, the Lai-Robbins bound is stated as:

$$
\liminf_{t \to \infty} \frac{L_t}{\ln t} \geq \sum_{a \mid \Delta_a > 0} \frac{\Delta_a}{D_{\text{KL}}(\mathcal{R}^a \|\| \mathcal{R}^{a^\star})},
$$

where $D_{\text{KL}}(\mathcal{R}^a || \mathcal{R}^{a^\star})$ is the Kullback–Leibler (KL) divergence between the reward distributions of a suboptimal arm $a$ and the optimal arm $a^\star$. Intuitively, this bound indicates that arms with smaller gaps ($\Delta_a$ close to zero) or similar reward distributions to the optimal arm (small KL divergence) inherently require more exploration, resulting in greater regret.

### Bernoulli Bandit Case

In practical scenarios such as Bernoulli bandits, where each action's reward distribution is Bernoulli($\mu_a$), the KL divergence has a closed-form expression:

$$
D_{\text{KL}}(\text{Bern}(\mu_a) \|\| \text{Bern}(\mu^\star)) = \mu^\star \ln \frac{\mu^\star}{\mu_a} + (1 - \mu^\star) \ln \frac{1 - \mu^\star}{1 - \mu_a}.
$$

For large $t$, the Lai-Robbins bound simplifies approximately to:

$$
L_t \gtrsim \sum_{a \mid \mu_a < \mu^\star} \frac{\ln t}{\mu^\star - \mu_a},
$$

clearly demonstrating the logarithmic lower bound on regret growth. Thus, no algorithm can improve beyond a logarithmic rate of regret growth for these problem instances.

### Problem-Dependent versus Minimax Regret

The regret bounds discussed so far are **problem-dependent**, reflecting intrinsic characteristics of specific problem instances (such as gaps between arms). Another view is the minimax regret, which considers the worst-case regret across all possible problem instances. For stochastic bandits with fixed reward distributions, the problem-dependent bound ($\Theta(\ln t)$) is generally more informative and achievable compared to the minimax bound, which typically scales as $\Theta(\sqrt{Kt})$ in adversarial settings.

Several algorithms, including Upper Confidence Bound (UCB) and Thompson Sampling, have been shown to achieve regret growth matching the logarithmic Lai-Robbins lower bound, both asymptotically and in some cases even in constant factors. This optimal performance contrasts starkly with naive strategies such as fixed $\varepsilon$-greedy methods, which incur linear regret due to continual exploration.
