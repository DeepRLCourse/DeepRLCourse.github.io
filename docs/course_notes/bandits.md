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







## The Exploration–Exploitation Trade-off

In sequential decision-making tasks, particularly in the multi-armed bandit problem, maintaining accurate estimates of the value or expected reward of each available action (often called an "arm") is essential. However, accurate estimation alone is insufficient. A fundamental and challenging issue emerges naturally from this setting: the exploration–exploitation trade-off. This trade-off encapsulates a strategic decision every agent must confront repeatedly: should it exploit its current knowledge by choosing actions known (or estimated) to yield high rewards, or should it explore uncertain options to gather more information and potentially discover even more rewarding choices?

### Formal Definition and Intuition

Formally, the exploration–exploitation trade-off can be characterized as follows. Consider a bandit problem with a set of arms $\mathcal{A} = \{1, 2, \dots, K\}$, each arm $i$ associated with a fixed but unknown reward distribution characterized by a mean reward $\mu_i$. At any time step $t$, the agent selects an arm $A_t \in \mathcal{A}$, observes a reward $R_t \sim \text{distribution}(\mu_{A_t})$, and updates its value estimates accordingly. If we denote the agent's estimate of the expected reward of arm $i$ at time $t$ by $\hat{Q}_t(i)$, the decision about which arm to pull next becomes critical.

The key tension arises because of incomplete knowledge: the agent does not initially know the true mean rewards $\mu_i$. Thus, it must decide between:

- **Exploitation**: Selecting the arm with the highest current estimated value $\hat{Q}_t(i)$ (greedy choice) to maximize immediate reward.
- **Exploration**: Selecting a less certain arm to refine value estimates and possibly discover a superior arm for future exploitation.

Intuitively, excessive exploitation risks converging prematurely to a suboptimal arm due to misleading early observations. On the other hand, excessive exploration continuously incurs opportunity costs by potentially sacrificing immediate rewards. Hence, effective strategies must delicately balance these competing objectives to achieve minimal long-term regret.

### Risks of Pure Exploitation and Exploration Strategies

Consider first a purely exploitative approach—commonly referred to as the **greedy strategy**. Under this policy, at every step after an initial brief exploration period, the agent always selects the arm that currently has the highest empirical mean reward:

$$
A_t = \arg\max_{i \in \mathcal{A}} \hat{Q}_t(i)
$$

At first glance, this might seem optimal, as the agent is consistently choosing the "best" known option. However, such a strategy is vulnerable to early randomness. For example, if the true best arm $i^*$ initially yields a low reward due to chance, while an inferior arm $j$ provides unusually high early rewards, the greedy policy will mistakenly favor the suboptimal arm $j$ indefinitely. Consequently, the agent fails to discover the superior reward potential of arm $i^*$, causing substantial long-term regret. Formally, it can be rigorously proven that the purely greedy strategy incurs linear regret:

$$
R(T) = \Theta(T), \quad \text{as } T \rightarrow \infty
$$

In contrast, a purely exploratory strategy, which chooses arms uniformly at random or with constant probability regardless of their past performance, guarantees discovery of each arm’s true expected value but at an excessive cost. Because exploration is indiscriminate, the agent continues to select suboptimal arms frequently, incurring unnecessary losses. This continuous exploration also leads to linear regret in expectation:

$$
R(T) = \Theta(T) \quad \text{(pure exploration strategy)}
$$

Therefore, neither extreme—pure exploitation nor pure exploration—is desirable. A systematic, controlled approach is required to reduce regret from linear to sublinear growth.

### Regret Minimization and the Concept of Optimism

Regret, denoted $R(T)$, measures the cumulative loss of reward compared to always choosing the best possible arm $i^*$ with mean reward $\mu^* = \max_i \mu_i$. Mathematically, regret after $T$ steps is defined as:

$$
R(T) = T \mu^* - \sum_{t=1}^{T} \mu_{A_t}
$$

Strategies addressing the exploration–exploitation trade-off aim for sublinear regret growth, typically logarithmic or polynomial, ensuring the average regret per step diminishes as time progresses. This goal motivates the concept of **optimism in the face of uncertainty**, a foundational principle guiding many effective algorithms. Optimism assumes uncertain actions potentially hold better rewards than currently estimated, encouraging exploration of less well-known arms. As the uncertainty around an arm's estimated value decreases (through repeated selection and reward observation), the optimism naturally decreases, favoring exploitation once the uncertainty sufficiently narrows.

### Common Approaches to Balancing Exploration and Exploitation

Several classic strategies systematically manage exploration and exploitation, each embodying optimism in a different way:

#### 1. $\epsilon$-Greedy Strategy

The $\epsilon$-greedy algorithm selects the greedy action with probability $1-\epsilon$, and explores randomly chosen arms uniformly with probability $\epsilon$. This approach guarantees continuous exploration but at a simple and fixed rate, allowing eventual convergence toward the optimal arm. The key limitation is the fixed exploration rate, which may remain unnecessarily high as uncertainty decreases, leading to avoidable regret.

#### 2. Optimistic Initial Values

This approach deliberately initializes all arms’ value estimates $\hat{Q}_0(i)$ optimistically high. The optimism encourages initial exploration since arms must be repeatedly tested to reduce inflated estimates toward their true values. Eventually, as real performance emerges clearly, exploitation naturally takes over. While effective initially, this method relies heavily on appropriate initial values and may lack flexibility at later stages.

#### 3. Upper Confidence Bound (UCB) Algorithms

UCB methods use statistical confidence intervals around the estimated values of arms. The algorithm selects actions by:

$$
A_t = \arg\max_{i} \left[ \hat{Q}_t(i) + \sqrt{\frac{2\ln t}{N_t(i)}} \right]
$$

where $N_t(i)$ denotes the number of times arm $i$ has been chosen by time $t$. The term added to the estimate is larger when arm $i$ is less explored (small $N_t(i)$), creating optimism toward uncertain arms. This systematic exploration results in a provably logarithmic regret bound, making UCB highly appealing from a theoretical perspective.

#### 4. Thompson Sampling (Bayesian Probability Matching)

Thompson Sampling employs a Bayesian framework. At each step, the agent draws random samples from posterior distributions representing its belief about arm values and chooses the arm associated with the highest sampled value. This probabilistic matching naturally balances exploration and exploitation, with uncertainty directly encoded in the posterior distributions. Thompson sampling frequently demonstrates excellent empirical and theoretical performance, often achieving state-of-the-art regret bounds.









## Exploration Strategies for Multi-Armed Bandits

Multi-armed bandit (MAB) problems embody the fundamental challenge of balancing exploration (gathering information about the uncertain environment) and exploitation (leveraging existing knowledge to maximize rewards). Several exploration strategies have emerged, each employing distinct mechanisms to navigate this critical trade-off. Below, we elaborate on two common strategies—**the ε-Greedy algorithm** and **Optimistic Initial Values**—examining their theoretical underpinnings, implementation specifics, and intuitive rationale.


### The $\boldsymbol{\epsilon}$-Greedy Algorithm

#### Overview and Motivation

The $\epsilon$-greedy algorithm is one of the most fundamental and widely used strategies for balancing the exploration-exploitation trade-off in sequential decision-making problems, particularly in the context of the stochastic multi-armed bandit problem. Its appeal lies in its simplicity and intuitive structure: the agent typically selects what appears to be the best action according to its current knowledge (exploitation), but occasionally takes a random action to gather more information about alternatives (exploration).

Formally, consider a $K$-armed bandit problem, where each arm $i \in \{1, \dots, K\}$ provides i.i.d. rewards drawn from an unknown distribution with mean $\mu_i$. The goal is to maximize the cumulative reward over time, or equivalently, minimize the *regret* with respect to always playing the optimal arm $i^* = \arg\max_i \mu_i$.

The $\epsilon$-greedy algorithm addresses this by injecting randomness into the action selection process. At each time step $t$, it behaves as follows:

$$
A_t = 
\begin{cases}
\text{random arm from } \{1,\dots,K\}, & \text{with probability } \epsilon, \\
\arg\max_i \hat{Q}_{t-1}(i), & \text{with probability } 1 - \epsilon.
\end{cases}
$$

Here, $\hat{Q}_{t-1}(i)$ is the estimated mean reward of arm $i$ based on observations up to time $t-1$.

#### Formal Definition

Let us define the following notation:

- Let $K$ be the number of arms.
- Let $X_{i, s}$ be the $s$-th observed reward from arm $i$.
- Let $N_t(i)$ denote the number of times arm $i$ has been selected up to time $t$.
- Let $\hat{Q}_t(i) = \frac{1}{N_t(i)} \sum_{s=1}^{N_t(i)} X_{i,s}$ denote the empirical mean reward for arm $i$ at time $t$.

At time $t+1$, the $\epsilon$-greedy algorithm proceeds as:

$$
A_{t+1} =
\begin{cases}
\text{randomly select an arm from } \{1, \dots, K\}, & \text{with probability } \epsilon, \\
\arg\max_i \hat{Q}_t(i), & \text{with probability } 1 - \epsilon.
\end{cases}
$$

The value of $\epsilon \in [0,1]$ is typically a small constant (e.g., $\epsilon = 0.1$), ensuring occasional exploration while primarily exploiting the current knowledge.

---

#### Intuition Behind the Algorithm

The core idea of $\epsilon$-greedy is to ensure that all arms are explored with non-zero probability. This addresses the fundamental problem of *uncertainty* in estimating the rewards of each arm. Initially, all estimates $\hat{Q}_t(i)$ are inaccurate due to limited samples. If the algorithm only exploits the current maximum, it risks becoming overconfident in suboptimal arms and permanently ignoring better alternatives.

Exploration allows the algorithm to collect more data about all arms, improving the estimates and preventing premature convergence to a suboptimal policy. Exploitation ensures that we are using the best known option most of the time, thus maximizing the expected reward in the short term.

The balance is governed by $\epsilon$: high $\epsilon$ means more exploration (potentially higher short-term regret), while low $\epsilon$ means more exploitation (potentially poor long-term performance if the optimal arm is missed early).

---

#### Regret Analysis with Constant $\epsilon$

The regret of a bandit algorithm at time $T$ is defined as:

$$
R(T) = T \mu^* - \mathbb{E} \left[ \sum_{t=1}^T \mu_{A_t} \right],
$$

where $\mu^* = \max_i \mu_i$ is the mean reward of the optimal arm.

For constant $\epsilon$, the agent explores with probability $\epsilon$ in each round. During exploration, it selects a random arm uniformly from the $K$ options. Hence, even as time progresses and the estimate $\hat{Q}_t$ of the optimal arm improves, the algorithm will still spend $\epsilon T$ steps (in expectation) exploring randomly.

Let $\Delta_i = \mu^* - \mu_i$ denote the expected reward *gap* between arm $i$ and the optimal arm. Then, the expected regret due to exploration is roughly:

$$
R_{\text{explore}}(T) \approx \epsilon T \cdot \Delta_{\text{avg}},
$$

where $\Delta_{\text{avg}} = \frac{1}{K} \sum_{i=1}^K \Delta_i$ is the average regret per random pull.

The expected regret due to exploitation is smaller. With enough exploration, the agent will learn to identify the optimal arm, and during the $(1 - \epsilon)T$ exploitation steps, it will mostly choose the correct arm. Hence, the dominant contribution to regret comes from exploration.

Thus, for constant $\epsilon$, we have:

$$
R(T) = \Theta(T), \quad \text{(linear regret)}
$$

and the *average* regret $\frac{R(T)}{T} \to \epsilon \Delta_{\text{avg}}$ as $T \to \infty$. Therefore, constant-$\epsilon$ greedy is *not asymptotically optimal*.

---

#### Convergence of $\hat{Q}_t$ Estimates

Despite its linear regret, constant-$\epsilon$ greedy does guarantee convergence of estimates. Since there is a fixed, non-zero probability of selecting each arm at every time step, the number of times any given arm $i$ is selected satisfies:

$$
\mathbb{E}[N_T(i)] \geq \epsilon \cdot \frac{T}{K}.
$$

By the Law of Large Numbers, this ensures:

$$
\hat{Q}_T(i) \xrightarrow{a.s.} \mu_i \quad \text{as } T \to \infty,
$$

for all $i$. Thus, in the limit, the algorithm identifies the optimal arm correctly, but continues to explore forever at a constant rate — causing the regret to grow linearly over time.

---

#### Decaying $\epsilon_t$ and Sublinear Regret

To improve performance, one can use a time-dependent exploration rate $\epsilon_t$ that decreases with $t$. The motivation is that early on, when little is known, exploration should be frequent. As estimates improve, less exploration is needed, and exploitation becomes safer.

Common choices for decaying schedules include:

- **Inverse time decay**: $\epsilon_t = \frac{1}{t}$
- **Logarithmic decay**: $\epsilon_t = \frac{c \ln t}{t}$, for some constant $c > 0$
- **Gap-aware decay**: $\epsilon_t = \min\left\{1, \frac{K}{t \Delta^2}\right\}$ (requires knowledge of gap $\Delta$)

Under such schedules, we can show that:

$$
R(T) = O(\ln T),
$$

i.e., the regret grows logarithmically, which is the best one can hope for in the stochastic setting (matching the lower bound of Lai and Robbins for the asymptotic regret of consistent algorithms).

**Sketch of proof (informal intuition):** With $\epsilon_t = \frac{c \ln t}{t}$, the total number of exploration steps up to time $T$ is approximately:

$$
\sum_{t=1}^T \epsilon_t = \sum_{t=1}^T \frac{c \ln t}{t} \leq c \ln^2 T.
$$

This means that suboptimal arms are chosen much less frequently over time, and the cumulative regret remains sublinear.






### Optimistic Initial Values

In the context of multi-armed bandit problems, where an agent must choose among several actions (or "arms") to maximize cumulative reward, one of the core challenges is managing the exploration-exploitation trade-off. That is, the agent must balance the need to **explore** lesser-known actions to gather information with the desire to **exploit** currently believed-to-be optimal actions to maximize rewards.

While a common strategy like **$\epsilon$-greedy** addresses this by injecting randomness into action selection, another elegant and deterministic alternative is **optimistic initialization**, or **optimistic initial values**. This approach leverages *prior optimism* to naturally induce exploration, without relying on explicit stochasticity.

---

#### Motivating Intuition

The intuition behind optimistic initial values stems from a simple psychological metaphor: the agent begins with **overly optimistic beliefs** about the potential payoff of every action. It “believes” every arm is excellent — better than it realistically could be — and therefore is compelled to try each arm at least once to confirm or refute this belief. Upon playing an arm and observing actual rewards (which are, on average, lower than the initial belief), the agent adjusts its estimate downward. Thus, exploration arises **not from randomness**, but from *disappointment* in inflated expectations.

This method of optimistic bias is especially powerful in **stationary environments**, where the underlying reward distributions do not change over time. In such settings, an intense burst of early exploration can suffice, after which the agent can greedily exploit the best-known option based on refined value estimates.

---

#### Formal Definition and Mathematical Formulation

Let us consider the standard $k$-armed bandit problem. Each arm $i \in \{1, 2, \dots, k\}$ yields stochastic rewards drawn from an unknown and stationary distribution with true mean $\mu_i$.

The agent maintains an estimate $\hat{Q}_t(i)$ of the mean reward for each arm $i$ at time $t$, which is updated incrementally as rewards are observed. The standard sample average update rule is:

$$
\hat{Q}_{t+1}(i) = \hat{Q}_t(i) + \alpha_t(i) \left( R_t(i) - \hat{Q}_t(i) \right),
$$

where:

- $R_t(i)$ is the reward observed after playing arm $i$ at time $t$,
- $\alpha_t(i)$ is the step size, typically set to $1/N_t(i)$ where $N_t(i)$ is the number of times arm $i$ has been selected by time $t$.

In **optimistic initialization**, we initialize the estimates as follows:

$$
\hat{Q}_0(i) = Q^+ \quad \text{for all } i,
$$

where $Q^+$ is a constant such that $Q^+ > \max_i \mu_i$, i.e., it exceeds all plausible true reward means. For example, if rewards are bounded in the interval $[0, 1]$, a typical choice is $Q^+ = 1$ or even slightly higher.

At each time step $t$, the agent selects the arm with the highest estimated value:

$$
A_t = \arg\max_i \hat{Q}_t(i),
$$

which is a purely greedy policy.

---

#### Behavioral Dynamics

Initially, all estimates $\hat{Q}_0(i) = Q^+$ are equal and maximal, so the agent arbitrarily picks one. Upon selecting an arm, the estimate is updated based on the reward received. Because actual rewards are typically lower than $Q^+$, the new estimate will decrease. Since all other arms still retain their high initial estimates, the agent is then drawn to try those next. This **cyclic effect** continues until all arms have been sampled and their estimates revised downward, in proportion to their observed performance.

Once each arm has been sampled sufficiently to provide reliable estimates of their true means, the arm with the highest empirical average is selected going forward. At this point, the agent effectively switches from exploration to exploitation — but crucially, **without any explicit exploration parameter**.

---

#### Comparison to $\epsilon$-Greedy

In contrast to $\epsilon$-greedy — where the agent continues to explore with fixed probability $\epsilon$ indefinitely — optimistic initialization focuses exploration into the **early stage** of learning. Once the overly optimistic estimates are corrected, the algorithm becomes purely greedy. This concentrated exploration phase often leads to **faster convergence** to optimal behavior, especially when the environment is stationary.

Furthermore, optimistic initialization **avoids persistent exploration** of obviously suboptimal arms, a common downside of $\epsilon$-greedy. This leads to reduced long-term regret in many practical scenarios.

---

#### Regret Analysis

Let us now examine the regret of optimistic initialization from a theoretical standpoint. Define the **regret** at time $T$ as:

$$
\text{Regret}(T) = T\mu^* - \sum_{t=1}^T \mathbb{E}[\mu_{A_t}],
$$

where $\mu^* = \max_i \mu_i$ is the mean reward of the optimal arm.

Even though the policy is greedy after a short initial phase, optimistic initialization ensures that every arm is sampled sufficiently many times to detect suboptimality. Suppose that $Q^+$ is set such that each suboptimal arm $i$ is pulled at most $O\left(\frac{1}{\Delta_i^2} \log T\right)$ times, where $\Delta_i = \mu^* - \mu_i$ is the suboptimality gap. This yields:

$$
\text{Regret}(T) = O\left( \sum_{i: \Delta_i > 0} \frac{\log T}{\Delta_i} \right),
$$

which matches the asymptotic regret bound of more sophisticated algorithms like **Upper Confidence Bound (UCB)**. Hence, optimistic initialization, despite its simplicity, can achieve logarithmic regret.

However, if $Q^+$ is set *too optimistically*, the agent may spend unnecessary time validating even poor arms. Conversely, if it is set *insufficiently optimistically*, some arms may not be explored at all. Therefore, **the choice of $Q^+$ must be made carefully**, ideally based on prior knowledge of the reward bounds.

