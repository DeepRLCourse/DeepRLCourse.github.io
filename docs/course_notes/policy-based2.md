---
comments: True
description: This page provides an in-depth exploration of policy-based methods in reinforcement learning, focusing on their theoretical foundations, practical implementations, and advantages over value-based methods. Topics include policy gradient theorem, variance reduction techniques, REINFORCE algorithm, actor-critic methods, and their applications in continuous action spaces. The content is enriched with mathematical proofs, examples, and visual aids to enhance understanding.
---

# Week 8: Policy-Based Methods


  

Reinforcement Learning (RL) focuses on training an agent to interact
with an environment by learning a policy $\pi_{\theta}(a | s)$ that
maximizes the cumulative reward. Policy gradient methods are a class of
algorithms that directly optimize the policy by adjusting the parameters
$\theta$ via gradient ascent.

  

## Why Policy Gradient Methods?

  

Unlike value-based methods (e.g., Q-learning), which rely on estimating
value functions, policy gradient methods:
- Can naturally handle stochastic policies, which are crucial in
environments requiring exploration.

- Work well in continuous action spaces, where discrete action methods
become infeasible.

- Can directly optimize differentiable policy representations, such as
neural networks.

- Avoid the need for an explicit action-value function approximation,
making them more robust in high-dimensional problems.

- Are capable of optimizing **parameterized policies** without
relying on action selection heuristics.

- Can incorporate entropy regularization to improve exploration and
prevent premature convergence to suboptimal policies.

- Allow for more **stable convergence** in some cases compared to
value-based methods, which may suffer from instability due to
bootstrapping.

- Can leverage **variance reduction techniques** (e.g., advantage
estimation, baseline subtraction) to improve learning efficiency.




## Policy Gradient

  

The goal of reinforcement learning is to find an optimal behavior
strategy for the agent to obtain optimal rewards. The **policy
gradient** methods target at modeling and optimizing the policy
directly. The policy is usually modeled with a parameterized function
respect to $\theta$, $\pi_{\theta}(a|s)$. The value of the reward
(objective) function depends on this policy and then various algorithms
can be applied to optimize $\theta$ for the best reward.

The reward function is defined as:

  

$$J(\theta) = \sum_{s \in  \mathcal{S}} d^{\pi}(s) V^{\pi}(s) = \sum_{s \in  \mathcal{S}} d^{\pi}(s) \sum_{a \in  \mathcal{A}} \pi_{\theta}(a|s) Q^{\pi}(s,a)$$

  

where $d^{\pi}(s)$ is the stationary distribution of Markov chain for
$\pi_{\theta}$ (on-policy state distribution under $\pi$). For
simplicity, the parameter $\theta$ would be omitted for the policy
$\pi_{\theta}$ when the policy is present in the subscript of other
functions; for example, $d^{\pi}$ and $Q^{\pi}$ should be
$d^{\pi_{\theta}}$ and $Q^{\pi_{\theta}}$ if written in full.

  

Imagine that you can travel along the Markov chain's states forever, and
eventually, as the time progresses, the probability of you ending up
with one state becomes unchanged --- this is the stationary probability
for $\pi_{\theta}$.
$d^{\pi}(s) = \lim_{t \to  \infty} P(s_t = s | s_0, \pi_{\theta})$ is the
probability that $s_t = s$ when starting from $s_0$ and following policy
$\pi_{\theta}$ for $t$ steps. Actually, the existence of the stationary
distribution of Markov chain is one main reason for why PageRank
algorithm works.

  

It is natural to expect policy-based methods are more useful in the
continuous space. Because there is an infinite number of actions and
(or) states to estimate the values for and hence value-based approaches
are way too expensive computationally in the continuous space. For
example, in *generalized policy iteration*, the policy improvement step
$\arg  \max_{a \in  \mathcal{A}} Q^{\pi}(s,a)$ requires a full scan of the
action space, suffering from the *curse of dimensionality*.

  

Using *gradient ascent*, we can move $\theta$ toward the direction
suggested by the gradient $\nabla_{\theta} J(\theta)$ to find the best
$\theta$ for $\pi_{\theta}$ that produces the highest return.

  

### Policy Gradient Theorem




Computing the gradient $\nabla_{\theta}J(\theta)$ is tricky because it
depends on both the action selection (directly determined by
$\pi_{\theta}$) and the stationary distribution of states following the
target selection behavior (indirectly determined by $\pi_{\theta}$).
Given that the environment is generally unknown, it is difficult to
estimate the effect on the state distribution by a policy update.

  

Luckily, the **policy gradient theorem** comes to save the world!
 It provides a nice reformation of the derivative of the
objective function to not involve the derivative of the state
distribution $d^{\pi}(\cdot)$ and simplify the gradient computation
$\nabla_{\theta}J(\theta)$ a lot.

  

$$\nabla_{\theta}J(\theta) = \nabla_{\theta} \sum_{s \in  \mathcal{S}} d^{\pi}(s) \sum_{a \in  \mathcal{A}} Q^{\pi}(s,a) \pi_{\theta}(a|s)$$

  

$$\propto  \sum_{s \in  \mathcal{S}} d^{\pi}(s) \sum_{a \in  \mathcal{A}} Q^{\pi}(s,a) \nabla_{\theta} \pi_{\theta}(a|s)$$

  

### Proof of Policy Gradient Theorem

  

This session is pretty dense, as it is the time for us to go through the
proof and figure out why the policy gradient theorem is correct.

 
???+ warning 
    This proof may be unnecessary for the first phase of the course. 
??? note "proof"


    We first start with the derivative of the state value function:

    $$
    \begin{aligned}
    \nabla_{\theta} V^{\pi}(s) &= \nabla_{\theta} \left( \sum_{a \in \mathcal{A}} \pi_{\theta}(a|s) Q^{\pi}(s,a) \right) \\
    &= \sum_{a \in \mathcal{A}} \left( \nabla_{\theta} \pi_{\theta}(a|s) Q^{\pi}(s,a) + \pi_{\theta}(a|s) \nabla_{\theta} Q^{\pi}(s,a) \right) \quad \text{; Derivative product rule.} \\
    &= \sum_{a \in \mathcal{A}} \left( \nabla_{\theta} \pi_{\theta}(a|s) Q^{\pi}(s,a) + \pi_{\theta}(a|s) \nabla_{\theta} \sum_{s', r} P(s',r|s,a) (r + V^{\pi}(s')) \right) \quad \text{; Extend } Q^{\pi} \text{ with future state value.} \\
    &= \sum_{a \in \mathcal{A}} \left( \nabla_{\theta} \pi_{\theta}(a|s) Q^{\pi}(s,a) + \pi_{\theta}(a|s) \sum_{s',r} P(s',r|s,a) \nabla_{\theta} V^{\pi}(s') \right) \\
    &= \sum_{a \in \mathcal{A}} \left( \nabla_{\theta} \pi_{\theta}(a|s) Q^{\pi}(s,a) + \pi_{\theta}(a|s) \sum_{s'} P(s'|s,a) \nabla_{\theta} V^{\pi}(s') \right) \quad \text{; Because } P(s'|s,a) = \sum_{r} P(s',r|s,a)
    \end{aligned}
    $$

    Now we have:

    $$
    \begin{aligned}
    \nabla_{\theta} V^{\pi}(s) &= \sum_{a \in \mathcal{A}} \left( \nabla_{\theta} \pi_{\theta}(a|s) Q^{\pi}(s,a) + \pi_{\theta}(a|s) \sum_{s'} P(s'|s,a) \nabla_{\theta} V^{\pi}(s') \right)
    \end{aligned}
    $$

    This equation has a nice recursive form, and the future state value function $V^{\pi}(s')$ can be repeatedly unrolled by following the same equation.

    Let's consider the following visitation sequence and label the probability of transitioning from state $s$ to state $x$ with policy $\pi_{\theta}$ after $k$ steps as $\rho^{\pi}(s \to x, k)$.

    $$
    s \xrightarrow{a \sim \pi_{\theta}(\cdot | s)} s' \xrightarrow{a' \sim \pi_{\theta}(\cdot | s')} s'' \xrightarrow{a'' \sim \pi_{\theta}(\cdot | s'')} \dots
    $$

    - When $k = 0$: $\rho^{\pi}(s \to s, k = 0) = 1$.

    - When $k = 1$, we scan through all possible actions and sum up the transition probabilities to the target state:

    $$
    \rho^{\pi}(s \to s', k = 1) = \sum_{a} \pi_{\theta}(a|s) P(s'|s,a).
    $$

    - Imagine that the goal is to go from state $s$ to $x$ after $k+1$ steps while following policy $\pi_{\theta}$. We can first travel from $s$ to a middle point $s'$ (any state can be a middle point, $s' \in S$) after $k$ steps and then go to the final state $x$ during the last step. In this way, we are able to update the visitation probability recursively:

    $$
    \rho^{\pi}(s \to x, k + 1) = \sum_{s'} \rho^{\pi}(s \to s', k) \rho^{\pi}(s' \to x, 1).
    $$

    Then we go back to unroll the recursive representation of $\nabla_{\theta}V^{\pi}(s)$! Let

    $$
    \phi(s) = \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi_{\theta}(a|s) Q^{\pi}(s,a)
    $$

    to simplify the maths. If we keep on extending $\nabla_{\theta}V^{\pi}(\cdot)$ infinitely, it is easy to find out that we can transition from the starting state $s$ to any state after any number of steps in this unrolling process and by summing up all the visitation probabilities, we get $\nabla_{\theta}V^{\pi}(s)$!

    $$
    \begin{aligned}
    \nabla_{\theta}V^{\pi}(s) &= \phi(s) + \sum_{a} \pi_{\theta}(a|s) \sum_{s'} P(s'|s,a) \nabla_{\theta}V^{\pi}(s') \\
    &= \phi(s) + \sum_{s'} \sum_{a} \pi_{\theta}(a|s) P(s'|s,a) \nabla_{\theta}V^{\pi}(s') \\
    &= \phi(s) + \sum_{s'} \rho^{\pi}(s \to s', 1) \nabla_{\theta}V^{\pi}(s') \\
    &= \phi(s) + \sum_{s'} \rho^{\pi}(s \to s', 1) \sum_{a \in \mathcal{A}} \left( \nabla_{\theta} \pi_{\theta}(a|s') Q^{\pi}(s',a) + \pi_{\theta}(a|s') \sum_{s''} P(s''|s',a) \nabla_{\theta}V^{\pi}(s'') \right) \\
    &= \phi(s) + \sum_{s'} \rho^{\pi}(s \to s', 1) \left[ \phi(s') + \sum_{s''} \rho^{\pi}(s' \to s'', 1) \nabla_{\theta}V^{\pi}(s'') \right] \\
    &= \phi(s) + \sum_{s'} \rho^{\pi}(s \to s', 1) \phi(s') + \sum_{s'} \rho^{\pi}(s \to s', 1) \sum_{s''} \rho^{\pi}(s' \to s'', 1) \nabla_{\theta}V^{\pi}(s'') \\
    &= \phi(s) + \sum_{s'} \rho^{\pi}(s \to s', 1) \phi(s') + \sum_{s''} \rho^{\pi}(s \to s'', 2) \nabla_{\theta}V^{\pi}(s'') \quad \text{; Consider } s' \text{ as the middle point for } s \to s''. \\
    &= \phi(s) + \sum_{s'} \rho^{\pi}(s \to s', 1) \phi(s') + \sum_{s''} \rho^{\pi}(s \to s'', 2) \phi(s'') + \sum_{s'''} \rho^{\pi}(s \to s''', 3) \nabla_{\theta}V^{\pi}(s''') \\
    &= \dots \quad \text{; Repeatedly unrolling the part of } \nabla_{\theta}V^{\pi}(\cdot) \\
    &= \sum_{x \in \mathcal{S}} \sum_{k=0}^{\infty} \rho^{\pi}(s \to x, k) \phi(x)
    \end{aligned}
    $$

    The nice rewriting above allows us to exclude the derivative of Q-value function, $\nabla_{\theta} Q^{\pi}(s,a)$. By plugging it into the objective function $J(\theta)$, we are getting the following:

    $$
    \begin{aligned}
    \nabla_{\theta}J(\theta) &= \nabla_{\theta}V^{\pi}(s_0) \\
    &= \sum_{s} \sum_{k=0}^{\infty} \rho^{\pi}(s_0 \to s, k) \phi(s) \quad \text{; Starting from a random state } s_0 \\
    &= \sum_{s} \eta(s) \phi(s) \quad \text{; Let } \eta(s) = \sum_{k=0}^{\infty} \rho^{\pi}(s_0 \to s, k) \\
    &= \left( \sum_{s} \eta(s) \right) \sum_{s} \frac{\eta(s)}{\sum_{s} \eta(s)} \phi(s) \quad \text{; Normalize } \eta(s), s \in \mathcal{S} \text{ to be a probability distribution.} \\
    &\propto \sum_{s} \frac{\eta(s)}{\sum_{s} \eta(s)} \phi(s) \quad \text{; } \sum_{s} \eta(s) \text{ is a constant} \\
    &= \sum_{s} d^{\pi}(s) \sum_{a} \nabla_{\theta} \pi_{\theta}(a|s) Q^{\pi}(s,a) \quad d^{\pi}(s) = \frac{\eta(s)}{\sum_{s} \eta(s)} \text{ is stationary distribution.}
    \end{aligned}
    $$

    In the episodic case, the constant of proportionality ($\sum_{s} \eta(s)$) is the average length of an episode; in the continuing case, it is 1. The gradient can be further written as:

    $$
    \begin{aligned}
    \nabla_{\theta}J(\theta) &\propto \sum_{s \in \mathcal{S}} d^{\pi}(s) \sum_{a \in \mathcal{A}} Q^{\pi}(s,a) \nabla_{\theta} \pi_{\theta}(a|s) \\
    &= \sum_{s \in \mathcal{S}} d^{\pi}(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(a|s) Q^{\pi}(s,a) \frac{\nabla_{\theta} \pi_{\theta}(a|s)}{\pi_{\theta}(a|s)} \quad \text{; Because } \ln(x)'=1/x \\
    &= \mathbb{E}_{\pi} [Q^{\pi}(s,a) \nabla_{\theta} \ln \pi_{\theta}(a|s)]
    \end{aligned}
    $$

    Where $\mathbb{E}_{\pi}$ refers to $\mathbb{E}_{s \sim d^{\pi}, a \sim \pi_{\theta}}$ when both state and action distributions follow the policy $\pi_{\theta}$ (on policy).

    The policy gradient theorem lays the theoretical foundation for various policy gradient algorithms. This vanilla policy gradient update has no bias but high variance. Many following algorithms were proposed to reduce the variance while keeping the bias unchanged.

    $$
    \nabla_{\theta}J(\theta) = \mathbb{E}_{\pi} [Q^{\pi}(s,a) \nabla_{\theta} \ln \pi_{\theta}(a|s)]
    $$

  



  

### Policy Gradient in Continuous Action Space 

  

In a continuous action space, the policy gradient theorem is given by:

  

$$\nabla_{\theta}J(\theta) = \mathbb{E}_{s \sim d^{\pi}, a \sim  \pi_{\theta}} \left[ Q^{\pi}(s,a) \nabla_{\theta} \ln  \pi_{\theta}(a|s) \right]$$

  

Since the action space is continuous, the summation over actions in the
discrete case is replaced by an integral:

  

$$\nabla_{\theta} J(\theta) = \int_{\mathcal{S}} d^{\pi}(s) \int_{\mathcal{A}} Q^{\pi}(s,a) \nabla_{\theta} \ln  \pi_{\theta}(a|s) \pi_{\theta}(a|s) \, da \, ds$$

  

where:

  

- $d^{\pi}(s)$ is the stationary state distribution under policy
$\pi_{\theta}$,

  

- $\pi_{\theta}(a|s)$ is the probability density function for the
continuous action $a$ given state $s$,

  

- $Q^{\pi}(s,a)$ is the state-action value function,

  

- $\nabla_{\theta} \ln  \pi_{\theta}(a|s)$ is the score function
(policy gradient term),

  

- The integral is taken over all possible states $s$ and actions $a$.

  

???+ example "Gaussian Policy Example"
  

    A common choice for a continuous policy is a Gaussian distribution:

    $$a \sim  \pi_{\theta}(a|s) = \mathcal{N}(\mu_{\theta}(s), \Sigma_{\theta}(s))$$

    where:
    
    - $\mu_{\theta}(s)$ is the mean of the action distribution,
    parameterized by $\theta$,


    - $\Sigma_{\theta}(s)$ is the covariance matrix (often assumed
    diagonal or fixed).

    For a Gaussian policy, the logarithm of the probability density is:

    $$\ln  \pi_{\theta}(a|s) = -\frac{1}{2} (a - \mu_{\theta}(s))^T \Sigma_{\theta}^{-1} (a - \mu_{\theta}(s)) - \frac{1}{2} \ln |\Sigma_{\theta}|$$

    Taking the gradient:

    $$\nabla_{\theta} \ln  \pi_{\theta}(a|s) = \Sigma_{\theta}^{-1} (a - \mu_{\theta}(s)) \nabla_{\theta} \mu_{\theta}(s)$$

    Thus, the policy gradient update becomes:

    $$\nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim d^{\pi}, a \sim  \pi_{\theta}} \left[ Q^{\pi}(s,a) \Sigma_{\theta}^{-1} (a - \mu_{\theta}(s)) \nabla_{\theta} \mu_{\theta}(s) \right]$$

  

### REINFORCE 

  

REINFORCE (Monte-Carlo policy gradient) relies on an estimated return by
**Monte-Carlo** methods using episode samples to update the policy
parameter $\theta$. REINFORCE works because the expectation of the
sample gradient is equal to the actual gradient:

$$
\begin{aligned}
\nabla_{\theta}J(\theta) &= \mathbb{E}_{\pi} \left[ Q^{\pi}(s,a) \nabla_{\theta} \ln \pi_{\theta}(a|s) \right] \\
&= \mathbb{E}_{\pi} \left[ G_t \nabla_{\theta} \ln \pi_{\theta}(A_t|S_t) \right] \quad \text{; Because } Q^{\pi}(S_t, A_t) = \mathbb{E}_{\pi} \left[ G_t \mid S_t, A_t \right]
\end{aligned}
$$
  

Therefore we are able to measure $G_t$ from real sample trajectories and
use that to update our policy gradient. It relies on a full trajectory
and that's why it is a Monte-Carlo method.

  

#### Algorithm 

  

The process is pretty straightforward:

  

1. Initialize the policy parameter $\theta$ at random.

  

2. Generate one trajectory on policy $\pi_{\theta}$: $S_1, A_1, R_2, S_2, A_2, \dots, S_T$.

  

3. For $t = 1, 2, \dots, T$:

    1. Estimate the return $G_t$;


    2. Update policy parameters: $\theta  \leftarrow  \theta + \alpha  \gamma^t G_t \nabla_{\theta} \ln  \pi_{\theta}(A_t|S_t)$

  

A widely used variation of REINFORCE is to subtract a baseline value
from the return $G_t$ to **reduce the variance of gradient estimation
while keeping the bias unchanged** (Remember we always want to do this
when possible).

  
For example, a common baseline is to subtract state-value from
action-value, and if applied, we would use **advantage** $A(s,a) = Q(s,a) - V(s)$ in the gradient ascent update. This [post](https://danieltakeshi.github.io/2017/03/28 going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/) nicely explained why a baseline works for reducing the variance, in addition to a set of fundamentals of policy gradient.

  



  

####  $G(s)$ in Continuous Action Space 

  

In the continuous setting, we define the **return** $G(s)$ as:

$$G(s) = \sum_{k=0}^{\infty} \gamma^k R(s_k, a_k), \quad s_0 = s, \quad a_k \sim  \pi_{\theta}(\cdot | s_k)$$
 

where:


- $R(s_k, a_k)$ is the reward function for state-action pair
$(s_k, a_k)$.

  

- $\gamma$ is the **discount factor**.

  

- $s_k$ evolves according to the environment dynamics.

  

- $a_k \sim  \pi_{\theta}(\cdot | s_k)$ means actions are sampled from
the policy.

  

#### Monte Carlo Approximation of $Q^{\pi}(s,a)$

  

In expectation, $G(s)$ serves as an **unbiased estimator** of the
state-action value function:

  

$$Q^{\pi}(s,a) = \mathbb{E} \left[ G(s) \middle| s_0 = s, a_0 = a \right]$$

  

Using this, we rewrite the policy gradient update as:

  

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim d^{\pi}, a \sim  \pi_{\theta}} \left[ G(s) \nabla_{\theta} \ln  \pi_{\theta}(a | s) \right]$$

  

#### Variance Reduction: Advantage Function

  

A **baseline** is often subtracted to reduce variance while keeping
the expectation unchanged:

  

$$A(s, a) = G(s) - V^{\pi}(s)$$

  

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim d^{\pi}, a \sim  \pi_{\theta}} \left[ A(s, a) \nabla_{\theta} \ln  \pi_{\theta}(a | s) \right]$$

  

where:

  

- $V^{\pi}(s) = \mathbb{E}_{a \sim  \pi_{\theta}(\cdot | s)} [Q^{\pi}(s,a)]$

is the **state value function**.

  

- $A(s,a)$ measures the **advantage** of taking action $a$ over
the expected policy action.

  

## Bias and Variance   
In this section we delve deeper into the bias and variance problem in RL especially in policy gradient 

### Monte Carlo Estimators in Reinforcement Learning

  

A Monte Carlo estimator is a method used to approximate the expected
value of a function $f(X)$ over a random variable $X$ with a given
probability distribution $p(X)$. The true expectation is:

  

$$E[f(X)] = \int f(x) p(x) \, dx$$

  

However, directly computing this integral may be complex. Instead, we
use Monte Carlo estimation by drawing $N$ independent samples
$X_1, X_2, \dots, X_N$ from $p(X)$ and computing:

  

$$\hat{\mu}_{MC} = \frac{1}{N} \sum_{i=1}^{N} f(X_i)$$

  

This estimator provides an approximation to the true expectation
$E[f(X)]$.

  

By the law of large numbers (LLN), as $N \to  \infty$, we have:

  

$$\hat{X}_N \to  \mathbb{E}[X] \quad  \text{(almost surely)}$$

  

Monte Carlo methods are commonly used in RL for estimating expected
rewards, state-value functions, and action-value functions.

  

### Bias in Policy Gradient Methods

  

Bias in reinforcement learning arises when an estimator systematically
deviates from the true value. In policy gradient methods, bias is
introduced due to function approximation, reward estimation, or gradient
computation errors.

  

#### Sources of Bias

  

-  **Function Approximation Bias:** Policy gradient methods often rely
on neural networks or other function approximators for policy
representation. Imperfect approximations introduce systematic
errors, leading to biased policy updates.

  

-  **Reward Clipping or Discounting:** Algorithms using reward clipping
or high discount factors ($\gamma$) can distort return estimates,
causing the learned policy to be biased toward short-term rewards.

  

-  **Baseline Approximation:** Variance reduction techniques like
baseline subtraction use estimates of expected returns. If the
baseline is inaccurately estimated, it introduces bias in the policy
gradient computation.


???+ example "Example of Bias"
    Consider a self-driving car optimizing for fuel efficiency. If the
    reward function prioritizes immediate fuel consumption over long-term
    efficiency, the learned policy may favor suboptimal strategies that
    minimize fuel use in the short term while missing globally optimal
    driving behaviors.



#### Biased vs. Unbiased Estimation

  

For example: The biased formula for the sample variance $S^2$ is given
by:

  

$$S^2_{\text{biased}} = \frac{1}{n} \sum_{i=1}^{n} (X_i - \overline{X})^2$$

  

This is an underestimation of the true population variance $\sigma^2$
because it does not account for the degrees of freedom in estimation.

Instead, the unbiased estimator is:

  

$$S^2_{\text{unbiased}} = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \overline{X})^2.$$

  

This unbiased estimator correctly accounts for variance in small sample
sizes, ensuring $\mathbb{E}[S^2_{\text{unbiased}}] = \sigma^2$.

  

### Variance in Policy Gradient Methods

  

Variance in policy gradient estimates refers to fluctuations in gradient
estimates across different training episodes. High variance leads to
instability and slow convergence.

  

#### Sources of Variance

  

-  **Monte Carlo Estimation:** REINFORCE estimates gradients using
complete episodes, leading to high variance due to trajectory
randomness.

  

-  **Stochastic Policy Outputs:** Policies represented as probability
distributions (e.g., Gaussian policies) introduce additional
randomness in gradient updates.

  

-  **Exploration Strategies:** Methods like softmax or epsilon-greedy
increase variance by adding stochasticity to action selection.



???+ example "Example of Variance"
    Consider a robotic arm learning to grasp objects. Due to high variance,
    in some episodes, it succeeds, while in others, minor variations cause
    failure. These inconsistencies slow down convergence.


### Techniques to Reduce Variance in Policy Gradient Methods

  

Several strategies help mitigate variance in policy gradient methods
while preserving unbiased gradient estimates.

  

#### Baseline Subtraction

  

A baseline function $b$ reduces variance without introducing bias:

  

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log  \pi_{\theta}(a_t | s_t) (G_t - b) \right].$$

  

A common choice for $b$ is the average return over trajectories:

  

$$b = \frac{1}{N} \sum_{i=1}^{N} G_i.$$

  

Since $b$ is independent of actions, it does not introduce bias in the
gradient estimate while reducing variance.

  
??? note "proof"

    $$\begin{aligned}
    E\left[\nabla_\theta  \log p_\theta(\tau) b\right] &= \int p_\theta(\tau) \nabla_\theta  \log p_\theta(\tau) b \, d\tau \\
    &= \int  \nabla_\theta p_\theta(\tau) b \, d\tau \\
    &= b \nabla_\theta  \int p_\theta(\tau) \, d\tau \\
    &= b \nabla_\theta  1 \\
    &= 0
    \end{aligned}$$

  



<!-- ![image](\assets\images\course_notes\policy-based\a4.png){width="0.8\\linewidth"} -->

<center> 
<img src="\assets\images\course_notes\policy-based\a4.png"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>


  

#### Causality Trick and Reward-to-Go Estimation

  

To ensure that policy updates at time $t$ are only influenced by rewards
from that time step onward, we use the causality trick:

  

$$\nabla_{\theta} J(\theta) \approx  \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log  \pi_{\theta}(a_{i,t} | s_{i,t}) \left( \sum_{t'=t}^{T} r(a_{i,t'}, s_{i,t'}) \right).$$

  

Instead of summing over all rewards, the reward-to-go estimate restricts
the sum to future rewards only:

  

$$Q(s_t, a_t) = \sum_{t'=t}^{T} \mathbb{E}_{\pi_{\theta}} [r(s_{t'}, a_{t'}) | s_t, a_t].$$

  

$$\nabla_{\theta} J(\theta) \approx  \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log  \pi_{\theta}(a_{i,t} | s_{i,t}) Q(s_{i,t}, a_{i,t}).$$

  

This prevents rewards from future time steps from affecting past
actions, reducing variance. This approach results in much lower variance
compared to the traditional Monte Carlo methods.

  



<!-- ![image](\assets\images\course_notes\policy-based\a1.png){width="0.4\\linewidth"} ![image](\assets\images\course_notes\policy-based\a2.png){width="0.4\\linewidth"} -->

<center> 
<img src="\assets\images\course_notes\policy-based\a1.png"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>

<center> 
<img src="\assets\images\course_notes\policy-based\a2.png"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>


  
??? note "proof"

    $$
    \begin{aligned}
    A_{t_0-1} &= s_{t_0-1}, a_{t_0-1}, \dots, a_0, s_0 \\
    \mathbb{E}_{A_{t_0-1}} &\left[ \mathbb{E}_{s_{t_0}, a_{t_0} | A_{t_0-1}} \left[ \nabla_{\theta} \log \pi_{\theta} (a_{t_0} | s_{t_0}) \sum_{t=0}^{t_0 - 1} r(s_t, a_t) \right] \right] \\
    U_{t_0-1} &= \sum_{t=0}^{t_0 - 1} r(s_t, a_t) \\
    &= \mathbb{E}_{A_{t_0-1}} \left[ U_{t_0-1} \mathbb{E}_{s_{t_0}, a_{t_0} | s_{t_0-1}, a_{t_0-1}} \nabla_{\theta} \log \pi_{\theta} (a_{t_0} | s_{t_0}) \right] \\
    &= \mathbb{E}_{A_{t_0-1}} \left[ U_{t_0-1} \mathbb{E}_{s_{t_0} | s_{t_0-1}, a_{t_0-1}} \mathbb{E}_{a_{t_0} | s_{t_0-1}, a_{t_0-1}, s_{t_0}} \nabla_{\theta} \log \pi_{\theta} (a_{t_0} | s_{t_0}) \right] \\
    &= \mathbb{E}_{A_{t_0-1}} \left[ U_{t_0-1} \mathbb{E}_{s_{t_0} | s_{t_0-1}, a_{t_0-1}} \mathbb{E}_{a_{t_0} | s_{t_0}} \nabla_{\theta} \log \pi_{\theta} (a_{t_0} | s_{t_0}) \right] \\
    &= \mathbb{E}_{A_{t_0-1}} \left[ U_{t_0-1} \mathbb{E}_{s_{t_0} | s_{t_0-1}, a_{t_0-1}} \mathbb{E}_{\pi_{\theta} (a_{t_0} | s_{t_0})} \nabla_{\theta} \log \pi_{\theta} (a_{t_0} | s_{t_0}) \right] \\
    \mathbb{E}_{\pi_{\theta} (a_{t_0} | s_{t_0})} &\nabla_{\theta} \log \pi_{\theta} (a_{t_0} | s_{t_0}) = 0 \\
    \mathbb{E}_{A_{t_0-1}}& \left[ \mathbb{E}_{s_{t_0}, a_{t_0} | A_{t_0-1}} \left[ \nabla_{\theta} \log \pi_{\theta} (a_{t_0} | s_{t_0}) \sum_{t=0}^{t_0 - 1} r(s_t, a_t) \right] \right] = 0
    \end{aligned}
    $$


  

#### Discount Factor Adjustment

  

The discount factor $\gamma$ helps reduce variance by weighting rewards
closer to the present more heavily:

  

$$G_t = \sum_{t' = t}^{T} \gamma^{t'-t} r(s_{t'}, a_{t'}).$$

  
??? note "proof"

    $$
    \begin{aligned}
    \nabla_{\theta} J(\theta) &\approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta} (a_{i,t} | s_{i,t}) \left( \sum_{t' = t}^{T} \gamma^{t' - t} r(s_{i,t'}, a_{i,t'}) \right) \\
    \nabla_{\theta} J(\theta) &\approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta} (a_{i,t} | s_{i,t}) \right) \left( \sum_{t=1}^{T} \gamma^{t-1} r(s_{i,t}, a_{i,t}) \right) \\
    \nabla_{\theta} J(\theta) &\approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta} (a_{i,t} | s_{i,t}) \left( \sum_{t' = t}^{T} \gamma^{t' - t} r(s_{i,t'}, a_{i,t'}) \right) \\
    \nabla_{\theta} J(\theta) &\approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \gamma^{t-1} \nabla_{\theta} \log \pi_{\theta} (a_{i,t} | s_{i,t}) \left( \sum_{t' = t}^{T} \gamma^{t' - t} r(s_{i,t'}, a_{i,t'}) \right)
    \end{aligned}
    $$

  

A lower $\gamma$ (e.g., 0.9) reduces variance but increases bias, while
a higher $\gamma$ (e.g., 0.99) improves long-term estimation but
increases variance. A balance is needed.

  

#### Advantage Estimation and Actor-Critic Methods

  

Actor-critic methods combine policy optimization (actor) with value
function estimation (critic). The advantage function is defined as:

  

$$A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t),$$

  

where the action-value function is:

  

$$Q^{\pi}(s_t, a_t) = \sum_{t' = t}^{T} \mathbb{E}_{\pi} [r(s_{t'}, a_{t'}) | s_t, a_t],$$

  

and the state-value function is:

  

$$V^{\pi}(s_t) = \mathbb{E}_{a_t \sim  \pi_{\theta}(a_t | s_t)} [Q^{\pi}(s_t, a_t)].$$

  

The policy gradient update using the advantage function becomes:

  

$$\nabla_{\theta} J(\theta) \approx  \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log  \pi_{\theta}(a_{i,t} | s_{i,t}) A^{\pi}(s_{i,t}, a_{i,t}).$$

  

This formulation allows for lower variance in policy updates while
leveraging learned state-value estimates. Actor-critic methods are
widely used in modern reinforcement learning due to their stability and
efficiency.

  





<center> 
<img src="\assets\images\course_notes\policy-based\a3.png"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>


#### Actor-Critic

  

Two main components in policy gradient methods are the policy model and
the value function. It makes a lot of sense to learn the value function
in addition to the policy since knowing the value function can assist
the policy update, such as by reducing gradient variance in vanilla
policy gradients. That is exactly what the Actor-Critic method does.

  

Actor-Critic methods consist of two models, which may optionally share
parameters:

  

-  **Critic**: Updates the value function parameters $w$. Depending on
the algorithm, it could be an action-value function $Q(s, a)$ or a
state-value function $V(s)$.

  

-  **Actor**: Updates the policy parameters $\theta$ for $\pi_{\theta}(a | s)$, in the direction suggested by the critic.

  

Let's see how it works in a simple action-value Actor-Critic algorithm:

  

1. Initialize policy parameters $\theta$ and value function parameters
$w$ at random.

  

2. Sample initial state $s_0$.

  

3. For each time step $t$:

    1. Sample reward $r_t$ and next state $s_{t+1}$.

  

    2. Then sample the next action $a_{t+1}$ from policy: $\pi_{\theta}(s_{t+1})$

  

    3. Update the policy parameters: $\theta  \leftarrow  \theta + \alpha  \nabla_{\theta} \log  \pi_{\theta}(a_t | s_t) Q(s_t, a_t)$

  

4. Compute the correction (TD error) for action-value at time $t$:

$$\delta_t = r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$$

  

5. Use it to update the parameters of the action-value function:

$$w \leftarrow w + \beta  \delta_t  \nabla_w Q(s_t, a_t)$$

  

6. Update $\theta$ and $w$.

  

Two learning rates, $\alpha$ and $\beta$, are predefined for policy and
value function parameter updates, respectively.

  


<!-- ![image](\assets\images\course_notes\policy-based\a6.png){width="0.9\\linewidth"} -->

<center> 
<img src="\assets\images\course_notes\policy-based\a6.png"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>





???+ example "Actor-Critic Architecture: Cartpole Example"

    Let's illustrate the Actor-Critic architecture with an example of a
    classic reinforcement learning problem: the *Cartpole* environment.


    <center> 
    <img src="\assets\images\course_notes\policy-based\cartpole.png"
        alt="pi estimation with monte carlo"
        style="float: center; margin-right: 10px;" 
        /> 
        </center>

    ---

    In the *Cartpole* environment, the agent controls a cart that can move
    horizontally on a track. A pole is attached to the cart, and the agent's
    task is to balance the pole upright for as long as possible.

    

    1.  **Actor (Policy-Based)**: The actor is responsible for learning the
    policy, which is the agent's strategy for selecting actions (left or
    right) based on the observed state (cart position, cart velocity,
    pole angle, and pole angular velocity).

    

    2.  **Critic (Value-Based)**: The critic is responsible for learning the
    value function, which estimates the expected total reward (return)
    from each state. The value function helps evaluate how good or bad a
    specific state is, which guides the actor's updates.

    

    3.  **Policy Representation**: For simplicity, let's use a neural
    network as the actor. The neural network takes the current state of
    the cart and pole as input and outputs the probabilities of
    selecting actions (left or right).

    

    4.  **Value Function Representation**: For the critic, we also use a
    neural network. The neural network takes the current state as input
    and outputs an estimate of the expected total reward (value) for
    that state.

    



    1.  **Collecting Experiences**: The agent interacts with the
    environment, using the current policy to select actions (left or
    right). As it moves through the environment, it collects
    experiences, including states, actions, rewards, and next states.

    

    2.  **Updating the Critic (Value Function)**: The critic learns to
    estimate the value function using the collected experiences. It
    optimizes its neural network parameters to minimize the difference
    between the predicted values and the actual rewards experienced by
    the agent.

    

    3.  **Calculating the Advantage**: The advantage represents how much
    better or worse an action is compared to the average expected value.
    It is calculated as the difference between the total return (reward)
    and the value function estimate for each state-action pair.

    

    4.  **Updating the Actor (Policy)**: The actor updates its policy to
    increase the probabilities of actions with higher advantages and
    decrease the probabilities of actions with lower advantages. This
    process helps the actor learn from the critic's feedback and improve
    its policy to maximize the expected rewards.

    

    5.  **Iteration and Learning**: The learning process is repeated over
    multiple episodes and iterations. As the agent explores and
    interacts with the environment, the actor and critic networks
    gradually improve their performance and converge to better policies
    and value function estimates.

    

    Through these steps, the Actor-Critic architecture teaches the agent how
    to balance the pole effectively in the *Cartpole* environment. The actor
    learns the best actions to take in different states, while the critic
    provides feedback on the quality of the actor's decisions. As a result,
    the agent converges to a more optimal policy, achieving longer balancing
    times and better performance in the task.




### Summary of Variance Reduction Methods


  

To summarize, the key methods for reducing variance in policy gradient

methods include:

  

-  **Baseline Subtraction:** Subtracting an average return baseline to
reduce variance while keeping gradients unbiased.

  

-  **Causality Trick and Reward-to-Go:** Using future rewards from time
step $t$ onward to prevent variance from irrelevant past rewards.

  

-  **Discount Factor Adjustment:** Adjusting $\gamma$ to balance
variance reduction and long-term reward optimization.

  

-  **Advantage Estimation:** Using the advantage function $A(s_t, a_t)$
instead of raw returns to stabilize learning.

  

-  **Actor-Critic Methods:** Combining policy gradient updates with
value function estimation to create more stable and efficient
training.

  

By employing these techniques, policy gradient methods can achieve more
stable and efficient learning with reduced variance.

  

## Concluding Remarks

  

Now that we have seen the principles behind a policy-based algorithm,
let us see how policy-based algorithms work in practice, and compare
advantages and disadvantages of the policy-based approach.

  

Let us start with the advantages. First of all, parameterization is at
the core of policy-based methods, making them a good match for deep
learning. For value- based methods, deep learning had to be retrofitted,
giving rise to complications. Second, policy-based methods can easily
find stochastic policies, whereas value- based methods find
deterministic policies. Due to their stochastic nature, policy- based
methods naturally explore, without the need for methods such as
$\epsilon$-greedy, or more involved methods that may require tuning to
work well. Third, policy-based methods are effective in large or
continuous action spaces. Small changes in $\theta$ lead to small
changes in $\pi$, and to small changes in state distributions (they are
smooth). Policy-based algorithms do not suffer (as much) from
convergence and stability issues that are seen in $\arg\max$-based
algorithms in large or continuous action spaces.

  

On the other hand, there are disadvantages to the episodic Monte Carlo
version of the REINFORCE algorithm. Remember that REINFORCE generates a
full random episode in each iteration before it assesses the quality.
(Value-based methods use a reward to select the next action in each time
step of the episode.) Because of this, policy-based methods exhibit low
bias since full random trajectories are generated. However, they are
also high variance, since the full trajectory is generated randomly,
whereas value-based methods use the value for guidance at each selection
step.

  

What are the consequences? First, policy evaluation of full trajectories
has low sample efficiency and high variance. As a consequence, policy
improvement happens infrequently, leading to slow convergence compared
to value-based methods. Second, this approach often finds a local
optimum, since convergence to the global optimum takes too long.

  

Much research has been performed to address the high variance of the
episode- based vanilla policy gradient. The enhancements that have been
found have greatly improved performance, so much so that policy-based
approaches---such as A3C, PPO, SAC, and DDPG---have become favorite
model-free reinforcement learning algorithms for many applications.

## Author(s)

<div class="grid cards" markdown>
-   ![Instructor Avatar](/assets/images/staff/Nima-Shirzady.jpg){align=left width="150"}
    <span class="description">
        <p>**Nima Shirzady**</p>
        <p>Teaching Assistant</p>
        <p>[shirzady.1934@gmail.com](mailto:shirzady.1934@gmail.com)</p>
        <p>
        [:fontawesome-brands-github:](https://github.com/shirzady1934){:target="_blank"}
        [:fontawesome-brands-linkedin-in:](https://www.linkedin.com/in/shirzady){:target="_blank"}
        </p>
    </span>
-   ![Instructor Avatar](/assets/images/staff/Hamidreza-Ebrahimpour.jpg){align=left width="150"}
    <span class="description">
        <p>**Hamidreza Ebrahimpour**</p>
        <p>Teaching Assistant</p>
        <p>[ebrahimpour.7879@gmail.com](mailto:ebrahimpour.7879@gmail.com)</p>
        <p>
        [:fontawesome-brands-github:](https://github.com/hamidRezA7878){:target="_blank"}
        [:fontawesome-brands-linkedin-in:](https://linkedin.com/in/hamidreza-ebrahimpour78){:target="_blank"}
        </p>
    </span>
-   ![Instructor Avatar](/assets/images/staff/Hesam-Hosseini.jpg){align=left width="150"}
    <span class="description">
        <p>**Hesam Hosseini**</p>
        <p>Teaching Assistant</p>
        <p>[hesam138122@gmail.com](mailto:hesam138122@gmail.com)</p>
        <p>
        [:fontawesome-brands-google-scholar:](https://scholar.google.com/citations?user=ODTtV1gAAAAJ&hl=en){:target="_blank"}
        [:fontawesome-brands-github:](https://github.com/Sam-the-first){:target="_blank"}
        [:fontawesome-brands-linkedin-in:](https://www.linkedin.com/in/hesam-hosseini-b57092259){:target="_blank"}
        </p>
    </span>
    
</div>


# References

1. [Reinforcement Learning Explained](http://172.27.48.15/Resources/Books/Nikolic%20L.%20Reinforcement%20Learning%20Explained.%20A%20Step-by-Step%20Guide...2023.pdf)

2. [An Introduction to Deep Reinforcement Learning](http://172.27.48.15/Resources/Books/Textbooks/An%20Introduction%20to%20Deep%20Reinforcement%20Learning.pdf)

3. [Deep Reinforcement Learning Processor Design for Mobile Applications](http://172.27.48.15/Resources/Books/Juhyoung%20Lee%2C%20Hoi-Jun%20Yoo%20-%20Deep%20Reinforcement%20Learning%20Processor%20Design%20for%20Mobile%20Applications-Springer%20%282023%29.pdf)

4. [REINFORCE — a policy-gradient based reinforcement Learning algorithm](https://medium.com/intro-to-artificial-intelligence/reinforce-a-policy-gradient-based-reinforcement-learning-algorithm-84bde440c816)

5. [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

6. [Deep Reinforcement Learning](http://172.27.48.15/Resources/Books/Textbooks/Aske%20Plaat%20-%20Deep%20Reinforcement%20Learning-arXiv%20%282023%29.pdf)

7. [Reinforcement Learning (BartoSutton)](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
