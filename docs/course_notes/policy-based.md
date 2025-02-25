# Introduction

Reinforcement Learning (RL) focuses on training an agent to interact
with an environment by learning a policy $\pi_{\theta}(a | s)$ that
maximizes the cumulative reward. Policy gradient methods are a class of
algorithms that directly optimize the policy by adjusting the parameters
$\theta$ via gradient ascent.

## Why Policy Gradient Methods?

Unlike value-based methods (e.g., Q-learning), which rely on estimating
value functions, policy gradient methods:

-   Can naturally handle stochastic policies, which are crucial in
    environments requiring exploration.

-   Work well in continuous action spaces, where discrete action methods
    become infeasible.

-   Can directly optimize differentiable policy representations, such as
    neural networks.

# Deriving the Policy Gradient Theorem

The Policy Gradient Theorem provides a fundamental result in RL,
allowing us to express the gradient of the expected return $J(\theta)$
in terms of the policy function.

## Expected Return and Gradient

The objective in RL is to maximize the expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \gamma^t R_t \right],$$

where $\tau = (s_0, a_0, s_1, a_1, ...)$ represents a trajectory sampled
from the policy.

The gradient of $J(\theta)$ is:

$$\nabla_{\theta} J(\theta) = \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \gamma^t R_t \right].$$

## Likelihood Ratio Trick

Since the expectation is taken over trajectories sampled from
$\pi_{\theta}$, we apply the likelihood ratio trick:

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \gamma^t R_t \nabla_{\theta} \log P(\tau) \right].$$

Using the probability of a trajectory:

$$P(\tau) = P(s_0) \prod_{t=0}^{T} \pi_{\theta}(a_t | s_t) P(s_{t+1} | s_t, a_t),$$

the log-derivative simplifies to:

$$\nabla_{\theta} \log P(\tau) = \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t | s_t).$$

Thus, the policy gradient reduces to:

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \sum_{t=0}^{T} G_t \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) \right],$$

where:

$$G_t = \sum_{k=t}^{T} \gamma^{k-t} R_k.$$

This is the policy gradient theorem, which forms the basis for
REINFORCE.

# Continuous Action Spaces

For continuous action spaces, we typically use a Gaussian distribution:

$$\pi_{\theta}(a | s) = \mathcal{N}(\mu_{\theta}(s), \sigma_{\theta}^2).$$

The log-likelihood of the Gaussian policy is:

$$\log \pi_{\theta}(a | s) = -\frac{(a - \mu_{\theta}(s))^2}{2\sigma_{\theta}^2} - \log (\sqrt{2\pi} \sigma_{\theta}).$$

Thus, the policy gradient update is:

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \left( \frac{a - \mu_{\theta}(s)}{\sigma_{\theta}^2} \right) \nabla_{\theta} \mu_{\theta}(s) G_t \right].$$

# The REINFORCE Algorithm

## Algorithm Overview

The REINFORCE algorithm is a Monte Carlo policy gradient method that
uses complete episode returns to estimate the policy gradient.

**Steps of REINFORCE:**

1.  **Initialize** policy parameters $\theta$.

2.  **Collect an episode**: Run the policy $\pi_{\theta}$ and store
    $(s_t, a_t, r_t)$ for all time steps $t$.

3.  **Compute returns**: For each time step, compute:

    $$G_t = \sum_{k=t}^{T} \gamma^{k-t} R_k.$$

4.  **Policy Update**: Update the parameters:

    $$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) G_t.$$

5.  **Repeat** for multiple episodes.

## Challenges and Variance Reduction

**Baseline Subtraction:** Using a baseline $b(s_t)$ reduces variance:

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) (G_t - b(s_t)) \right].$$

A common choice is:

$$b(s_t) = V^{\pi}(s_t), \quad A(s_t, a_t) = G_t - V^{\pi}(s_t).$$

## Entropy Regularization

To encourage exploration, we introduce entropy regularization:

$$J_{\text{entropy}}(\theta) = J(\theta) + \beta H(\pi_{\theta}),$$

where:

$$H(\pi_{\theta}) = - \sum_{a} \pi_{\theta}(a | s) \log \pi_{\theta}(a | s).$$

## Natural Policy Gradient

Instead of using vanilla gradient ascent, we use the natural gradient:

$$\nabla_{\theta}^{\text{nat}} J(\theta) = F^{-1} \nabla_{\theta} J(\theta),$$

where $F$ is the Fisher Information Matrix.

# Bias in Policy Gradient Methods

Bias in RL occurs when an estimator systematically deviates from the
true value. In policy gradient methods, bias arises due to function
approximation, reward estimation, or gradient computation.

## Sources of Bias

-   **Function Approximation Bias:** When neural networks or linear
    functions approximate the policy or value function, they may
    introduce systematic errors. For instance, underestimating a value
    function can lead to suboptimal policy updates.

-   **Reward Clipping or Discounting:** Some algorithms use clipped
    rewards or high discount factors ($\gamma$), which introduce bias in
    estimating long-term returns.

-   **Baseline Approximation:** The use of an estimated baseline (e.g.,
    $V^{\pi}(s)$) in variance reduction can introduce bias if the
    baseline is poorly estimated.

## Example of Bias

Consider a self-driving car learning to optimize fuel efficiency. If the
reward function overemphasizes immediate fuel consumption rather than
long-term efficiency, the learned policy may prioritize short-term gains
while missing globally optimal strategies, leading to biased learning.

# Variance in Policy Gradient Methods

Variance in policy gradient estimates refers to the fluctuation in
gradient estimates across different training episodes. High variance can
lead to instability and slow convergence.

## Sources of Variance

-   **Monte Carlo Estimation:** The REINFORCE algorithm computes
    gradients based on entire episodes, leading to high variance due to
    random sampling of trajectories.

-   **Stochastic Policy Outputs:** Policies represented as probability
    distributions (e.g., Gaussian policies) can introduce randomness in
    gradient updates.

-   **Exploration Strategies:** Random action selection, such as using
    softmax or epsilon-greedy exploration, increases variability in
    learning updates.

## Example of Variance

Consider a robotic arm learning to pick up objects. Due to high
variance, in some training episodes it may accidentally grasp the object
correctly, while in others it fails due to slight variations in initial
positioning. These fluctuations in learning updates slow down
convergence.

# Monte Carlo Estimators in Reinforcement Learning

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

By the law of large numbers (LLN), as $N \to \infty$, we have:

$$\hat{X}_N \to \mathbb{E}[X] \quad \text{(almost surely)}$$

Monte Carlo methods are commonly used in RL for estimating expected
rewards, state-value functions, and action-value functions.

# Biased vs. Unbiased Estimation

The biased formula for the sample variance $S^2$ is given by:

$$S^2_{\text{biased}} = \frac{1}{n} \sum_{i=1}^{n} (X_i - \overline{X})^2$$

This is an underestimation of the true population variance $\sigma^2$
because it does not account for the degrees of freedom in estimation.
Instead, the unbiased estimator is:

$$S^2_{\text{unbiased}} = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \overline{X})^2.$$

This unbiased estimator correctly accounts for variance in small sample
sizes, ensuring $\mathbb{E}[S^2_{\text{unbiased}}] = \sigma^2$.

# Balancing Bias and Variance

Reducing bias often increases variance, and vice versa. The goal is to
find a balance between the two.

## Strategies for Bias Reduction

-   Using more expressive function approximators (e.g., deeper neural
    networks).

-   Improving reward estimation techniques (e.g., using learned value
    functions).

## Strategies for Variance Reduction

-   **Baseline Subtraction:** Introducing a baseline function $b(s_t)$
    to reduce variance without affecting bias.

-   **Reward-to-Go:** Instead of using the full return, using the
    reward-to-go estimator reduces variance.

-   **Actor-Critic Methods:** Combining value function estimation with
    policy updates stabilizes learning.
