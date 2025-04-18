## Notation

| Symbol | Meaning |
|--------|---------|
| $s \in \mathcal{S}$ | States. |
| $a \in \mathcal{A}$ | Actions. |
| $r \in \mathcal{R}$ | Rewards. |
| $S_t, A_t, R_t$ | State, action, and reward at time step $t$ of one trajectory. I may occasionally use $s_t, a_t, r_t$ as well. |
| $\gamma$ | Discount factor; penalty to uncertainty of future rewards; $0 < \gamma \leq 1$. |
| $G_t$ | Return; or discounted future reward; $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$. |
| $P(s',r\|s,a)$ | Transition probability of getting to the next state $s'$ from the current state $s$ with action $a$ and reward $r$. |
| $\pi(a\|s)$ | Stochastic policy (agent behavior strategy); $\pi_\theta(\cdot)$ is a policy parameterized by $\theta$. |
| $\mu(s)$ | Deterministic policy; we can also label this as $\pi(s)$, but using a different letter gives better distinction so that we can easily tell when the policy is stochastic or deterministic without further explanation. Either $\pi$ or $\mu$ is what a reinforcement learning algorithm aims to learn. |
| $V(s)$ | State-value function measures the expected return of state $s$; $V_w(\cdot)$ is a value function parameterized by $w$. |
| $V^\pi(s)$ | The value of state $s$ when we follow a policy $\pi$; $V^\pi(s) = \mathbb{E}_{a\sim\pi}[G_t\|S_t = s]$. |
| $Q(s,a)$ | Action-value function is similar to $V(s)$, but it assesses the expected return of a pair of state and action $(s,a)$; $Q_w(\cdot)$ is a action value function parameterized by $w$. |
| $Q^\pi(s,a)$ | Similar to $V^\pi(\cdot)$, the value of (state, action) pair when we follow a policy $\pi$; $Q^\pi(s,a) = \mathbb{E}_{a\sim\pi}[G_t\|S_t = s, A_t = a]$. |
| $A(s, a)$ | Advantage function, $A(s, a) = Q(s,a) - V(s)$;  it can be considered as another version of Q-value with lower variance by taking the state-value off as the baseline.

# Policy Gradient


<!-- ## What Are Policy Gradient Methods?   -->

Policy Gradient (PG) methods are a class of **Reinforcement Learning (RL)** algorithms that **optimize a policy $\pi_{\theta}(a | s)$ directly**  by following the gradient of expected reward with respect to the policy parameters $\theta$. Unlike value-based methods (e.g., Q-Learning), which learn a value function and derive a policy from it, PG methods **parameterize the policy** and adjust its parameters to maximize the expected return.  

### The Main Idea  

The core idea behind Policy Gradients is simple:  


!!! abstract ""
    **If an action leads to a better outcome, make it more likely; if it leads to a worse outcome, make it less likely.**


Instead of estimating value functions and selecting actions based on them, PG methods **directly learn a policy** and update it using gradient ascent on the expected return.  

### Why Use Policy Gradients?  

1. **Natural for Stochastic Policies**: Some problems require stochasticity (e.g., games with imperfect information). PG methods handle this naturally.  
2. **Continuous Action Spaces**: Unlike Q-learning, which struggles with high-dimensional or continuous actions, PG methods can easily parameterize policies for such spaces.  
3. **Better Convergence Properties**: While not always guaranteed, PG methods often exhibit more stable convergence compared to value-based methods in certain environments.  
4. **Direct Optimization**: Since PG methods optimize the policy directly, they can avoid some pitfalls of value-based approaches (e.g., overestimation bias in Q-learning).  

### Advantages of Policy Gradient Methods  

<center>

| **Advantage** | **Description** |
|--------------|----------------|
| **Handles High-Dimensional Actions** | Works well in continuous spaces (e.g., robotics). |
| **Stochastic Policies** | Can represent optimal stochastic policies naturally. |
| **No Max-Operator Issues** | Avoids maximization bias present in Q-learning. |
| **Compatible with Neural Networks** | Easily integrates with deep learning frameworks. |

</center>
 

## Policy Gradient Theorem
The objective in RL is to maximize the **expected return** $J(\theta)$, where $\theta$ are the policy parameters.  

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]
$$

Here, $\tau$ is a trajectory, and $R(\tau)$ is the total reward. The policy gradient theorem provides a way to compute the gradient of $J(\theta)$ (i.e. $\nabla_\theta J(\theta)$). This gradient tells us how to adjust $\theta$ to increase the likelihood of high-reward trajectories. 
Rewriting the expectation over trajectories using the stationary distribution of states:

$$J(\theta) = \sum_{s \in  \mathcal{S}} d^{\pi}(s) V^{\pi}(s) = \sum_{s \in  \mathcal{S}} d^{\pi}(s) \sum_{a \in  \mathcal{A}} \pi_{\theta}(a|s) Q^{\pi}(s,a)$$

  

where $d^{\pi}(s)$ is the stationary distribution of Markov chain for
$\pi_{\theta}$ (on-policy state distribution under $\pi$). For
simplicity, the parameter $\theta$ would be omitted for the policy
$\pi_{\theta}$ when the policy is present in the subscript of other
functions; for example, $d^{\pi}$ and $Q^{\pi}$ should be
$d^{\pi_{\theta}}$ and $Q^{\pi_{\theta}}$ if written in full.

The policy gradient theorem provides a way to compute the gradient of $J(\theta)$:  

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \, R(\tau) \right]
$$

This gradient tells us how to adjust $\theta$ to increase the likelihood of high-reward trajectories.  

- $\nabla_\theta \log \pi_\theta(a_t | s_t)$ is the **direction to increase the probability** of action $a_t$ in state $s_t$.  
- $R(\tau)$ **scales** this update:  
    - If the reward is high, the action is reinforced.  
    - If the reward is low, the action is suppressed.  




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

  



  

## Policy Gradient in Continuous Action Space 

  

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

  
## The Policy Gradient Theorem: Estimation, Bias, and Variance  

So far, we derived the **Policy Gradient Theorem**:  

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \, R(\tau) \right]
$$

This gives us the *theoretical* gradient for improving our policy. However, in practice, we **never compute the exact expectation**—instead, we **estimate it from samples**. This turns policy gradient into a **statistical estimation problem**, where we must carefully consider:  

1. **Bias**: Does our estimator systematically over/underestimate the true gradient?  
2. **Variance**: How noisy is our estimator?  

### Why Does This Matter?  

- **High Bias** ⟶ The policy may converge to a suboptimal solution.  
- **High Variance** ⟶ Training becomes unstable; many samples are needed for reliable updates.  


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

  

-  **Monte Carlo Estimation:** Some algorithms (e.g. REINFORCE) estimates gradients using complete episodes, leading to high variance due to trajectory randomness.

  

-  **Stochastic Policy Outputs:** Policies represented as probability
distributions (e.g., Gaussian policies) introduce additional
randomness in gradient updates.

  

-  **Exploration Strategies:** Methods like softmax or epsilon-greedy
increase variance by adding stochasticity to action selection.



???+ example "Example of Variance"
    Consider a robotic arm learning to grasp objects. Due to high variance,
    in some episodes, it succeeds, while in others, minor variations cause
    failure. These inconsistencies slow down convergence.

## Policy Gradient Algorithms

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

  

!!! note "REINFORCE Algorithm" 
    **for trajectory $i$:**

    - sample $\left\{\tau^i\right\}$ from $\pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right)$ (run the policy)

    - $\nabla_\theta J(\theta) \approx \sum_i\left(\sum_t \nabla_\theta \log \pi_\theta\left(\mathbf{a}_t^i \mid \mathbf{s}_t^i\right)\right)\left(\sum_t r\left(\mathbf{s}_t^i, \mathbf{a}_t^i\right)\right)$

    - $\theta \leftarrow \theta+\alpha \nabla_\theta J(\theta)$

#### Understanding the Variance Problem in REINFORCE

In the REINFORCE algorithm's gradient estimate:

$$\nabla_\theta J(\theta) \approx \sum_i\left(\underbrace{\sum_t \nabla_\theta \log \pi_\theta\left(\mathbf{a}_t^i \mid \mathbf{s}_t^i\right)}_{\text{policy score}}\right)\left(\underbrace{\sum_t r\left(\mathbf{s}_t^i, \mathbf{a}_t^i\right)}_{\text{total trajectory reward}}\right)$$

While being an **unbiased estimation**, it suffers from high variance problem. The second term - the total trajectory reward $\sum_t r(\mathbf{s}_t^i, \mathbf{a}_t^i)$ - is the primary source of high variance because:

1. **Cumulative Nature**: Small differences in individual rewards compound exponentially over a trajectory's length
2. **No Relative Scaling**: Absolute reward magnitudes directly affect gradient step sizes
3. **Trajectory-Wide Credit Assignment**: All actions in the trajectory get weighted by the same total reward, regardless of their actual contribution

???+ example "Example of High Variance"

    Consider two scenarios:

    - A trajectory with small positive rewards summing to +100
    - A trajectory with large alternating (+1000, -900) rewards also summing to +100

    While both have the same total reward, the second case's individual transitions have much higher variance. The gradient estimate weights all actions by this unstable total reward signal.

!!! danger

    The variance grows because:

    1. The return $R(\tau) = \sum_{t=0}^T r_t$ accumulates variance from each timestep

    2. For long horizons, the variance scales approximately with $O(T^2)$
    Here's a concise derivation showing why the variance of Monte Carlo policy gradient estimates scales as $O(T^2)$ with horizon length $T$:

        - **Variance of Monte Carlo Estimation**

            Consider the **total return** over a trajectory:

            $$
            R(\tau) = \sum_{t=0}^{T-1} r_t
            $$

            The **policy gradient estimator** is:

            $$
            g = \sum_{t=0}^{T-1} \left( \nabla_\theta \log \pi_\theta(a_t|s_t) \right) R(\tau)
            $$

        - **Reward Covariance Structure**  

            Assume rewards are correlated across timesteps (true in RL), with:

            $$
            \text{Var}(r_t) = \sigma^2, \quad \text{Cov}(r_t, r_{t'}) = \rho \sigma^2 \quad (\rho > 0)
            $$

        - **Variance of Total Return**  

            $$
            \text{Var}(R(\tau)) = \sum_{t=0}^{T-1} \text{Var}(r_t) + 2 \sum_{t<t'} \text{Cov}(r_t, r_{t'}) = T \sigma^2 + T(T-1) \rho \sigma^2
            $$

            For large $T$, this dominates as:

            $$
            \text{Var}(R(\tau)) \approx \rho \sigma^2 T^2 \quad \text{(since } T^2 \gg T \text{)}
            $$

        - **Policy Gradient Variance**  

            The score function $\nabla_\theta \log \pi_\theta$ is typically correlated with $R(\tau)$. If we approximate $\text{Var}(g) \propto \text{Var}(R(\tau))$ (since the score function terms are multipliers), we get:
            
            $$
            \text{Var}(g) = O(T^2)
            $$


    3. The gradient estimate multiplies this already-high-variance term by another stochastic quantity (the policy score)



This explains why vanilla REINFORCE often requires an impractical number of samples to converge reliably in complex environments. Several strategies help mitigate variance in policy gradient methods
while preserving unbiased gradient estimates.


#### Variance Reduction: Baseline Subtraction

  

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
    &= 0 \quad ;\text{No bias is introduced}
    \end{aligned}$$

???+ example "Example"

    **Imagine grading students on a curve:**  

    - Without a baseline (absolute grading): Score = 950 (A+) or 920 (A) → Huge difference in updates  

    - With a baseline (grading relative to average=900): Score = 50 (A+) or 20 (A) → More reasonable updates  

The baseline centers the rewards around their average value, so:  
1. **Smaller magnitude** → Less extreme gradient steps  
2. **Relative comparisons preserved** → Good actions still get boosted more than bad ones  
3. **Same expected gradient** (unbiased) but with less wild fluctuation  

!!! note "Intuition"

    It's like looking at *how much better than average* each action was, rather than its raw score.



<!-- ![image](\assets\images\course_notes\policy-based\a4.png){width="0.8\\linewidth"} -->

<center> 
<img src="\assets\images\course_notes\policy-based\a4.png"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>


  

#### Variance Reduction: Causality Trick and Reward-to-Go Estimation

  

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
    \mathbb{E}_{A_{t_0-1}}& \left[ \mathbb{E}_{s_{t_0}, a_{t_0} | A_{t_0-1}} \left[ \nabla_{\theta} \log \pi_{\theta} (a_{t_0} | s_{t_0}) \sum_{t=0}^{t_0 - 1} r(s_t, a_t) \right] \right] = 0 \quad ;\text{No bias is introduced}
    \end{aligned}
    $$


  

#### Variance Reduction: Discount Factor Adjustment

  

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

  

#### Variance Reduction: Advantage Estimation and Actor-Critic Methods

  

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

#### **N-Step Returns and Generalized Advantage Estimation (GAE)**

*The Bias-Variance Tradeoff in Policy Gradients*

Policy gradient methods rely on estimating the expected return to guide policy updates. However, different ways of estimating this return lead to different bias-variance properties:

1. **Monte Carlo (full-episode) returns**  

    - **Low bias** (correct in expectation)  
    - **High variance** (depends on entire trajectory)  

2. **1-step TD (Temporal Difference) returns**  

    - **High bias** (bootstraps from a value estimate)  
    - **Low variance** (only depends on one transition)  

**N-step returns** provide a middle ground between these extremes.

#### N-Step Returns: Balancing Bias and Variance
The **N-step return** combines the first *N* steps of actual rewards and then bootstraps the rest using a value function:

$$
R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})
$$

- **When $n=1$:** Pure TD (high bias, low variance)  
- **When $n=\infty$:** Monte Carlo (low bias, high variance)  
- **Intermediate $n$:** Smooth interpolation  

#### Why N-Step Helps?

- **Reduces variance** compared to full Monte Carlo (shorter reward sequences).  
- **Reduces bias** compared to 1-step TD (less reliance on bootstrapping).  


#### **Generalized Advantage Estimation (GAE): The Best of All Worlds**
Instead of choosing a single *N*, **GAE** combines *all* N-step returns using an exponential weighting scheme controlled by a parameter $\lambda \in [0,1]$:

$$
A_t^{GAE(\gamma, \lambda)} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

#### Key Properties of GAE

- **$\lambda = 0$ → Pure TD (1-step advantage)**  
  - Highest bias, lowest variance.  
- **$\lambda = 1$ → Monte Carlo advantage**  
  - Lowest bias, highest variance.  
- **Intermediate $\lambda$ (e.g., 0.9-0.99)**  
  - Balances bias and variance effectively.  

#### Why GAE is Powerful

1. **Smoothly interpolates between TD and Monte Carlo**  
2. **Automatically weights different N-step advantages**  
3. **One hyperparameter ($\lambda$) controls bias-variance tradeoff**  


### **Actor-Critic Methods: Bridging Policy and Value Learning**  

#### **From REINFORCE to Actor-Critic**  
Policy gradient methods like REINFORCE rely on **Monte Carlo returns**, which are unbiased but **high-variance**.  
Actor-Critic methods address this by:  

1. **Learning a value function (Critic)** to estimate expected returns more efficiently.  
2. **Using this estimate (Advantage)** to guide policy (Actor) updates with lower variance.  

This creates a **two-network system**:  

- **Actor (Policy Network)** → Decides which actions to take.  
- **Critic (Value Network)** → Evaluates how good states/actions are.  

<center>

## **Why Actor-Critic?**  
| Method          | Pros | Cons |  
|----------------|------|------|  
| **REINFORCE**  | Simple, unbiased | High variance, slow learning |  
| **Pure Value-Based (e.g., DQN)**  | Stable, efficient | Struggles with continuous actions |  
| **Actor-Critic**  | Best of both: **Low variance + handles continuous actions** | More complex, risk of instability |  

</center>

### **Key Insight**  
The Critic **reduces variance** by replacing noisy Monte Carlo returns with a learned value estimate:  


$$
\nabla_\theta J(\theta) \approx \mathbb{E} \left[ \nabla_\theta \underbrace{\log \pi_\theta(a_t|s_t)}_{\text{Actor}} \cdot \underbrace{(Q(s_t,a_t) - V(s_t))}_{\text{Advantage (Critic)}} \right]
$$  

- If **Advantage > 0**, the action was better than expected → increase its probability.  
- If **Advantage < 0**, the action was worse → decrease its probability.  

## **Types of Actor-Critic Methods**  

1. **Vanilla Actor-Critic**  

    - Uses **1-step TD advantage** (low variance, but biased).  

2. **Advantage Actor-Critic (A2C/A3C)**  

    - Parallel agents + multi-step returns for stability.  

3. **Trust Region Methods (e.g., PPO, TRPO)**  

    - Constrain policy updates to prevent instability.  


#### Why Actor-Critic Dominates Modern RL  

- ✅ **Lower variance** than pure policy gradients.  
- ✅ **More sample-efficient** than value-based methods.  
- ✅ **Handles continuous action spaces** naturally.  

!!! danger

    **Tradeoff:** Requires tuning two networks (Actor + Critic), which can be unstable without proper tricks (e.g., target networks, policy constraints).  



Here is a nice summary of a general form of policy gradient methods borrowed from the GAE (general advantage estimation) paper [Schulman et al., 2016](https://arxiv.org/abs/1506.02438)



<center> 
<img src="\assets\images\course_notes\policy-based\general_form_policy_gradient.png"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>

### **Advanced Policy Gradients: From REINFORCE to Natural Policy Gradient**

In this section, we'll build upon the foundational *REINFORCE* algorithm and explore more advanced techniques that address its limitations, culminating in the Natural Policy Gradient method.

#### Recap: The Policy Gradient Update Using the Advantage Function

Before we dive into advanced methods, let's recall the basic policy gradient update using the advantage function:


$$\nabla_{\theta} J(\theta) \approx  \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log  \pi_{\theta}(a_{i,t} | s_{i,t}) A^{\pi}(s_{i,t}, a_{i,t}).$$


#### **Policy Gradient as Policy Iteration**

An insightful perspective is viewing policy gradient methods as approximate policy iteration:

1. **Evaluate** the current policy by estimating advantages $A^{\pi}(s_{i,t}, a_{i,t})$
2. **Improve** the policy using these estimates

This connects to the classic policy iteration algorithm, but with function approximation.

#### Policy Improvement

We begin by recalling the definition of the expected return (objective function) under a parameterized policy $\pi_{\theta}$:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]
$$

Now, consider the scenario where we update the policy from $\pi_{\theta}$ to a new policy $\pi_{\theta'}$. The difference in expected returns, which quantifies the effect of this update, is given by:

$$
J(\theta') - J(\theta) = \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[\sum_{t=0}^{T} \gamma^t A^{\pi_\theta}(s_t, a_t)\right]
$$

This equation expresses the **policy improvement step**; it tells us that the expected return of the updated policy $\pi_{\theta'}$ is improved if actions that have a positive advantage (i.e., better than expected under $\pi_\theta$) are favored more in $\pi_{\theta'}$.


??? note "proof"

    $$
    \begin{aligned}
    &\text{claim}: J\left(\theta^{\prime}\right)-J(\theta)=E_{\tau \sim p_{\theta^{\prime}}(\tau)}\left[\sum_t \gamma^t A^{\pi_\theta}\left(\mathbf{s}_t, \mathbf{a}_t\right)\right]\\
    &\begin{aligned}
    J\left(\theta^{\prime}\right)-J(\theta) & =J\left(\theta^{\prime}\right)-E_{\mathbf{s}_0 \sim p\left(\mathbf{s}_0\right)}\left[V^{\pi_\theta}\left(\mathbf{s}_0\right)\right] \\
    & =J\left(\theta^{\prime}\right)-E_{\tau \sim p_{\theta^{\prime}}(\tau)}\left[V^{\pi_\theta}\left(\mathbf{s}_0\right)\right] \\
    & =J\left(\theta^{\prime}\right)-E_{\tau \sim p_{\theta^{\prime}}(\tau)}\left[\sum_{t=0}^{\infty} \gamma^t V^{\pi_\theta}\left(\mathbf{s}_t\right)-\sum_{t=1}^{\infty} \gamma^t V^{\pi_\theta}\left(\mathbf{s}_t\right)\right] \\
    & =J\left(\theta^{\prime}\right)+E_{\tau \sim p_{\theta^{\prime}}(\tau)}\left[\sum_{t=0}^{\infty} \gamma^t\left(\gamma V^{\pi_\theta}\left(\mathbf{s}_{t+1}\right)-V^{\pi_\theta}\left(\mathbf{s}_t\right)\right)\right] \\
    & =E_{\tau \sim p_{\theta^{\prime}}(\tau)}\left[\sum_{t=0}^{\infty} \gamma^t r\left(\mathbf{s}_t, \mathbf{a}_t\right)\right]+E_{\tau \sim p_{\theta^{\prime}}(\tau)}\left[\sum_{t=0}^{\infty} \gamma^t\left(\gamma V^{\pi_\theta}\left(\mathbf{s}_{t+1}\right)-V^{\pi_\theta}\left(\mathbf{s}_t\right)\right)\right] \\
    & =E_{\tau \sim p_{\theta^{\prime}}(\tau)}\left[\sum_{t=0}^{\infty} \gamma^t\left(r\left(\mathbf{s}_t, \mathbf{a}_t\right)+\gamma V^{\pi_\theta}\left(\mathbf{s}_{t+1}\right)-V^{\pi_\theta}\left(\mathbf{s}_t\right)\right)\right] \\
    & =E_{\tau \sim p_{\theta^{\prime}}(\tau)}\left[\sum_{t=0}^{\infty} \gamma^t A^{\pi_\theta}\left(\mathbf{s}_t, \mathbf{a}_t\right)\right]
    \end{aligned}
    \end{aligned}
    $$


The challenge in computing this expectation arises because it is taken over the state distribution induced by the new policy $\pi_{\theta'}$, whereas our available samples come from the original policy $\pi_{\theta}$. 

To address this discrepancy, we can apply **importance sampling**, allowing us to express the expectation in terms of samples from $\pi_{\theta}$:

$$
\begin{aligned}
\mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[\sum_{t=0}^{T} \gamma^t A^{\pi_\theta}(s_t, a_t)\right] &= \sum_t \mathbb{E}_{s_t \sim p_{\theta'}(s_t)}\left[\mathbb{E}_{a_t \sim \pi_{\theta'}(a_t|s_t)}\left[ \gamma^t A^{\pi_\theta}(s_t, a_t)\right]\right] \\
&= \sum_t \mathbb{E}_{s_t \sim p_{\theta'}(s_t)}\left[\mathbb{E}_{a_t \sim \pi_{\theta}(a_t|s_t)}\left[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t)\right]\right] 
\end{aligned}
$$

However, this expression still depends on the state distribution $p_{\theta'}(s_t)$, which we do not have direct samples from.

The key question is: **Can we approximate this expectation by ignoring the discrepancy between $p_{\theta'}$ and $p_{\theta}$?** 
That is, can we make the following approximation?  

$$
\sum_t \mathbb{E}_{s_t \sim p_{\theta'}(s_t)}\left[\mathbb{E}_{a_t \sim \pi_{\theta}(a_t|s_t)}\left[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t)\right]\right] \approx \sum_t \mathbb{E}_{s_t \sim p_{\theta}(s_t)}\left[\mathbb{E}_{a_t \sim \pi_{\theta}(a_t|s_t)}\left[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t)\right]\right] 
$$

The approximation holds **if** the state distributions induced by the two policies are similar, i.e., $p_{\theta'}(s_t) \approx p_{\theta}(s_t)$. The critical observation is that if the new policy $\pi_{\theta'}$ is sufficiently close to the old policy $\pi_{\theta}$, then their corresponding state distributions will also be close.  

??? note "proof"
    **Claim**: $p_\theta\left(\mathbf{s}_t\right)$ is close to $p_{\theta^{\prime}}\left(\mathbf{s}_t\right)$ when $\pi_\theta$ is close to $\pi_{\theta^{\prime}}$

    **Simple case**: assume $\pi_\theta$ is a deterministic policy $\mathbf{a}_t=\pi_\theta\left(\mathbf{s}_t\right)$

    $\pi_{\theta^{\prime}}$ is close to $\pi_\theta$ if $\pi_{\theta^{\prime}}\left(\mathbf{a}_t \neq \pi_\theta\left(\mathbf{s}_t\right) \mid \mathbf{s}_t\right) \leq \epsilon$

    $$
    p_{\theta^{\prime}}\left(\mathbf{s}_t\right)=(\underbrace{(1-\epsilon)^t}_{\text{no mistakes probability}} p_\theta\left(\mathbf{s}_t\right)+\left(1-(1-\epsilon)^t\right)) \underbrace{p_{\text {mistake}}(\mathbf{s}_t)}_{\text{some other distribution}}
    $$

    $$
    \left|p_{\theta^{\prime}}\left(\mathbf{s}_t\right)-p_\theta\left(\mathbf{s}_t\right)\right|=\left(1-(1-\epsilon)^t\right)\left|p_{\text {mistake }}\left(\mathbf{s}_t\right)-p_\theta\left(\mathbf{s}_t\right)\right| \leq 2\left(1-(1-\epsilon)^t\right) \leq 2 \epsilon t
    $$

    useful identity:
     $(1-\epsilon)^t \geq 1-\epsilon t$ for $\epsilon \in[0,1]$

    **not a great bound, but a bound!**


    **General case**: assume $\pi_\theta$ is an arbitrary distribution $\pi_{\theta^{\prime}}$ is close to $\pi_\theta$ if $\left|\pi_{\theta^{\prime}}\left(\mathbf{a}_t \mid \mathbf{s}_t\right)-\pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right)\right| \leq \epsilon$ for all $\mathbf{s}_t$

    Useful lemma: if $\left|p_X(x)-p_Y(x)\right|=\epsilon$, exists $p(x, y)$ such that $p(x)=p_X(x)$ and $p(y)=p_Y(y)$ and $p(x=y)=1-\epsilon$

    $\Rightarrow p_X(x)$ "agrees" with $p_Y(y)$ with probability $\epsilon$

    $\Rightarrow \pi_{\theta^{\prime}}\left(\mathbf{a}_t \mid \mathbf{s}_t\right)$ takes a different action than $\pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right)$ with probability at most $\epsilon$

    $$
    \begin{aligned}
    \left|p_{\theta^{\prime}}\left(\mathbf{s}_t\right)-p_\theta\left(\mathbf{s}_t\right)\right|=\left(1-(1-\epsilon)^t\right)\left|p_{\text {mistake }}\left(\mathbf{s}_t\right)-p_\theta\left(\mathbf{s}_t\right)\right| & \leq 2\left(1-(1-\epsilon)^t\right) \\
    & \leq 2 \epsilon t
    \end{aligned}
    $$


    $\pi_{\theta^{\prime}}$ is close to $\pi_\theta$ if $\left|\pi_{\theta^{\prime}}\left(\mathbf{a}_t \mid \mathbf{s}_t\right)-\pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right)\right| \leq \epsilon$ for all $\mathbf{s}_t$


    $$
    \begin{aligned}
    &\left|p_{\theta^{\prime}}\left(\mathbf{s}_t\right)-p_\theta\left(\mathbf{s}_t\right)\right| \leq 2 \epsilon t \\
    & E_{p_{\theta^{\prime}}\left(\mathbf{s}_t\right)}\left[f\left(\mathbf{s}_t\right)\right]=\sum_{\mathbf{s}_t} p_{\theta^{\prime}}\left(\mathbf{s}_t\right) f\left(\mathbf{s}_t\right) \geq \sum_{\mathbf{s}_t} p_\theta\left(\mathbf{s}_t\right) f\left(\mathbf{s}_t\right)-\left|p_\theta\left(\mathbf{s}_t\right)-p_{\theta^{\prime}}\left(\mathbf{s}_t\right)\right| \max _{\mathbf{s}_t} f\left(\mathbf{s}_t\right) \\
    & \geq E_{p_\theta\left(\mathbf{s}_t\right)}\left[f\left(\mathbf{s}_t\right)\right]-2 \epsilon t \max _{\mathbf{s}_t} f\left(\mathbf{s}_t\right) \\
    & \sum_t E_{\mathbf{s}_t \sim p_{\theta^{\prime}}\left(\mathbf{s}_t\right)}\left[E_{\mathbf{a}_t \sim \pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right)}\left[\frac{\pi_{\theta^{\prime}}\left(\mathbf{a}_t \mid \mathbf{s}_t\right)}{\pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right)} \gamma^t A^{\pi_\theta}\left(\mathbf{s}_t, \mathbf{a}_t\right)\right]\right] \geq \\
    & \sum_t E_{\mathbf{s}_t \sim p_{\theta}\left(\mathbf{s}_t\right)}\left[E_{\mathbf{a}_t \sim \pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right)}\left[\frac{\pi_{\theta^{\prime}}\left(\mathbf{a}_t \mid \mathbf{s}_t\right)}{\pi_\theta\left(\mathbf{a}_t \mid \mathbf{s}_t\right)} \gamma^t A^{\pi_\theta}\left(\mathbf{s}_t, \mathbf{a}_t\right)\right]\right] - \sum_t 2\epsilon t C \quad ;C = O(T r_{max}) \text{ or } O(\frac{r_{max}}{1-\gamma})
    \end{aligned}
    $$


    maximizing this maximizes a bound on the thing we want!
    
Thus, when policy updates are small, this assumption is reasonable, and the approximation allows us to compute the expectation using samples from $p_{\theta}(s_t)$ instead of requiring samples from $p_{\theta'}(s_t)$.


#### Constrained Policy Optimization

This leads to a constrained optimization problem:

$$
\theta' = \arg\max_{\theta'} \underbrace{\sum_t \mathbb{E}_{s_t \sim p_\theta(s_t)}\left[\mathbb{E}_{a_t \sim \pi_\theta(a_t|s_t)}\left[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t)\right]\right]}_{\bar{A}(\theta')}
$$

$$
\text{such that } D_{KL}(\pi_{\theta'}(\cdot|s_t) \| \pi_\theta(\cdot|s_t)) \leq \epsilon
$$

The KL divergence constraint ensures policies remain close, making our approximation valid. For small enough $\epsilon$, this is **guaranteed** to improve $J(\theta') - J(\theta)$

#### Solving the Constrained Problem

We can solve this using Lagrangian duality:

$$
\mathcal{L}(\theta', \lambda) = \text{objective} - \lambda (D_{KL}(\pi_{\theta'}\|\pi_\theta) - \epsilon)
$$

1. Maximize $\mathcal{L} \text{ w.r.t. } \theta'$ (partial optimization)
2. Update $\lambda$ based on constraint violation:

$$
   \lambda \leftarrow \lambda + \alpha (D_{KL} - \epsilon)
$$

### **Natural Policy Gradient**  

Instead of performing unconstrained gradient ascent on the policy objective, the **Natural Policy Gradient** approach optimizes a local **linear approximation** of the advantage while ensuring that the new policy does not deviate too much from the current policy in terms of KL divergence. This is formulated as the following constrained optimization problem:  

$$
\theta' = \arg\max_{\theta'} \nabla_\theta \bar{A}(\theta)^T (\theta' - \theta)
$$

$$
\text{s.t. } D_{KL}(\pi_{\theta'}\|\pi_\theta) \leq \epsilon
$$

Where $\bar{A}(\theta') = \sum_t \mathbb{E}_{s_t \sim p_\theta(s_t)}\left[\mathbb{E}_{a_t \sim \pi_\theta(a_t|s_t)}\left[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t)\right]\right]$
The objective aims to find a new parameter $\theta'$ that maximally improves the policy while ensuring that the policy update remains within a **trust region** to prevent drastic changes.

We now show that the gradient of the expected advantage function is equal to the gradient of the policy objective $J(\theta)$, i.e.,  

$$
\nabla_\theta \bar{A}(\theta) = \nabla_\theta J(\theta).
$$


??? note "proof"

    $$
    \begin{aligned}
    \nabla_{\theta'} \bar{A}(\theta') & = \nabla_{\theta'} \sum_t \mathbb{E}_{s_t \sim p_{\theta}(s_t)}\left[\mathbb{E}_{a_t \sim \pi_{\theta}(a_t|s_t)}\left[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \gamma^t A^{\pi_\theta}(s_t, a_t)\right]\right] \\
    & = \sum_t \mathbb{E}_{s_t \sim p_{\theta}(s_t)}\left[\mathbb{E}_{a_t \sim \pi_{\theta}(a_t|s_t)}\left[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \gamma^t \nabla_{\theta'}\log \pi_{\theta'}(a_t|s_t) A^{\pi_\theta}(s_t, a_t)\right]\right] \quad ;\text{log trick}
    \end{aligned}
    $$

    $$
    \Longrightarrow  \nabla_{\theta} \bar{A}(\theta) =  \sum_t \mathbb{E}_{s_t \sim p_{\theta}(s_t)}\left[\mathbb{E}_{a_t \sim \pi_{\theta}(a_t|s_t)}\left[ \gamma^t \nabla_{\theta}\log \pi_{\theta}(a_t|s_t) A^{\pi_\theta}(s_t, a_t)\right]\right] = \nabla_\theta J(\theta)
    $$



Thus, we can **rewrite the original optimization problem** as:

$$
\theta' = \arg\max_{\theta'} \nabla_\theta J(\theta)^T (\theta' - \theta)
$$

$$
\text{s.t. } D_{KL}(\pi_{\theta'}\|\pi_\theta) \leq \epsilon.
$$

To simplify the constraint, we approximate the KL divergence using a **second-order Taylor expansion** around $\theta$. The KL divergence between two nearby policies can be expanded as:

$$
D_{KL}(\pi_{\theta'} \| \pi_{\theta}) \approx \frac{1}{2} (\theta' - \theta)^T F(\theta) (\theta' - \theta),
$$



where $F(\theta)$ is the **Fisher Information Matrix (FIM)**, defined as:

$$
F(\theta) = \mathbb{E}_{s_t \sim p_{\theta}(s_t), a_t \sim \pi_{\theta}(a_t|s_t)} \left[ \nabla_\theta \log \pi_{\theta}(a_t | s_t) \nabla_\theta \log \pi_{\theta}(a_t | s_t)^T \right].
$$

This approximation holds when the policy update is small, which is precisely the regime in which we operate.

??? note "proof"

    The KL divergence between two policies parameterized by $\theta'$ and $\theta$ is:

    $$
    D_{KL}(\pi_{\theta'}\|\pi_\theta) = \mathbb{E}_{a \sim \pi_{\theta'}} \left[\log \frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}\right]
    $$

    To understand its behavior **when $\theta'$ is close to $\theta$ (trust region)**, we can examine its Taylor expansion around $\theta$.

    The first-order term in the Taylor expansion is:

    $$
    \nabla_\theta D_{KL}(\pi_{\theta}\|\pi_\theta) = 0
    $$

    This vanishes because:

    - At $\theta' = \theta$, the KL divergence is minimized (equals zero)
    - The gradient at a minimum is zero

    Since the first-order term is zero, the dominant term becomes the second-order approximation:

    $$
    D_{KL}(\pi_{\theta'}\|\pi_\theta) \approx \frac{1}{2} (\theta' - \theta)^T H(\theta) (\theta' - \theta)
    $$

    where $H(\theta)$ is the Hessian of $D_{KL}$ evaluated at $\theta$.

    Remarkably, the Hessian of the KL divergence is exactly the Fisher information matrix $F(\theta)$:

    $$
    F(\theta) = \mathbb{E}_{a \sim \pi_\theta} \left[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T\right]
    $$


    - Start with the definition of KL divergence:

    $$
    D_{KL}(\pi_{\theta'}\|\pi_\theta) = \mathbb{E}_{a \sim \pi_{\theta'}} \left[\log \pi_{\theta'}(a|s) - \log \pi_\theta(a|s)\right]
    $$

    - Take the gradient with respect to $\theta$:

    $$
    \nabla_\theta D_{KL} = -\mathbb{E}_{a \sim \pi_{\theta'}} \left[\nabla_\theta \log \pi_\theta(a|s)\right]
    $$

    - At $\theta' = \theta$:

    $$
    \nabla_\theta D_{KL}(\pi_\theta\|\pi_\theta) = -\mathbb{E}_{a \sim \pi_\theta} \left[\nabla_\theta \log \pi_\theta(a|s)\right] = 0 \quad ; \text{The expected score is zero}
    $$

    

    - Now take the Hessian (derivative of the gradient):

    $$
    H(\theta) = \nabla_\theta^2 D_{KL}(\pi_\theta\|\pi_\theta) = \mathbb{E}_{a \sim \pi_\theta} \left[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T\right] = F(\theta)
    $$

    This follows because:

    $$
    \nabla_\theta \mathbb{E}[\nabla_\theta \log \pi_\theta] = \mathbb{E}[\nabla_\theta^2 \log \pi_\theta] + \mathbb{E}[\nabla_\theta \log \pi_\theta \nabla_\theta \log \pi_\theta^T]
    $$

    and the first term vanishes since:

    $$
    \mathbb{E}[\nabla_\theta^2 \log \pi_\theta] = \int \pi_\theta \nabla_\theta^2 \log \pi_\theta da = \int \nabla_\theta^2 \pi_\theta da = \nabla_\theta^2 \int \pi_\theta da = \nabla_\theta^2 1 = 0
    $$


#### Geometric Interpretation

This approximation shows that the KL divergence behaves locally like a squared distance in parameter space, with the Fisher matrix defining the local geometry:

- The Fisher matrix $F(\theta)$ captures how changes in parameters affect the policy distribution
- It accounts for the fact that some parameter changes affect the policy more than others
- The resulting quadratic form gives a natural **distance** measure between policies

Thus, we replace the KL divergence constraint with:

$$
\frac{1}{2} (\theta' - \theta)^T F(\theta) (\theta' - \theta) \leq \epsilon.
$$



Now, our optimization problem becomes:

$$
\theta' = \arg\max_{\theta'} \nabla_\theta J(\theta)^T (\theta' - \theta)
$$

$$
\text{s.t. } \frac{1}{2} (\theta' - \theta)^T F(\theta) (\theta' - \theta) \leq \epsilon.
$$

Using the **method of Lagrange multipliers**, we introduce:

$$
\mathcal{L}(\theta', \lambda) = \nabla_\theta J(\theta)^T (\theta' - \theta) - \frac{\lambda}{2} (\theta' - \theta)^T F(\theta) (\theta' - \theta) + \lambda \epsilon.
$$

Setting the derivative with respect to $\theta'$ to zero:

$$
\nabla_\theta J(\theta) - \lambda F(\theta) (\theta' - \theta) = 0.
$$

Solving for $\theta'$:

$$
\theta' - \theta = \frac{1}{\lambda} F(\theta)^{-1} \nabla_\theta J(\theta).
$$

Substituting into the constraint:

$$
\frac{1}{2} \left( \frac{1}{\lambda} \nabla_\theta J(\theta)^T F(\theta)^{-1} \nabla_\theta J(\theta) \right) \leq \epsilon.
$$

Solving for $\lambda$:

$$
\lambda = \sqrt{\frac{\nabla_\theta J(\theta)^T F(\theta)^{-1} \nabla_\theta J(\theta)}{2\epsilon}}.
$$

Finally, substituting back, we obtain the **Natural Policy Gradient update rule**:

$$
\theta' = \theta + \alpha F(\theta)^{-1} \nabla_\theta J(\theta),
$$

where $\alpha$ is a step-size parameter and equals to $\frac{1}{\lambda}$

!!! success "Intuition"

    Take this figure [[Peters & Schaal 2003]](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JanPeters/peters-ICHR2003.pdf) as an example:

    <center> 
    <img src="\assets\images\course_notes\policy-based\natural_policy_gradient.png"
        style="float: center; margin-right: 10px;" 
        /> 
        </center>

    - **Vanilla Policy Gradients** $(\theta' = \theta + \alpha \nabla_\theta J(\theta))$
        - The gradient updates **do not align with the curvature** of the objective function.
        - The updates tend to be more **anisotropic**, meaning they are not adjusted for the scaling of different directions in parameter space.
        - The movement appears **inefficient**, leading to slower convergence and potentially unstable updates.

    - **Natural Policy Gradients** $(\theta' = \theta + \alpha F(\theta)^{-1} \nabla_\theta J(\theta))$
        - The updates **consider the geometry of the policy space**.
        - The movement is **more direct towards the optimal region**, respecting the curvature.
        - The updates **scale properly** in different directions, leading to more stable and efficient convergence.

    #### But Why Natural Policy Gradient Improves Performance?
    1. **Accounts for the Geometry of the Policy Space**  
        - In standard gradient descent (VPG), updates are made directly in the parameter space without considering how changes in policy parameters affect the actual policy distribution.
        - NPG **re-scales the gradients using the Fisher Information Matrix (FIM)**, effectively **whitening the gradient updates** and making them more **natural** to the policy's probability distribution.

    2. **Better Step-Size Control via KL Constraint**  
        - The KL divergence constraint ensures that each update remains within a **trust region**, preventing excessively large updates that can destabilize learning.
        - This avoids issues like **policy collapse** (where large updates drastically change behavior and degrade performance).

    3. **Faster and More Stable Convergence**  
        - Since NPG **aligns with the curvature**, it results in **larger effective steps** in directions where learning is slower and **smaller steps** in directions where the policy is sensitive.
        - This leads to more **direct movement towards optimal policies**, as seen in the right-hand plot.

    4. **Mitigates the Problem of Poorly Scaled Gradients**  
        - Vanilla gradients suffer from **poor scaling in high-dimensional spaces**.
        - NPG corrects this by preconditioning the gradients using the **inverse Fisher Information Matrix**, making learning more stable and efficient.

### **Trust Region Policy Optimization (TRPO)**  

In the previous section, we derived the **Natural Policy Gradient (NPG)** and showed how it accounts for the geometry of the policy space by incorporating the **Fisher Information Matrix (FIM)**. However, while NPG improves convergence and stability over vanilla policy gradients, it does not **strictly** enforce the KL divergence constraint, leading to potential **constraint violations** and **poor policy updates**.  

To address this issue, **Trust Region Policy Optimization (TRPO)** introduces a modification:  

- Instead of using an **approximate KL constraint via Fisher Information**, it **directly enforces** the constraint through a constrained optimization procedure.  

- It **performs a backtracking line search** to ensure that the update remains within the **trust region** while still improving performance.  


Recall that in **Natural Policy Gradient**, we derived the following update rule:

$$
\theta' = \theta + \alpha F(\theta)^{-1} \nabla_\theta J(\theta)
$$

where:  

- $F(\theta)$ is the **Fisher Information Matrix** (approximating the KL divergence),  
- $\alpha$ is the step-size chosen to satisfy the approximate trust region constraint.  

This formulation works well, but the **KL constraint is only approximately satisfied**, meaning that:  

1. The **Taylor series expansion of KL divergence** introduces approximation errors.  
2. The **trust region size can still be violated**, leading to **unstable updates**.  

Thus, TRPO refines this approach by ensuring that the **KL constraint is strictly enforced**.  


TRPO modifies the **NPG optimization problem** to ensure that the KL divergence **exactly** satisfies the trust region constraint:

$$
\theta' = \arg\max_{\theta'} \mathbb{E}_{s_t \sim p_{\theta}(s_t), a_t \sim \pi_{\theta}(a_t|s_t)} \left[ \frac{\pi_{\theta'}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)} A^{\pi_\theta}(s_t, a_t) \right]
$$

$$
\text{s.t. } D_{KL}(\pi_{\theta'}\|\pi_\theta) \leq \epsilon.
$$

#### **Why Does This Improve Over NPG?**

- The **NPG update assumes a quadratic KL divergence approximation**, which can be **inaccurate for large updates**.
- TRPO **directly enforces** the KL constraint **without relying on second-order approximations**.
- Instead of just taking a single update step, **TRPO performs a constrained optimization procedure** that ensures each update **improves performance while respecting the KL divergence limit**.


To solve this **constrained optimization problem**, we use the **Lagrangian formulation**:

$$
\mathcal{L}(\theta', \lambda) = \mathbb{E}_{s_t, a_t} \left[ \frac{\pi_{\theta'}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)} A^{\pi_\theta}(s_t, a_t) \right] - \lambda \left( D_{KL}(\pi_{\theta'} \| \pi_{\theta}) - \epsilon \right).
$$

This leads to the **search direction** given by solving:

$$
F(\theta) (\theta' - \theta) = \nabla_\theta J(\theta).
$$

This is **exactly the Natural Policy Gradient equation**! However, if we stop here, we would be using **NPG**, which does not strictly enforce the constraint.


The **key difference in TRPO** is that instead of taking a single fixed step as in NPG, **we refine the step size using a backtracking line search**:

1. **Compute a candidate step** using the NPG update:

    $$
    \theta' = \theta + \alpha F(\theta)^{-1} \nabla_\theta J(\theta).
    $$

2. **Perform backtracking line search**:
    - Start with the full step-size $\alpha = 1$.
    - If the KL constraint is violated ($D_{KL}(\pi_{\theta'} \| \pi_{\theta}) > \epsilon$), reduce $\alpha$ (e.g., multiply by a factor $\beta < 1$).
    - Continue reducing $\alpha$ until the constraint is satisfied.

#### Why Does Backtracking Line Search Help?

- The **approximate KL constraint from NPG** can **violate** the trust region.  
- By **backtracking**, TRPO **ensures that every update remains in the feasible region** while still maximizing improvement.  
- This prevents **overly aggressive policy updates** that can degrade performance.  

The final update rule in TRPO is:

$$
\theta' = \theta + \alpha_k F(\theta)^{-1} \nabla_\theta J(\theta),
$$

where $\alpha_k$ is **determined by the backtracking line search** to ensure:

$$
D_{KL}(\pi_{\theta'} \| \pi_{\theta}) \leq \epsilon.
$$

<center>

| **Method** | **Update Rule** | **KL Constraint Handling** | **Pros** | **Cons** |
|------------|---------------|-------------------------|---------|---------|
| **Vanilla Policy Gradient (VPG)** | $\theta' = \theta + \alpha \nabla_\theta J(\theta)$ | No constraint | Simple but unstable | Poorly scaled updates |
| **Natural Policy Gradient (NPG)** | $\theta' = \theta + \alpha F(\theta)^{-1} \nabla_\theta J(\theta)$ | Approximate constraint (Fisher Information) | More stable, better scaling | KL constraint may still be violated |
| **Trust Region Policy Optimization (TRPO)** | $\theta' = \theta + \alpha_k F(\theta)^{-1} \nabla_\theta J(\theta)$ | Strict constraint via **backtracking line search** | Stable updates, enforces KL constraint | Computationally expensive and sample inefficient |

</center>


### **Proximal Policy Optimization (PPO)**  

We have covered **Natural Policy Gradient (NPG)** and **Trust Region Policy Optimization (TRPO)**, both of which aim to improve stability and performance over **Vanilla Policy Gradient (VPG)** by incorporating **trust region constraints**.  

While **TRPO is powerful**, it requires solving a **constrained optimization problem** using a second-order method, which can be **computationally expensive**.  

To address this, **Proximal Policy Optimization (PPO)** simplifies TRPO by:  

- **Removing the need for a constrained optimization solver**,  
- **Using a clipping mechanism to enforce trust regions efficiently**,  
- **Making the algorithm easier to implement and more scalable**.

TRPO optimizes the **surrogate advantage function**:

$$
J(\theta') = \mathbb{E}_{s_t, a_t} \left[ \frac{\pi_{\theta'}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)} A^{\pi_\theta}(s_t, a_t) \right]
$$

subject to the trust region constraint:

$$
D_{KL}(\pi_{\theta'} \| \pi_{\theta}) \leq \epsilon.
$$

This ensures stable updates, but solving a constrained optimization problem is **computationally expensive**. **Can we enforce the trust region in a simpler way?**  

#### **PPO: The Key Idea**  
Instead of explicitly solving for an optimal update **with a KL constraint**, PPO introduces a **clipping mechanism** that prevents updates from deviating too far from the old policy.  

PPO modifies the TRPO objective by introducing a **clipping function** to restrict how much the new policy $\pi_{\theta'}$ can deviate from the old policy $\pi_{\theta}$.

First, define the **importance sampling ratio**:

$$
r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}.
$$

This measures the **change in policy probability** for an action after updating the policy.  

The **PPO objective function** is:

$$
J_{\text{PPO}}(\theta) = \mathbb{E}_{s_t, a_t} \left[ \min \left( r_t(\theta) A^{\pi_{\theta_{\text{old}}}}(s_t, a_t), \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A^{\pi_{\theta_{\text{old}}}}(s_t, a_t) \right) \right].
$$

#### **The Clipping Mechanism**


<figure style="text-align: center;">
  <img src="\assets\images\course_notes\policy-based\ppo_clip.png" 
       style="margin-right: 10px;" />
  <figcaption> From <a href="https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl" 
    style="color: #0066cc; text-decoration: none;"> Stack Overflow</a></figcaption>
</figure>

- The first term $r_t(\theta) A^{\pi_{\theta_{\text{old}}}}$ is the **standard policy gradient update**.
- The second term **clamps** $r_t(\theta)$ within $[1 - \epsilon, 1 + \epsilon]$, preventing updates that shift the policy too much.
- **This ensures that updates are small, enforcing a trust region without requiring an explicit KL constraint**.



#### **Why Does This Work?**

- **If $r_t(\theta) A > 0$** (action probability increases and advantage is positive), the update is performed **as long as $r_t(\theta)$ stays within $1 \pm \epsilon$**.
- **If $r_t(\theta)$ moves outside $1 \pm \epsilon$**, the clipped term stops further updates, preventing excessive changes.
- This effectively **penalizes large updates**, **stabilizing** the training process without the need for complex constrained optimization.

<center>

| **Algorithm** | **Trust Region Enforcement** | **Update Complexity** | **Advantages** | **Disadvantages** |
|--------------|--------------------------|-----------------|--------------|----------------|
| **TRPO** | KL divergence constraint $D_{KL}(\pi_{\theta'} \| \pi_{\theta}) \leq \epsilon$ | Solves a constrained optimization problem | Strict KL enforcement, stable | Computationally expensive, complex |
| **PPO (Clipped)** | Clipping function $\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)$ | Simple gradient ascent | Easy to implement, scalable | Less strict trust region |

</center>

#### **Why PPO is Preferred Over TRPO?**

1. **Easier to Implement**
    - No need to solve a constrained optimization problem like in TRPO.
    - Uses standard gradient ascent with clipping.

2. **Computationally Efficient**  
    - TRPO requires **computing and inverting the Fisher Information Matrix**, which is expensive.
    - PPO **only requires simple clipping operations**, making it faster.

3. **Stable and Effective in Practice** 
    - Enforces a trust region without requiring second-order optimization.
    - Empirically, **PPO performs as well as, or even better than, TRPO**, while being easier to use.


!!! success ""

    🔥 **PPO is now the standard choice for on-policy reinforcement learning, used in OpenAI's Gym, robotics, and many real-world applications.** 🚀

### **Deep Deterministic Policy Gradient (DDPG): Off-Policy Actor-Critic**  

So far, we have covered **on-policy** actor-critic methods like **TRPO and PPO**, which require collecting new data with the current policy for each update. While these methods are stable, they are **sample inefficient** because they cannot reuse old experiences.  

Now, let’s move to **off-policy** actor-critic methods, specifically **Deep Deterministic Policy Gradient (DDPG)**, which improves sample efficiency by storing and reusing past experiences.  


DDPG extends the **Q-learning framework** to **continuous action spaces** by using a function approximator (a deep neural network) for the **Q-function**:

$$
Q^\pi(s, a) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a \right].
$$

Since **DQN** is designed for discrete actions (where you can select the optimal action via $\max_a Q(s, a)$), we need a way to optimize actions in continuous spaces.  

**Solution:** Instead of taking the max over discrete actions, DDPG **directly parameterizes the policy $\mu_{\theta}(s)$ and trains it to maximize $Q(s, a)$**.

The **policy is updated using the deterministic policy gradient theorem**:

$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \nabla_a Q^\pi(s_t, a) \big|_{a = \mu_{\theta}(s_t)} \nabla_{\theta} \mu_{\theta}(s_t) \right].
$$

This means the **policy is updated in the direction that increases $Q(s, a)$** by following the gradient of the Q-function.

??? note "proof"
    
    We consider a **deterministic policy** $\mu_\theta: \mathcal{S} \rightarrow \mathcal{A}$, which maps each state $s$ deterministically to an action $a$.

    The goal is to maximize the expected return:

    $$
    J(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ Q^{\mu_\theta}(s, \mu_\theta(s)) \right],
    $$

    where:

    - $\mathcal{D}$ is a **replay buffer distribution** or the **state visitation distribution** induced by the policy.
    - $Q^{\mu}(s, a)$ is the **action-value function** under the policy $\mu$.

    The **Deterministic Policy Gradient Theorem** states:

    $$
    \nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \nabla_a Q^\mu(s, a) \big|_{a = \mu_{\theta}(s)} \nabla_{\theta} \mu_{\theta}(s) \right].
    $$

    Unlike the stochastic case, this gradient does **not** involve the log-derivative trick. Instead, it directly leverages the **chain rule** through the deterministic policy.

    We start from the objective:

    $$
    J(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ Q^{\mu_\theta}(s, \mu_\theta(s)) \right].
    $$

    We apply the gradient to the objective using the chain rule:

    $$
    \nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{s \sim \mathcal{D}} \left[ Q^{\mu_\theta}(s, \mu_\theta(s)) \right].
    $$

    Assuming $s$ is sampled from a fixed distribution $\mathcal{D}$ (e.g., a replay buffer or steady-state distribution), we can move the gradient inside the expectation:

    $$
    \nabla_\theta J(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \nabla_\theta Q^{\mu_\theta}(s, \mu_\theta(s)) \right].
    $$

    We now differentiate $Q^{\mu_\theta}(s, \mu_\theta(s))$ using the chain rule:

    $$
    \nabla_\theta Q^{\mu_\theta}(s, \mu_\theta(s)) = \nabla_a Q^{\mu}(s, a) \big|_{a = \mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s).
    $$

    So the full gradient becomes:

    $$
    \nabla_\theta J(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \nabla_a Q^{\mu}(s, a) \big|_{a = \mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s) \right].
    $$

    **Intuition**

    - We are optimizing how the policy $\mu_\theta(s)$ chooses actions **in states visited by the current policy**.
    - The gradient tells us **how much changing the policy parameters would increase the expected return**, by:
    - Measuring how $Q(s, a)$ changes with $a$.
    - Multiplying by how $a = \mu_\theta(s)$ changes with $\\theta$.

    This gives a **direct signal** to adjust the policy in directions that **increase expected reward**.



DDPG follows a **soft policy iteration approach**, similar to Conservative Policy Iteration:

1. **Policy Evaluation (Critic Update)**  

    - The **Q-function** (critic) is updated using the Bellman equation:

    $$
    y_t = r_t + \gamma Q_{\phi'}(s_{t+1}, \mu_{\theta'}(s_{t+1})).
    $$

    - The critic is trained by minimizing the **mean squared Bellman error**:

    $$
    L(\phi) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q_{\phi}(s, a) - y_t \right)^2 \right].
    $$

2. **Policy Improvement (Actor Update)**  

    - The actor updates via gradient ascent on $Q(s, a)$:

    $$
    \nabla_{\theta} J(\theta) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \nabla_a Q_{\phi}(s_t, a) \big|_{a = \mu_{\theta}(s_t)} \nabla_{\theta} \mu_{\theta}(s_t) \right].
    $$


To prevent instability in Q-learning updates, DDPG **uses target networks**:

- **Slow-moving target networks** $Q_{\phi'}(s, a)$ and $\mu_{\theta'}(s)$ are maintained.  

- They are updated using a **soft update rule**:

    $$
    \phi' \leftarrow \tau \phi + (1 - \tau) \phi', \quad \theta' \leftarrow \tau \theta + (1 - \tau) \theta'.
    $$

where $\tau$ is a small value (e.g., 0.005), ensuring **smooth updates** and preventing divergence.

<figure style="text-align: center;">
  <img src="\assets\images\course_notes\policy-based\DDPG.png" 
       style="margin-right: 10px;" />
  <figcaption> DDPG Algorithm <a href="https://arxiv.org/pdf/1509.02971" 
    style="color: #0066cc; text-decoration: none;"> (Lillicrap, et al., 2015)</a></figcaption>
</figure>


#### **Why is DDPG "Deterministic"?**  
DDPG builds on the **Deterministic Policy Gradient (DPG) theorem**, which uses a **deterministic policy** instead of a stochastic one.  

- **Stochastic Policy (used in PPO, TRPO, A2C, etc.)**:  

    $$
    a \sim \pi_{\theta}(a | s)
    $$

    - The policy outputs a **distribution** over actions.  
    - This is useful for **exploration** but makes training **less stable**.  

- **Deterministic Policy (used in DDPG)**:  

    $$
    a = \mu_{\theta}(s)
    $$

    - The policy **directly maps** states to actions.  
    - No randomness in action selection (hence "deterministic").  
    - **More stable updates**, but **requires explicit exploration**.  


#### **Why Does Deterministic Policy Affect Exploration?**  
Since DDPG's policy is **deterministic**, it **always selects the same action for a given state**. This creates a major problem:  

- **In on-policy methods**, exploration happens naturally because the policy samples different actions due to its stochastic nature.  

- **In DDPG, without stochasticity, the agent may get stuck in suboptimal behavior and fail to explore effectively.**  

To encourage exploration, we add **random noise** to the actions **during training**:  

$$
a_t = \mu_{\theta}(s_t) + \mathcal{N}_t
$$

where **$\mathcal{N}_t$** is a noise process.  

**Common choices for noise:**  

1. **Ornstein-Uhlenbeck (OU) noise**  
    - **Time-correlated** noise that simulates realistic exploration (e.g., in robotic control).  
    - Used in **early DDPG implementations**.  
    - Defined as: 

        $$
        \mathcal{N}_{t+1} = \theta (\mu - \mathcal{N}_t) + \sigma W_t
        $$

        where $W_t$ is a Wiener process.  

        ??? abstract 

            **What is a Wiener Process?**

            The **Wiener process** (also known as **Brownian motion**) is a continuous-time stochastic process with the following properties:

            - Starts at zero: $\mathcal{W}_0 = 0$.
            - Has **independent increments**.
            - Increments are normally distributed: $\mathcal{W}_{t+1} - \mathcal{W}_t \sim \mathcal{N}(0, \Delta t)$.
            - Paths are continuous but nowhere differentiable.

            In the context of the Ornstein-Uhlenbeck process, the term $W_t$ denotes **Gaussian noise sampled over time**, and can be interpreted as a discrete-time approximation of the Wiener process. This introduces **time-correlated** randomness, unlike independent Gaussian noise, making it more suitable for environments like physical control tasks where smooth actions are preferred.

            **Why Use the Wiener Process in DDPG?**

            In DDPG, actions are chosen by a **deterministic policy**, which means there’s no inherent exploration like in stochastic policies. To encourage exploration, noise is added to the actions during training.

            The **Ornstein-Uhlenbeck process**, driven by a Wiener process, is used because:
            
            - It produces **temporally correlated noise**, meaning the noise values change smoothly over time.
            - This leads to **more natural, realistic exploration** in physical systems (e.g., robotics), where abrupt changes in action are undesirable or infeasible.
            - The Wiener process ensures that the noise evolves in a **stochastic yet continuous** way, helping the agent explore the action space more effectively than with uncorrelated (white) Gaussian noise.

            Thus, using OU noise (based on a Wiener process) improves stability and realism in continuous control tasks during the exploration phase.


2. **Gaussian Noise**  
    - Simpler alternative to OU noise.  
    - Just adds zero-mean Gaussian noise:  

        $$
        \mathcal{N}_t \sim \mathcal{N}(0, \sigma^2)
        $$

    - Works well in **modern implementations** of DDPG.  

#### **Why Off-Policy? Experience Replay in DDPG**  

DDPG is **off-policy**, meaning it **stores experiences** and learns from **past** data, unlike on-policy methods that require fresh samples.  

**Replay Buffer** ($\mathcal{D}$):  

- Stores **past experiences** $(s, a, r, s')$.  
- Allows the agent to **reuse old samples**, improving sample efficiency.  
- **Breaks correlation** between consecutive experiences, making learning more stable.  

**Why is this important?**  

- Deterministic policies update using gradient descent, which requires **diverse data** to prevent overfitting.  

- Without a replay buffer, updates would depend too much on recent actions, leading to poor generalization.  


<center>

| **Feature**         | **DDPG (Off-Policy)** | **PPO/TRPO (On-Policy)** |
|--------------------|-------------------|-------------------|
| **Policy Type**    | Deterministic      | Stochastic        |
| **Exploration**    | Noise added to actions | Inherent in stochastic policy |
| **Sample Efficiency** | High (reuses past data) | Low (requires new samples) |
| **Update Stability** | More stable      | Can be unstable   |
| **Trust Region**   | No                | Yes (TRPO, PPO use KL/clipping) |

</center>


### **Soft Actor-Critic (SAC)**

#### Challenges and motivation of SAC
1. Previous Off-policy methods like DDPG often struggle with exploration , leading to suboptimal policies. SAC overcomes this by introducing entropy maximization, which encourages the agent to explore more efficiently.
2. Sample inefficiency is a major issue in on-policy algorithms like Proximal Policy Optimization (PPO), which require a large number of interactions with the environment. SAC, being an off-policy algorithm, reuses past experiences stored in a replay buffer, making it significantly more sample-efficient.
3. Another challenge is instability in learning, as methods like DDPG can suffer from overestimation of Q-values. SAC mitigates this by employing twin Q-functions (similar to TD3) and incorporating entropy regularization, leading to more stable and robust learning.

In essence, SAC seeks to maximize the entropy in policy, in addition to the expected reward from the environment. The entropy in policy can be interpreted as randomness in the policy.

???+ note "what is entropy?"

    We can think of entropy as how unpredictable a random variable is. If a random variable always takes a single value then it has zero entropy because it’s not unpredictable at all. If a random variable can be any Real Number with equal probability then it has very high entropy as it is very unpredictable.
    <center> 
    <img src="\assets\images\course_notes\policy-based\entropy.jpg"
        alt="entropy"
        style="float: center; margin-right: 10px;" 
        /> 
        </center>

    <!-- ![Figure 9](..//assets//images//course_notes//policy-based//entropy.jpg) -->



    *probability distributions with low entropy have a tendency to greedily sample certain values, as the probability mass is distributed relatively unevenly*.


#### Maximum Entropy Reinforcement Learning

In Maximum Entropy RL, the agent tries to optimize the policy to choose the right action that can receive the highest sum of reward and long term sum of entropy. This enables the agent to explore more and avoid converging to local optima.

!!! note "reason"

    We want a high entropy in our policy to encourage the policy to assign equal probabilities to actions that have same or nearly equal Q-values(allow the policy to capture multiple modes of good policies), and also to ensure that it does not collapse into repeatedly selecting a particular action that could exploit some inconsistency in the approximated Q function. Therefore, SAC overcomes the  problem by encouraging the policy network to explore and not assign a very high probability to any one part of the range of actions.

The objective function of the Maximum entropy RL is as shown below:

$$
J(\pi_{\theta}) = \mathbb{E}_{\pi_{\theta}} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) + \alpha H(\pi(\cdot | s_t)) \right]
$$

and the optimal policy is:

$$
\pi^* = \arg\max_{\pi_{\theta}} \mathbb{E}_{\pi_{\theta}} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) + \alpha H(\pi(\cdot | s_t)) \right]
$$


**$\alpha$** is the temperature parameter that balances between exploration and exploitation.




#### Overcoming Exploration Bias in Multimodal Q-Functions



Here we want to explain the concept of a **multimodal Q-function** in reinforcement learning (RL), where the **Q-value function**, $Q(s, a)$, represents the expected cumulative reward for taking action $a$ in state $s$. 

In this context, the robot is in an initial state and has two possible passages to follow, which result in a **bimodal Q-function** (a function with two peaks). These peaks correspond to two different action choices, each leading to different potential outcomes.


#### **Standard RL Approach**
<center> 
<img src="\assets\images\course_notes\policy-based\SAC1.png" 
    alt="sac1"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>

<!-- ![Figure 10](..//assets//images//recitation//week4//SAC1.png) -->

The **grey curve** represents the Q-function, which has two peaks, indicating two promising actions.A conventional RL approach typically assumes a unimodal (single-peaked) policy distribution, represented by the **red curve**.

This policy distribution is modeled as a Gaussian $\mathcal{N}(\mu(s_t), \Sigma)$ centered around the highest Q-value peak.

This setup results in exploration bias where the agent primarily explores around the highest peak and ignores the lower peak entirely.


#### Improved Exploration Strategy
Instead of using a unimodal Gaussian policy, a Boltzmann-weighted policy is used.
The **policy distribution (green shaded area)** is proportional to $\exp(Q(s_t, a_t))$, meaning actions are sampled based on their Q-values. 

This approach allows the agent to explore multiple high-reward actions and avoids the bias of ignoring one passage.
As a result, the agent recognizes both options, increasing the chance of finding the optimal path.
<center> 
<img src="\assets\images\course_notes\policy-based\SAC2.png"
    alt="SAC2"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>

<!-- ![Figure 11](..//assets//images//recitation//week4//SAC2.png) -->


This concept is relevant for **actor-critic RL methods** like **Soft Actor-Critic (SAC)**, which uses entropy to encourage diverse exploration.
 

#### **Soft Policy**
- Soft policy:


$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t)\sim\rho_{\pi}} \left[ r(s_t, a_t) + \alpha\mathcal{H}(\pi(\cdot | s_t)) \right]
$$

With new objective function we need to define Value funciton and Q-value funciton again. 



- Soft Q-value funciton:

$$
Q(s_t, a_t) = r(s_t, a_t) + \gamma\mathbb{E}_{s_{t+1}\sim p}\left[ V(s_{t+1}) \right]
$$

- Soft Value function:

$$
V(s_t) = \mathbb{E}_{a_t\sim \pi} \left[Q(s_t, a_t) - \text{log}\space\pi(a_t|s_t)\right]
$$

#### Soft policy iteration
Soft Policy Iteration is an entropy-regularized version of classical policy iteration, which consists of:

1. *Soft Policy Evaluation*: Estimating the soft Q-value function under the current policy.
2. *Soft Policy Improvement*:  Updating the policy to maximize the soft Q-value function,incorporating entropy regularization.

This process iteratively improves the policy while balancing exploration and exploitation.

#### Soft Policy Evaluation (Critic Update)
The goal of soft policy evaluation is to compute the expected return of a given policy $\pi$ under the maximum entropy objective, which modifies the standard Bellman equation by adding an entropy term. (SAC explicitly learns the Q-function for the current policy)

The soft Q-value function for a policy $\pi$ is updated using a modified Bellman operator $T^{\pi}$:

$$
T^\pi Q(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1} \sim p} [V(s_{t+1})]
$$

with substitution of $V$ we have :

$$
T^\pi Q(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{\substack{s_{t+1} \sim p \\ a_{t+1} \sim \pi}} [Q(s_{t+1,}, a_{t+1}) - \text{log}\space\pi(a_{t+1}|s_{t+1})]
$$

**Key Result: Soft Bellman Backup Convergence**  

**Theorem:** By repeatedly applying the operator $T^\pi$, the Q-value function converges to the true soft Q-value function for policy $\pi$:  

$$
Q_k \to Q^\pi \text{ as } k \to \infty
$$

Thus, we can estimate $Q^\pi$ iteratively.

---
#### Soft Policy Improvement (Actor Update)
Once the Q-function is learned, we need to improve the policy using a gradient-based update. This means:

- Instead of directly maximizing Q-values, the policy is updated to optimize a modified objective that balances reward maximization and exploration.
- The update is off-policy, meaning the policy can be trained using past experiences stored in a replay buffer, rather than requiring fresh samples like on-policy methods (e.g., PPO).

The update is based on an ***exponential function of the Q-values***:

$$
\pi^*(a | s) \propto \exp (Q^\pi(s, a))
$$

which means the optimal policy is obtained by normalizing $\exp (Q^\pi(s, a))$ over all actions:

$$
\pi^*(a | s) = \frac{\exp (Q^\pi(s, a))}{Z^\pi(s)}
$$

where $Z^\pi(s)$ is the partition function that ensures the distribution sums to 1.

For the policy improvement step, we update the policy distribution towards the softmax distribution for the current Q function.

$$
\pi_{\text{new}} = \arg \min_{\pi' \in \Pi} D_{\text{KL}} \left( \pi'(\cdot | s) \, \bigg|\bigg| \, \frac{\exp (Q^\pi(s, \cdot))}{Z^\pi(s)} \right)
$$


**Key Result: Soft Policy Improvement Theorem**  
The new policy $\pi_{\text{new}}$ obtained via this update improves the expected soft return:

$$
Q^{\pi_{\text{new}}}(s, a) \geq Q^{\pi_{\text{old}}}(s, a) \quad \forall (s, a)
$$

Thus, iterating this process leads to a better policy.

#### Convergence of Soft Policy Iteration
By alternating between soft policy evaluation and soft policy improvement, soft policy iteration converges to an optimal maximum entropy policy within the policy class $\Pi$:

$$
\pi^* = \arg \max_{\pi \in \Pi} \sum_t \mathbb{E}[r_t + \alpha H(\pi(\cdot | s_t))]
$$


However, this exact method is only feasible in the ***tabular setting***. For ***continuous control***, we approximate it using function approximators.


#### Soft Actor-Critic Method

For complex learning domains with high-dimensional and/or continuous state-action spaces, it is mostly impossible to find exact solutions for the MDP. Thus, we must leverage function approximation (i.e. neural networks) to find a practical approximation to soft policy iteration. then we use stochastic gradient descent (SGD) to update parameters of these networks.

we model the value functions as expressive neural networks, and the policy as a Gaussian distribution over the action space with the mean and covariance given as neural network outputs with the current state as input.

#### **Soft Value function ($V_{\psi}(s)$)**

A separate soft value function which helps in stabilising the training process. The soft value function approximator minimizes the squared residual error as follows:

$$
J_V(\psi) = \mathbb{E}_{s_{t} \sim \mathcal{D}} \left[ \frac{1}{2} \left( V_{\psi}(s_t) - \mathbb{E}_{a \sim \pi_{\phi}} [Q_{\theta}(s_t, a_t) - \log \pi_{\phi}(a_t | s_t)] \right)^2 \right]
$$

It means the learning of the state-value function $V$, is done by minimizing the squared difference between the prediction of the value network and expected prediction of Q-function with the entropy of the policy, $\pi$.

  - $D$ is the distribution of previously sampled states and actions, or a replay buffer.

**Gradient Update for $V_{\psi}(s)$**

$$
\hat{\nabla}_{\psi} J_V(\psi) = \nabla_{\psi} V_{\psi}(s_t) \left( V_{\psi}(s_t) - Q_{\theta}(s_t, a_t) + \log \pi_{\phi}(a_t | s_t) \right)
$$

where the actions are sampled according to the current policy, instead of the replay buffer.

#### **Soft Q-funciton ($Q_{\theta}(s, a)$)**

We minimize the soft Q-function parameters by using the soft Bellman residual provided here:

$$
J_Q(\theta) = \mathbb{E}_{(s_{t}, a_t) \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q_{\theta}(s_t, a_t) - \hat{Q}(s_t, a_t)\right)^2 \right]
$$

with : 

$$
\hat{Q}(s_t, a_t) = r(s_t, a_t) + \gamma \space \mathbb{E}_{s_{t+1} \sim p} [V_{\bar{\psi}}(s_{t+1})]
$$

**Gradient Update for $Q_{\theta}$**:

$$
\hat{\nabla}_\theta J_Q(\theta) = \nabla_{\theta} Q_{\theta}(s_t, a_t) \left( Q_{\theta}(s_t, a_t) - r(s_t, a_t) - \gamma V_{\bar{\psi}}(s_{t+1}) \right)
$$

A **target value function** $V_{\bar{\psi}}$ (exponentially moving average of $V_{\psi}$) is used to stabilize training.

---

???+ note "More Explanation About Target Networks"

    The use of target networks is motivated by a problem in training $V$ network. If you go back to the objective functions in the Theory section, you will find that the target for the $Q$ network training depends on the $V$ Network and the target for the $V$ Network depends on the $Q$ network (this makes sense because we are trying to enforce Bellman Consistency between the two functions). Because of this, the $V$ network has a target that’s indirectly dependent on itself which means that the $V$ network’s target depends on the same parameters we are trying to train. This makes training very unstable.


    The solution is to use a set of parameters which comes close to the parameters of the main $V$ network, but with a time delay. Thus we create a second network which lags the main network called the target network. There are two ways to go about this.

    1. The first way is to have the target network copied over from the main network regularly after a set number of steps -> **Periodic Hard Update**


    $$
    \theta^{-} \leftarrow \theta
    $$

    2. The other way is to update the target network by Polyak averaging (a kind of moving averaging) itself and the main network. -> **Soft Update**

    $$
    \theta^{-} \leftarrow \tau \theta + (1-\tau) \theta^{-}
    $$
    
    <center> 
    <img src="\assets\images\course_notes\policy-based\target_network1.png"
        alt="target1"
        style="float: center; margin-right: 10px;" 
        /> 
        </center>
    <center> 
    <img src="\assets\images\course_notes\policy-based\target_network_2.png"
        alt="target2"
        style="float: center; margin-right: 10px;" 
        /> 
        </center>

    <!-- ![Figure 12](..//assets//images//course_notes//policy-based//target_network1.png)
    <!-- ![Figure 13](..//assets//images//course_notes//policy-based//target_network_2.png) --> -->

    *source: [concept target network in category reinforcement learning](https://livebook.manning.com/concept/reinforcement-learning/target-network#:~:text=By%20using%20target%20networks%2C%20we,a%20new%20one%20is%20set.)*

---

#### **Policy network ($\pi_{\phi}(a|s)$)**

The policy $\pi_{\phi}(a | s)$ is updated using the soft policy improvement step, minimizing the KL-divergence:

$$
J_{\pi}(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ D_{\text{KL}} \left( \pi_{\phi}(\cdot | s_t) \bigg\| \frac{\exp(Q_{\theta}(s_t, \cdot))}{Z_{\theta}(s_t)} \right) \right]
$$

Instead of solving this directly, SAC reparameterizes the policy using:

$$
a_t = f_{\phi}(\epsilon_t; s_t)
$$

This trick is used to make sure that sampling from the policy is a differentiable process so that there are no problems in backpropagating the errors.  $\epsilon_t$ is random noise vector sampled from fixed distribution (e.g., Spherical Gaussian).


???+ note "Why Reparameterization is Needed?"

    In reinforcement learning, the policy $\pi(a | s)$ often outputs a probability distribution over actions rather than deterministic actions. The standard way to sample an action is:

    $$
    a_t \sim \pi_{\phi}(a_t | s_t)
    $$

    However, this sampling operation blocks gradient flow during backpropagation, preventing efficient training using stochastic gradient descent (SGD).

    Instead of directly sampling $a_t$ from $\pi_{\phi}(a_t | s_t)$, we transform a simple noise variable into an action:

    $$
    a_t = f_{\phi}(\epsilon_t, s_t)
    $$

    - $\epsilon_t \sim \mathcal{N}(0, I)$ is sampled from a fixed noise distribution (e.g., a Gaussian).
    - $f_{\phi}(\epsilon_t, s_t)$ is a differentiable function (often a neural network) that maps noise to an action.

    For a Gaussian policy in SAC, the action is computed as:

    $$
    a_t = \mu_{\phi}(s_t) + \sigma_{\phi}(s_t) \cdot \epsilon_t
    $$

    So instead of sampling from $\mathcal{N}(\mu, \sigma^2)$ directly, we sample from a fixed standard normal and transform it using a differentiable function.This makes the policy differentiable, allowing gradients to flow through $\mu_{\phi}$ and $\sigma_{\phi}$.

    <center> 
    <img src="\assets\images\course_notes\policy-based\trick.png"
        alt="trick"
        style="float: center; margin-right: 10px;" 
        /> 
        </center>
    <center> 
    <img src="\assets\images\course_notes\policy-based\trick2.png"
        alt="trick2"
        style="float: center; margin-right: 10px;" 
        /> 
        </center>

    <!-- ![Figure 14](..//assets//images//course_notes//policy-based//trick.png)
    ![Figure 15](..//assets//images//course_notes//policy-based//trick2.png) -->

    - **Continuous Action Generation**

    In a continuous action space soft actor-critic agent, the neural network in the actor takes the current observation and generates two outputs, one for the mean and the other for the standard deviation. To select an action, the actor randomly selects an unbounded action from this Gaussian distribution. If the soft actor-critic agent needs to generate bounded actions, the actor applies tanh and scaling operations to the action sampled from the Gaussian distribution.

    During training, the agent uses the unbounded Gaussian distribution to calculate the entropy of the policy for the given observation.
    <center> 
    <img src="\assets\images\course_notes\policy-based\bounded.png"
        alt="bounded"
        style="float: center; margin-right: 10px;" 
        /> 
        </center>

    <!-- ![Figure 16](..//assets//images//course_notes//policy-based//bounded.png) -->

    - **Discrete Action Generation**

    In a discrete action space soft actor-critic agent, the actor takes the current observation and generates a categorical distribution, in which each possible action is associated with a probability. Since each action that belongs to the finite set is already assumed feasible, no bounding is needed.

    During training, the agent uses the categorical distribution to calculate the entropy of the policy for the given observation.




if we rewrite the equation we have:

$$
J_{\pi}(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}, \epsilon_t \sim \mathcal{N}} \left[ \text{log}\space\pi_{\phi} \left(f_{\phi}(\epsilon_t; s_t) | s_t \right) - Q_{\theta}(s_t,f_{\phi}(\epsilon_t; s_t) \right]
$$

where $\pi_{\phi}$ is defined implicitly in terms of $f_{\phi}$, and we have
noted that the partition function is independent of $\phi$ and can
thus be omitted.

**Policy Gradient Update**

$$
\hat{\nabla}_{\phi} J_{\pi}(\phi) = \nabla_{\phi} \log \pi_{\phi}(a_t | s_t) + \left( \nabla_{a_t} \log \pi_{\phi}(a_t | s_t)- \nabla_{a_t} Q_{\theta}(s_t, a_t) \right) \nabla_{\phi} f_{\phi}(\epsilon_t; s_t)
$$


!!! note "SAC main steps"


      1. **Q-function Update**:
        
        $$
        Q(s, a) \leftarrow r(s, a) + \mathbb{E}_{s' \sim p, a' \sim \pi} \left[ Q(s', a') - \log \pi(a' | s') \right].
        $$

        This update converges to $Q^\pi$, the Q-function under the current policy $\pi$.

      2. **Policy Update**:

        $$
        \pi_{\text{new}} = \arg\min_{\pi'} D_{KL} \left( \pi'(\cdot | s) \Bigg\| \frac{1}{Z} \exp Q^{\pi_{\text{old}}}(s, \cdot) \right).
        $$

        In practice, only one gradient step is taken on this objective to ensure stability.

      3. **Interaction with the Environment**:  
        Collect more data by interacting with the environment using the updated policy.



!!! note "**SAC Pseudocode** "

    Initialize parameter vectors $\psi, \bar{\psi}, \theta, \phi$

    **for** each iteration **do**  
    &nbsp;&nbsp;&nbsp;&nbsp;**for** each environment step **do**  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$a_t \sim \pi_{\phi}(a_t | s_t)$  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$s_{t+1} \sim p(s_{t+1} | s_t, a_t)$  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathcal{D} \gets \mathcal{D} \cup \{(s_t, a_t, r(s_t, a_t), s_{t+1})\}$  
    &nbsp;&nbsp;&nbsp;&nbsp;**end for**  

    &nbsp;&nbsp;&nbsp;&nbsp;**for** each gradient step **do**  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\psi \gets \psi - \lambda \hat{\nabla}_{\psi} J_V(\psi)$  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\theta_i \gets \theta_i - \lambda_Q \hat{\nabla}_{\theta_i} J_Q(\theta_i) \quad \text{for } i \in \{1,2\}$  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\phi \gets \phi - \lambda_{\pi} \hat{\nabla}_{\phi} J_{\pi}(\phi)$  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\bar{\psi} \gets \tau \psi + (1 - \tau) \bar{\psi}$  
    &nbsp;&nbsp;&nbsp;&nbsp;**end for**  
    **end for**


### **Summary**  

SAC is an **off-policy** algorithm that improves upon DDPG by introducing **entropy regularization**, which encourages exploration and robustness. It uses **two Q-networks (double Q-learning)** to mitigate overestimation bias and a **stochastic policy** for improved exploration.  

- **Advantages:**  
    - Better exploration due to entropy regularization.  
    - More stable than DDPG because of double Q-learning.  
    - Handles high-dimensional continuous control tasks efficiently.  
    - More sample-efficient than PPO.  

- **Disadvantages:**  
    - Higher computational cost due to multiple Q-function updates.  
    - Requires careful tuning of entropy coefficient $\alpha$.  
    - Slower than DDPG in deterministic settings where exploration is not an issue.  

### **Comparison Table: PPO vs. DDPG vs. SAC**  
<center>

| Algorithm | Type | Sample Efficiency | Stability | Exploration |
|-----------|------|------------------|----------|------------|
| **PPO**  | On-policy | Low | High | Moderate | Large-scale RL, discrete & continuous actions |
| **DDPG** | Off-policy | High | Moderate | Weak (deterministic) | Continuous control, robotics |
| **SAC**  | Off-policy | High | High | Strong (entropy regularization) |

</center>

## References

1. [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

2. [CS285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/)

3. [Reinforcement Learning Explained](http://172.27.48.15/Resources/Books/Nikolic%20L.%20Reinforcement%20Learning%20Explained.%20A%20Step-by-Step%20Guide...2023.pdf)

4. [An Introduction to Deep Reinforcement Learning](http://172.27.48.15/Resources/Books/Textbooks/An%20Introduction%20to%20Deep%20Reinforcement%20Learning.pdf)

5. [Deep Reinforcement Learning Processor Design for Mobile Applications](http://172.27.48.15/Resources/Books/Juhyoung%20Lee%2C%20Hoi-Jun%20Yoo%20-%20Deep%20Reinforcement%20Learning%20Processor%20Design%20for%20Mobile%20Applications-Springer%20%282023%29.pdf)

6. [REINFORCE — a policy-gradient based reinforcement Learning algorithm](https://medium.com/intro-to-artificial-intelligence/reinforce-a-policy-gradient-based-reinforcement-learning-algorithm-84bde440c816)

7. [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

8. [Deep Reinforcement Learning](http://172.27.48.15/Resources/Books/Textbooks/Aske%20Plaat%20-%20Deep%20Reinforcement%20Learning-arXiv%20%282023%29.pdf)

9. [Reinforcement Learning (BartoSutton)](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)