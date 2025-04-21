# Stochastic Bandits

## Core Assumptions
A **stochastic bandit** is a collection of distributions 

\[
\nu = (P_a : a \in \mathcal{A}),
\]

where \(\mathcal{A}\) is the set of available actions.  Over \(n\) rounds (possibly \(n=\infty\)), the interaction proceeds as follows:

1. At round \(t\), the learner chooses an action \(A_t \in \mathcal{A}\).  
2. The environment samples a reward,\(X_t \sim P_{A_t}\) and reveals \(X_t\) to the learner.  
3. This yields a history \((A_1, X_1, \dots, A_n, X_n)\).  

We require two key assumptions on this sequence:

- **(a) Environment sampling**
The reward at time \(t\) depends only on the chosen arm \(A_t\). In other

\[
X_t \,\bigl\lvert\, A_1, X_1,\dots,A_{t-1},X_{t-1},A_t\;\sim\;P_{A_t}.
\]  


- **(b) Non‑anticipation (Learner’s policy)**
The learner’s choice can only depend on past observations, not future ones. in other words:

\[
A_t \,\bigl\lvert\, A_1,X_1,\dots,A_{t-1},X_{t-1}
\;\sim\;\pi_t\bigl(\cdot\mid A_1,X_1,\dots,X_{t-1}\bigr),
\]  

## The Learning Objective

The learner aims to **maximize** the total reward

\[
S_n = \sum_{t=1}^n X_t,
\]

but this isn’t a straightforward optimization for three main reasons:

1. **Unknown horizon**  
   The number of rounds \(n\) may not be known in advance.  
2. **Random outcomes**  
   Even if all reward distributions were known, \(S_n\) is still a random variable requiring a utility measure.  
3. **Unknown distributions**  
   The learner must discover each arm’s reward law via interaction.

Points 1 and 2 can often be managed—e.g.\ by designing a policy for a guessed \(n\) and adapting it—so the core challenge is point 3.

## Beyond The Learning Objective

Beyond maximizing the raw sum \(S_n=\sum_{t=1}^n X_t\), we need a way to **measure** how well a learner performs **relative** to the best possible strategy.  A common approach is to compare the learner’s cumulative reward to that of the best fixed action in hindsight.  Concretely, define the *regret* after \(n\) rounds as

\[
R_n \;=\; \max_{a\in\mathcal A}\sum_{t=1}^n \mathbb{E}[X_t \mid A_t=a]\;-\;\sum_{t=1}^n X_t.
\]

Intuitively, \(R_n\) quantifies the loss from not always playing the optimal arm.  We aim to design policies for which \(R_n\) grows as slowly as possible (ideally sublinearly in \(n\)), so that average regret \(R_n/n\to0\).  A precise definition and discussion of **regret bounds** will be given later.


### Why Concentration Bounds Matter

Since rewards are random, even a policy with low expected regret may occasionally suffer large deviations.  To guarantee that \(R_n\) stays small **with high probability**, we need **concentration inequalities** (e.g.\ Hoeffding’s or Bernstein’s inequalities).  These tools let us bound the probability that empirical averages deviate from their means, which in turn controls how soon the learner can distinguish a good arm from a suboptimal one.

---



## Concentration Bounds

Concentration bounds are mathematical inequalities that provide upper (or lower) bounds on the probability that a random variable deviates from its expected value. They are particularly useful in the context of bandit problems, where we want to understand how our algorithm's estimates of the expected rewards can vary due to randomness.

### Concentration Inequalities

Concentration inequalities are mathematical tools that provide bounds on how a random variable deviates from some central value (like its mean). They are particularly useful in the context of bandit problems, where we want to understand how our algorithm's estimates of the expected rewards can vary due to randomness.
Concentration inequalities help us quantify the uncertainty in our estimates and provide guarantees on the performance of our algorithms. They are essential for deriving regret bounds and understanding the performance of bandit algorithms.

### Markov's Inequality

Markov's inequality is a simple but powerful tool for bounding the probability that a non-negative random variable exceeds a certain value. It's a fundamental result in probability theory and is often used in probabilistic analysis.

!!! danger "Markov's Inequality"

    Let $X$ be a non-negative random variable and let $a > 0$. Then, the probability that $X$ is greater than or equal to $a$ is bounded by the expected value of $X$ divided by $a$:

    $$
        P(X \geq a) \leq \frac{E[X]}{a}
    $$

!!! Note

    This inequality is particularly useful when we have a ==non-negative random variable== and we want to bound the probability of it being ==large==. It provides a simple way to relate the expected value of the random variable to its tail behavior.

!!! Proof

    To prove Markov's inequality, we start with the definition of the expected value of a non-negative random variable $X$:

    $$
        E[X] = \int_0^\infty x f_X(x) dx
    $$

    where $f_X(x)$ is the probability density function of $X$.
    Now, we can split the integral into two parts: one for $x < a$ and one for $x \geq a$:

    $$
        E[X] = \int_0^a x f_X(x) dx + \int_a^\infty x f_X(x) dx
    $$

    The first part of the integral is non-negative, so we can ignore it for our purposes. Now, we can bound the second part of the integral:

    $$
        E[X] \geq \int_a^\infty x f_X(x) dx
    $$

    Now, note that the integral domain is $[a, \infty)$, hence $x \geq a$ for all $x$ in this domain. Therefore, we can replace $x$ with $a$ in the integral:

    $$
    E[X] \geq \int_a^\infty x f_X(x) dx \geq \int_a^\infty a f_X(x) dx
    $$

    Finally, we can factor out $a$ from the integral:

    $$
        E[X] \geq a \int_a^\infty f_X(x) dx = a P(X \geq a)
    $$

    Deviding both sides by $a$ gives us the desired result:

    $$
        P(X \geq a) \leq \frac{E[X]}{a}
    $$

This means that the probability of $X$ being greater than or equal to $a$ is at most the expected value of $X$ divided by $a$.
This inequality is useful in bandit problems because it allows us to bound the probability of large deviations from the expected reward, which can help us understand the performance of our algorithms. However, it is important to note that Markov's inequality is quite loose and may not provide tight bounds in many cases.

You may ask what does "loose" mean? Let's consider a simple example:

!!! example "Markov's Inequality Example"

    Let $X$ be a random variable that takes the value 0 with probability 0.9 and the value 10 with probability 0.1. Then, we have:

    $$
        E[X] = 0 \cdot 0.9 + 10 \cdot 0.1 = 1
    $$

    Now, let's apply Markov's inequality for $a = 5$:

    $$
        P(X \geq 5) \leq \frac{E[X]}{5} = \frac{1}{5} = 0.2
    $$

    However, in reality, $P(X \geq 5) = 0.1$, which is much smaller than the bound provided by Markov's inequality.

We will discuss more about the tightness of the bounds in the next sections, but for now, keep in mind that Markov's inequality is a loose bound that can be useful in certain situations, but it may not provide the best estimates for the probability of large deviations.

### Chebyshev's Inequality

Chebyshev's inequality is a powerful tool for bounding the probability that a random variable deviates from its ==mean==. It is particularly useful when we have a random variable with a known variance, and we want to understand ==how likely it is to be far from its expected value==. It is a more refined version of Markov's inequality, as it takes into account the variance of the random variable, and it doesn't require the random variable to be non-negative.

!!! danger "Chebyshev's Inequality"

    Let $X$ be a random variable with mean $\mu = E[X]$ and variance $\sigma^2 = Var(X)$. Then, for any $k > 0$, the probability that $X$ deviates from its mean by more than $k$ standard deviations is bounded by:

    $$
        P(|X - \mu| \geq k \sigma) \leq \frac{1}{k^2}
    $$

Note that some books may write this as:

$$
    P(|X - \mu| \geq k) \leq \frac{Var(X)}{k^2}
$$

which is equivalent to the above inequality when we set $k = k \sigma$.

!!! Note

    This inequality is particularly useful when we have a random variable with a known mean and variance, and we want to bound the probability of it being far from its expected value. It provides a more refined estimate than Markov's inequality, as it takes into account the variance of the random variable.

!!! Note

    Chebyshev's inequality is a general result that applies to any random variable with a ==finite mean and variance==. It is not specific to any particular distribution, which makes it a powerful tool in probability theory.

Chebyshev's inequality can be easily derived from Markov's inequality. Let's break it down:

!!! proof "Chebyshev's Inequality Proof"

    Start with the definition of variance:

    $$
        Var(X) = E[(X - \mu)^2] = E[X^2] - \mu^2
    $$

    Now, we can apply Markov's inequality to the non-negative random variable $(X - \mu)^2$:

    $$
        P(|X - \mu| \geq k \sigma) = P((X - \mu)^2 \geq k^2 \sigma^2) \leq \frac{E[(X - \mu)^2]}{k^2 \sigma^2}
    $$

    Substitute the expression for variance into the inequality:

    $$
        P(|X - \mu| \geq k \sigma) \leq \frac{Var(X)}{k^2 \sigma^2}
    $$

    Finally, note that $\sigma^2 = Var(X)$. Therefore, we can simplify the inequality:

    $$
        P(|X - \mu| \geq k \sigma) \leq \frac{1}{k^2}
    $$

Compared to Markov's inequality, Chebyshev's inequality is more refined as it accounts for the variance of the random variable. However, it still provides a loose bound and may not yield accurate estimates for the probability of large deviations in many cases.

!!! example "Chebyshev's Inequality Example"

    Let $X$ be a normal random variable with mean $\mu = 0$ and variance $\sigma^2 = 1$,i.e., $X \sim N(0, 1)$. We can apply Chebyshev's inequality to bound the probability that $X$ deviates from its mean by more than $a$ standard deviations:

    $$
        P(|X - 0| \geq a) \leq \frac{1}{a^2}
    $$

    However, we know that for a normal distribution, the probability of $X$ being more than $a$ standard deviations away from the mean is much smaller than the bound provided by Chebyshev's inequality. To see this, we can calculate the actual probability using the cumulative distribution function (CDF) of the normal distribution, keeping in mind that normal distributions is symmetric:

    $$
        P(|X| \geq a) = 2 \cdot P(X \geq a) = 2 \cdot (1 - \Phi(a))
    $$

    where $\Phi(a)$ is the CDF of the standard normal distribution. Note that $\Phi(a)$ has exponential tails, which means that the actual probability of $X$ being more than $a$ standard deviations away from the mean is much smaller than the bound provided by Chebyshev's inequality, which has polynomial tails.

### Can't we just use Cental Limit Theorem?

Now that we have find out that Markov's and Chebyshev's inequalities are loose bounds, you may wonder why we don't just use the Central Limit Theorem (CLT) to bound the probability of large deviations. Remark that CLT states that the sum of a large number of independent and identically distributed (i.i.d.) random variables converges to a normal distribution. While this is true, the CLT only provides asymptotic results, meaning that it applies when the number of samples is very large.

For a better illustration, consider the following theorems:

!!! danger "Lidenberg-Levy Central Limit Theorem"

    Let $X_1, X_2, \ldots, X_N$ be i.i.d. random variables with mean $\mu$ and variance $\sigma^2$. Then, define sum $S_N = X_1 + X_2 + \ldots + X_N$. Define $Z_N$ to be normalized version of $S_N$:

    $$
        Z_N = \frac{S_N - E[S_N]}{\sqrt{Var(S_N)}} = \frac{1}{\sigma \sqrt{N}} \sum_{i=1}^N (X_i - \mu)
    $$

    Lidenberg-Levy CLT states that $Z_N$ converges in distribution to a standard normal random variable as $N \to \infty$.

!!! danger "Berry-Esseen Central Limit Theorem"

    Let $X_1, X_2, \ldots, X_N$ be i.i.d. random variables with mean $\mu$ and variance $\sigma^2$. Then, define sum $S_N = X_1 + X_2 + \ldots + X_N$. Define $Z_N$ to be normalized version of $S_N$:

    $$
        Z_N = \frac{S_N - E[S_N]}{\sqrt{Var(S_N)}} = \frac{1}{\sigma \sqrt{N}} \sum_{i=1}^N (X_i - \mu)
    $$

    Berry-Esseen theorem states that the convergence of $Z_N$ to a standard normal random variable is not only asymptotic but also has a rate of convergence. Specifically, it provides an upper bound on the difference between the distribution of $Z_N$ and the standard normal distribution.

    $$
        |P(Z_N \leq x) - \Phi(x)| \leq C \frac{\rho}{\sigma^3 N}
    $$

    where $\rho$ is the third absolute moment of the random variable $X_i$, and $C$ is a constant.

You don't need to worry about the details of the Berry-Esseen theorem, but the important point is that it provides a rate of convergence for the Central Limit Theorem (CLT). The CLT approximates the sum of independent random variables $S_N = X_1 + \dots + X_N$ by a normal distribution, which has very light, exponentially decaying tails. However, the rate of convergence is $O(1/\sqrt{N})$, meaning that the probability of large deviations may still be significant even for large $N$. This slow decay of the approximation error—slower than linear—undermines the desired concentration properties of $S_N$, preventing it from achieving the same light-tailed behavior as the normal distribution.

Now that we have discussed the limitations of CLT, let's move on to some other concentration inequalities that provide tighter bounds for the probability of large deviations.

### Hoeffding's Inequality

Hoeffding's inequality is a powerful concentration inequality that provides bounds on the probability of large deviations for sums of independent random variables.

!!! Info "Definition (Symmetric Bernoulli distribution)"

    A random variable $X$ has symmetric Bernoulli *distribution* (also called *Rademacher distribution*) if it takes values $-1$ and $1$ with probabilities $\frac{1}{2}$ each, i.e.

    $$
    \mathbb{P} \{ X = -1 \} = \mathbb{P} \{ X = 1 \} = \frac{1}{2}.
    $$

    Clearly, a random variable $X$ has (usual) Bernoulli distribution with parameter $\frac{1}{2}$ if and only if $Z = 2X - 1$ has symmetric Bernoulli distribution.

!!! Danger "Theorem 2.2.2 (Hoeffding’s inequality)"

    Let $X_1, \dots, X_N$ be independent symmetric Bernoulli random variables, and let $a = (a_1, \dots, a_N) \in \mathbb{R}^N$. Then, for any $t > 0$, we have

    $$
    \mathbb{P} \left\{ \sum_{i=1}^{N} a_i X_i \geq t \right\} \leq \exp \left( \frac{-t^2}{2 \| a \|_2^2} \right).
    $$

Note that the above theorem is a special case of Hoeffding's inequality for symmetric Bernoulli random variables. The general form of Hoeffding's inequality applies to any bounded independent random variables, and it provides a more general result. Also pay attention to the exponential decay of the probability bound, which is much faster than the polynomial decay provided by Chebyshev's inequality. For now, let's focus on the special case of symmetric Bernoulli random variables and provide a proof for the above theorem. This proof is provided from ==Vershynin's High Dimensional Probability book==, which is a great resource for understanding concentration inequalities and their applications in high-dimensional spaces.

!!! Proof

    Let us recall how we deduced Chebyshev’s inequality : we squared both sides and applied Markov’s inequality. Let us do something similar here. But instead of squaring both sides, let us multiply by a fixed parameter $\lambda > 0$(to be chosen later) and exponentiate. This gives

    $$
    \mathbb{P} \left\{ \sum_{i=1}^{N} a_i X_i \geq t \right\} = \mathbb{P} \left\{ \exp \left( \lambda \sum_{i=1}^{N} a_i X_i \right) \geq \exp(\lambda t) \right\}
    $$

    $$
    \leq e^{-\lambda t} \, \mathbb{E} \exp \left( \lambda \sum_{i=1}^{N} a_i X_i \right).
    \tag{1.1}
    $$

    In the last step we applied Markov’s inequality.

    We thus reduced the problem to bounding the *moment generating function* (MGF) of the sum $\sum_{i=1}^{N} a_i X_i$. As we recall from the basic probability course, the MGF of the sum is the product of the MGF’s of the terms; this follows immediately from independence. In other words,

    $$
    \mathbb{E} \exp \left( \lambda \sum_{i=1}^{N} a_i X_i \right) = \prod_{i=1}^{N} \mathbb{E} \exp(\lambda a_i X_i).
    \tag{1.2}
    $$

    Let us fix $i$. Since $X_i$ takes values $-1$ and $1$ with probabilities $1/2$ each, we have

    $$
    \mathbb{E} \exp(\lambda a_i X_i) = \frac{\exp(\lambda a_i) + \exp(-\lambda a_i)}{2} = \cosh(\lambda a_i).
    $$

    !!! note "Bounding the hyperbolic cosine"

        It's easy to show that:

        $$
        \cosh(x) \leq \exp(x^2/2) \quad \text{for} \quad x \in \mathbb{R}.
        $$

        DIY (Compare the Taylor’s expansions of both sides.)

    This bound shows that

    $$
    \mathbb{E} \exp(\lambda a_i X_i) \leq \exp(\lambda^2 a_i^2/2).
    $$

    Substituting this into (1.2) and then into (1.1), we obtain

    $$
    \mathbb{P} \left\{ \sum_{i=1}^N a_i X_i \geq t \right\} \leq e^{-\lambda t} \prod_{i=1}^N \exp(\lambda^2 a_i^2/2) = \exp \left( -\lambda t + \frac{\lambda^2}{2} \sum_{i=1}^N a_i^2 \right) = \exp \left( -\lambda t + \frac{\lambda^2}{2} \right).
    $$

    In the last identity, we used the assumption that $\|a\|_2 = 1$.

    This bound holds for arbitrary $\lambda > 0$. It remains to optimize in $\lambda$; the minimum is clearly attained for $\lambda = t$. With this choice, we obtain

    $$
    \mathbb{P} \left\{ \sum_{i=1}^N a_i X_i \geq t \right\} \leq \exp(-t^2/2).
    $$

    This completes the proof of Hoeffding’s inequality.

Hoefding's bound has two other forms, which we introduce here for completeness, but proofs are omitted.

!!! danger "Hoeffding’s inequality, two-sided"

    Let $X_1, \ldots, X_N$ be independent symmetric Bernoulli random variables, and $a = (a_1, \ldots, a_N) \in \mathbb{R}^N$. Then, for any $t > 0$, we have

    $$
    \mathbb{P} \left\{ \left| \sum_{i=1}^N a_i X_i \right| \geq t \right\} \leq 2 \exp \left( -\frac{t^2}{2 \|a\|_2^2} \right).
    $$

    !!! Note "Proof sketch"
        Use:

        $$
        \mathbb{P} \{ |S| \geq t\} = \mathbb{P} \{ S \geq t\} + \mathbb{P} \{-S \geq t\}.
        $$

!!! danger "Hoeffding's inequality for general bounded random variables"

    Let $X_1, \ldots, X_N$ be independent random variables. Assume that $X_i \in [m_i, M_i]$ almost surely for every $i$. Then, for any $t > 0$, we have

    $$
    \mathbb{P} \left\{ \sum_{i=1}^N (X_i - \mathbb{E}X_i) \geq t \right\} \leq \exp \left( -\frac{2t^2}{\sum_{i=1}^N (M_i - m_i)^2} \right)
    $$
    !!! Note "Proof sketch"
        Use the fact that $X_i$ is bounded to construct a symmetric Bernoulli random variable $Y_i$ such that $X_i = m_i + (M_i - m_i) Y_i$. Then, apply Hoeffding's inequality for the symmetric Bernoulli random variables $Y_i$.

Hoeffding's inequality for a general bounded random variable is a powerful tool for bounding the probability of large deviations of the sum of independent random variables. Note that this senario happens often in bandit problems, where we have a finite number of arms and we want to bound the probability of large deviations of the sum of rewards from the expected value.

### Chernoff Bound

Chernoff Bound is a powerful concentration inequality that provides sharp exponential bounds on the tail probabilities for sums of independent random variables. In the context of bandit problems, this bound is particularly useful because it quantifies the likelihood that the observed sum of rewards (or losses) significantly deviates from its expected value. Intuitively, the bound tells us that the probability of witnessing a large deviation decays exponentially as the deviation increases, which is a much stronger guarantee than those offered by bounds like Chebyshev's inequality. This property is instrumental when constructing high-confidence estimates for the rewards of different arms.

In bandit algorithms, such as those based on upper confidence bounds (UCB), the Chernoff Bound allows us to build tight confidence intervals around the estimated rewards. These intervals serve as a guide for choosing which arm to pull next by balancing the exploration of less tried arms with the exploitation of arms that have previously shown good performance. Thus, Chernoff's exponential decay of tail probabilities helps ensure that the algorithm's estimates remain reliable even as the exploration continues, leading to more effective regret minimization over time.

!!! danger "Chernoff's inequality"

    Let \( X_i \) be independent Bernoulli random variables with parameters \( p_i \). Consider their sum

    \[
    S_N = \sum_{i=1}^N X_i
    \]

    and denote its mean by

    \[
    \mu = \mathbb{E} S_N.
    \]

    Then, for any \( t > \mu \), we have

    \[
    \mathbb{P} \{ S_N \geq t \} \leq e^{-\mu} \left( \frac{e\mu}{t} \right)^t.
    \]

    In particular, for any \( t \geq e^2 \mu \) we have

    \[
    \mathbb{P} \{ S_N \geq t \} \leq e^{-t}.
    \tag{2.6}
    \]

!!! Proof

    We will use the same method - based on moment generating function - as we did in the proof of Hoeffding's inequality. We repeat the first steps of that argument, leading to (1.1) and (1.2) - multiply both sides of the inequality \( S_N \geq t \) by a parameter \(\lambda\), exponentiate, and then use Markov's inequality and independence. This yields

    \[
    \mathbb{P} \{ S_N \geq t \} \leq e^{-\lambda t} \prod_{i=1}^{N} \mathbb{E} \exp(\lambda X_i).
    \tag{2.1}
    \]

    It remains to bound the MGF of each Bernoulli random variable \( X_i \) separately. Since \( X_i \) takes value 1 with probability \( p_i \) and 0 with probability \( 1-p_i \), we have

    \[
    \mathbb{E} \exp(\lambda X_i) = e^{\lambda} p_i + (1 - p_i) = 1 + (e^{\lambda} - 1)p_i \leq \exp \left[ (e^{\lambda} - 1)p_i \right].
    \]

    In the last step, we used the numeric inequality \( 1 + x \leq e^x \). Consequently,

    \[
    \prod_{i=1}^{N} \mathbb{E} \exp(\lambda X_i) \leq \exp \left[ (e^{\lambda} - 1) \sum_{i=1}^{N} p_i \right] = \exp \left[ (e^{\lambda} - 1) \mu \right].
    \]

    Substituting this into (2.1), we obtain

    \[
    \mathbb{P} \{ S_N \geq t \} \leq e^{-\lambda t} \exp \left[ (e^{\lambda} - 1) \mu \right].
    \]

    This bound holds for any \(\lambda > 0\). Substituting the value \(\lambda = \ln(t/\mu)\) which is positive by the assumption \( t > \mu \) and simplifying the expression, we complete the proof.

### Sub-Gaussian Random Variables

Sub-Gaussian random variables decay ==at least as fast as Gaussian ones==, meaning their tails are exponentially light. This ensures that large deviations from the mean are unlikely, a property crucial for concentration inequalities like Hoeffding's and Chernoff's bounds. Such bounds are extensively used to provide robust statistical guarantees in high-dimensional statistics, machine learning, and theoretical computer science, even when the underlying distribution is not normal.

!!! tip "Bounded random variables are sub-Gaussian"

    Every bounded random variable is sub-Gaussian. This is because a bounded random variable has a finite variance, and thus it satisfies the sub-Gaussian condition.

So if you have a bounded random variable, you can use the sub-Gaussian properties to derive concentration inequalities and bounds on the tail probabilities. Now that we have discussed the properties of sub-Gaussian random variables, let's define them rigorously. Note that ==Vershynin== provides a more general definition of sub-Gaussian random variables - actually you can use 5 equivalent definitions - but we will use the one that is most commonly used in the literature.

!!! danger "Definition (Sub-Gaussian random variable)"

    A random variable \(X\) is said to be sub-Gaussian if there exists a constant \(C > 0\) such that for all \(t > 0\), we have:

    $$
        P(|X| \geq t) \leq 2 \exp \left( -\frac{t^2}{C^2} \right).
    $$

    This means that the tails of the distribution of \(X\) decay at least as fast as the tails of a Gaussian distribution with variance \(C^2\).

In stochastic multi-armed bandits, rewards with subgaussian noise allow concentration inequalities like Hoeffding’s to derive optimal exploration-exploitation trade-offs, ensuring \(O(\sqrt{T \log T})\) regret. This applies to bounded rewards (e.g., [0,1]) or Gaussian noise, simplifying analysis.

They also enable **adaptive algorithms for unknown variance** (e.g., Variance-UCB) and **linear bandits** (e.g., LinUCB), where subgaussianity ensures fast convergence of least-squares estimates. Even heavy-tailed rewards can be clipped to enforce subgaussianity, making them tractable. The light-tailed property is key for robustness and privacy in bandits.

## Regret Bound

Regret is a measure of how well an algorithm performs compared to the best possible strategy in hindsight. In the context of bandit problems, regret quantifies the difference between the ==expected== reward of the optimal arm and the ==expected== reward of the arm chosen by the algorithm.

Regret is a crucial concept in bandit problems, as it allows us to evaluate the performance of our algorithms and compare different strategies. The goal of most bandit algorithms is to minimize regret over time, which means that we want to maximize the expected reward while minimizing the difference between our algorithm's performance and the optimal performance.

In mathematical terms, if ==$r^*$ is the expected reward== of the optimal arm and $r_t$ is the expected reward of the arm chosen by the algorithm at time $t$, then the regret at time $t$ is defined as:

!!! Info "Regret definition"

    $$
        R_t = r^* - r_t
    $$

    where $r_i$ is the expected reward of the arm chosen by the algorithm at time $i$.

The total regret over $T$ rounds is defined as:

!!! Info "Total Regret definition"

    $$
        R_T = \sum_{t=1}^T R_t = \sum_{t=1}^T (r^* - r_t) = T r^* - \sum_{t=1}^T r_t
    $$

!!! warning "Why expected reward?"

    The term **expected reward** is central to multi-armed bandit algorithms because it captures the _long-term average performance_ of an arm, accounting for randomness in outcomes. Bandit algorithms aim to maximize cumulative rewards, but since observed rewards are stochastic (e.g., noisy clicks or purchases), the **expectation** (mean) provides a stable measure to compare arms. For example, UCB and Thompson Sampling rely on estimating \(\mathbb{E}[r_i]\) for each arm \(i\) to guide decisions, ensuring optimal exploration-exploitation trade-offs.

    Alternatively, there are other ways to define regret, such as using the maximum reward of the chosen arm or the maximum reward of the optimal arm. However, there is one more elegent reason to use expected reward: it allows us to focus on the *long-term performance* of the algorithm, rather than the short-term fluctuations. This is particularly important in bandit problems, where we want to understand how well our algorithm performs over time, rather than just in a single round.

    If you are confused, let's first define a well-studied branch of Bandit algorithms, called **Adversarial Bandits**. and see why the term *expected reward* is here to rescue.

    !!! danger "Adversarial Bandits"

        In adversarial bandits, the rewards are chosen by an adversary, and the algorithm has no prior knowledge about the distribution of the rewards. In this case, the algorithm must make decisions based on the observed rewards, and it cannot rely on any assumptions about the underlying distribution.

    Now, let's consider an adversarial bandit problem where the rewards are chosen by me, and you can pull the arms.

    !!! example "Why not min $r_t$ nor max $r_t$"

        For simplicity, consider the case where there are only two arms, and the rewards are either 0 or 1. If I choose the rewards in such a way that one arm always gives a reward of 1 and the other arm always gives a reward of 0, then the expected reward of the optimal arm is 1, and the expected reward of the chosen arm is either 0 or 1. In this case, if you define regret as the maximum reward of the chosen arm, then you will have a regret of 0, with non-zero probability. Then if you define regret as the minimum reward of the chosen arm, then you will have a regret of 1, with non-zero probability. In both cases, the regret is not a good measure of the performance of the algorithm, because it does not capture the difference between the expected reward of the optimal arm and the expected reward of the chosen arm.

Now that we have find out regret of an algorithm is actually a random variable, it's very rational that we shall try to find out that ==on average== how does algorithm perform.

### Regret Bound definition

Regret bounds offer simplified, worst-case guarantees on algorithm performance, making it easier to analyze and compare strategies regardless of specific outcomes. They provide an upper limit on potential regret over time, which is especially valuable when exact regret values are difficult to calculate and when ensuring robust performance is critical.

Regret bounds are typically expressed in terms of the number of rounds $T$ and the number of arms $K$. A common form of regret bound is:
!!! Info "Regret Bound definition"

    $$
        R_T \leq C \cdot f(T, K)
    $$

    where $C$ is a constant, $f(T, K)$ is a function of the number of rounds $T$ and the number of arms $K$, and $R_T$ is the total regret over $T$ rounds.

Now, of course, evey algorithm makes mistakes to some extent, and uses it to learn. In fact, there is a graet quote by Paul Tobin, that says:

> "If there is not folly in the world, then the world itself is folly. You must understand that mistakes are not always regrets."

In literature, people often classify the regret of algorithms by it's ratio over $T$. Note how this table summarizes the different types of regret growth and their corresponding efficiency levels:

| **Regret Growth** | **Efficiency**       | **Definition**                              | **Example Algorithm**  | **Use Case**             |
| ----------------- | -------------------- | ------------------------------------------- | ---------------------- | ------------------------ |
| **Linear**        | Inefficient          | Regret grows as \(O(T)\)                    | Random Selection       | Worst-case, no learning  |
| **Sublinear**     | Efficient            | Regret grows slower than \(O(T)\)           | ε-Greedy               | Basic exploration        |
| **Logarithmic**   | Optimal              | Regret grows as \(O(\log T)\)               | UCB, Thompson Sampling | Stochastic bandits       |
| **Square-root**   | Near-optimal         | Regret grows as \(O(\sqrt{T})\)             | EXP3                   | Adversarial bandits      |
| **Polynomial**    | Moderately efficient | Regret grows as \(O(T^\alpha), \alpha < 1\) | Gradient Bandit        | Continuous action spaces |

## UCB (Upper Confidence Bound)

The UCB strategy manages the exploration-exploitation trade-off in multi-armed bandit problems. This tutorial covers two main variants—UCB1 and UCB2—and explains their regret bounds.

---

### UCB1

UCB1 is a simple yet powerful algorithm that selects arms based on the following idea:

- Each arm's prior performance is measured by the sample mean reward.
- A bonus term, which decreases as the arm is played more often, is added to ensure that less frequently chosen arms are explored.

#### Algorithm Steps

1. **Initialization:**  
   Play each arm at least once to collect initial rewards.

2. **Selection in Subsequent Rounds:**  
   For every new round, select the arm _i_ that maximizes the following expression:

   - **Expression:**  
     **Sample Mean** + **Exploration Bonus**

   - **Exploration Bonus Formula:**  
     $$\sqrt{\frac{2\ln(\text{total plays})}{\text{plays of arm } i}}$$

#### Python Example:

Below is a sample implementation of UCB1 in Python:

```py
import math
import random

def ucb1(num_rounds, arms):
    total_rounds = len(arms)
    counts = [1] * len(arms)
    rewards = [arm() for arm in arms]

    for t in range(total_rounds, num_rounds):
        ucb_values = [
                (rewards[i] / counts[i]) + math.sqrt(2 * math.log(t + 1) / counts[i])
                for i in range(len(arms))
        ]
        chosen_arm = ucb_values.index(max(ucb_values))
        reward = arms[chosen_arm]()
        counts[chosen_arm] += 1
        rewards[chosen_arm] += reward

    return counts, rewards

arms = [
    lambda: random.gauss(1, 1),
    lambda: random.gauss(2, 1),
    lambda: random.gauss(1.5, 1)
]

counts, rewards = ucb1(1000, arms)
print("Arm plays:", counts)
```

#### Explanation:

- The algorithm starts by initializing counts and rewards.
- In every round, it calculates the UCB value for each arm and chooses the one with the highest value.
- The exploration bonus ensures arms with fewer plays are selected more often to gather more information.

---

### UCB2

UCB2 is a variant that modifies the exploration bonus to achieve a smoother trade-off between exploration and exploitation. Instead of updating after every round, UCB2 uses **epochs** or rounds where an arm is played a predetermined number of times depending on a parameter ρ (rho).

#### Key Differences from UCB1:

- **Epoch-based updates:** Rather than updating at each pull, UCB2 commits to an arm for a series of rounds.
- **Parameter ρ:** Controls the length of each epoch, balancing between exploration and exploitation. A smaller ρ leads to more frequent updates similar to UCB1.

#### High-Level Steps:

1. For each arm, determine the epoch length based on its play count.
2. Play the chosen arm for the duration of its epoch.
3. Update the statistics after each epoch.

UCB2 can provide theoretical improvements under certain conditions and is especially useful when more stable arm selection over short windows is preferred.

---

### Regret Bound for UCB1

For a multi-armed bandit problem with $K$ arms, let $\mu^*$ be the expected reward of the optimal arm and $\mu_i$ be the expected reward of arm $i$. Define the gap for each suboptimal arm as $\Delta_i = \mu^* - \mu_i$. Then the regret of UCB1 after $T$ plays, $R(T)$, can be bounded as follows:

For every suboptimal arm $i$ (with $\Delta_i > 0$),

$$
E[n_i(T)] \leq \frac{8 \ln T}{\Delta_i^2} + \left(1 + \frac{\pi^2}{3}\right),
$$

and hence the total expected regret satisfies

$$
R(T) = \sum_{i: \Delta_i > 0} \Delta_i \cdot E[n_i(T)] \leq \sum_{i: \Delta_i > 0} \left( \frac{8 \ln T}{\Delta_i} + \left(1 + \frac{\pi^2}{3}\right)\Delta_i \right).
$$


This shows that UCB1 attains a **logarithmic regret**, which is order-optimal up to constant factors.

!!! Proof 

    #### 1. **Initialization**  
    Each arm is pulled once, ensuring that initial estimates are available.

    #### 2. **Confidence Bounds & Chernoff-Hoeffding Inequality**
    For each arm $i$, the UCB1 algorithm computes

    $$
    UCB_i(t) = \hat{\mu}_i(t) + \sqrt{\frac{2 \ln t}{n_i(t)}},
    $$

    where $\hat{\mu}_i(t)$ is the sample mean reward of arm $i$ and $n_i(t)$ is the number of times arm $i$ has been played until time $t$.

    Using the **Chernoff-Hoeffding bound**, one can show that, with high probability, the true mean $\mu_i$ lies within the confidence interval given by $\hat{\mu}_i(t) \pm \sqrt{\frac{2 \ln t}{n_i(t)}}$.

    #### 3. **Bounding the Number of Suboptimal Pulls**
    Assume arm $i$ is suboptimal ($\mu_i < \mu^*$). If arm $i$ is selected at time $t$, then it must be that

    $$
    \hat{\mu}_i(t) + \sqrt{\frac{2 \ln t}{n_i(t)}} \geq \mu^*.
    $$

    By rearranging terms and using concentration bounds, it implies that the number of times $n_i(T)$ that arm $i$ is played is bounded by a term proportional to $\ln T$ divided by $\Delta_i^2$. More precisely, one can show that

    $$
    E[n_i(T)] \leq \frac{8 \ln T}{\Delta_i^2} + \text{constant}.
    $$

    #### 4. **Total Regret Bound**
    The expected regret when arm $i$ is pulled is $\Delta_i$, so summing over all suboptimal arms gives

    $$
    R(T) = \sum_{i: \Delta_i > 0} \Delta_i E[n_i(T)] \leq \sum_{i: \Delta_i > 0} \left( \frac{8 \ln T}{\Delta_i} + \text{constant} \times \Delta_i \right).
    $$

    This derivation uses a careful application of the **union bound** over the time steps and the **concentration inequality** to control the probability of overestimating the reward of a suboptimal arm. The constants (like $8$ and $\pi^2/3$) arise from detailed analysis in the original proofs by Auer et al.

---

Thus, **UCB1 enjoys a regret bound of order O(ln T)** for **T** rounds, which is optimal for many stochastic bandit problems.

References:

- Vershynin, R. (2018). High-dimensional probability: An introduction with applications in data science. Cambridge University Press.
- Tor Lattimore, Csaba Szepesvari. Bandit Algorithms. Cambridge University Press, 2020.