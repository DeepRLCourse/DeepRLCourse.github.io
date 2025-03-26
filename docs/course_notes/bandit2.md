## Concentration Bounds

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







### Chernoff Bound

Chernoff bounds are a family of inequalities that provide exponentially decreasing bounds on the tail probabilities of sums of independent random variables. Chernoff bounds are more refined than Chebyshev's inequality and provide tighter bounds on the tail probabilities, especially for sums of independent random variables.

Chernoff bounds are particularly useful when we have a large number of independent random variables and we want to bound the probability of their sum deviating from its expected value, which is a common scenario in bandit problems. 

!!! danger "Chernoff Bound"
    Let $X_1, X_2, \ldots, X_n$ be independent random variables with $E[X_i] = \mu_i$. Let $\mu = \sum_{i=1}^n \mu_i$. Then, for any $\delta > 0$, we have:

    $$
        P\left(\sum_{i=1}^n X_i \geq (1 + \delta) \mu\right) \leq e^{-\frac{\delta^2}{2 + \delta} \mu}
    $$

    and

    $$
        P\left(\sum_{i=1}^n X_i \leq (1 - \delta) \mu\right) \leq e^{-\frac{\delta^2}{2} \mu}
    $$
    
    where $\delta > 0$ is a parameter that controls the level of deviation from the expected value.

### Hoeffding's Inequality

### Sub-Gaussian Random Variables

### Bernstein's Inequality

## Regret Bound

### Regret Bound definition

## UCB (Upper Confidence Bound)

### UCB1

### UCB2

### Regret Bound for UCB
