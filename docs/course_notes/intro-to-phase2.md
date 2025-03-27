# Introduction to RL in Depth
## goal
get deeper into the concepts. learn new math that are usfull for RL 

## overview


### value iteration
As mentioned in the first part of the course, optimal value functions, $V^*(s)$ or  $Q^*(s, a)$ must satisfy Bellman's optimality equations:

$$
Q^*(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} Q^*(s_{t+1}, a') \;\middle|\; s_t = s, a_t = a \right]
$$

$$
V^*(s) = \mathbb{E} \left[ R_{t+1} + \gamma V^*(s_{t+1}) \;\middle|\; s_t = s \right]
$$



Therefore by finding $V^*(s)$ and $Q^*(s, a)$ through Bellman optimality equations, one can determine optimal policy. Specifically, dynamic programming (DP) methods are obtained by simply turning these equations into an update rule.

Value iteration simply finds $V^*(s)$ by iterating through these 2 equations, the update rule can be written as a simple backup operation : 

$$
V_{k+1}(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s), s' \sim P(\cdot \mid s, a)} \left[ R(s, a, s') + \gamma V_k(s') \right]
$$


For the sake of conviniency we define the Bellman operator for having more intuition on how fixed point algorithms workout!

$$Q \xrightarrow{\mathcal{T}} Q \xrightarrow{\mathcal{T}} Q \xrightarrow{\mathcal{T}} Q$$

!!! abstract "Definition"

    $\textbf{Bellman Operator}$ _A Bellman optimality operator_ $\mathcal{T}
    :\mathbb{R}^{|S|} \to \mathbb{R}^{|S|}$ _is an operator that satisfies: for any_ $V 
    \in \mathbb{R}^{|S|}$,

    $$
    (\mathcal{T} V)(s) = \max_{a} \left[ r(s, a) + \gamma \mathbb{E}_{s' \sim T(s' \mid
    s, a)} V(s') \right].
    $$





Value iteration can thus be represented as recursively applying the Bellman optimality 
operator:

$$
V_{k+1} = \mathcal{T} V_k. 
$$

The Bellman optimality operator $\mathcal{T}$ has several excellent properties. It is 
easy to verify that $V^*$ is a fixed point of $\mathcal{T}$, i.e., $\mathcal{T} V^* = 
V^*$. Another important property is that $\mathcal{T}$ is a contraction mapping.

### contraction mapping
!!! note "Theorem"


     $\textit{Contraction Mapping Theorem}$ $\mathcal{T}$ _is a contraction mapping under
     sup-norm_
     $\|\cdot\|_\infty$,
        _i.e.,  there exists_ $\gamma \in [0,1]$ _such that_


    $$
     \|\mathcal{T}U - \mathcal{T}V\|_\infty \leq \gamma \|U - V\|_\infty, \quad \forall
     U, V \in \mathbb{R}^{|S|}.
    $$


    ???+ note "Proof"

        To prove this property, we need the following lemma:

        ???+ info "**Lemma**"

            $$
            \begin{aligned}
            \left| \max_{a} f(a) - \max_{a} g(a) \right| \leq \max_{a} | f(a) - g(a) |.
            \end{aligned}
            $$

            This Lemma is proved as follows. Assume without loss of generality that
             $\max_{a}
            f(a) \geq \max_{a} g(a)$, and denote $a^* = \arg\max_{a} f(a)$. Then,
            
            $$
            \begin{aligned}
            \left| \max_{a} f(a) - \max_{a} g(a) \right| = \max_{a} f(a) - \max_{a} g(a) =
             f(a^*) - \max_{a} g(a) \leq f(a^*) - g(a^*) \leq \max_{a} | f(a) - g(a) |.
            \end{aligned} 
            $$ 

        We now proceed to prove Theorem 2. For any state $s$, we have

        $$
        \begin{aligned}
        | \mathcal{T}V(s) - \mathcal{T}U(s) | &=  
        \left| \max_{a} \left[ r(s, a) + \gamma \mathbb{E}_{s' \sim T(s' \mid s,a)} V(s') \right]
        - \max_{a} \left[ r(s, a) + \gamma \mathbb{E}_{s' \sim T(s' \mid s,a)} U(s') \right] \right| \\
        &\leq \max_{a} \left| \gamma \mathbb{E}_{s' \sim T(s' \mid s,a)} \left[ V(s') - U(s') \right] \right| \\
        &= \left| \gamma \mathbb{E}_{s' \sim T(s' \mid s, a^*)} \left[ V(s') - U(s') \right] \right|
        \quad \text{where } a^* \text{ is the argmax of the RHS above} \\
        &\leq \gamma \max_{s'} |V(s') - U(s')| \\
        &= \gamma \|V - U\|_\infty.
        \end{aligned}
        $$

        Since the above holds for any state $s$, it also holds for the state maximizing the LHS, such that:

        $$
        \|\mathcal{T}V - \mathcal{T}U\|_\infty \leq \gamma \|V - U\|_\infty.
        $$

Given $$\mathcal{T}$$ is a contraction mapping it is easy to prove the convergence of value iteration.

!!! note "Theorem"


     


    ???+ note "Proof"

        




        




         







### policy iterration




### liptishtness


### DDPG 
### natural gradient
### policy gradient
### kl
### TRPO theory
### SAC theory

### conceteration bound
### hofdding
### regret bound
### ucb

## notaion
here is a the notation we will use thgoht this part:


## prequestions
you may need to know the follwing concepts two better undestad this part:



