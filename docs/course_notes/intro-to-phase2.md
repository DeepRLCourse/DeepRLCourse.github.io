# Introduction to RL in Depth
## goal
get deeper into the concepts. learn new math that are usfull for RL

## overview


### value iteration
After getting familiar with state value function and state-action value function in the first section of the course, by introducing value iteration, policy improvement algorithms we try getting deeper into value based algorithms have a better comperehention on how actually these algorithms converge to an optimal solution in a MDP setup.

#### Relation between $Q^\pi(s,a)$ and $V^\pi(s)$
For further understanding of value iteration and policy improvement, better take a moment and derive $Q^\pi(s,a)$ from $V^\pi(s)$ and vice versa.

$$
Q^{\pi}(s,a) = r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' | s,a) V^{\pi}(s')
$$

$$
V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a | s) Q^{\pi}(s,a)
$$


!!! tip "Relation between $Q^\pi(s,a)$ and $V^\pi(s)$ "

    $$
    \begin{aligned}
    Q^{\pi}(s,a) &= \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 = s, a_0 = a, \pi \right] \\
    &= r(s,a) + \mathbb{E} \left[ \sum_{t=1}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 = s, a_0 = a, \pi \right] \\
    &= r(s,a) + \gamma \sum_{s' \in \mathcal{S}} {P(s' | s,a)} \mathbb{E} \left[ \sum_{t=1}^{\infty}
    \gamma^{t-1} r(s_t, a_t) \mid s_0 = s, s_1 = s', a_0 = a, \pi \right] \\
    &= r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' | s,a) \mathbb{E} \left[ {\sum_{t=1}^{\infty}
    \gamma^{t-1} r(s_t, a_t) \mid s_1 = s', \pi} \right] 
    \quad {\text{(Markov assumption)}} \\
    &= r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' | s,a) \mathbb{E} \left[ {\sum_{t=0}^{\infty} 
    \gamma^t r(s_t, a_t) \mid s_0 = s', \pi} \right] 
    \quad {\text{(i.e., } V^{\pi}(s')\text{)}} \\
    &= r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' | s,a) V^{\pi}(s') \, \square
    \end{aligned}
    $$


!!! abstract "Theorem Bellman Optimality Equation"


    For a deterministic optimal policy \(\pi^*\), the Bellman optimality equation can be expressed as follows.

    ### For the state-value function:
    $$
    V^*(s) = \max_{a \in \mathcal{A}} \left[ r(s,a) + \gamma \, \mathbb{E}_{s' \sim P(\cdot \mid s,a)}\bigl[V^*
    (s')\bigr] \right]
    $$

    ### For the action-value function:
    $$
    Q^*(s,a) = r(s,a) + \gamma \, \mathbb{E}_{s' \sim P(\cdot \mid s,a)}\left[ \max_{a' \in \mathcal{A}} Q^*(s',a')
    \right]
    $$



As mentioned in the first part of the course, optimal value functions, $V^*(s)$ or  $Q^*(s, a)$ must satisfy Bellman's optimality equations.

- Finding $\pi^\star$ can be done by first computing $V^\star$ or $Q^\star$.
- Note that we can directly get a (deterministic and stationary) optimal policy from $Q^\star$:

$$
\pi^\star(s) = \arg\max_{a \in \mathcal{A}} Q^\star(s,a).
$$

#### why optimal policy $\pi^*$ exists?

!!! abstract "Theorem Optimal policy existance"

    For an infinite horizon MDP_ $M = (\mathcal{S}, \mathcal{A}, P, t, \mu, \gamma)$, _there exists a stationary and
     deterministic policy_ $\pi$ _such that for any_ $s \in \mathcal{S}$ _and_ $a \in \mathcal{A}$, _we have_

    $$
    V^\pi(s) = V^\star(s), \quad Q^\pi(s,a) = Q^\star(s,a).
    $$




    


???+ abstract "Proof"

    
    Assume a start from \((s_0, a_0, r_0, s_1) = (s, a, r, s')\), then:
    Define the _offset_ policy \( \pi_t(a_t \mid h_t) := \pi(a_{t+1} = a \mid (s_0, a_0) = (s, a), h_t) \). The
    Markov property implies  

    $$
    \mathbb{E} \left[ \sum_{t=1}^{\infty} \gamma^t r(s_t, a_t) \mid (s_0, a_0, r_0, s_1) = (s, a, r, s') \right]  
    = \gamma V^\pi(s').
    $$

    Since for all \( (s_0, a_0, r_0) = (s, a, r) \), the set \( \{\pi' \mid \Pi\} \) is just \( \Pi \) itself.

    Show that the optimal value from \( s_1 \) onward is independent of \( (s_0, a_0, r_0) = (s, a, r) \):

    $$
    \max_{\pi \in \Pi} \mathbb{E} \left[ \sum_{t=1}^{\infty} \gamma^t r(s_t, a_t) \mid (s_0, a_0, r_0, s_1) = (s, a, r,
    s') \right]  
    = \gamma \max_{\pi \in \Pi} V^\pi(s') = \gamma V^*(s').
    $$
   

    Let \( \pi(s) = \arg\max_{a \in A} \max_{\pi' \in \Pi} Q^{\pi'}(s,a) \), and show that this deterministic and
    randomized policy is optimal:

    $$
    V^*(s_0) = \max_{\pi' \in \Pi} \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid s_0 = s, \pi \right]
    $$

    $$
    = \max_{\pi' \in \Pi} \mathbb{E} \left[ r(s_0, a_0) + \sum_{t=1}^{\infty} \gamma^t r(s_t, a_t) \mid (s_0, a_0, r_0,
     s_1), \pi \right]
    $$

    $$
    = \max_{\pi' \in \Pi} \mathbb{E} \left[ r(s_0, a_0) + \mathbb{E} \left[ \sum_{t=1}^{\infty} \gamma^t r(s_t, a_t)
    \mid (s_0, a_0, r_0, s_1), \pi \right] \Big| (s_0, a_0), \pi \right].
    $$

    

    $$
    \leq \max_{\pi' \in \Pi} \mathbb{E} \left[ r(s_0, a_0) + V^*(s_1) \mid (s_0, a_0), \pi \right]
    $$

    By the definition of \( \pi \):

    $$
    = \mathbb{E} \left[ r(s_0, a_0) + V^*(s_1) \mid \pi \right].
    $$

    Since  

    $$
    V^*(s_0) \leq \mathbb{E} \left[ r(s_0, a_0) + V^*(s_1) \mid \pi \right] \leq \mathbb{E} \left[ r(s_0, a_0) + \gamma
    r(s_1, a_1) + \gamma^2 V^*(s_1) \mid \pi \right] \leq \dots
    $$

    we obtain  

    $$
    V^\pi = V^*,
    $$
   
    which proves that the proposed \( \pi \) is optimal.




Therefore by finding $V^*(s)$ and $Q^*(s, a)$ through Bellman optimality equations, one can determine optimal policy. Specifically, dynamic programming (DP) methods are obtained by simply turning these equations into an update rule.

Value iteration simply finds $V^*(s)$ by iterating through these 2 equations, the update rule can be written as a simple backup operation : 

$$
V_{k+1}(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s), s' \sim P(\cdot \mid s, a)} \left[ R(s, a, s') + \gamma V_k(s') \right]
$$


For the sake of conviniency we define the Bellman operator for having more intuition on how fixed point algorithms workout!

$$V^0 \xrightarrow{\mathcal{T^\star}} V^1 \xrightarrow{\mathcal{T^\star}} V^2 ... \xrightarrow{\mathcal{T^\star}} V^{k-1} \xrightarrow{\mathcal{T^\star}} V^{k} $$

!!! abstract "Definition Bellman Operator"

    ### Bellman's Optimality Operator
    A Bellman optimality operator $\mathcal{T}
    :\mathbb{R}^{|S|} \to \mathbb{R}^{|S|}$ _is an operator that satisfies: for any_ $V 
    \in \mathbb{R}^{|S|}$,

    $$
    (\mathcal{T} V)(s) = \max_{a} \left[ r(s, a) + \gamma \mathbb{E}_{s' \sim P(s' \mid
    s, a)} V(s') \right].
    $$

    ### Bellman's Expectation Operator
    A *Bellman expectation operator for policy* $\pi$, or $\mathcal{T}^\pi: \mathbb{R}^{|S|} \to \mathbb{R}^{|S|}$, *is
    an operator that satisfies:*  
    _for any_ $V \in \mathbb{R}^{|S|}$,  

    $$(\mathcal{T}^\pi V)(s) = \mathbb{E}_{a \sim \pi(a | s), s' \sim T(s' | s, a)} \left[ r(s, a) + \gamma V^\pi(s')
    \right].
    $$



    



Value iteration can thus be represented as recursively applying the Bellman optimality 
operator:

$$
V_{k+1} = \mathcal{T} V_k. 
$$



#### Bellman Operator Properties

- The optimal value function **$\mathbf{V}^\star$** is the **fixed point** of $\mathcal{T}$, i.e.,  

    $$
    \mathcal{T} \mathbf{V}^\star = \mathbf{V}^\star.
    $$

- The Bellman optimality operator is a $\gamma$-contraction mapping w.r.t. $\ell_{\infty}$-norm.  
- The Bellman operator is also monotonic (component-wise):  

    $$
    \mathbf{V}_1 \leq \mathbf{V}_2 \Rightarrow \mathcal{T} \mathbf{V}_1 \leq \mathcal{T} \mathbf{V}_2.
    $$


- We can define a similar Bellman operator on the $Q$-function and show similar properties.


### Contraction mapping
!!! note "Theorem"


     _The Bellman optimality operator_ $\mathcal{T}$ _defined above is a_ $\gamma$-_contraction mapping under_
      $\ell_{\infty}$-_norm, i.e., for any_  
    $\mathbf{V}', \mathbf{V} \in \mathbb{R}^{|\mathcal{S}|}$, _we have_

    $$
    \|\mathcal{T} \mathbf{V}' - \mathcal{T} \mathbf{V} \|_{\infty} \leq \gamma \|\mathbf{V}' - \mathbf{V} \|_{\infty},
    $$

    _where_  

    $$
    \|\mathbf{x}\|_{\infty} := \max_i |x_i|.
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

        We now proceed to the proof.
        For any $\mathbf{V}' , \mathbf{V} \in \mathbb{R}^{|S|}$ and $s \in S$, we have

        $$
        \begin{aligned}
        &\left| \left( \mathcal{T}\mathbf{V}' \right) (s) - \left( \mathcal{T}\mathbf{V} \right) (s) \right| \\
        &= \left| \max_{a \in A} \left[ r(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) \mathbf{V}' (s') \right] -
        \max_{a'
        \in A} \left[ r(s, a') + \gamma \sum_{s' \in S} P(s' | s, a') \mathbf{V} (s') \right] \right| \\
        &\leq \max_{a \in A} \left| \left( r(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) \mathbf{V}' (s') \right) -
        \left(
        r(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) \mathbf{V} (s') \right) \right| \\
        &= \max_{a \in A} \gamma \sum_{s' \in S} P(s' | s, a) \left| \mathbf{V}' (s') - \mathbf{V} (s') \right| \\
        &\leq \left\| \mathbf{V}' - \mathbf{V} \right\|_{\infty} \max_{a \in A} \gamma \sum_{s' \in S} P(s' | s, a) \\
        &= \gamma \left\| \mathbf{V}' - \mathbf{V} \right\|_{\infty},
        \end{aligned}
        $$


So we can conclude that finding $V^\star$ or reaching out the optimal policy **$\mathbf{\pi}^\star$** is equivalent to finding a fixed point for $\mathcal{T}$ and value iteration can therefore be viewd as fixed point iteration.

Given $\mathcal{T}$ is a contraction mapping it is easy to prove the convergence of value iteration.



!!! note "Theorem"

    Value iteration algorithm attains a linear convergence rate , i.e,

    $$
    \|V^k - V^\star\|_{\infty} \leq \gamma^t \|V - V^\star\|_{\infty}.
    $$


    ???+ note "Proof"

        $$
        \|V^k - V^\star\|_{\infty} = \|\mathcal{T} V^{k-1} - \mathcal{T} V^\star\|_{\infty} 
        \leq \gamma \|V^{k-1} - V^\star\|_{\infty} 
        \leq \cdots 
        \leq \gamma^t \|V_0 - V^\star\|_{\infty}.
        $$  

        as it's obvious the norm is always greater than 0 , so by using the envelope theorem :

        $$
        \lim_{k \to \infty} \gamma^t \|V_0 - V^\star\|_{\infty}  =  0
        $$

    

After obtaining $V^*$ via value iteration, we can derive an optimal policy using the **greedy policy**:

$$
\pi^*(s) = \arg\max_{a \in A} \left[ r(s, a) + \gamma \sum_{s' \in S} P(s'|s, a)V^*(s') \right].
$$

Alternatively, we can use $Q$-value iteration to compute $\pi^*$ directly:

$$
\pi^*(s) = \arg\max_{a \in A} Q^*(s, a).
$$

By using these 2 equations that has been mentioned before :   

$$
Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in S} P(s'|s, a)V^\pi(s'),
$$

$$
V^\pi(s) = \sum_{a \in A} \pi(a | s) Q^\pi(s, a).
$$

The $Q$-value iteration framework provides insights for handling **model-free** scenarios in later discussions.


        




         







### policy iterration

The policy iteration algorithm, for discounted MDPs, starts from an arbitrary policy $\pi_{0}$, and repeats the following iterative procedure: for $k = 0, 1, 2, \ldots$

1. **Policy evaluation**. Compute $Q^{\pi_{k}}$.  
2. **Policy improvement**. Update the policy:  

$$
\pi_{k+1}(s) = \arg\max_{a \in A} Q^{\pi_{k}}(s, a).
$$

In each iteration, we compute the $Q$-value function of $\pi_{k}$, using the analytical form given in Equation (2), and update the policy to be greedy with respect to this new $Q$-value. The first step is often called *policy evaluation*, and the second step is often called *policy improvement*.

**Lemma** *We have that:*  
1. $Q^{\pi_{k+1}} \geq \mathcal{T}Q^{\pi_{k}} \geq Q^{\pi_{k}}$  
2. $\|Q^{\pi_{k+1}} - Q^{\star}\|_{\infty} \leq \gamma \|Q^{\pi_{k}} - Q^{\star}\|_{\infty}$

!!! note "Proof"

        
    First, show \( \mathcal{T}Q^{\pi_{k}} \geq Q^{\pi_{k}} \). Since policies are deterministic, \( V^{\pi_{k}}(s) =
    Q^{\pi_{k}}(s, \pi_{k}(s)) \). Hence:  

    $$
    \begin{aligned}
    \mathcal{T}Q^{\pi_{k}}(s, a) 
    &= r(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot | s, a)}\left[\max_{a'} Q^{\pi_{k}}(s', a')\right] \\
    &\geq r(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot | s, a)}\left[Q^{\pi_{k}}(s', \pi_{k}(s'))\right] \\
    &= Q^{\pi_{k}}(s, a).
    \end{aligned}
    $$  

    Next, prove \( Q^{\pi_{k+1}} \geq \mathcal{T}Q^{\pi_{k}} \). Observe \( Q^{\pi_{k+1}} \geq Q^{\pi_{k}} \):  

    $$
    \begin{aligned}
    Q^{\pi_{k}} 
    &= r + \gamma P^{\pi_{k}} Q^{\pi_{k}} \\
    &\leq r + \gamma P^{\pi_{k+1}} Q^{\pi_{k}} \quad \text{(greedy policy \(\pi_{k+1}\))} \\
    &\leq \sum_{t=0}^{\infty} \gamma^{t} (P^{\pi_{k+1}})^{t} r \quad \text{(recursion)} \\
    &= Q^{\pi_{k+1}}.
    \end{aligned}
    $$  

    Using this:  

    $$
    \begin{aligned}
    Q^{\pi_{k+1}}(s, a) 
    &= r(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot | s, a)}\left[Q^{\pi_{k+1}}(s', \pi_{k+1}(s'))\right] \\
    &\geq r(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot | s, a)}\left[Q^{\pi_{k}}(s', \pi_{k+1}(s'))\right] \\
    &= r(s, a) + \gamma \mathbb{E}_{s' \sim P(\cdot | s, a)}\left[\max_{a'} Q^{\pi_{k}}(s', a')\right] \\
    &= \mathcal{T}Q^{\pi_{k}}(s, a).
    \end{aligned}
    $$  

    For the second claim:  

    $$
    \begin{aligned}
    \|Q^{\star} - Q^{\pi_{k+1}}\|_{\infty} 
    &\leq \|Q^{\star} - \mathcal{T}Q^{\pi_{k}}\|_{\infty} \\
    &= \|\mathcal{T}Q^{\star} - \mathcal{T}Q^{\pi_{k}}\|_{\infty} \quad \text{(since \( \mathcal{T}Q^{\star} = Q^{\star}
    \))} \\
    &\leq \gamma \|Q^{\star} - Q^{\pi_{k}}\|_{\infty} 
    \end{aligned}
    $$


Now using this lemma we can build up these 2 properties by extending our theory to Bellman expectation operator:

**Property 1:** For any $\pi$,

$$
\mathcal{T}^{\pi_{k+1}} Q^{\pi_k} \geq \mathcal{T}^\pi Q^{\pi_k}
$$

where $Q^{\pi_k}$ is the Q-function for policy $\pi_k$, and $\mathcal{T}^{\pi_{k+1}}$ is the Bellman operator for $\pi_{k+1}$.  

**Property 2:**

$$
\mathcal{T}^{\pi_{k+1}} Q^{\pi_k} \leq Q^{\pi_{k+1}}
$$

!!! note "Theorem"

    ####Policy Iteration gets no worse on every iteration:

    $$
    \|Q^* - Q^{\pi_{k+1}}\|_{\infty} \leq \gamma \|Q^* - Q^{\pi_k}\|_{\infty}
    $$

!!! note "Proof"

    $$
    \begin{aligned}
    0 \leq Q^* - Q^{\pi_{k+1}} 
    &\leq (Q^* - \mathcal{T}^{\pi_{k+1}} Q^{\pi_k}) + (\mathcal{T}^{\pi_{k+1}} Q^{\pi_k} - Q^{\pi_{k+1}}) \\
    &\leq (Q^* - \mathcal{T}^{\pi_{k+1}} Q^{\pi_k}) \quad \text{(by Property 2)}
    \end{aligned}
    $$

    Since $Q^* = \mathcal{T} Q^*$, substitute:

    $$
    \begin{aligned}
    Q^* - Q^{\pi_{k+1}} 
    &\leq \mathcal{T} Q^* - \mathcal{T}^{\pi_{k+1}} Q^{\pi_k} \\
    &= \mathcal{T}^{\pi^*} Q^* - \mathcal{T}^{\pi_{k+1}} Q^{\pi_k} \quad \text{(as $\pi^*$ is optimal)}
    \end{aligned}
    $$

    Apply Property 1 with $\pi = \pi^*$:

    $$
    \begin{aligned}
    Q^* - Q^{\pi_{k+1}} 
    &\leq \mathcal{T}^{\pi^*} Q^* - \mathcal{T}^{\pi^*} Q^{\pi_k} \\
    &\leq \gamma \|Q^* - Q^{\pi_k}\|_{\infty} \quad \text{(contraction property)}.
    \end{aligned}
    $$



**A** $(\mathcal{T}^{\pi_{k+1}}Q^{\pi_k})(s,a) = \mathbb{E} \left[ \sum_{h=1}^\infty \gamma^{h-1} r_h \bigg| s_1 = s, a_1 = a, a_2 \sim \pi_{k+1}, a_{3: \infty} \sim \pi_k \right]$  

The initial state and action are given as inputs to the $Q$-function. The next action follows $\pi_{k+1}$, and all future actions follow $\pi_k$.  



**B** $(\mathcal{T}^{\pi}Q^{\pi_k})(s,a) = \mathbb{E} \left[ \sum_{h=1}^\infty \gamma^{h-1} r_h \bigg| s_1 = s, a_1 = a, a_2 \sim \pi, a_{3: \infty} \sim \pi_k \right]$$  

This generalizes (A) where $\pi = \pi_k$.  



**C** $Q^{\pi_{k+1}}(s,a) = \mathbb{E} \left[ \sum_{h=1}^\infty \gamma^{h-1} r_h \bigg| s_1 = s, a_1 = a, a_2 \sim \pi_{k+1}, a_{3: \infty} \sim \pi_{k+1} \right]$ 

This is the standard $Q$-function definition for policy $\pi_{k+1}$.  

 
- **Property 1**: $A \geq B$ (intuitive, as $A$ uses the improved policy $\pi_{k+1}$ for $a_2$).  
- **Property 2**: $C \geq A$ (true because $C$ uses $\pi_{k+1}$ for all steps, while $A$ does not).  

Now we can prove policy improvement in other words : 
First, consider the monotonic property of $\mathcal{T} ^\pi$.
For any $Q \leq Q' \rightarrow \mathcal{T}Q \leq \mathcal{T}Q'$. Then it follows that:  

$$Q^{\pi_k} = \mathcal{T}^{\pi_k}Q^{\pi_k} \leq \mathcal{T}Q^{\pi_k} = \mathcal{T}^{\pi_{k+1}}Q^{\pi_k}$$  

So we can show:  

$$Q^{\pi_k} \leq \mathcal{T}^{\pi_{k+1}}Q^{\pi_k} \leq \mathcal{T}^{\pi_{k+1}}(\mathcal{T}^{\pi_{k+1}}Q^{\pi_k}) \leq \ldots \leq (\mathcal{T}^{\pi_{k+1}})^{\infty}Q^{\pi_k} = Q^{\pi_{k+1}}$$  

Noting here that:  

$$\lim_{h \to \infty}(\mathcal{T}^{\pi_{k+1}})^hQ^{\pi_k} = Q^{\pi_{k+1}}$$  



### $\epsilon$-Greedy Policy Improvement

!!! note "Theorem"

    For any $\epsilon$-greedy policy $\pi$, the $\epsilon$-greedy policy $\pi'$ with respect to $Q_{\pi}$ is an
    improvement:  

    $$
    V_{\pi'}(s) \geq V_{\pi}(s)
    $$

!!! note "Proof"

    \[  
    \begin{aligned}  
    Q_{\pi}(s, \pi'(s)) &= \sum_{a \in A} \pi'(a|s) Q_{\pi}(s, a) \\  
    &= \frac{\epsilon}{m} \sum_{a \in A} Q_{\pi}(s, a) + (1 - \epsilon) \max_{a \in A} Q_{\pi}(s, a) \\  
    &\geq \frac{\epsilon}{m} \sum_{a \in A} Q_{\pi}(s, a) + (1 - \epsilon) \sum_{a \in A} \frac{\pi(a|s) - \epsilon/m}{1
    - \epsilon} Q_{\pi}(s, a) \\  
    &= \sum_{a \in A} \pi(a|s) Q_{\pi}(s, a) = V_{\pi}(s)  
    \end{aligned}  
    \]  















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



