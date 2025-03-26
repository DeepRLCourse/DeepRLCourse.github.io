# Introduction to RL in Depth
## goal
get deeper into the concepts. learn new math that are usfull for RL

## overview

### value iteration
As mentioned in the first part of course optimal value functions, $V^*(s)$ or  $Q^*(s, a)$ must satisfy Bellman's optimality equations:

$$
Q^*(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} Q^*(s_{t+1}, a') \mid s_t = s, a_t = a \right]
$$
,
$$
V^*(s) = \mathbb{E} \left[ R_{t+1} + \gamma V^*(s_{t+1}) \mid s_t = s \right] 
$$
.
Therefore by finding $V^*(s)$ and $Q^*(s, a)$ through Bellman optimality equations, one can determine optimal policy. Specifically, dynamic programming (DP) methods are obtained by simply turning these equations into an update rule.

Value iteration simply finds $V^*(s)$ by iterating through these 2 equations, the update rule can be written as a simple backup operation : 

$$
V_{k+1}(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s), s' \sim P(\cdot \mid s, a)} \left[ R(s, a, s') + \gamma V_k(s') \right]
$$


For the sake of conviniency we define the Bellman operator for having more intuition on how fixed point algorithms workout!

$Q \xrightarrow{\text{$\mathcal{T} V(s)$}} Q \xrightarrow{\text{$\mathcal{T} V(s)$}} Q \xrightarrow{\text{$\mathcal{T} V(s)$}} Q$

!!! note "Theorem"

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

    ??? note "Inner Note"

        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
        nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
        massa, nec semper lorem quam in massa.


### policy iterration




### liptishtness
### contraction mapping

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



