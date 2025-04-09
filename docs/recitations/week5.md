---
description: This page contains the recitation materials for Week 5 of the Deep Reinforcement Learning course. You can find links to the recitation recordings and slides.
comments: True
---


# Week 5: Model-Based Methods

### Screen Record

<iframe width="996" height="560" src="https://www.youtube.com/embed/VOTmlx4_sTs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

### Recitation Notes

#### **Table of Contents**
- [Week 5: Model-Based Methods](#week-5-model-based-methods)
- [**Lecture Notes on Model-Based Reinforcement Learning**](#lecture-notes-on-model-based-reinforcement-learning)
  - [**Table of Contents**](#table-of-contents)
  - [1. **Introduction to Model-Based Reinforcement Learning**](#1-introduction-to-model-based-reinforcement-learning)
  - [2. **Stochastic Optimization**](#2-stochastic-optimization)
    - [2.1 **Problem Formulation**](#21-problem-formulation)
    - [2.2 **Sample-Based Methods**](#22-sample-based-methods)
    - [2.3 **Example Problem: Fitting a Linear Regression Model for Transition Prediction**](#23-example-problem-fitting-a-linear-regression-model-for-transition-prediction)
      - [**Iteration 1**](#iteration-1)
        - [**1) Predictions \& Errors**](#1-predictions--errors)
        - [**2) Gradients**](#2-gradients)
        - [**3) Update**](#3-update)
      - [**Iteration 2**](#iteration-2)
        - [**1) Predictions \& Errors**](#1-predictions--errors-1)
        - [**2) Gradients**](#2-gradients-1)
        - [**3) Update**](#3-update-1)
      - [**Result**](#result)
  - [3. **Cross-Entropy Method (CEM)**](#3-cross-entropy-method-cem)
    - [3.1 **Algorithmic Steps**](#31-algorithmic-steps)
    - [3.2 **Intuition and Variants**](#32-intuition-and-variants)
    - [3.3 **Example Problem: 1D Action Optimization**](#33-example-problem-1d-action-optimization)
  - [4. **Monte Carlo Tree Search (MCTS)**](#4-monte-carlo-tree-search-mcts)
    - [4.1 **Basic Components**](#41-basic-components)
    - [4.2 **PUCT / UCB for Trees**](#42-puct--ucb-for-trees)
    - [4.3 **Example Problem: MCTS for a Simple Game (Tic-Tac-Toe)**](#43-example-problem-mcts-for-a-simple-game-tic-tac-toe)
      - [**Game Overview**](#game-overview)
      - [**Notation for Game States**](#notation-for-game-states)
      - [**Simulation #1**](#simulation-1)
        - [**Step 1: Selection (from the root)**](#step-1-selection-from-the-root)
        - [**Step 2: Expansion**](#step-2-expansion)
        - [**Step 3: Simulation (Rollout)**](#step-3-simulation-rollout)
    - [**Step 4: Backpropagation**](#step-4-backpropagation)
      - [**Simulation #2**](#simulation-2)
        - [**Step 1: Selection (from root)**](#step-1-selection-from-root)
        - [**Step 2: Expansion**](#step-2-expansion-1)
        - [**Step 3: Simulation (Rollout)**](#step-3-simulation-rollout-1)
        - [**Step 4: Backpropagation**](#step-4-backpropagation-1)
      - [**After Two Simulations**](#after-two-simulations)
  - [5. **Model Predictive Control (MPC)**](#5-model-predictive-control-mpc)
    - [5.1 **General Framework**](#51-general-framework)
    - [5.2 **MPC in Reinforcement Learning**](#52-mpc-in-reinforcement-learning)
    - [5.3 **Example Problem: Double Integrator System**](#53-example-problem-double-integrator-system)
  - [6. **Uncertainty Estimation**](#6-uncertainty-estimation)
    - [6.1 **Sources of Uncertainty**](#61-sources-of-uncertainty)
    - [6.2 **Methods of Estimation**](#62-methods-of-estimation)
    - [6.3 **Implications for Model-Based RL**](#63-implications-for-model-based-rl)
    - [6.4 **Example Problem: Gaussian Process for Next-State Prediction**](#64-example-problem-gaussian-process-for-next-state-prediction)
  - [7. **Dyna-Style Algorithms**](#7-dyna-style-algorithms)
    - [7.1 **Sutton’s Dyna Architecture**](#71-suttons-dyna-architecture)
    - [7.2 **Integrating Planning, Acting, and Learning**](#72-integrating-planning-acting-and-learning)
    - [7.3 **Example Problem: Dyna-Q in a 3-State Chain Environment**](#73-example-problem-dyna-q-in-a-3-state-chain-environment)
  - [8. **References**](#8-references)

---

#### 1. **Introduction to Model-Based Reinforcement Learning**

In reinforcement learning, an agent interacts with an environment modeled (or approximated) as a Markov Decision Process $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$. Model-based RL involves explicitly learning or using a model of the environment—transition dynamics $P(s'|s,a)$ and/or reward $R(s,a)$—to **plan** actions. 

**Why Model-Based?**  
- **Sample Efficiency**: If collecting data is expensive, using a model for planning can reduce the necessary real-world interactions.  
- **Exploration**: A model allows the agent to simulate potential scenarios offline and direct exploration more effectively.  
- **Robustness and Safety**: In safety-critical applications (e.g., robotics, autonomous driving), planning with a model can help avoid catastrophic outcomes during training.  

In the sections that follow, we will delve into foundational methods and demonstrate their use in simple, illustrative problems.

---

#### 2. **Stochastic Optimization**

##### 2.1 **Problem Formulation**

A **stochastic optimization** problem aims to optimize an objective function under uncertainty:

$$\min_{\theta} \; \mathbb{E}_{x \sim \mathcal{D}}[ f(\theta, x) ],$$

where
- $\theta$ are parameters (could be neural network weights, policy parameters, etc.),
- $\mathcal{D}$ is a (possibly unknown) data distribution or environment dynamics,
- $f(\theta, x)$ is a cost (or negative reward) function.

In model-based RL, such problems appear when we:
1. **Train a model** $\hat{P}_\theta(s'|s,a)$ to predict transitions by minimizing some loss $\mathcal{L}(\theta)$.
2. **Optimize a policy** using predicted trajectories.

##### 2.2 **Sample-Based Methods**

Because $\mathcal{D}$ or $f(\theta, x)$ might be complex or high-dimensional, **sample-based approaches** are common:

1. **Stochastic Gradient Descent (SGD)**: 
   - Evaluate $\nabla_\theta f(\theta, x^{(i)})$ on mini-batches of samples from $\mathcal{D}$.  
   - Update $\theta \leftarrow \theta - \alpha \nabla_\theta f(\theta, x^{(i)})$.

2. **Population-Based / Evolutionary Algorithms**:
   - Maintain a population of candidate solutions $\{\theta_1, \theta_2, \ldots\}$.  
   - Evaluate fitness and use selection, crossover, mutation to evolve better solutions.

3. **Simulated Annealing**:
   - Iteratively propose a new solution and accept/reject based on a temperature parameter that decreases over time, allowing for occasional acceptance of worse solutions to escape local minima.

##### 2.3 **Example Problem: Fitting a Linear Regression Model for Transition Prediction**

**Problem Setup**  
- We have a small environment: state $s\in \mathbb{R}$ is 1D, action $a\in \{-1, 1\}$.  
- The true environment dynamics is $s_{t+1} = s_t + 0.5 \, a + \epsilon_t$, where $\epsilon_t \sim \mathcal{N}(0, 0.1^2)$.  
- We collect $N=100$ transitions $\{(s^{(i)}, a^{(i)}, s'^{(i)})\}_{i=1}^N$.  
- We want to fit a *linear model*: $\hat{s}_{t+1} = w_0 + w_1 s_t + w_2 a_t$.  

**Cost Function** 

$$\mathcal{L}(w_0, w_1, w_2) = \sum_{i=1}^N \bigl(s'^{(i)} - (w_0 + w_1 s^{(i)} + w_2 a^{(i)}) \bigr)^2.$$

**Stochastic Gradient Descent Approach**  

1. **Initialize** parameters $\theta = (w_0, w_1, w_2)$ randomly.  
2. **Loop** until convergence:

    - Sample a mini-batch $\mathcal{B}$ of transitions from the dataset.
    - Compute the gradient:

        $$\nabla_\theta \mathcal{L}_\mathcal{B}(\theta) = \sum_{(s,a,s') \in \mathcal{B}} 2 \, (s' - \hat{s}) \, (-1) \nabla_\theta \hat{s},$$

        where $\hat{s} = w_0 + w_1 s + w_2 a$.

    - **Update**:

        $$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}_\mathcal{B}(\theta).$$

---

###### **Iteration 1**

####### **1) Predictions & Errors**

1. $\hat{s}^{(1)} = 0.20 + 0.90\cdot0.0 + 0.30\cdot(+1)=0.50$; error $e^{(1)}=0.55-0.50=0.05.$  
2. $\hat{s}^{(2)} = 0.20 + 0.90\cdot1.0 + 0.30\cdot(-1)=0.80$; error $e^{(2)}=0.40-0.80=-0.40.$

####### **2) Gradients**

- Point #1: $\nabla = -2e^{(1)}[1,\,s,\,a] = -2\cdot0.05\,[1,\,0.0,\,+1] = [-0.10,0,-0.10].$  
- Point #2: $\nabla = -2e^{(2)}[1,\,s,\,a] = -2\cdot(-0.40)\,[1,\,1.0,\,-1] = [+0.80,+0.80,-0.80].$

**Sum**: $[0.70,\,0.80,\,-0.90].$

####### **3) Update**

$$
(w_0,w_1,w_2) \leftarrow (0.20,0.90,0.30)\;-\;0.1\times(0.70,\,0.80,\,-0.90)
$$
$$
= (0.13,\,0.82,\,0.39).
$$

###### **Iteration 2**

####### **1) Predictions & Errors**

1. $\hat{s}^{(1)}=0.13 + 0.82\cdot0.0 + 0.39\cdot(+1)=0.52$; $e^{(1)}=0.55-0.52=0.03.$  
2. $\hat{s}^{(2)}=0.13 + 0.82\cdot1.0 + 0.39\cdot(-1)=0.56$; $e^{(2)}=0.40-0.56=-0.16.$

####### **2) Gradients**

- Point #1: $[-2\cdot0.03,\,-2\cdot0.03\cdot0,\,-2\cdot0.03\cdot1]=[-0.06,\,0,\,-0.06].$  
- Point #2: $[-2\cdot(-0.16),\,-2\cdot(-0.16)\cdot1,\,-2\cdot(-0.16)\cdot(-1)]=[+0.32,+0.32,-0.32].$

**Sum**: $[0.26,\,0.32,\,-0.38].$

####### **3) Update**

$$
(w_0,w_1,w_2)\leftarrow(0.13,\,0.82,\,0.39)-0.1\times(0.26,\,0.32,\,-0.38)
$$
$$
= (0.104,\,0.788,\,0.428).
$$

---

###### **Result**

After two steps, $\theta$ moved from $(0.20,\,0.90,\,0.30)$ to $(0.104,\,0.788,\,0.428)$. With more iterations (and more data), the model converges near $(0,1.0,0.5)$, reflecting the true dynamics $s_{t+1} = s_t + 0.5\,a + \epsilon_t$.

---

#### 3. **Cross-Entropy Method (CEM)**

The **Cross-Entropy Method (CEM)** is a population-based algorithm that iteratively refines a sampling distribution over possible solutions, focusing on an “elite” set of the highest-performing samples.

##### 3.1 **Algorithmic Steps**

1. **Parameterize a Distribution**: Let $q(\theta \mid \phi)$ be a distribution over solutions $\theta$. Often, we use a multivariate Gaussian with parameters $\phi = \{\mu, \Sigma\}$.  
2. **Sample Solutions**: Draw $\{\theta_1,\ldots,\theta_M\}$ from $q(\theta|\phi)$.  
3. **Evaluate Solutions**: Compute an objective $J(\theta_i)$ for each sample (e.g., cumulative reward, negative cost).  
4. **Select Elites**: Pick the top $K$ samples (or a top percentage).  
5. **Update Distribution**: Update $\phi$ (e.g., $\mu, \Sigma$) to fit the elite set.  
6. **Iterate**: Continue sampling from the updated distribution until convergence or a maximum iteration limit.

##### 3.2 **Intuition and Variants**

- **Intuition**: By repeatedly sampling solutions and focusing on the best ones, we “zoom in” on promising regions.  
- **Quantile Selection**: Instead of picking the top $K$, one can pick all solutions above a certain performance threshold.  
- **Regularization**: Add a fraction of the old mean/covariance to stabilize updates (avoiding collapsing to a single point).

##### 3.3 **Example Problem: 1D Action Optimization**

**Scenario**  

- We have a 1D system with state $s_0 = 0$.  
- We can choose a *single* action $a\in\mathbb{R}$ that transitions the system to $s_1 = s_0 + a$. Then a reward is given by $r(s_1) = -(s_1 - 2)^2$.  
- We want to find the action $a^*$ that maximizes the reward (equivalently minimizes the negative reward):

$$a^* = \arg\max_a \; - (0 + a - 2)^2.$$

The optimal solution is obviously $a^*=2$.

But let's see how CEM would handle this *without* an analytical solution.

**CEM Steps**  
1. **Initialize** a Gaussian distribution for $a$: $\mu=0$, $\sigma^2=4$.  
2. **Sample** 20 actions: $\{a_1,\ldots,a_{20}\}$.  
3. **Evaluate** each action: $J(a_i)= - (a_i - 2)^2.$  
4. **Select Elites**: Suppose we pick the top 5 actions.  
5. **Update** $\mu,\sigma$ to the mean and std of these top 5 actions.  
6. **Iterate**: After a few iterations, $\mu$ converges near $2$.

A table might look like:

| Iteration | Sample Actions             | Best 5 (Elites)                 | New Mean (μ) | New Std (σ) |
| --------- | -------------------------- | ------------------------------- | ------------ | ----------- |
| 1         | $\{-4, 1.2, 3.1, ...\}$  | $\{2.9, 3.1, 1.8, 2.4, 1.6\}$ | 2.36         | 0.53        |
| 2         | $\{1.9, 2.6, 2.1, ...\}$ | $\{2.0, 2.1, 2.4, 2.6, 1.9\}$ | 2.2          | 0.24        |
| ...       | ...                        | ...                             | ...          | ...         |

The method “homes in” on the correct action $a^*\approx 2$.

---

#### 4. **Monte Carlo Tree Search (MCTS)**

**Monte Carlo Tree Search** is a powerful method for decision-making in discrete action spaces (notably games). It builds a partial search tree by random simulations (rollouts) and updates action values based on the simulation outcomes.

##### 4.1 **Basic Components**

1. **Selection**: Traverse down the tree from the root state, choosing actions that balance exploration and exploitation (e.g., UCB).  
2. **Expansion**: When reaching a leaf node, expand one or more children.  
3. **Simulation (Rollout)**: From the new child node, simulate a default policy until a terminal state (or a depth limit) is reached.  
4. **Backpropagation**: Propagate the final simulation outcome (win/loss or reward) back up the tree, updating visit counts and action-value estimates.

##### 4.2 **PUCT / UCB for Trees**

A common selection formula is:

$$\text{UCT}(s,a) = \bar{Q}(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}}.$$

- $\bar{Q}(s,a)$: average return of choosing $a$ at $s$.  
- $N(s)$: visits to state $s$.  
- $N(s,a)$: visits to the action $a$ from state $s$.  
- $c$: exploration constant.

In **AlphaZero**, a variation called **PUCT** (Polynomial Upper Confidence Trees) adds a prior probability $\pi(a|s)$ from a policy network:

$$\text{PUCT}(s,a) = \bar{Q}(s,a) + c \,\pi(a\mid s)\,\frac{\sqrt{N(s)}}{1+N(s,a)}.$$

##### 4.3 **Example Problem: MCTS for a Simple Game (Tic-Tac-Toe)**

To illustrate how Monte Carlo Tree Search (MCTS) proceeds in a small game like Tic-Tac-Toe, let’s walk step-by-step through **two** simulated MCTS iterations. We will track how a single simulation (playout) is done each time, and how the search tree is incrementally updated. 

---

###### **Game Overview**

- **Board**: 3x3 grid.
- **Players**: 
  - **X** (the MCTS agent we are training), who always goes first.
  - **O** (the opponent), which we will assume plays randomly in this example.
- **Rewards** (from X’s perspective):
  - +1 if X eventually wins,
  - -1 if O wins,
  - 0 for a draw.

We will show how:
1. The **Selection** step chooses which moves to explore based on visit statistics.
2. The **Expansion** step adds new nodes (child states) to the tree.
3. The **Simulation** (rollout) step plays randomly until a terminal board state is reached.
4. The **Backpropagation** step updates the node statistics.

> **Note**: Real MCTS typically runs many (hundreds or thousands) of simulations per move. We’ll illustrate just **two** simulations for clarity.

###### **Notation for Game States**


We’ll represent the board states with a simple ASCII layout. For example:

    ```
    1 | 2 | 3 
    ---+---+--- 
    4 | 5 | 6 
    ---+---+--- 
    7 | 8 | 9
    ```


- Positions $1,2,3,4,5,6,7,8,9$ can be empty, 'X', or 'O'.
- The **root state** (empty board) has all 9 positions empty.

---

###### **Simulation #1**

####### **Step 1: Selection (from the root)**

- **Root State**: Empty board. We have no Q-values (no prior visits).  
- MCTS typically breaks ties randomly when no node has a better value or visit count.  
- **Possible moves** for X: 9 (positions 1 through 9).  

Let's assume the selection step picks **the center move** (position 5) for X.  
- Because all moves are identical from the MCTS perspective at the start, we choose position 5 arbitrarily (or randomly among the 9).

####### **Step 2: Expansion**

- We’ve now arrived at the child node corresponding to X having placed a mark in position 5.  
- Board now (child node):

    ```
    1 | 2 | 3
    ---+---+---
    4 | X | 6
    ---+---+---
    7 | 8 | 9
    ```

- We expand from this node. In basic MCTS, we typically **expand one child** for the next player (O).  
- So let's add a single child for O’s move. But O can choose any of the 8 empty positions.  

**For simplicity,** say the expansion picks O’s move at position 1.

####### **Step 3: Simulation (Rollout)**

We now have a partially specified board:

    ```
    O | 2 | 3 
    ---+---+--- 
    4 | X | 6 
    ---+---+---
    7 | 8 | 9
    ```

From here, the **simulation** step means:
1. We keep playing until the game ends (win/loss/draw),
2. **Both** players choose moves randomly (in a real MCTS rollout, we typically do a “default policy” or random play).

Let’s illustrate a possible random sequence:

1. **X**’s random move: positions available are $\{2,3,4,6,7,8,9\}$. Suppose X picks position 9.  

    ```
    O | 2 | 3
    ---+---+---
    4 | X | 6
    ---+---+---
    7 | 8 | X
    ```

2. **O** picks randomly from the remaining $\{2,3,4,6,7,8\}$. Suppose O goes in position 2.

    ```
    O | O | 3
    ---+---+---
    4 | X | 6
    ---+---+---
    7 | 8 | X
    ```


3. **X** picks from $\{3,4,6,7,8\}$. Suppose X picks position 3.  

    ```
    O | O | X
    ---+---+---
    4 | X | 6
    ---+---+---
    7 | 8 | X
    ```

- Check if X wins: Right now, no row/column/diagonal is complete.  
4. **O** picks from $\{4,6,7,8\}$. Suppose O picks position 6.  

    ```
    O | O | X
    ---+---+---
    4 | X | O
    ---+---+---
    7 | 8 | X
    ```

- No win yet.  
5. **X** picks from $\{4,7,8\}$. Suppose X picks position 7.  

    ```
    O | O | X
    ---+---+---
    4 | X | O
    ---+---+---
    X | 8 | X
    ```

- Now X has positions \{3,5,7,9\}. That **is** a winning combination on a diagonal? Actually, 3-5-7 is not a diagonal; the diagonals are (1,5,9) and (3,5,7). 
- **Check**: 3,5,7 *does indeed form a diagonal* (top-right to bottom-left). So X has a **win**.

Hence, the random rollout ends with **X winning** at step 5.  
- **Return** from X’s perspective: +1

##### **Step 4: Backpropagation**

We backpropagate the **reward** (+1 for an X win) up the search tree:

1. **Child Node (O at pos 1)**: 
- Visit count: $N = 1$
- Sum of returns: $W = +1$  (since from X’s perspective, that rollout was a success)
- Average Q-value: $\bar{Q} = +1/1 = +1$

2. **Parent Node (X at pos 5)**:
- Visit count: $N = 1$
- Sum of returns: $W = +1$
- $\bar{Q} = +1$

3. **Root Node** (empty board):
- Visit count: $N = 1$
- Sum of returns: $W = +1$
- $\bar{Q} = +1$

*(We typically store separate counts for each edge or node, but conceptually the result is the same: each visited node has +1 to its total returns.)*

Thus, after **one** simulation, the move “X at center” has 100% success rate in our limited data. The child move “O at pos 1” also has an average value of +1 (though that’s from X’s perspective, which might seem counterintuitive for O’s move, but in MCTS we track the perspective of the player to move at that node).

---

---

###### **Simulation #2**

Let’s now do a **second** MCTS simulation. We start again at the **root** (empty board).

####### **Step 1: Selection (from root)**

Now, the root node has a single visited child: “X at pos 5” with a Q-value of +1.  
- All **other 8 moves** are unvisited (Q=0, visits=0).  
- With typical MCTS selection criteria (like UCB), we might still explore an unvisited move. 
  However, because “X at pos 5” has a very high (maximum) average return so far, let’s assume the selection again picks “X at pos 5”.

####### **Step 2: Expansion** 

From the state with “X at center,” we already have an expanded child “O at pos 1.”  
- But O could also move to positions 2,3,4,6,7,8,9.  
- Typically, MCTS expansions add at least one **unvisited** move for O.  
- Let’s say we expand “O at pos 9” this time as a new child node.

####### **Step 3: Simulation (Rollout)**

From the state:
```
   1 | 2 | 3
  ---+---+---
   4 | X | 6
  ---+---+---
   7 | 8 | O
```
(That is X in 5, O in 9.)

We do another random rollout:

1. **X** picks from $\{1,2,3,4,6,7,8\}$. Suppose X picks position 1.
   ```
   X | 2 | 3
   ---+---+---
   4 | X | 6
   ---+---+---
   7 | 8 | O
   ```
2. **O** picks from $\{2,3,4,6,7,8\}$. Suppose O picks position 2.
   ```
   X | O | 3
   ---+---+---
   4 | X | 6
   ---+---+---
   7 | 8 | O
   ```
3. **X** picks from $\{3,4,6,7,8\}$. Suppose X picks position 6.
   ```
   X | O | 3
   ---+---+---
   4 | X | X
   ---+---+---
   7 | 8 | O
   ```
4. **O** picks from $\{3,4,7,8\}$. Suppose O picks position 4.
   ```
   X | O | 3
   ---+---+---
   O | X | X
   ---+---+---
   7 | 8 | O
   ```
5. **X** picks from $\{3,7,8\}$. Suppose X picks 3:
   ```
   X | O | X
   ---+---+---
   O | X | X
   ---+---+---
   7 | 8 | O
   ```
   - Check if X has 3 in a row: Not yet (positions X has are {1,3,5,6} – no row/column/diagonal complete).
6. **O** picks from $\{7,8\}$. Suppose O picks 7:
   ```
   X | O | X
   ---+---+---
   O | X | X
   ---+---+---
   O | 8 | O
   ```
   - O has positions {2,4,7,9}. Check if O wins: 
     - 2,4,7 is not a line, 4,7,9 is not a line, 2,7,9 is not a line, etc. So no win yet.
7. **X** picks the last available spot 8:
   ```
   X | O | X
   ---+---+---
   O | X | X
   ---+---+---
   O | X | O
   ```
   - Now the board is full. Check final lines:
     - X’s positions: {1,3,5,6,8} 
       - No winning triple among those positions (1,3,5) not a line, 3,5,6 not a line, etc.
     - O’s positions: {2,4,7,9}
       - Also no line. 
   - It’s a **draw**.

**Result** from X’s perspective: **0** (draw).

####### **Step 4: Backpropagation**

We backpropagate the draw reward (0) up the tree.

- **Child node**: “O at pos 9.”  
  - Visits: $N = 1$  
  - Sum of returns: $W = 0$  
  - $\bar{Q} = 0$

- **Parent node**: “X at pos 5.”  
  - Previously: $\bar{Q} = +1$, $N=1$, sum of returns $W=+1$.  
  - Now we add one more visit with reward 0: new $N=2$, new sum $W = +1 + 0 = +1$.  
  - Updated average $\bar{Q} = \frac{+1}{2} = +0.5$.

- **Root node**:  
  - Previously: $\bar{Q} = +1$, $N=1$, sum of returns = +1.  
  - Now: $N=2$, sum of returns = +1+0 = +1.  
  - $\bar{Q} = 1/2 = +0.5$.

---

###### **After Two Simulations**

Here is a simplified view of the **search tree** and stats:

- **Root node** (empty board):  
  - Visits: $N=2$, Q-value: $+0.5$.  
  - Has 9 possible children (8 unvisited, 1 visited).  

  - Child (X at pos 5):
    - Visits: 2, $\bar{Q}=+0.5$.
    - Children (O’s moves):
      - “O at pos 1”: Visits=1, $\bar{Q}=+1$.  
      - “O at pos 9”: Visits=1, $\bar{Q}=0$.  
      - Others unvisited.

In real MCTS, we keep running more simulations. Because “X at pos 5” has a higher average return so far than unvisited moves, it will continue to be selected often. Within that node, O’s possible moves get expanded one by one. Over many simulations, the tree grows, and the average returns converge to reflect the probability of winning from each state.


---

#### 5. **Model Predictive Control (MPC)**

Model Predictive Control (MPC), also known as Receding Horizon Control, has a long history in industrial process control. It optimizes over a **finite horizon** at each time step, applies the first action, then **re-plans**.

##### 5.1 **General Framework**

1. **Predictive Model** $\hat{P}$: We assume we have a model $\hat{P}(s_{t+1}|s_t,a_t)$.  
2. **Finite-Horizon Optimization**:

    $$\min_{a_0, \dots, a_{H-1}} \sum_{t=0}^{H-1} c(s_t, a_t) \quad \text{subject to} \; s_{t+1} = \hat{P}(s_t,a_t),$$

    plus an optional terminal cost or constraint.  
3. **Execute the First Action**: Apply $a_0^*$ to the real system.  
4. **Observe New State**: Shift the horizon window and repeat.

##### 5.2 **MPC in Reinforcement Learning**

- In RL contexts, MPC can be used if we have a learned model $\hat{P}_\theta$.  
- At each step, we solve an **optimal control problem** using the learned model for a short horizon.  
- This is especially common in **continuous control** tasks (e.g., robot arms, drones), where well-tuned MPC can be more stable than naive policy-based approaches.

##### 5.3 **Example Problem: Double Integrator System**

**System**  
- Continuous state: $(x, \dot{x})$.  
- Action: acceleration $u$.  
- Dynamics:  

$$\begin{cases}
x_{t+1} = x_t + \dot{x}_t \Delta t \\
\dot{x}_{t+1} = \dot{x}_t + u_t \Delta t
\end{cases}$$

**Goal**: Regulate to $(x, \dot{x}) = (0,0)$ with minimal cost:

$$c(x, \dot{x}, u) = x^2 + (\dot{x})^2 + \lambda u^2.$$

**MPC Steps**  
1. **Horizon = $H$** (e.g., 5 steps).  
2. **At each time**:
   - Solve $\min \sum_{t=0}^{H-1} \left[ x_t^2 + (\dot{x}_t)^2 + \lambda u_t^2 \right]$.  
   - Subject to the above discrete dynamics.  
   - Use a standard solver or even the Cross-Entropy Method to find $\{u_0,\dots,u_{H-1}\}$.  
   - Apply only $u_0$.  
   - Observe new state $(x_1,\dot{x}_1)$.  
   - Repeat.  

Over time, this *receding horizon* approach will bring the double integrator to the origin while balancing control effort.

---

#### 6. **Uncertainty Estimation**

When learning a model of the environment, we often have **uncertainty** about model parameters or about inherent stochasticity. Accounting for this uncertainty can greatly improve planning and exploration.

##### 6.1 **Sources of Uncertainty**
1. **Epistemic** (model) uncertainty: Due to limited data or model expressiveness.  
2. **Aleatoric** (intrinsic) uncertainty: Irreducible randomness in the environment (e.g., sensor noise).

##### 6.2 **Methods of Estimation**
- **Bayesian Neural Networks**: Place a prior over weights, approximate posterior with VI or MCMC.  
- **Ensembles**: Train multiple networks on bootstrapped data; measure variance across predictions.  
- **Gaussian Processes (GPs)**: Provide predictive means and variances with a kernel-based prior.

##### 6.3 **Implications for Model-Based RL**
- **Exploration**: Target states/actions where uncertainty is high to gather more data.  
- **Risk-Sensitivity**: Adjust the policy if high variance could lead to catastrophic failures.  
- **Conservative Model-Based Planning**: If uncertain, the model can yield higher cost or lower reward estimates to encourage caution.

##### 6.4 **Example Problem: Gaussian Process for Next-State Prediction**

**Scenario**  
- 1D environment with unknown dynamics: $s_{t+1} = g(s_t) + \epsilon$. We suspect $g(\cdot)$ is smooth.  
- Collect data: $\{(s^{(i)}, s'^{(i)})\}$.  
- Use a Gaussian Process (GP) with a kernel $k(s, s')$ to predict $s_{t+1}$.

**GP Training**  
1. **Choose Kernel**: e.g., RBF $k(s,s')=\exp\left(-\frac{(s-s')^2}{2l^2}\right)$.  
2. **Compute Posterior**: $p(s'_*|s_*, X, y) = \mathcal{N}(\bar{\mu}, \bar{\sigma}^2)$, where $\bar{\mu},\bar{\sigma}^2$ come from GP regression formulas.

**Outcome**  
- We get not just a prediction $\bar{\mu}$ for next-state but also a standard deviation $\bar{\sigma}$.  
- In planning, we can account for $\bar{\sigma}$ by preferring actions with lower predicted variance or exploring high-variance regions.

---

#### 7. **Dyna-Style Algorithms**

Dyna is a classic framework by Richard Sutton that **integrates**:
- **Direct Reinforcement Learning** from real experience,
- **Model Learning** of transitions/rewards,
- **Planning** using the learned model.

##### 7.1 **Sutton’s Dyna Architecture**

1. **Experience**: Interact with the environment, collecting transitions $(s,a,r,s')$.  
2. **Model**: Update the model $ \hat{P}(s'|s,a) $, $ \hat{R}(s,a) $.  
3. **Replay / Planning**: Sample “imaginary” transitions from $\hat{P}$ to update the policy or value function.  

##### 7.2 **Integrating Planning, Acting, and Learning**
- Each real-world step triggers **k** planning updates.  
- This can dramatically increase data efficiency because each real transition can spawn many synthetic updates.

##### 7.3 **Example Problem: Dyna-Q in a 3-State Chain Environment**

**Environment Setup**  

- **States**: $S_0, S_1, S_2$.  
- **Actions**: left (L), right (R).  
- **Transitions**:  
    - From $S_0$: R leads to $S_1$, L does nothing.  
    - From $S_1$: R leads to $S_2$, L leads back to $S_0$.  
    - From $S_2$: terminal (or loops with 0 reward if we want a continuing environment).  
- **Rewards**: +1 upon reaching $S_2$. Otherwise 0.  

**Dyna-Q Algorithm**  

```python
Initialize Q(s,a) arbitrarily
Initialize model M(s,a) # store transitions and rewards
alpha = 0.1
gamma = 0.99
num_episodes = 100
k = 5 # number of planning updates each step

for episode in range(num_episodes):
    s = S_0
    while s != S_2:
        # 1. Choose action
        a = epsilon_greedy(Q[s, :])
        
        # 2. Observe real transition
        s_next, r = environment_step(s,a)
        
        # 3. Update Q with real transition
        Q[s,a] = Q[s,a] + alpha * (r + gamma * max(Q[s_next,:]) - Q[s,a])
        
        # 4. Update the model
        M.store(s,a, s_next, r)
        
        # 5. Planning (k steps)
        for i in range(k):
            s_rand, a_rand = M.sample_previously_visited()
            s_sim, r_sim = M.predict(s_rand, a_rand)
            Q[s_rand,a_rand] = Q[s_rand,a_rand] + alpha * (
                r_sim + gamma * max(Q[s_sim,:]) - Q[s_rand,a_rand]
            )
        
        # 6. Move on
        s = s_next
```

**Intuition**
- Each real transition is used both to directly update Q and to **improve the model**.  
- Then **k planning steps** use the model to “hallucinate” transitions, effectively multiplying the benefit of each real step.  
- The agent learns the optimal policy (right-right from $S_0$ to reach $S_2$) in fewer real interactions than a purely model-free approach.

---

#### 8. **References**

Below is a comprehensive list of references mentioned, plus additional readings for deeper insights.

1. **General Reinforcement Learning**
    - Sutton, R.S. & Barto, A.G. (2018). [*Reinforcement Learning: An Introduction* (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html). MIT Press.
    - Bertsekas, D. (2012). *Dynamic Programming and Optimal Control*. Athena Scientific.
    - Spall, J.C. (2003). *Introduction to Stochastic Search and Optimization*. Wiley.

2. **Stochastic Optimization**
    - Spall, J.C. (2003). *Introduction to Stochastic Search and Optimization*. Wiley.
    - Bertsekas, D. & Tsitsiklis, J. (1996). *Neuro-Dynamic Programming*. Athena Scientific.

3. **Cross-Entropy Method**
    - Rubinstein, R.Y. & Kroese, D.P. (2004). *The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation, and Machine Learning*. Springer.
    - De Boer, P.-T., Kroese, D.P., Mannor, S., & Rubinstein, R.Y. (2005). “A tutorial on the cross-entropy method”. *Annals of Operations Research*, 134(1), 19–67.

4. **Monte Carlo Tree Search (MCTS)**
    - Kocsis, L. & Szepesvári, C. (2006). “Bandit based Monte-Carlo Planning”. In *ECML*.
    - Coulom, R. (2007). “Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search”. In *Computers and Games*.
    - Silver, D. et al. (2016). “Mastering the game of Go with deep neural networks and tree search”. *Nature*, 529(7587), 484–489.
    - Silver, D. et al. (2017). “Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm”. *arXiv preprint arXiv:1712.01815*.

5. **Model Predictive Control (MPC)**
    - Camacho, E.F. & Bordons, C. (2004). *Model Predictive Control*. Springer.
    - Williams, G. et al. (2017). “Information Theoretic MPC for Model-Based Reinforcement Learning”. In *ICRA*.
    - Bertsekas, D. (2012). *Dynamic Programming and Optimal Control*. Athena Scientific.

6. **Uncertainty Estimation**
    - Deisenroth, M.P. & Rasmussen, C.E. (2011). “PILCO: A Model-Based and Data-Efficient Approach to Policy Search”. In *ICML*.
    - Blundell, C. et al. (2015). “Weight Uncertainty in Neural Networks”. In *ICML*.
    - Lakshminarayanan, B. et al. (2017). “Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles”. In *NIPS*.
    - Rasmussen, C.E. & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

7. **Dyna-Style Algorithms**
    - Sutton, R.S. (1991). “Dyna, an integrated architecture for learning, planning, and reacting”. *SIGART Bulletin*, 2(4), 160–163.
    - Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

#### Author(s)

<div class="grid cards" markdown>
-   ![Instructor Avatar](/assets/images/staff/Naser-Kazemi.jpg){align=left width="150"}
    <span class="description">
        <p>**Naser Kazemi**</p>
        <p>Teaching Assistant</p>
        <p>[naserkazemi2002@gmail.com](mailto:naserkazemi2002@gmail.com)</p>
        <p>
        [:fontawesome-brands-github:](https://github.com/naser-kazemi){:target="_blank"}
        </p>
    </span>
</div>