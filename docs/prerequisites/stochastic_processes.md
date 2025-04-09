# Stochastic Processes

Have you ever happened to explore a reinforcement learning paper? Well, if you have, you might have come across the term "Markov Decision Process (MDP)", typically defined as a tuple of the form $\langle S, A, P, R, \gamma \rangle$. In this section, we will explore the components of this tuple and how they relate to stochastic processes.

In this prerequisite, we will cover the following topics:

- **Stochastic Processes**: Basic definitions and examples.

- **Markov Decision Process (MDP)**: Modeling sequential decision-making under uncertainty.


## Stochastic Process
A stochastic process is a collection of random variables indexed by time or space. It is a mathematical object used to describe systems that evolve over time in a probabilistic manner. Stochastic processes are widely used in various fields, including finance, physics, and reinforcement learning.
### Definition
A stochastic process is defined as a family of random variables $\{X_t\}_{t \in T}$, where $T$ is an index set (often representing time). Each random variable $X_t$ represents the state of the system at time $t$. The index set can be discrete (e.g., $T = \{0, 1, 2, \ldots\}$) or continuous (e.g., $T = [0, \infty)$).

### Examples
1. **Discrete-Time Markov Chain (DTMC)**: A stochastic process where the future state depends only on the current state and not on the past states. Formally, a DTMC is defined by a set of states $S$ and a transition probability matrix $P$, where $P_{ij} = P(X_{t+1} = j | X_t = i)$.
2. **Continuous-Time Markov Chain (CTMC)**: A stochastic process where transitions between states occur continuously over time. The transition rates are defined by a rate matrix $Q$, where $Q_{ij} = \lim_{h \to 0} \frac{P(X_{t+h} = j | X_t = i)}{h}$.
3. **Poisson Process**: A stochastic process that counts the number of events occurring in a fixed interval of time or space, where the events occur independently and at a constant average rate $\lambda$. The number of events in an interval of length $t$ follows a Poisson distribution with parameter $\lambda t$.
4. **Brownian Motion**: A continuous-time stochastic process that models the random motion of particles suspended in a fluid. It is characterized by continuous paths and independent increments, where the increment $X_{t+s} - X_t$ follows a normal distribution with mean 0 and variance $s$.
5. **Random Walk**: A stochastic process that describes a path consisting of a succession of random steps. In a simple symmetric random walk, at each time step, the process moves one step to the left or right with equal probability.
6. **Markov Chain**: A stochastic process that satisfies the Markov property, meaning that the future state depends only on the current state and not on the past states. It is defined by a set of states and transition probabilities between them.
7. **Hidden Markov Model (HMM)**: A statistical model that represents a system with unobserved (hidden) states. It consists of a Markov chain for the hidden states and an observation model that relates the hidden states to observed data. HMMs are widely used in speech recognition, natural language processing, and bioinformatics.
8. **Gaussian Process**: A collection of random variables, any finite number of which have a joint Gaussian distribution. It is used in machine learning for regression and classification tasks, where the function values are modeled as a Gaussian process with a mean function and a covariance function (kernel).

But why one should care about stochastic processes? Well, they are almost everywhere in the world around us. For example, the daily progression of weather often behaves like a [Discrete-Time Markov Chain (DTMC)](https://en.wikipedia.org/wiki/Markov_chain "Definition of Markov Chains, including DTMCs"), where the chance of having sun or rain tomorrow primarily depends just on today's weather state. This 'memoryless' property is central to Markovian systems. When events can happen at any instant, like customers arriving or being served in a bank queue, we might use a [Continuous-Time Markov Chain (CTMC)](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain "Definition of CTMCs") to model the number of people waiting. Even the core algorithm behind Google Search, PageRank, models web surfing as a user randomly clicking links – a large-scale application of [Markov Chains](https://en.wikipedia.org/wiki/Markov_chain "General concept of Markov Chains"). Furthermore, when the underlying process state is hidden, like deducing the sequence of spoken words from just the audio signal in speech recognition, we rely on [Hidden Markov Models (HMMs)](https://en.wikipedia.org/wiki/Hidden_Markov_model "Definition and applications of HMMs").

Other types of stochastic processes describe different kinds of randomness. The number of independent events occurring over time, like calls arriving at a support center within an hour or radioactive decay events detected per minute, is often modeled by a [Poisson Process](https://en.wikipedia.org/wiki/Poisson_process "Definition and properties of Poisson Processes"), assuming a constant average rate. The seemingly erratic, jiggling path of a tiny particle suspended in fluid, known as [Brownian Motion](https://en.wikipedia.org/wiki/Brownian_motion "Physical phenomenon and mathematical model"), provides a fundamental model for continuous random movement and is famously used (though as an approximation) to describe fluctuations in financial markets. A simpler, often discrete, version of random movement is the [Random Walk](https://en.wikipedia.org/wiki/Random_walk "Concept and types of Random Walks"), exemplified by the proverbial drunkard's staggering path or basic models of step-by-step changes in stock prices.

Finally, some processes help us model uncertainty about entire functions or spatial distributions. A [Gaussian Process (GP)](https://en.wikipedia.org/wiki/Gaussian_process "Mathematical definition and use in machine learning") isn't just about a single random variable but defines a probability distribution over functions. This is incredibly useful in fields like [geostatistics](https://en.wikipedia.org/wiki/Geostatistics "Branch of statistics focused on spatial data") for interpolating values like temperature or mineral deposits between sampled locations, providing not just an estimate but also a measure of uncertainty for any point in the area of interest. They are also foundational in Bayesian optimization, helping tune complex models efficiently.

Also, if you are intrested in deeplearning theory, you might want to check out the [Neural Tangent Kernel](https://arxiv.org/abs/1806.07572) (NTK) paper, which is a Gaussian process that describes the behavior of neural networks in the infinite-width limit. 

!!! danger "Spoiler"
    Neural Tangent Kernel (NTK) is almost everywhere in the deep learning theory literature. Almost everything we know about deep learning mathematically is based on the NTK. So, if you are interested in deep learning theory, you owe it to yourself to understand the NTK.



### Markov Chain
A Markov chain is a type of stochastic process that satisfies the Markov property, which states that the future state of the process depends only on its current state and not on its past states. Formally, a Markov chain is defined by a set of states and transition probabilities between those states.

!!! danger "Defenition"
    
    A Markov chain is defined as a tuple $\langle S, P \rangle$, where:

    - $S$ is a finite or countable set of states.
    - $P$ is a transition probability matrix, where $P_{ij} = P(X_{t+1} = j | X_t = i)$ represents the probability of transitioning from state $i$ to state $j$ in one time step.

Markov chains have some good properties that make them useful for modeling a wide range of systems. Let's explore some of these properties:

1. **Memoryless Property**: The future state depends only on the current state, not on the past states. This is the essence of the Markov property.
2. **Stationary Distribution**: A Markov chain may converge to a stationary distribution, which is a probability distribution over states that remains unchanged as the process evolves. This is particularly useful for long-term predictions. We will discuss this in more detail later.
3. **Irreducibility**: A Markov chain is irreducible, meaning it is possible to reach any state from any other state in a finite number of steps. This property ensures that the chain can explore the entire state space.

#### Simple Example
Let's consider a simple example of a Markov chain with three states, representing whether conditions; sunny, cloudy, or rainy. 

![Markov Chain Example](/assets/images/prerequisites/stochastic_processes/Markov-chain.png){ align=center width=50% }

As you can guess from the image, the states are represented by circles, and the arrows indicate the possible transitions between states. The numbers on the arrows represent the transition probabilities. For a better structure, let's write the transition matrix for this Markov chain. The transition matrix is a square matrix that describes the probabilities of moving from one state to another in one time step. Each entry in the matrix represents the probability of transitioning from one state to another. In other words, the entry in row $i$ and column $j$ of the matrix represents the probability of transitioning from state $i$ to state $j$. The sum of the probabilities in each row must equal 1, as they represent all possible transitions from that state.

In this example, the transition probabilities are as follows:
states are sunny, cloudy, and rainy. Lets define the states as follows:

- $S^0$: Sunny
- $S^1$: Cloudy
- $S^2$: Rainy

Thus, the transition probability matrix $P$ is given by:

$$
P = \begin{bmatrix}
0.8 & 0.1 & 0.1 \\
0.4 & 0.4 & 0.2 \\
0.2 & 0.3 & 0.5
\end{bmatrix}
$$

Now, let's say we start in the sunny state $S^0$. Also, let's denote the state at time $t$ as $S_t$. In other words, $S_0 = S^0$. The next state $S_1$ will be determined by the transition probabilities in the matrix $P$. For example, if we are in the sunny state $S^0$, there is an 80% chance that we will stay in the sunny state, a 10% chance that we will transition to the cloudy state, and a 10% chance that we will transition to the rainy state.

Now, we can use the transition matrix to calculate the probabilities of being in each state at time $t=1$. Also, let's abuse the notation a bit and denote the state at time $t$ as a vector $S_t$ instead of a single state. So, we can write the state at time $t=0$ as a vector $S_0 = [1, 0, 0]$, where the first element represents the probability of being in the sunny state, the second element represents the probability of being in the cloudy state, and the third element represents the probability of being in the rainy state. Then, we can calculate the state at time $t=1$ as follows:

$$
S_1 = S_0 \cdot P = [1, 0, 0] \cdot \begin{bmatrix}
0.8 & 0.1 & 0.1 \\
0.4 & 0.4 & 0.2 \\
0.2 & 0.3 & 0.5
\end{bmatrix} = [0.8, 0.1, 0.1]
$$

This means that at time $t=1$, there is an 80% chance of being in the sunny state, a 10% chance of being in the cloudy state, and a 10% chance of being in the rainy state.

Now, we can repeat this process to calculate the state at time $t=2$:

$$
S_2 = S_1 \cdot P = [0.8, 0.1, 0.1] \cdot \begin{bmatrix}
0.8 & 0.1 & 0.1 \\
0.4 & 0.4 & 0.2 \\
0.2 & 0.3 & 0.5
\end{bmatrix} = [0.7, 0.15, 0.15]
$$

But how about the whether at time $t=k$? We can calculate the state at time $t=k$ as follows:

$$
S_k = S_0 \cdot P^k
$$

Now, another great property of Markov chains is that they can converge to a stationary distribution. This means that as $k$ approaches infinity, the state vector $S_k$ will converge to a fixed distribution, regardless of the initial state. In other words, the probabilities of being in each state will stabilize over time. But how to find this stationary distribution? The stationary distribution $\pi$ is a probability vector that satisfies the following equation:

$$
\pi P = \pi
$$

Look's familiar, right? This is the same equation we used to calculate to find eigenvalues and eigenvectors. In fact, the stationary distribution is the left eigenvector of the transition matrix $P$ corresponding to the eigenvalue 1.

Let's calculate the stationary distribution for our example. We can do this by solving the following system of equations:

``` py title="calculate_stationary_distribution.py"
    import numpy as np
    P = np.array([[0.8, 0.1, 0.1],
                [0.4, 0.4, 0.2],
                [0.2, 0.3, 0.5]])

    # Calculate the eigenvalues and eigenvectors of the matrix
    eigenvalues, eigenvectors = np.linalg.eig(P)

    # Print the eigenvalues and eigenvectors
    print("Eigenvalues:")
    print(eigenvalues)
    print("Eigenvector 1:")
    print(eigenvectors[:, 0])
    print("Eigenvector 2:")
    print(eigenvectors[:, 1])
    print("Eigenvector 3:")
    print(eigenvectors[:, 2])
    # Calculate the stationary distribution
    # The stationary distribution is the eigenvector corresponding to the eigenvalue 1
    stationary_distribution = eigenvectors[:, 0]
    # Normalize the stationary distribution
    stationary_distribution /= np.sum(stationary_distribution)
    # Print the stationary distribution
    print("Stationary distribution:")
    print(stationary_distribution)
```
this shall give you $[0.333, 0.33, 0.33]$ as the stationary distribution. This means that as time goes to infinity, the probabilities of being in each state will converge to approximately 33% for each state. If this is not making sense, you can look at cat and mouse example at [Wikipedia](https://en.wikipedia.org/wiki/Stochastic_matrix#Example:_Cat_and_mouse)

## Markov Decision Process (MDP)

A Markov Decision Process (MDP) is a mathematical framework used to model sequential decision-making problems where outcomes are partly random and partly under the control of a decision maker. In many ways, an MDP is an extension of a Markov chain with the inclusion of actions and rewards.

### Components of an MDP
An MDP is typically defined as a tuple ⟨S, A, P, R, γ⟩, where:

- **S (States)**: Represents the set of all possible states in the environment. Each state conveys information about the current situation.
- **A (Actions)**: Denotes the set of actions available to the agent. The choices made by the agent influence the path through the state space.
- **P (Transition Probability)**: A function that defines the probability of moving from one state to another given an action. Formally, P(s' | s, a) is the probability of arriving at state s' when action a is taken in state s.
- **R (Reward Function)**: Specifies the immediate reward received after transitioning from state s to state s' due to action a. This reward quantifies the immediate benefit or cost of taking an action at a state.
- **γ (Discount Factor)**: A scalar between 0 and 1 that discounts future rewards compared to immediate rewards. A smaller value makes the agent more shortsighted, while a value closer to 1 encourages planning for the long-term.

### Understanding MDPs
In an MDP, the decision-making process is often guided by a policy, which is a strategy mapping states to actions. The goal is to find an optimal policy that maximizes the cumulative reward over time. This is frequently formulated as maximizing the expected sum of discounted rewards:

$$
  V(s) = E[ R_t + γR_{t+1} + γ^2R_{t+2} + … | s_t = s ]
$$

where $V(s)$ represents the value of starting in state $s$ under the optimal policy.

### How MDPs Extend Markov Chains
While a Markov chain models the evolution of states with fixed probabilities, an MDP introduces choices at each state thanks to actions. This allows for different trajectories, as the transition probabilities now depend on both state and action.

In other words, the transition probabilities in an MDP is now defined as follows:

$$
P(s' | s, a) = P(X_{t+1} = s' | X_t = s, A_t = a)
$$

Although the Markov property still holds, the agent's actions influence the transition probabilities. This is a key distinction that allows MDPs to model decision-making processes where the agent has control over its actions.

!!! Note 

    Almost and ==almost== everywhere throughout the course, this transition probability will be deterministic! for simplicity. But in real life, this is not the case. in other words, this is the influence of real word chance, human-level approximation, and so on. So, the transition probability is not deterministic. But for simplicity, we will assume that the transition probability is deterministic. ==You may not want to do this in exams if it's not mentioned explicitly, your exam is a real-world environment.==

### Relation to graphical models
Graphical models provide a visual and mathematical framework for understanding the dependencies within an MDP. By representing states, actions, and their probabilistic transitions as nodes and directed edges, these models offer insights into the inherent structure of decision-making problems. This perspective not only clarifies how various components of an MDP interact, but also enables the application of inference algorithms to efficiently compute optimal policies in reinforcement learning. If you are interested in this topic, you can check out the [RL and Control as probabilistic inference](https://arxiv.org/abs/1805.00909) paper, which provides a comprehensive overview of how graphical models can be applied to MDPs and reinforcement learning.

For instance, in the below simple graphical model, the states and actions are represented as nodes, ==which is infact very different from the usual state:node and action:edge representation==. The directed edges indicate the ==direct== dependencies between states and actions.  
![Graphical Model of MDP](/assets/images/prerequisites/stochastic_processes/MDP_GM.png){ align=center width=50% }

### Example of MDP: Grid World

Consider a simple grid world where an agent navigates within a 3x3 grid. The objective is to reach a terminal state while minimizing the cost incurred at each step.

**States (S):**  
Each cell in the grid represents a state. For instance, a state can be represented as a coordinate $(i, j)$ with $i, j ∈ \{1, 2, 3\}$.

**Actions (A):**  
The available actions are:  
- Up  
- Down  
- Left  
- Right  

**Transition Function (P):**  
Transitions are deterministic:  
- When an agent attempts to move in a given direction, it moves to the adjacent cell in that direction unless the move would take it outside the grid.  
- If a move would cross the grid boundary, the agent remains in the same state.

For example, if the agent is at state $(2, 2)$ and takes action Up, it transitions to state $(1, 2)$. Conversely, if the agent is at $(1, 1)$ and takes action Left, it stays at $(1, 1)$.

**Reward Function (R):**  
- Each move incurs a reward of $-1$.  
- Reaching the terminal state, for example, state $(3, 3)$, gives a reward of $0$ and ends the episode.

**Discount Factor (γ):**  
We use a discount factor of $0.9$, balancing immediate and future rewards.

**Example Run:**  
Assume the agent starts at state $(1, 1)$ and follows a policy that moves it right and then down:
1. From $(1, 1)$, move Right to $(1, 2) ⟹$ Reward: $-1$  
2. From $(1, 2)$, move Down to $(2, 2) ⟹$ Reward: $-1$  

The cumulative discounted reward (return) is calculated as:  

$$
V = -1 + 0.9 × (-1) = -1 - 0.9 = -1.9
$$

This example demonstrates how the grid world is modeled as a Markov Decision Process, clearly defining states, actions, deterministic transitions, rewards, and the discounting of future rewards.

For more reading on MDPs, you can check out the [Medium article](https://sanchittanwar75.medium.com/markov-chains-and-markov-decision-process-e91cda7fa8f2) or [this](https://builtin.com/machine-learning/markov-decision-process) article.
