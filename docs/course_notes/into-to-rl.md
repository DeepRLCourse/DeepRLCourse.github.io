---
comments: True
description: This page provides an introduction to Reinforcement Learning, covering key concepts, comparisons with other AI methods, and foundational frameworks like MDPs and POMDPs.
---

# Week 1: Introduction to RL

## Why Do We Need Reinforcement Learning?

Reinforcement Learning (RL) is a subfield of artificial intelligence (AI) that focuses on training agents to make sequences of decisions by interacting with an environment. Unlike other AI methods, RL is particularly well-suited for problems where the agent must learn optimal behavior through trial and error, often in dynamic and uncertain environments.

### Limitations of Other AI Methods

1. **Supervised Learning**: 
   - **What it does**: Supervised learning requires a labeled dataset where the correct output (label) is provided for each input. The model learns to map inputs to outputs based on this data.
   - **Limitation**: In many real-world scenarios, obtaining a labeled dataset is impractical or impossible. For example, consider training a robot to walk. It's not feasible to provide a labeled dataset of all possible states and actions the robot might encounter.

2. **Unsupervised Learning**:
   - **What it does**: Unsupervised learning deals with unlabeled data and tries to find hidden patterns or intrinsic structures within the data.
   - **Limitation**: While unsupervised learning can identify patterns, it doesn't provide a way to make decisions or take actions based on those patterns. For instance, clustering similar states in a game doesn't tell the agent how to win the game.

3. **Traditional Control Methods**:
   - **What it does**: Traditional control methods are designed to maintain a system's state within a desired range using predefined rules.
   - **Limitation**: These methods require a precise model of the environment and are not adaptable to complex, changing environments.

### Where Reinforcement Learning Shines

Reinforcement Learning excels in scenarios where:

- **No Labeled Data is Available**: RL agents learn by interacting with the environment and receiving feedback in the form of rewards or penalties. This eliminates the need for a pre-labeled dataset.
  
- **Sequential Decision-Making is Required**: RL is designed to handle problems where decisions are made in a sequence, and each decision affects the future state of the environment. For example, in a game like Go or Chess, each move affects the board's state and influences future moves.

- **The Environment is Dynamic and Uncertain**: RL agents can adapt to changing environments and learn optimal policies even when the environment is not fully known or is stochastic. For instance, an RL agent can learn to navigate a maze even if the maze's layout changes over time.

- **Non-i.i.d. Data**: RL is capable of handling non-independent and identically distributed (non-i.i.d.) data. In many real-world scenarios, data points are not independent (e.g., the state of the environment at one time step depends on previous states) and may not be identically distributed (e.g., the distribution of states changes over time). A notable example is **robotic control**, where an autonomous robot learns to walk. Each movement directly affects the next state, and the terrain may change dynamically, requiring the RL agent to adapt its policy based on sequential dependencies. RL agents can learn from such data by considering the temporal dependencies and adapting to the evolving data distribution.

- **Lack of Input Data**: In supervised learning, the model is provided with input-output pairs $(x, y)$, where $x$ is the input data and $y$ is the corresponding label. In RL, not only are the labels (correct actions) not provided, but the "$x$"s (input states) are also not explicitly given to the model. The agent must actively interact with the environment to observe and collect these states.
  
   - **Example**: Consider training an RL agent to play a video game. In supervised learning, you would need a dataset of game states ($x$) and the corresponding optimal actions ($y$). In RL, the agent starts with no prior knowledge of the game states or actions. It must explore the game environment, observe the states, and learn which actions lead to higher rewards. This process of discovering and learning from the environment is what makes RL uniquely powerful for tasks where the input data is not readily available.

### Key Difference Between RL and Supervised Learning

The primary distinction between RL and supervised learning is that, while feedback is provided in RL, the exact correct answer or action is not explicitly given. Let's delve deeper into this concept and explore the importance of exploration in RL, along with its challenges.

#### Supervised Learning:
  - In supervised learning, you are presented with a bunch of data and told exactly what the answer for each data point is. Your model adjusts its parameters to get the prediction right for the data you trained on, and the goal is to generalize to unseen data, but is doesn't try to do **better** than the data.
  - **Example**: In a image classification task, the model is given images along with their correct labels (e.g., "cat" or "dog"). The model learns to predict the label for new, unseen images based on this labeled data.

- [**Imitation Learning**](https://sites.google.com/view/icml2018-imitation-learning/): A specialized form of supervised learning used in decision-making problems is **Imitation Learning (IL)**. In IL, the model learns by mimicking expert demonstrations. The training data consists of state-action pairs provided by an expert, and the model learns to replicate the expert's behavior. Unlike traditional supervised learning, IL is applied to sequential decision-making tasks, making it a bridge between supervised learning and RL.



#### Reinforcement Learning
- **Agent-Environment Interaction**: In RL, the agent interacts with the environment, takes actions, receives rewards, and adapts its policy based on these rewards. However, unlike supervised learning, the agent is never told which action was the right one for a given state or what the correct policy is for a given task. In other words, there are no labels! The agent must learn the optimal actions using the learning signals provided by the reward.
  
- **Exploration**: Exploration is a critical component of RL. Since the agent is not provided with labeled data or explicit instructions, it must explore the environment to discover which actions yield the highest rewards. This exploration allows the agent to learn from its experiences and improve its policy over time. Without exploration, the agent might get stuck in suboptimal behaviors, never discovering better strategies.

- **Drawbacks and Difficulties of Exploration**: While exploration is essential, it comes with its own set of challenges. One major difficulty is that **wrong exploration** can lead to poor learning outcomes. If the agent explores suboptimal or harmful actions excessively, it may reinforce bad behaviors, leading to a failure in learning the optimal policy. This is often summarized by the phrase **"garbage in, garbage out"**—if the agent explores poorly and collects low-quality data, the resulting policy will also be of low quality.

  - **Example**: Consider an RL agent learning to navigate a maze. If the agent spends too much time exploring dead ends or repeatedly taking wrong turns, it may fail to discover the correct path to the goal. The agent's policy will be based on these poor explorations, resulting in a suboptimal or even failing strategy.

- **Balancing Exploration and Exploitation**: A key challenge in RL is balancing exploration (trying new actions to discover their effects) and exploitation (using known actions that yield high rewards). Too much exploration can lead to inefficiency, while too much exploitation can cause the agent to miss out on better strategies. Techniques like **epsilon-greedy policies** and **Thompson sampling** are often used to strike this balance.

- **Outperforming Human Intelligence**: Despite the challenges, when exploration is done effectively, the agent can discover novel strategies and solutions that humans might not consider. Since the data is collected by the agent itself through exploration, an RL agent can even **outperform** human intelligence and execute impressive actions that no one has thought of before.

- **Example**: In a game of chess, the RL agent might receive a reward for winning the game but won't be told which specific move led to the victory. It must figure out the sequence of optimal moves through trial and error. By exploring different moves and learning from the outcomes, the agent can develop a strategy that maximizes its chances of winning. However, if the agent explores ineffective moves too often, it may fail to learn a winning strategy.

### Example: Autonomous Driving

Consider the task of autonomous driving. 

- **Supervised Learning Approach**: You would need a massive labeled dataset of all possible driving scenarios, including rare events like a child running into the street. This is impractical.
  - **Imitation Learning**: In autonomous driving, IL can be used to train a model by observing human drivers. The model is provided with data on how human drivers react in various driving scenarios (e.g., steering, braking, accelerating). The model learns to mimic these actions, effectively reducing the problem to a supervised learning task. However, the model's performance is limited by the quality of the expert demonstrations and may not discover strategies that outperform the expert.

- **Reinforcement Learning Approach**: An RL agent can learn to drive by interacting with a simulated environment. It receives rewards for safe driving and penalties for accidents or traffic violations. Over time, the agent learns an optimal policy for driving without needing a labeled dataset.


## Key Concepts in Reinforcement Learning

To understand RL, it's essential to grasp some fundamental concepts: **state**, **action**, **reward**, and **policy**. Let's explore each of these concepts in detail, using an example to illustrate their roles.

### 1. State ($\mathbf{s}_t$)

- **Definition**: A **state** represents the current situation or configuration of the environment at a given time. It encapsulates all the relevant information that the agent needs to make a decision.
- **Example**: Consider a self-driving car navigating through a city. The state could include the car's current position, speed, the positions of other vehicles, traffic lights, and pedestrians. All these factors together define the state of the environment at any moment.

### 2. Action ($\mathbf{a}_t$)

- **Definition**: An **action** is a decision or move made by the agent that affects the environment. The set of all possible actions that an agent can take is called the **action space**.
- **Example**: In the self-driving car scenario, possible actions might include accelerating, braking, turning left, turning right, or maintaining the current speed. Each action changes the state of the environment, such as the car's position or speed.

### 3. Reward ($r_t$)

- **Definition**: A **reward** is a feedback signal that the agent receives from the environment after taking an action. The reward indicates the immediate benefit or cost of the action taken. The goal of the agent is to maximize the cumulative reward over time. The reward can be provided by an **expert** or learned from demonstrations. This can be achieved by directly copying observed behavior or inferring rewards from observed behavior ([**Inverse RL**](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/inverseRL.pdf)) .

- **Example**: For the self-driving car, rewards could be assigned as follows:
  - **Positive Reward**: Reaching the destination safely (+100), obeying traffic rules (+10).
  - **Negative Reward**: Colliding with another vehicle (-100), running a red light (-50).

- **Sparse Reward Environments**

   - **Definition**: In **sparse reward environments**, the agent receives rewards very infrequently. Instead of getting feedback after every action, the agent might only receive a reward after completing a long sequence of actions or achieving a significant milestone. For example, in a game, the agent might only receive a reward upon winning or losing, with no feedback provided during the game.

   - **Challenges**:
      - **Difficulty in Learning**: Sparse rewards make it challenging for the agent to learn which actions lead to positive outcomes. Since the agent receives little to no feedback during most of its interactions, it struggles to associate specific actions with rewards. This can lead to slow or ineffective learning.
      - **Exploration**: In sparse reward environments, the agent must explore extensively to discover actions that yield rewards. Without frequent feedback, the agent may take a long time to stumble upon the correct sequence of actions, making the learning process inefficient.
      - **Credit Assignment Problem**: Determining which actions in a sequence contributed to a reward is difficult in sparse reward settings. The agent may not be able to accurately attribute the reward to the correct actions, leading to suboptimal policies.

   - **Example**: Consider an RL agent learning to play a complex strategy game where the only reward is given at the end of the game (e.g., +1 for winning and -1 for losing). The agent must explore countless moves and strategies without any intermediate feedback, making it challenging to learn effective strategies. The agent might take a very long time to discover the sequence of actions that leads to a win.
### 4. Policy ($\pi_{\theta}$)

- **Definition**: A **policy** is a strategy or set of rules that the agent follows to decide which action to take in a given state. It maps states to actions and can be deterministic (always choosing a specific action in a state) or stochastic (choosing actions based on a probability distribution).
- **Example**: In the self-driving car example, a policy might dictate that the car should slow down when it detects a pedestrian crossing the street or stop when it encounters a red traffic light. The policy is what the agent learns and optimizes to maximize cumulative rewards.

<center> 
<img src="\assets\images\course_notes\into-to-rl\RL_Framework.png"
     alt=""
     style="float: center; margin-right: 10px;" 
     />
      Fig1. RL Framework </center>


## The Anatomy of Reinforcement Learning

An RL agent's training is done through an interactive process between the agent and the environment. This process involves several key steps that allow the agent to learn and improve its decision-making capabilities. 

### 1. **Evaluate the Action: "How well was our choice of (x, y, z); execute gripping!"**

- This step involves assessing the effectiveness of the action taken by the agent. The agent performs an action, such as gripping an object at a specific location (x, y, z), and then evaluates how well this action was executed.

### 2. **Fit a Model**

- Fitting a model refers to creating a representation of the environment or the task based on the data collected from interactions. This model helps the agent predict the outcomes of future actions and understand the dynamics of the environment.

### 3. **Estimate the Return: "Gripping objects estimate the return"**

- Estimating the return involves calculating the expected cumulative reward from a given state or action. The return is a key concept in RL, as the agent aims to maximize this cumulative reward over time.

### 4. **Understand Physical Consequences: "Or ... when I grip some object on a specific location, what happens physically; i.e. how its (x, y, z)'s change?"**

- This step emphasizes the importance of understanding the physical consequences of actions. The agent needs to know how its actions (like gripping an object at a specific location) affect the environment, particularly the object's position (x, y, z).

### 5. **Improve the Policy: "Improve the policy 'best guess'"**

- Improving the policy involves refining the agent's strategy for choosing actions based on the feedback received from the environment. The "best guess" refers to the agent's current understanding of the optimal actions, which is continuously updated as the agent learns.


<center> 
<img src="\assets\images\course_notes\into-to-rl\RL_Anatomy.png"
     alt=""
     style="float: center; margin-right: 10px;" 
     />
      Fig2. The Anatomy of RL </center>


## Comparing Reinforcement Learning with Other Learning Methods

To better understand the unique characteristics of Reinforcement Learning, it's helpful to compare it with other learning methods. The table below provides a comparison of RL with other approaches such as Supervised Learning (SL), Unsupervised Learning (UL), and Imitation Learning (IL). Let's break down the table and discuss the key differences and similarities.

### Comparison Table
 
  |    | AI Planning | SL  | UL  | RL  | IL  |
  |---|---|---|---|---|---|
  | **Optimization**    | X    |    |    | X   | X   |
  | **Learns from experience** |    | X   | X   | X   | X   |
  | **Generalization**    | X    | X   | X   | X   | X   |
  | **Delayed Consequences** | X    |    |    | X   | X   |
  | **Exploration**    |    |    |    | X   |    | 

### Key Concepts

1. **Optimization**
    - **AI Planning**: Involves optimizing a sequence of actions to achieve a goal.
    - **RL**: Focuses on optimizing policies to maximize cumulative rewards.
    - **IL**: Also involves optimization, often by learning from demonstrations.

2. **Learns from Experience**
    - **SL**: Learns from labeled data, where each input has a corresponding output.
    - **UL**: Learns from unlabeled data by identifying patterns.
    - **RL**: Learns by interacting with the environment and receiving feedback.
    - **IL**: Learns from demonstrations of good policies.

3. **Generalization**
    - All methods (AI Planning, SL, UL, RL, IL) aim to generalize from training data to new, unseen situations.

4. **Delayed Consequences**
    - **AI Planning**: Considers the long-term effects of actions.
    - **RL**: Takes into account the delayed consequences of actions to maximize future rewards.
    - **IL**: Can also consider delayed consequences if the demonstrations include long-term strategies.

5. **Exploration**
    - **RL**: Requires exploration of the environment to discover optimal policies.
    - Other methods (SL, UL, IL) typically do not involve exploration in the same way.
    - In particular, IL is limited to the data and experiences provided by the expert model. Since IL relies on demonstrations, it cannot leverage the benefits of exploration to discover better strategies beyond what the expert has demonstrated. In contrast, RL has the capability to explore and learn from trial and error, allowing it to outperform expert demonstrations in some cases by discovering novel, more efficient policies.
## Planning vs Learning

Two fundamental problems in sequential decision making:

1. **Reinforcement learning**:
    - The environment is initially **unknown**
    - The agent **interacts** with the environment
    - The agent **improves** its policy

2. **Planning**:
    - A model of the environment is **known**
    - The agent performs computations with its model (w**ithout any external
    interaction**)
    - The agent **improves** its policy, a.k.a. deliberation, reasoning, introspection, pondering, thought, search 

## Introduction to Markov Decision Processes (MDPs)

Markov Decision Processes (MDPs) are a fundamental framework used in Reinforcement Learning to model decision-making problems. MDPs provide a mathematical foundation for describing an environment in which an agent interacts, takes actions, and receives rewards. Let's break down the components of an MDP.

### Components of an MDP

An MDP is defined by the following components:

1. **Set of States ($S$)**: 
    - The set of all possible states that the environment can be in. A state $s \in S$ represents a specific configuration or situation of the environment at a given time.

2. **Set of Actions ($A$)**: 
    - The set of all possible actions that the agent can take. An action $a \in A$ is a decision made by the agent that affects the environment.

3. **Transition Function ($P(s' | s, a)$)**:
    - The transition function defines the probability of transitioning from state $s$ to state $s'$ when action $a$ is taken. This function captures the dynamics of the environment.
    - **Markov Property**: The transition function satisfies the Markov property, which states that the future state $s'$ depends only on the current state $s$ and action $a$, and not on the sequence of states and actions that preceded it. This is why the process is called "Markovian."

4. **Reward Function ($R(s, a, s')$)**:
    - The reward function specifies the immediate reward received by the agent after transitioning from state $s$ to state $s'$ by taking action $a$. The reward is a scalar value that indicates the benefit or cost of the action.

5. **Start State ($s_0$)**:
    - The initial state from which the agent starts its interaction with the environment.

6. **Discount Factor ($\gamma$)**:
    - The discount factor $\gamma$ (where $0 \leq \gamma \leq 1$) determines the present value of future rewards. A discount factor close to 1 makes the agent prioritize long-term rewards, while a discount factor close to 0 makes the agent focus on immediate rewards.

7. **Horizon ($H$)**:
    - The horizon $H$ represents the time horizon over which the agent interacts with the environment. It can be finite (fixed number of steps) or infinite (continuous interaction).

### Episodes and Environment Types  
- An **episode** is a sequence of interactions that starts from an initial state and ends when a terminal condition is met.  
- **Episodic Environments**: The interaction consists of episodes, meaning the agent's experience is divided into separate episodes with a clear start and end (e.g., playing a game with levels or a robot completing a task like picking up an object).  
- **Non-Episodic (Continuous) Environments**: There is no clear termination, and the agent continuously interacts with the environment without resetting (e.g., stock market trading, autonomous vehicle navigation).  

### Difference Between Horizon and Episode  
- The **episode** refers to a full sequence of interactions that has a clear beginning and an end.  
- The **horizon ($H$)** defines the length of time the agent considers while making decisions, which can be within a single episode (in episodic environments) or over an ongoing interaction (in non-episodic environments).  
- In **finite-horizon episodic tasks**, the episode length is usually equal to the horizon. However, in infinite-horizon tasks, the agent keeps interacting with the environment indefinitely.

### The Goal of Reinforcement Learning

The goal of RL in the context of an MDP is to find a policy $\pi$ that maximizes the expected cumulative reward over time. The policy $\pi$ is a function that maps states to actions, and it can be deterministic or stochastic.

#### Objective:
$$
\max_{\pi} \mathbb{E}\left[\sum_{t=0}^{H} \gamma^{t} R(S_t, A_t, S_{t+1}) \mid \pi\right]
$$

- **Expected Value**: The use of the expected value $\mathbb{E}$ is necessary because the transition function $P(s' | s, a)$ is probabilistic. The agent does not know exactly which state $s'$ it will end up in after taking action $a$ in state $s$. Therefore, the agent must consider the expected reward over all possible future states, weighted by their probabilities.

- **Cumulative Reward**: The agent aims to maximize the sum of discounted rewards over time. The discount factor $\gamma$ ensures that the agent balances immediate rewards with future rewards. A higher $\gamma$ makes the agent more farsighted, while a lower $\gamma$ makes it more shortsighted.

### Example: Grid World

Consider a simple grid world where the agent must navigate from a start state to a goal state while avoiding obstacles. The states $S$ are the grid cells, the actions $A$ are movements (up, down, left, right), and the reward function $R(s, a, s')$ provides positive rewards for reaching the goal and negative rewards for hitting obstacles. The transition function $P(s' | s, a)$ defines the probability of actually moving to a neighboring cell when an action is taken.

<center> 
<img src="\assets\images\course_notes\into-to-rl\gridworld.png"
     alt=""
     style="float: center; margin-right: 10px;" 
     /> 
     
   Fig3. Grid World Game </center>


### An Example of Gridworld  
In the provided Gridworld example, the agent starts from the yellow square and has to navigate to a goal while avoiding the cliff. The rewards are:
- **+1 for reaching the close exit**
- **+10 for reaching the distant exit**
- **-10 penalty for stepping into the cliff** (red squares)

<center> 
<img src="\assets\images\course_notes\into-to-rl\gridworld_example.png"
     alt=""
     style="float: center; margin-right: 10px;" 
     /> 
     
   Fig4. Grid World Example </center>

The agent's choice depends on:
- The **discount factor ($\gamma$)**, which determines whether it prioritizes short-term or long-term rewards.
- The **noise level**, which introduces randomness into actions.

Depending on the values of $\gamma$ and noise, the agent's behavior varies:
1. **$\gamma$ = 0.1, noise = 0.5:**  
   - The agent **prefers the close exit (+1) but takes the risk of stepping into the cliff (-10).**  
2. **$\gamma$ = 0.99, noise = 0:**  
   - The agent **prefers the distant exit (+10) while avoiding the cliff (-10).**  
3. **$\gamma$ = 0.99, noise = 0.5:**  
   - The agent **still prefers the distant exit (+10), but due to noise, it risks the cliff (-10).**  
4. **$\gamma$ = 0.1, noise = 0:**  
   - The agent **chooses the close exit (+1) while avoiding the cliff.**  

### Stochastic Policy  

Another source of randomness in MDPs comes from **stochastic policies**. Unlike the transition function $P(s' | s, a)$, which describes the environment’s inherent randomness in executing actions, a **stochastic policy** $\pi(a | s)$ defines the probability of selecting an action $a$ when in state $s$. This means that even if the environment were fully deterministic, the agent itself may act probabilistically.

#### **Example: Gridworld with a Stochastic Policy**  
Consider a modified version of the previous Gridworld example. Instead of always choosing the action with the highest expected return, the agent follows a **stochastic policy** where it selects each possible action with a certain probability:

- With **99% probability**, the agent follows its optimal policy.
- With **1% probability**, it selects a random action.


**Transition Probability $P(s'|s, a)$**  is the probability that taking action $a$ in state $s$ results in transitioning to state $s'$. This is determined by the **environment**. If the environment is slippery, moving "right" from $(2,2)$ may lead to $(2,3)$ with 80% probability, but also to $(3,2)$ with 20%.
**Stochastic Policy $\pi(a | s)$** determines the probability that the agent **chooses** action $a$ when in state $s$. This is determined by the **agent's strategy**.

- If the policy $\pi(a | s)$ is deterministic, the agent **always selects the same action** in a given state.
- If the policy $\pi(a | s)$ is stochastic, the agent **introduces randomness in its decision-making process**, which can be beneficial in **exploration** and **avoiding local optima**.

### Graphical Model of MDPs
MDPs can be represented graphically as a sequence of **states ($\mathbf{s}$)**, **actions ($\mathbf{a}$)**, and **transitions ($p$)**:

- The agent starts at state $\mathbf{s}_1$.
- It selects an action $\mathbf{a}_1$, which moves it to $\mathbf{s}_2$ based on the probability $p(\mathbf{s}_2 | \mathbf{s}_1, \mathbf{a}_1)$.
- The process continues, forming a **decision-making chain** where each action influences future states and rewards.

<center> 
<img src="\assets\images\course_notes\into-to-rl\graphical_MDP.png"
     alt=""
     style="float: center; margin-right: 10px;" 
     /> 
     
   Fig5. Graphical Model of MDPs </center>

## Partially Observable MDPs (POMDPs)
In real-world scenarios, the agent may not have full visibility of the environment, leading to **Partially Observable MDPs (POMDPs)**.
- **Hidden states:** The true state $\mathbf{s}_t$ is not fully known to the agent.
- **Observations ($O$):** Instead of directly knowing $\mathbf{s}_t$, the agent receives a noisy or incomplete observation $\mathbf{o}_t$.
- **Decision-making challenge:** The agent must infer the state from past observations and actions.

<center> 
<img src="\assets\images\course_notes\into-to-rl\POMDP.png"
     alt=""
     style="float: center; margin-right: 10px;" 
     /> 
     
   Fig6. Partially Observable MDPs (POMDPs) </center>

### Policy as a Function of State ($\mathbf{s}_t$) or Observation ($\mathbf{o}_t$)

   - **Fully Observed Policy**: When the agent has access to the full state $\mathbf{s}_t$, the policy is denoted as $\pi_{\theta}(\mathbf{a}_t | \mathbf{s}_t)$. This means the action $\mathbf{a}_t$ is chosen based on the current state $\mathbf{s}_t$.
   - **Partially Observed Policy**: When the agent only has access to observations $\mathbf{o}_t$, the policy is denoted as $\pi_{\theta}(\mathbf{a}_t | \mathbf{o}_t)$. This means the action $\mathbf{a}_t$ is chosen based on the current observation $\mathbf{o}_t$.
   
      - **Observation**: At each time step, the agent receives an observation $\mathbf{o}_t$ that provides partial information about the current state $\mathbf{s}_t$. For example, $\mathbf{o}_1$ is the observation corresponding to state $\mathbf{s}_1$.

      - **Policy**: The policy $\pi_{\theta}$ maps observations to actions. For instance, $\pi_{\theta}(\mathbf{a}_1 | \mathbf{o}_1)$ determines the action $\mathbf{a}_1$ based on the observation $\mathbf{o}_1$.
<center> 
<img src="\assets\images\course_notes\into-to-rl\policy_POMDP.png"
     alt=""
     style="float: center; margin-right: 10px;" 
     /> 
     
   Fig7. POMDP Policy </center>

## Utility Function in Reinforcement Learning

In Reinforcement Learning, the **utility function** plays a central role in evaluating the long-term desirability of states or state-action pairs. The utility function quantifies the expected cumulative reward that an agent can achieve from a given state or state-action pair, following a specific policy. Let's explore the concept of the utility function and its mathematical formulation.

### Definition of Utility Function

The utility function measures the expected cumulative reward that an agent can accumulate over time, starting from a particular state or state-action pair, and following a given policy. There are two main types of utility functions:

1. **State Value Function ($V^{\pi}(s)$)**:
    - The state value function $V^{\pi}(s)$ represents the expected cumulative reward when starting from state $s$ and following policy $\pi$ thereafter.
    - Mathematically, it is defined as:
        $$V^{\pi}(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(S_t, A_t, S_{t+1}) \mid S_0 = s, \pi \right]$$
    - Here, $\gamma$ is the discount factor, and $R(S_t, A_t, S_{t+1})$ is the reward received at time $t$.

2. **Action-Value Function ($Q^{\pi}(s, a)$)**:
    - The action-value function $Q^{\pi}(s, a)$ represents the expected cumulative reward when starting from state $s$, taking action $a$, and following policy $\pi$ thereafter.
    - Mathematically, it is defined as:
        $$Q^{\pi}(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(S_t, A_t, S_{t+1}) \mid S_0 = s, A_0 = a, \pi \right]$$

### Why is the Utility Function Important?

The utility function is crucial for several reasons:

- **Policy Evaluation**: It allows the agent to evaluate how good a particular policy $\pi$ is by estimating the expected cumulative reward for each state or state-action pair.
- **Policy Improvement**: By comparing the utility of different actions, the agent can improve its policy by choosing actions that lead to higher cumulative rewards.
- **Optimal Policy**: The ultimate goal of RL is to find the optimal policy $\pi^*$ that maximizes the utility function for all states or state-action pairs.

### Bellman Equation for Utility Functions

The utility functions satisfy the **Bellman equation**, which provides a recursive relationship between the value of a state (or state-action pair) and the values of its successor states. The Bellman equation is fundamental for solving MDPs and is used in many RL algorithms.

1. **Bellman Equation for State Value Function**:

$$V^{\pi}(s) = \sum_{a} \pi(a | s) \sum_{s'} P(s' | s, a) \left[ R(s, a, s') + \gamma V^{\pi}(s') \right]$$

* This equation states that the value of a state $s$ under policy $\pi$ is the expected immediate reward plus the discounted value of the next state $s'$.

2. **Bellman Equation for Action-Value Function**:

$$Q^{\pi}(s, a) = \sum_{s'} P(s' | s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a' | s') Q^{\pi}(s', a') \right]$$

* This equation states that the value of taking action $a$ in state $s$ under policy $\pi$ is the expected immediate reward plus the discounted value of the next state $s'$ and action $a'$.

### Example: Grid World

Consider the Grid World example where the agent navigates to a goal while avoiding obstacles. The utility function helps the agent evaluate the long-term desirability of each cell (state) in the grid:

- **State Value Function ($V^{\pi}(s)$)**: The agent calculates the expected cumulative reward for each cell, considering the rewards for reaching the goal and penalties for hitting obstacles.
- **Action-Value Function ($Q^{\pi}(s, a)$)**: The agent evaluates the expected cumulative reward for each possible action (up, down, left, right) in each cell, helping it decide the best action to take.

<center> 
<img src="\assets\images\course_notes\into-to-rl\gridworld_V.png"
     alt=""
     style="float: center; margin-right: 10px;" 
     /> 
     
   Fig8. An example of estimated $V^{\pi}(s)$ values in grid world  </center>

   <center> 
<img src="\assets\images\course_notes\into-to-rl\gridworld_Q.png"
     alt=""
     style="float: center; margin-right: 10px;" 
     /> 
     
   Fig9. An example of estimated $Q^{\pi}(s, a)$ values in grid world </center>


## Author(s)

<div class="grid cards" markdown>
-   ![Instructor Avatar](/assets/images/staff/Masoud-Tahmasbi.jpg){align=left width="150"}
    <span class="description">
        <p>**Masoud Tahmasbi**</p>
        <p>Teaching Assistant</p>
        <p>[masoudtahmasbifard@gmail.com](mailto:masoudtahmasbifard@gmail.com)</p>
        <p>
        [:fontawesome-brands-google-scholar:](https://scholar.google.com/citations?hl=en&user=BUiXXIYAAAAJ){:target="_blank"}
        [:fontawesome-brands-github:](https://github.com/masoudtfard){:target="_blank"}
        [:fontawesome-brands-linkedin-in:](https://www.linkedin.com/in/masoud-tahmasbi-fard/){:target="_blank"}
        </p>
    </span>
</div>