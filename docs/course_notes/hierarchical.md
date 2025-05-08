# Hierarchical RL
## Introduction
### A Taste of Hierarchy: The Cake Recipe Analogy
Imagine your friend just baked the most delicious cake you've ever had, and naturally, you ask for the recipe. One might assume this is a simple request—just tell me how to make it! But as it turns out, providing effective instructions is not as straightforward as it seems.

How detailed should your friend be? Should they explain how to preheat the oven, or assume you know that already? Do they need to break down every micro-movement involved in slicing a carrot?
Probably not.

Consider a more elaborate dish like beef bourguignon. At some point in the recipe, it says, “Cut 4 carrots into slices.” To a human, that instruction is clear and sufficient. There's no need to specify every muscle contraction required to pick up a knife or place the carrots on a board. Humans naturally operate at a *pertinent level of abstraction*, applying high-level instructions that encapsulate many low-level actions.

This insight reveals something profound about how intelligent systems—humans, toddlers, even primates—handle complex tasks. They don't plan or act at the level of atomic operations. Instead, they organize behavior into **hierarchies**, leveraging **temporal abstraction** to chunk sequences of actions into meaningful units.

### Temporal Abstraction in Humans
Developmental psychology provides converging evidence that both children and adults rely on cognitive systems that intuitively support this kind of abstraction. For instance, toddlers playing open-ended games, like building block towers, often set intermediate **sub-goals**—such as placing a wide block at the base before stacking others—without being explicitly taught to do so. This behavior involves selecting and pursuing goals across different time scales.

In the same way, slicing an onion in our cooking example is a **temporally extended action**: it may take several individual steps, depending on how fine the cut needs to be, yet we still think of it as one coherent action.

### From Intuition to Reinforcement Learning
This natural human strategy of breaking tasks into manageable chunks forms the core idea behind **Hierarchical Reinforcement Learning (HRL)**. Just as humans don't need to reason about every keystroke when writing a sentence, HRL aims to allow agents to operate using high-level abstractions over actions, called **options**, **skills**, or **sub-policies**.
By integrating temporal abstraction into reinforcement learning, we enable agents to:

- Plan and act over multiple time scales
- Reuse skills across tasks
- Learn more efficiently in complex environments

### Why Is Reinforcement Learning Hard?
Reinforcement Learning offers a compelling framework for training agents to interact with environments through trial and error. And yet, despite its success in controlled domains like board games and synthetic benchmarks, RL remains strikingly limited when applied to complex, real-world problems.

To illustrate this, consider a famous developmental psychology experiment by Warneken and Tomasello (full video [here](https://www.youtube.com/watch?v=Z-eU5xZW7cU)). In this study, an 18-month-old child watches an adult drop a clothespin while attempting to hang clothes. Remarkably, the child quickly grasps the goal and hands the object back—without having seen the task before.

<center> 
<img src="\assets\images\course_notes\HRL\GIF1.gif"
    style="float: center; margin-right: 10px;" 
    /> 
    </center>

What’s astonishing is that the child understands:
The goal of the adult

- The constraints of the environment
- The appropriate actions to take
- The plan needed to resolve the issue

At one point, the child even looks at the adult’s hands to infer how the task will conclude. This is a level of reasoning—incorporating physics, social cues, and foresight—that current RL agents struggle to achieve.

### The Cracks in the Foundation of Flat RL
Despite decades of progress, standard RL—or “flat” RL—struggles to scale. Here's a breakdown of some of its most pressing weaknesses:

🚫 **Sample Inefficiency**

RL methods typically require an enormous number of interactions with the environment to learn useful policies. Since agents start from scratch, they must rediscover knowledge in every new task—even if the new task is nearly identical to previous ones.

*How HRL helps:* Subtasks and high-level skills can be reused across tasks, improving transfer learning and overall efficiency.

🌌 **Scaling to Large Spaces**

In large action or state spaces, RL becomes computationally intractable. This is often referred to as the curse of dimensionality.

*How HRL helps:* By decomposing large tasks into smaller sub-tasks, HRL makes the learning problem more tractable and scalable.

🎯 **Poor Generalization**

Even when agents master a task, they often fail to generalize to variations of that task. Trained policies tend to be brittle and overly specialized.

*How HRL helps:* With well-defined subgoals and abstractions, HRL encourages agents to learn more modular and generalizable policies.

🧠 **Lack of Abstraction**
Flat RL treats the state-action space as one undifferentiated mass. This results in long decision sequences and delayed rewards, making credit assignment extremely difficult.

*How HRL helps:* Temporal and state abstraction allows RL agents to operate over higher-level units of behavior, simplifying planning and reasoning.

!!! danger

    ⚠️ Without hierarchy, an RL agent must learn policies equivalent to controlling every motor neuron for slicing a carrot. With hierarchy, it can simply call the "slice carrot" skill.


### Flat RL Is a Long Road with Weak Signals
In flat RL, the agent must traverse long sequences of low-level actions to accomplish a goal. This has two consequences:

- **Reward delay**: Rewards are sparse and only arrive after long chains of behavior, making them hard to assign correctly.

- **Learning cost**: The longer the effective planning horizon, the higher the computational and sample cost for discovering good policies.

This makes standard RL deeply inefficient and often unworkable in real-world scenarios.

### Hierarchy Is Not a New Idea
Interestingly, the idea of hierarchy has long been known to improve planning and decision-making. In the 1970s, techniques like:

- **Hierarchical Task Networks (HTNs)**
- **Macro-actions**
- **State abstraction methods**

...were already shown to offer exponential speedups in planning. More recent lines of research have introduced tools like:

- **Subgoal discovery**
- **Intrinsic motivation**
- **Artificial curiosity**

Yet, despite this wealth of theory and intuition, the integration of hierarchical ideas into modern, effective RL algorithms is still an ongoing research challenge.

## Foundations of Hierarchical Reinforcement Learning (HRL)

As we’ve seen, standard reinforcement learning suffers from critical limitations when faced with long time horizons, sparse rewards, and large state/action spaces. Hierarchical Reinforcement Learning (HRL) emerges as a powerful paradigm to address these challenges by allowing agents to reason and act at multiple levels of temporal abstraction.

In essence, HRL enables agents to not just choose atomic actions, like "move left" or "pick up object," but also higher-level macro-actions such as "navigate to room" or "open the door," each of which may span multiple primitive actions.

### A Shift in Perspective: From RL to HRL

Before we dive into the mathematical formalism, it's worth revisiting the core problem that both RL and HRL are trying to solve: the **Markov Decision Process (MDP)**. In flat RL, we model the environment as an MDP.
However, to support **temporally extended actions**, HRL builds upon a generalized version of MDPs known as the **Semi-Markov Decision Process (SMDP)**.

### The Semi-Markov Decision Process (SMDP)

In an SMDP, actions can span variable amounts of time. Thus, instead of modeling transitions as:

$$
P(s' \mid s, a)
$$

We model them as:

$$
P(s', \tau \mid s, a)
$$

Where $\tau$ is the duration (i.e., the number of steps) the action $a$ takes to complete.

This allows us to define **macro-actions**, also called **options**, that consist of a policy, a termination condition, and an initiation set. These options serve as building blocks for higher-level decision-making.


### Hierarchical Learning Dynamics

Let’s break down the dynamics of hierarchical control using the following elements:
<center>

| Symbol          | Meaning                                            |
| --------------- | -------------------------------------------------- |
| $a$           | Primitive action                                   |
| $\sigma$      | Subroutine or macro-action                         |
| $\pi$         | High-level (meta) policy                           |
| $\pi_{\sigma}$ | Low-level policy used within subroutine $\sigma$ |
| $V$           | Value function over states under $\pi$           |
| $V_a$        | Value function over primitive actions              |

</center>

<figure style="text-align: center;">
  <img src="\assets\images\course_notes\HRL\HRL_Dynamics.png" 
       style="margin-right: 10px;" />
  <figcaption> From <a href="https://www.princeton.edu/~yael/Publications/RibasFernandesSolwayEtAl2011.pdf" 
    style="color: #0066cc; text-decoration: none;"> A Neural Signature of f Hierarchical Reinforcement Learning. </a></figcaption>
</figure>
The agent alternates between macro-actions and primitive actions, each governed by policies at different levels of abstraction. A hierarchical policy $\pi$ selects a macro-action $\sigma$, which is executed by a lower-level policy $\pi_{\sigma}$ that produces primitive actions until the macro-action terminates.

### Why Temporal Abstraction Works

By allowing the agent to plan over macro-actions:

* The effective planning horizon shrinks, since one macro-action may handle dozens of time steps.
* The value function can be updated less frequently, reducing sample complexity.
* High-level reasoning becomes more human-like, as the agent focuses on *what* to do, while lower levels handle *how* to do it.

Great! Let’s now dive into the **Options Framework** and **MAXQ**, two foundational approaches in Hierarchical Reinforcement Learning (HRL). This post builds directly on our previous introduction to HRL and SMDPs.

## The Options Framework and MAXQ

### The Options Framework

A central idea in HRL is that agents can make decisions not just at the level of primitive actions, but at a **higher level of abstraction** using *options*—temporally extended courses of action.

This idea is formally encapsulated in the **Options Framework** [Sutton et al., 1999](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf), which extends the classical MDP by including **macro-actions**, known as **options**.

#### What is an Option?

An option $\omega$ is a tuple:

$$
\omega = \left( \mathcal{I}_\omega, \pi_\omega, \beta_\omega \right)
$$

Where:

* $\mathcal{I}_{\omega} \subseteq \mathcal{S}$: **Initiation set** – the states in which the option can be initiated.
* $\pi_{\omega}: \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$: **Intra-option policy** – the low-level policy followed while the option is active.
* $\beta_{\omega}: \mathcal{S} \rightarrow [0,1]$: **Termination condition** – gives the probability that the option will terminate in a state.


#### From MDP to Options over MDP

The **Options Framework** operates on a discrete-time MDP but introduces a layer of macro-actions that overlay this structure. The diagram below shows this idea:

<figure style="text-align: center;">
  <img src="\assets\images\course_notes\HRL\options.png" 
       style="margin-right: 10px;" />
  <figcaption> Options Framework over MDP. <a href="https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf" 
    style="color: #0066cc; text-decoration: none;"> Sutton et al., 1999. </a></figcaption>
</figure>


* **Top (MDP)**: Classic RL, where each state transition is governed by a primitive action at every time step. Time is discrete and discounts are homogeneous.
* **Middle (SMDP)**: Time is modeled as a sequence of discrete *events* rather than ticks. Actions (or options) can span variable durations. Discounting depends on the interval.
* **Bottom (Options over MDP)**: Options add a layer over the MDP—each blue arrow represents a high-level option that consists of multiple primitive actions. This creates *overlaid discrete events*, governed by the option’s termination.

In this framework, the agent chooses an **option**, which internally selects **primitive actions** using its own policy until it terminates, at which point the high-level controller chooses a new option.

#### Why Options Matter

Options provide:

* **Temporal abstraction**: Act for multiple time steps in one decision.
* **Transferability**: Reusable subpolicies for related tasks.
* **Reduced planning complexity**: Fewer decisions at the high level.

In this setup, we can still apply value iteration or policy gradient methods, treating options as actions in an SMDP.


### The MAXQ Framework

While the Options Framework defines what macro-actions are, **MAXQ** provides a structure to *learn* them hierarchically.

MAXQ [[Dietterich, 2000]](http://matt.colorado.edu/teaching/RL/readings/dietterich%201998%20ICML%20maxQ.pdf) is both a **decomposition** of the value function and a **hierarchical policy** representation.


#### MAXQ Decomposition

In MAXQ, the value function is decomposed into a **hierarchy of subtasks**, each with its own value function.

Each subtask $i$ has:

* A **completion function** \$C\_i(s,a)\$: expected reward for completing task \$i\$ after executing subtask \$a\$ in state \$s\$.
* A **value function** \$V\_i(s)\$: expected total reward to complete task \$i\$ from state \$s\$.

The total value of a task is:

$$
Q_i(s,a) = V_a(s) + C_i(s,a)
$$

Where:

* $V_a(s)$ is the value of completing subtask $a$ starting from $s$
* $C_i(s,a)$ is the expected cumulative reward for finishing task $i$ once $a$ is done

This decomposition makes the credit assignment problem easier by breaking it into manageable, local learning problems.


#### How MAXQ Works Intuitively

MAXQ builds a **task graph**:

* Root task: the full problem (e.g., "make beef bourguignon")
* Internal tasks: subtasks (e.g., "cut carrots", "boil sauce")
* Leaf tasks: primitive actions (e.g., "move hand", "grab knife")

Each node in the graph is associated with its own policy and value function, learned independently but linked through completion functions.


<figure style="text-align: center;">
  <img src="\assets\images\course_notes\HRL\MAXQ.png" 
       style="margin-right: 10px;" />
  <figcaption> Example of a MAXQ hierarchy.</figcaption>
</figure>

## Large Language Models in Hierarchical Reinforcement Learning

In recent years, Large Language Models (LLMs) have emerged as powerful general-purpose tools for reasoning, planning, and knowledge retrieval. A growing body of work explores how these models can support HRL agents, particularly in solving complex, long-horizon tasks that are otherwise intractable for traditional RL methods.

Reinforcement Learning suffers from:

- Sparse and delayed rewards
- Poor generalization to new environments
- Inefficient exploration in large or complex state spaces

HRL helps alleviate some of these issues by introducing temporal abstraction, decomposing complex tasks into reusable sub-tasks. However, even HRL struggles with **exploration**, **scaling** (as the number of skills increases), and **knowledge transfer**.

This is where **LLMs** come in. These models encode rich, world-level priors and reasoning capabilities, which can be used to guide high-level decision-making.

### LLM-Guided High-Level Exploration
One promising recent framework by [Prakash et al.,](https://openreview.net/forum?id=Gv04zPxvCq) proposes using LLMs to guide high-level decision-making in HRL, without relying on them during deployment. Here's how it works:

- The agent has access to a finite library of low-level skills, each with a natural language description, e.g., "pick up red block".

- The high-level task is described in natural language, like "build a tower with red blocks".

- The agent receives observations from the environment, which are translated into a simple language trajectory $(l_{traj})$ of recent actions or states.


In this framework, the LLM is not responsible for generating actions directly. Instead, it answers a series of **binary questions** for each available skill:

> *Given the goal ($l_{goal\_inst}$) and recent trajectory ($l_{traj}$), should the agent execute skill $l_{skill_i}$?*

Each answer is **"yes"** or **"no"**, converted to **1** or **0**, respectively.

These outputs are collected into a binary vector:

$$
F_\text{LLM} = [f_1, f_2, \ldots, f_n]
$$

Then, a **log-softmax** is applied to compute a prior distribution over skills:

$$
p_{\text{CS}} = \log \left( \text{softmax}(F_{\text{LLM}}) \right)
$$

### Biasing High-Level Exploration

This **common-sense prior** $p_{\text{CS}}$ is used to bias the skill-selection process in the high-level policy. The actual logits used for sampling an action from the policy become:

$$
\text{logits} = \text{PolicyHead}(s) + \lambda \cdot p_{\text{CS}}
$$

Where:

* $s$ is the current state
* $\lambda$ is a weight that starts at **1** and is **annealed to 0** during training

By training the agent this way, the model **gradually reduces its reliance on the LLM**, enabling **efficient learning** during training while remaining **LLM-free during deployment**.

# Goal-Conditioned Reinforcement Learning (GCRL)

Imagine you're training a robotic arm. Some days, you want it to pick up a cup. Other days, it needs to press a button or move an object to a particular location. These tasks may share dynamics (e.g. how the arm moves), but differ in what we want the agent to accomplish. Wouldn’t it be inefficient to train a separate policy for each task from scratch?

This is where **Goal-Conditioned Reinforcement Learning (GCRL)** shines.

## Introduction
### Motivation: An Illustrative Example

Let’s consider a robot named **Robo**, operating in a 2D grid world:

* **State**: Robo's position on the grid, e.g. $(x, y)$
* **Action**: Move up/down/left/right
* **Goal**: A specific position on the grid, e.g. $g = (x_g, y_g)$

Each episode, Robo is assigned a different target location to reach. If we trained Robo with a classic RL setup, we'd need a *separate policy* for each possible goal — very inefficient!

Instead, we can train **a single policy** that **takes the goal as input** and **learns to generalize across goals**. This is the central idea of **goal-conditioned RL**.

<figure style="text-align: center;">
  <img src="\assets\images\course_notes\HRL\Robo.png" 
       style="margin-right: 10px;" />
  <figcaption>  Typical representations of goals in GCRL. <a href="https://arxiv.org/pdf/2201.08299" 
    style="color: #0066cc; text-decoration: none;"> Liu et al., 2022. </a></figcaption>
</figure>

<!-- 
### Formal Setup

In GCRL, we reformulate the Markov Decision Process (MDP) to include goals:

#### Standard MDP:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma)
$$

#### Goal-Conditioned MDP:

We introduce a **goal space** $\mathcal{G}$ and a goal distribution $p(g)$:

$$
\mathcal{M}_g = (\mathcal{S}, \mathcal{A}, P, r_g, \gamma, p(g))
$$

Where:

* $g \in \mathcal{G}$ is the goal,
* $r_g(s, a)$ is the **goal-conditioned reward function**, e.g.

    $$r_g(s) = 
    \begin{cases}
    1 & \text{if } s \approx g \\
    0 & \text{otherwise}
    \end{cases}
    $$

* Policies become **goal-conditioned policies**:

    $$\pi(a \mid s, g)$$

This policy can *generalize* to new goals at test time if trained properly. -->


### Comparison with Hierarchical Reinforcement Learning (HRL)

GCRL and HRL are two distinct but sometimes overlapping approaches to tackling complex tasks in reinforcement learning. Here's a breakdown of their key differences:   

#### **Goal-Conditioned Reinforcement Learning (GCRL)**:

- **Focus**: Learning policies that are conditioned on achieving specific goals. The agent's policy π(s,g) takes both the current state (s) and a desired goal (g) as input and outputs an action.   
- **Task Specification**: Tasks are defined by desired goals, which can be states, images, or other representations of a target outcome. The reward function is often based on whether the agent has achieved the specified goal.   
- **Learning Objective**: To learn a single, versatile policy that can achieve a variety of goals within the same environment. This allows for generalization across different tasks.   
- **Key Idea**: Instead of learning separate policies for each task, GCRL aims to learn a universal policy that understands how to reach different outcomes.   
- **Example**: Training a robot arm to reach various target positions in its workspace using a single policy that takes the current joint angles and the desired target position as input.


#### **Hierarchical Reinforcement Learning (HRL)**:

- **Focus**: Decomposing complex tasks into a hierarchy of sub-tasks or levels of control. This involves learning policies at different levels of abstraction.   
- **Task Decomposition**: The overall task is broken down into smaller, more manageable sub-goals or abstract actions (often called "options" or "skills").   
- **Learning Objective**: To learn a high-level policy that selects sub-goals or sub-tasks and lower-level policies that execute these sub-goals using primitive actions.
- **Key Idea**: By introducing a hierarchy, HRL aims to improve exploration, credit assignment, and learning efficiency in tasks with long horizons or sparse rewards. The hierarchy provides temporal abstraction, allowing the agent to reason at different time scales.   
- **Example**: Training an autonomous vehicle. A high-level policy might decide on a sequence of waypoints (sub-goals), while a lower-level policy controls the steering, acceleration, and braking to reach these waypoints.


<center>

| Feature                 | Goal-Conditioned RL                       | Hierarchical RL                                 |
| ----------------------- | ----------------------------------------- | ----------------------------------------------- |
| **Policy Input**        | Conditioned on a goal $g$               | Uses high-level and low-level policies          |
| **Structure**           | Flat policy, goal is part of input        | Explicit hierarchy (manager and worker)         |
| **Goal Semantics**      | External or internally sampled            | Subgoals usually generated by high-level policy |
| **Learning Complexity** | Easier to implement; more straightforward | Complex credit assignment across levels         |
| **Reuse across tasks**  | High — generalizes to new goals           | Medium — depends on subgoal definitions         |

</center>

It's important to note that GCRL and HRL are not mutually exclusive. In fact, they can be combined. For instance, in a hierarchical system, the lower-level policies responsible for achieving sub-goals can be goal-conditioned. The high-level policy sets the sub-goals, and the lower-level policies learn to reach these sub-goals from different states. This combination can leverage the benefits of both approaches, leading to more efficient and versatile learning in complex, multi-goal environments. GCRL can *serve as a component* in hierarchical RL. The high-level policy in HRL can generate **goals** for a GCRL-style low-level policy. This hybrid approach is common in **goal-conditioned HRL frameworks**, such as [HIRO](https://arxiv.org/pdf/1805.08296) and [FeUdal Networks](https://arxiv.org/abs/1703.01161).

In essence, GCRL is about learning *what* to achieve, while HRL is about learning *how* to achieve it by breaking down the process.

## Mathematical Formulation

### Standard RL as MDP

First, let's recall how standard RL problems are modeled as Markov Decision Processes (MDPs):

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, r, \gamma, \rho_0 \rangle
$$

Where:

- $\mathcal{S}$: State space
- $\mathcal{A}$: Action space
- $\mathcal{T}$: Transition function
- $r$: Reward function
- $\gamma$: Discount factor
- $\rho_0$: Initial state distribution

The objective is to learn a policy $\pi: \mathcal{S} \rightarrow \mathcal{A}$ that maximizes:

$$
J(\pi) = \mathbb{E}_{\substack{a_t \sim \pi(\cdot|s_t) \\ s_{t+1} \sim \mathcal{T}(\cdot|s_t,a_t)}} \left[\sum_t \gamma^t r(s_t, a_t)\right]
$$

### Goal-Augmented MDP (GA-MDP)

GCRL extends this formulation by augmenting the MDP with goal information:

$$
\mathcal{M}_{GA} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, r, \gamma, \rho_0, \mathcal{G}, p_g, \phi \rangle
$$

New components:

- $\mathcal{G}$: Goal space
- $p_g$: Goal distribution
- $\phi: \mathcal{S} \rightarrow \mathcal{G}$: Mapping from states to goals

The reward function now depends on the goal: $r: \mathcal{S} \times \mathcal{A} \times \mathcal{G} \rightarrow \mathbb{R}$

The objective becomes:

$$
J(\pi) = \mathbb{E}_{\substack{a_t \sim \pi(\cdot|s_t,g) \\ g \sim p_g \\ s_{t+1} \sim \mathcal{T}(\cdot|s_t,a_t)}} \left[\sum_t \gamma^t r(s_t, a_t, g)\right]
$$

As depicted below, the
agent in GCRL is required to either learn to master multiple tasks simultaneously, or decompose the long-term and hardreaching goals into intermediate sub-goals while learning to reach them with a unified policy, or achieve both of them. For example, to navigate a robot or manipulate an object, the
goals are typically defined as the target positions to reach. To tackle such a challenge, the formulation of GCRL augments the MDP with an extra tuple$(\mathcal{G}, p_g, \phi)$ as a goal-augmented MDP (GA-MDP). 

<figure style="text-align: center;">
  <img src="\assets\images\course_notes\HRL\GAMDP.png" 
       style="margin-right: 10px;" />
  <figcaption>  The mission of policy in complex RL problems. a) Learn
    to achieve multiple tasks with one single policy; b) Decompose the
    long-term, hard-reaching goals into easily obtained sub-goals. <a href="https://arxiv.org/pdf/2201.08299" 
    style="color: #0066cc; text-decoration: none;"> Liu et al., 2022. </a></figcaption>
</figure>

### Key Definitions

1. **Desired Goal**: The target task provided by the environment or generated by the agent
2. **Achieved Goal**: The goal actually reached by the agent at the current state
3. **Behavioral Goal**: The goal used for sampling in a rollout episode

## Goal Representations

Goals can be represented in various forms depending on the task:

### 1. Feature Vectors
Most common representation, especially in robotics:

- Target positions (FetchPush, FetchPickAndPlace)
- Target velocities (HalfCheetah)
- Object orientations (HandManipulateBlock)

### 2. Image Goals
Used for more complex visual tasks:

- Target image of a scene
- Game screenshots (Atari, Minecraft)
- Real-world camera images

An encoder model, e.g., VAE can be utilized to encode image goals

### 3. Language Goals
Natural language instructions:

- "Push the blue block to the corner"
- "Run at 5 m/s"
- "Open the drawer"

An embedding model can be utilized to encode linguistic goals.

## Core Challenges in GCRL

### 1. Sparse Rewards
The typical reward function is binary:

$$
r_g(s_t, a_t, g) = \mathbb{1}(\|\phi(s_{t+1}) - g\| \leq \epsilon)
$$

This leads to:

- No learning signal for most of the state space
- Difficult exploration problems

**Solutions**:

- Reward shaping: Provide denser rewards using domain knowledge (e.g., negative distance to goal),

$$
\tilde{r}_g(s_t, a_t, g) = -d(\phi(s_{t+1}), g).
$$

May create local optima if poorly designed.

- Hindsight Experience Replay (HER) method: Relabel failed trajectories as successful for the goals actually achieved.The key benefit is that it turns failures into useful training data.

- Intrinsic motivation: Add exploration bonuses (e.g., curiosity, novelty).

### 2. Multi-Task Learning
The agent must generalize across different goals:

$$
L(f) = \mathbb{E}_{p_g}[L_i(f)]
$$

Challenges include:

- Catastrophic forgetting
- Negative transfer between tasks
- Imbalanced goal distribution

**Solutions**:

- Parameter Isolation: Use separate network components for different goals, e.g., Modular architectures or mixture-of-experts.

- Gradient Modulation: Balance gradient updates across tasks, e.g., [PCGrad](https://arxiv.org/abs/2001.06782), [GradNorm](https://arxiv.org/abs/1711.02257).

Fortunately, different goals in the same environment always possess similar properties (e.g., the dimension, the structure, and even the way to compute rewards are the same). This makes it possible for the goal-conditioned policy to share parameters across various tasks (i.e., goals $g$) to derive
a generalized solution in GCRL.

## Algorithmic Approaches in Goal-Conditioned RL: A Deep Dive

GCRL introduces several key algorithmic approaches to tackle the challenges of learning policies that generalize across multiple goals. Below, we explore three fundamental methods in detail: **Universal Value Function Approximators (UVFA)**, **Hindsight Experience Replay (HER)**, and **Goal-Conditioned Supervised Learning (GCSL)**.

### **Universal Value Function Approximators (UVFA)**

[UVFA](https://proceedings.mlr.press/v37/schaul15.pdf) extends traditional value functions by conditioning them on goals, allowing a single policy to generalize across multiple tasks. Instead of learning separate value functions for each goal, UVFA learns a unified function $V(s, g)$ or $Q(s, a, g)$ that estimates expected returns for any state-goal pair.

#### **Key Components**

- **Goal-Augmented Input**: The value function takes both the state $s$ and goal $g$ as inputs.

- **Generalization Across Goals**: Learns a shared representation for different goals, enabling transfer learning.

- **Multi-Task Optimization**: Optimizes a single objective over a distribution of goals $p(g)$.

#### **Mathematical Formulation**
The **goal-conditioned value function** is defined as:

$$
V^\pi(s_t, g) = \mathbb{E} \left[ \sum_{k=t}^\infty \gamma^{k-t} r(s_k, a_k, g) \Big| s_t, g \right]
$$

Similarly, the **goal-conditioned Q-function** is:

$$
Q^\pi(s_t, a_t, g) = \mathbb{E} \left[ r(s_t, a_t, g) + \gamma V^\pi(s_{t+1}, g) \right]
$$

#### **Implementation Details**

- **Architecture**: Typically a neural network with concatenated state-goal inputs.
- **Training**: Uses off-policy TD-learning (e.g., DQN, DDPG) with goal-relabeling.
- **Challenge**: Requires careful normalization of goal spaces to ensure stable learning.

### **2. Hindsight Experience Replay (HER)**

[HER](https://arxiv.org/abs/1707.01495) addresses **sparse rewards** by relabeling failed trajectories as if they were successful for different goals. If an agent fails to reach goal $g$ but reaches $g'$, HER stores the experience as a success for $g'$.

#### **Key Components**

- **Goal Relabeling**: Replaces the original goal in failed episodes with achieved goals.

- **Multi-Goal Learning**: Enables learning from failures by treating them as successes for alternative goals.

- **Compatibility**: Works with any off-policy RL algorithm (e.g., DDPG, SAC).

!!! note "**Algorithm Steps** "

    1. **Collect Trajectory**: Execute policy $\pi(a|s, g)$ for goal $g$.
    2. **Store Original Data**: Save transitions $(s_t, a_t, r_t, s_{t+1}, g)$.
    3. **Relabel Goals**: For each transition, substitute $g$ with future achieved states $g' = \phi(s_{t+k})$.
    4. **Recompute Rewards**: Update $r_t$ based on the new goal $g'$.

#### **Mathematical Formulation**
Given a trajectory $\tau = (s_0, a_0, ..., s_T)$, HER generates new transitions:

$$
(s_t, a_t, r(s_t, a_t, g'), s_{t+1}, g')
$$

where $g' = \phi(s_T)$ (final state) or other intermediate states.

#### **Why It Works**

- **Data Efficiency**: Converts failures into useful training samples.
- **Exploration Boost**: Encourages discovering diverse behaviors.
- **Scalability**: Works with high-dimensional goal spaces (e.g., images).


### **3. Goal-Conditioned Supervised Learning (GCSL)**
[GCSL](https://arxiv.org/abs/1912.06088) simplifies GCRL by framing it as a **supervised learning problem**. Instead of learning a value function, it directly trains a policy $\pi(a|s, g)$ to predict actions that lead to the goal.

#### **Key Components**

- **Behavioral Cloning**: Mimics actions that led to successful goal-reaching.

- **No Value Function**: Avoids complex RL optimization.

- **Works Offline**: Can learn from pre-collected datasets (no environment interaction needed).

#### **Mathematical Formulation**
The objective is to maximize the likelihood of actions that reached the goal:

$$
J(\pi) = \mathbb{E}_{(s, a, g) \sim \mathcal{D}} \left[ \log \pi(a|s, g) \right]
$$

where $\mathcal{D}$ contains transitions where action $a$ led to goal $g$ from state $s$.

!!! note "**Algorithm Steps** "

    1. **Collect Dataset**: Gather $(s, a, g)$ tuples from successful trajectories.

    2. **Supervised Training**: Train $\pi(a|s, g)$ via maximum likelihood.

    3. **Optional Fine-Tuning**: Combine with RL for improved performance.

#### **Advantages**

- **Simplicity**: Easier to implement than RL-based methods.
- **Stability**: Avoids issues like exploration and credit assignment.
- **Scalability**: Works well with large offline datasets.

#### **Limitations**

- **Suboptimal Data**: Performance depends on the quality of the dataset.
- **No Exploration**: Requires additional mechanisms for discovering new behaviors.

<center>

| **Method**       | **Key Idea** | **Strengths** | **Weaknesses** |
|------------------|-------------|---------------|----------------|
| **UVFA** | Learns a single value function for all goals. | Generalizes across tasks. | Requires careful goal normalization. |
| **HER** | Relabels failed trajectories as successes. | Highly data-efficient. | Needs off-policy RL for best results. |
| **GCSL** | Supervised learning over goal-reaching actions. | Simple and stable. | Limited by dataset quality. |

</center>