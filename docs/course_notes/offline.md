# Week 8:  Offline RL


In modern AI, we’ve seen the power of large-scale pretraining in supervised learning. For example, models trained on **ImageNet** have become the backbone of many computer vision applications, enabling easy fine-tuning for downstream tasks with limited data. These are known as **foundation models**.

This success leads to an exciting question:

> can we replicate this paradigm in reinforcement learning (RL)?

Imagine a general-purpose RL model trained on a large and diverse set of offline experience logs from many tasks. This model could then be **fine-tuned or adapted** to new environments with minimal or no online interaction. Such an approach would dramatically reduce the cost and risk associated with online RL, and enable more scalable and accessible applications of RL in the real world.

This vision is the core motivation behind **Offline Reinforcement Learning**.

---


## What is Offline Reinforcement Learning?

## 🔍 What is Offline RL?

Offline RL can be viewed as a **data-driven** approach to reinforcement learning, more akin to supervised learning in terms of setup. Given a static dataset of transitions:

\[
\mathcal{D} = \{(s_i, a_i, s'_i, r_i)\}
\]

the agent must learn the best policy it can **only from this data**. No new data can be collected. This requires the learning algorithm to build an understanding of the environment dynamics and reward structure, entirely from offline data.

- We treat \(\mathcal{D}\) as the **training set**.
- The learned policy \(\pi(a|s)\) must be evaluated **when deployed**, not during training.
- State and action samples in \(\mathcal{D}\) follow the distribution:

\[
  s \sim d^{\pi_\beta}(s), \quad a \sim \pi_\beta(a|s)
\]

This setting is sometimes called:
- **Fully off-policy RL** (no new data)
- **Batch RL** (learning from a fixed dataset or "batch")
- **Offline RL** (term we use for clarity)

---
---
before delving into offline RL realm let's see why!

## Why Offline RL Matters

Offline RL is especially valuable when online interaction is :

- ⚠️ **Dangerous**: e.g., autonomous driving, healthcare
- 💸 **Expensive**: e.g., robotics, industrial control
- 🐢 **Slow**: e.g., scientific processes, education
- 🚫 **Impossible**: e.g., using historical logs, simulations no longer available

these reason are similar to those of inverse RL and Imitation learning.

Offline RL allows learning from existing data safely and efficiently—an essential step toward building **scalable RL systems**.

---


## Offline RL vs. Traditional RL

| Feature                 | Online RL                          | Offline RL                          |
|-------------------------|-------------------------------------|--------------------------------------|
| Data collection         | Requires live environment access   | Uses a static dataset                |
| Exploration             | Yes                                | No                                  |
| Training safety         | Risky                              | Safe                                |
| Reusability             | Low                                | High                                |
| Analogy to supervised ML| Train-from-scratch                 | Pretraining & fine-tuning           |

---

## Key Challenges in Offline RL




As we will see throghut this note despite its promise, Offline RL introduces several difficult challenges:

### 🛑 1. No Exploration

The most direct difficulty is that **exploration is impossible**. Since the policy cannot interact with the environment during training, it cannot:

- Discover new high-reward transitions
- Learn about parts of the state-action space that were not visited in the dataset

If the offline dataset does not contain examples from important regions of the environment, **the agent has no way to discover or learn about them**.

This challenge is important but generally assumed to be out of scope (i.e., we assume the dataset is “good enough”), so more emphasis is placed on the subtler, algorithmic difficulties.

---

### ❓ 2. Offline RL is About Counterfactual Reasoning

At its core, offline RL requires reasoning about **"what if"** scenarios:
> What would happen if the agent took actions **not seen** in the dataset?

Since we hope the learned policy will **outperform** the behavior that generated the dataset, it must **imagine and evaluate alternative action sequences**—even if those actions were never actually taken.

This is fundamentally a **counterfactual inference problem**. Unlike supervised learning (which assumes data is i.i.d. from the test distribution), offline RL must:

- Generalize to new policies
- Predict outcomes under a distribution **different** from the one observed

This makes offline RL **incompatible with many standard machine learning assumptions**, and challenges current learning methods.

---

### 🔄 3. Distributional Shift in Counterfactual Evaluation

When a policy \(\pi(a|s)\) learned offline differs from the behavior policy \(\pi_\beta(a|s)\), the new policy will visit **different states** and **take different actions** than those seen in the dataset.

This leads to **distributional shift**, which causes:

- Policy, value function, or model to operate on **out-of-distribution inputs**
- Inaccurate predictions due to lack of data support
- Potentially unbounded errors when maximizing expected return

> For example, a Q-function trained on dataset \(\mathcal{D}\) might be accurate near \(\pi_\beta\), but unreliable elsewhere. If \(\pi\) deviates from \(\pi_\beta\), value estimates can become invalid.

To combat this, many offline RL methods **constrain the learned policy** to stay close to the behavior policy, e.g., by bounding the KL-divergence between \(\pi\) and \(\pi_\beta\). This helps reduce extrapolation errors and improve reliability.

---

Specialized algorithms are needed to address these problems and ensure safe, stable training. Another reason why we need you to take this course seariously!

## 📊 3. Offline Evaluation and RL via Importance Sampling

One of the most fundamental steps in offline reinforcement learning is evaluating how good a new policy \(\pi\) is—**using only trajectories collected by a different policy** \(\pi_\beta\). This problem is known as **off-policy evaluation (OPE)**.

### 🎯 Goal:
Estimate the expected return \( J(\pi) \) of a policy \(\pi\), given only data sampled from a behavior policy \(\pi_\beta\).

---

## 🔁 3.1 Off-Policy Evaluation via Importance Sampling

A natural approach is to use **importance sampling (IS)** to reweight trajectories from \(\pi_\beta\) so they reflect what would happen under \(\pi\). The basic estimator is:

\[
J(\pi) = \mathbb{E}_{\tau \sim \pi_\beta} \left[ \left( \prod_{t=0}^H \frac{\pi(a_t | s_t)}{\pi_\beta(a_t | s_t)} \right) \sum_{t=0}^H \gamma^t r_t \right]
\]

This provides an **unbiased** estimator of return but suffers from **high variance**, especially for long horizons, due to the product of many importance weights.

---

### 🧮 Per-Decision IS

To reduce variance, we can apply **per-decision importance sampling**:

\[
J(\pi) = \mathbb{E}_{\tau \sim \pi_\beta} \left[ \sum_{t=0}^H \left( \prod_{i=0}^t \frac{\pi(a_i | s_i)}{\pi_\beta(a_i | s_i)} \right) \gamma^t r_t \right]
\]

This estimator has lower variance but is still sensitive to the divergence between \(\pi\) and \(\pi_\beta\).

---

### 📉 Weighted Importance Sampling

To stabilize training, the weights can be **normalized**:

\[
w'_i = \frac{w_i}{\sum_j w_j}
\]

This yields a **biased but lower-variance** estimator. These tradeoffs between bias and variance are a key consideration in offline RL.

---

### 🧠 Control Variates and Doubly Robust Estimators

If we have an estimate of the Q-function \(\hat{Q}^\pi(s,a)\), we can reduce variance further using **control variates**. The **doubly robust estimator** combines model-based predictions with importance sampling:

\[
J(\pi) = \frac{1}{n} \sum_{i=1}^n \sum_{t=0}^H \left( w_i^t (r_t^i - \hat{Q}^\pi(s_t^i, a_t^i)) + \mathbb{E}_{a \sim \pi}[\hat{Q}^\pi(s_t^i, a)] \right)
\]

- **Unbiased** if either the model is correct or IS weights are accurate
- **Low variance**, especially with good value function approximators

These estimators form the backbone of several high-confidence policy evaluation methods (e.g., MAGIC, Fitted Q Evaluation, WDR).

---

## ✅ Summary

| Method                        | Bias         | Variance      | Notes                                      |
|-----------------------------|--------------|---------------|--------------------------------------------|
| Vanilla IS                  | Unbiased     | Very high     | Exponential variance with horizon          |
| Per-Decision IS             | Unbiased     | Lower         | Still unstable with long horizons          |
| Weighted IS                 | Biased       | Lower         | Normalizes weights, better practical use   |
| Doubly Robust (DR)         | Unbiased*    | Low           | Best of both worlds with good Q estimates  |

These evaluation methods allow us to not only **assess policies offline**, but also **train policies** using policy gradients or selection mechanisms, as we'll see next.

---









