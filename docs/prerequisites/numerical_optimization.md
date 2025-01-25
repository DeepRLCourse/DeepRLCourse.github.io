# Numerical Optimization

## Convex Optimization
### Convex Sets and Functions
- Definition and properties of convex sets
- Convex functions and their properties (e.g., Jensen’s inequality)
- Examples of convex functions in RL (e.g., quadratic functions, log-sum-exp)

### Convex Optimization Problems
- Formulation of convex optimization problems
- Global vs. local optima
- Examples in RL (e.g., policy evaluation, value function approximation)

### Lagrange Multipliers and Duality
- Lagrangian function and dual problem
- Strong duality and Slater’s condition
- Applications in RL (e.g., constrained policy optimization)

---

## Gradient-Based Methods
### Gradient Descent
- Basic algorithm and intuition
- Stochastic gradient descent (SGD) and mini-batch gradient descent
- Convergence properties and challenges (e.g., learning rate selection)

### Newton’s Method
- Basic algorithm and intuition
- Hessian matrix and second-order optimization
- Applications in RL (e.g., natural policy gradient)

### Quasi-Newton Methods
- BFGS and L-BFGS algorithms
- Advantages over Newton’s method (e.g., no need for Hessian computation)
- Applications in RL for large-scale optimization

---

## Constrained Optimization
### Karush-Kuhn-Tucker (KKT) Conditions
- Statement and intuition
- Necessary and sufficient conditions for optimality
- Applications in RL (e.g., constrained MDPs, safe RL)

### Projected Gradient Descent
- Basic algorithm and intuition
- Applications in RL for constrained policy updates

### Penalty and Barrier Methods
- Quadratic penalty method
- Interior-point (barrier) methods
- Applications in RL for handling constraints

---

## Trust Region Optimization
### Trust Region Methods
- Basic idea and intuition
- Trust region subproblem and its solution
- Comparison with line search methods

### Trust Region Policy Optimization (TRPO)
- KL divergence constraint and surrogate objective
- Algorithm and implementation details
- Applications in RL for stable policy updates

### Proximal Policy Optimization (PPO)
- Clipped surrogate objective as an approximation to TRPO
- Advantages over TRPO (e.g., simplicity, scalability)
- Applications in RL for efficient policy optimization

