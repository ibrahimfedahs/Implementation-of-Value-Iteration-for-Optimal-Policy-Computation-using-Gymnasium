# Implementation of Value Iteration for Optimal Policy Computation using Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement

Develop a Value Iteration agent that finds the optimal policy for navigating a custom FrozenLake environment. The objective is to maximize the expected cumulative reward while safely reaching the goal and avoiding holes.




## Software Requirements
- Python 3.x
- Gymnasium
- NumPy
- Matplotlib
- Jupyter Notebook / Google Colab / VS Code

---

## Environment Description

The FrozenLake environment is a grid-world problem where the agent starts at the **Start (S)** state and aims to reach the **Goal (G)** while avoiding **Holes (H)**. Safe tiles are represented by **F**. The environment is stochastic (`is_slippery=True`), meaning the agent may not always move in the intended direction.

Custom Environment:
```text
S F F F
F F F F
F H F H
F F F G
```
---


## MDP Representation

- **States (S):** 16 grid cells (4 × 4)
- **Actions (A):**
  - 0 → Left
  - 1 → Down
  - 2 → Right
  - 3 → Up
- **Transition Probability (P):** Defined by the FrozenLake environment.
- **Reward (R):**
  - Goal = 1
  - All other states = 0
- **Discount Factor (γ):** 0.99

---

## Theory

Value Iteration is a Dynamic Programming algorithm used to solve Markov Decision Processes. It repeatedly updates the value of every state using the Bellman Optimality Equation until the values converge. Once convergence is achieved, the optimal policy is obtained by selecting the action that maximizes the expected future reward from each state.

---

## Algorithm

1. Initialize all state values to zero.
2. Repeat until convergence:
   - Compute the value of every action for each state.
   - Update the state value with the maximum action value.
3. Stop when the maximum value change is less than the threshold.
4. Extract the optimal policy by selecting the action with the highest value.
5. Display the optimal value function and policy.

---
## Python Program


```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env_desc = [
    "SFFF",
    "FFFF",
    "FHFH",
    "FFFG"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
env = env.unwrapped

# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    iteration = 0

    while True:
        delta = 0

        for s in range(n_states):
            action_values = np.zeros(n_actions)

            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            best_value = np.max(action_values)
            delta = max(delta, abs(best_value - V[s]))
            V[s] = best_value

        iteration += 1

        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        action_values = np.zeros(n_actions)

        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (
                    reward + gamma * V[next_state] * (not done)
                )

        policy[s] = np.argmax(action_values)

    return V, policy, iteration

# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------

V, policy, iteration = value_iteration(env)

# -------------------------------------------------
# Display Output
# -------------------------------------------------

print("Name: Marxin Lijo M")
print("Register Number: 212223240085")
print("Value Iteration Completed")
print("Number of Iterations:", iteration)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))

action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

plt.figure(figsize=(6,6))
plt.imshow(V.reshape(4,4), cmap="viridis")
plt.colorbar(label="State Value")
plt.title("Optimal Value Function")
plt.show()

env.close()
```

---


## Output

<img width="845" height="296" alt="image" src="https://github.com/user-attachments/assets/73dd2a50-9600-45db-b57f-fd6bacd35987" />


---

## Result
```text
The Value Iteration algorithm was successfully implemented on the custom FrozenLake environment. The optimal state-value function and policy were computed using the Bellman Optimality Equation, enabling the agent to choose the best action from every state.

```
---

## Inference
```text
The Value Iteration algorithm efficiently computes the optimal policy by repeatedly updating state values until convergence. States closer to the goal obtain higher values, while hole states have lower values. The resulting policy maximizes the expected cumulative reward and provides the optimal path to the goal in the FrozenLake environment.


```
---

