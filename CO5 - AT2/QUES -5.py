import numpy as np
import matplotlib.pyplot as plt

rows = 5
cols = 5

gamma = 0.9

actions = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

symbols = ["↑", "↓", "←", "→"]

obstacles = [(1, 1), (1, 3), (3, 1), (3, 3)]

goal = (0, 4)

value = np.zeros((rows, cols))

policy = np.random.randint(0, 4, (rows, cols))

for obstacle in obstacles:
    policy[obstacle] = -1

policy[goal] = -1

while True:

    # Policy Evaluation
    for _ in range(100):

        new_value = value.copy()

        for r in range(rows):
            for c in range(cols):

                if (r, c) in obstacles or (r, c) == goal:
                    continue

                action = policy[r, c]

                dr, dc = actions[action]

                nr = r + dr
                nc = c + dc

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    nr, nc = r, c

                if (nr, nc) in obstacles:
                    nr, nc = r, c

                reward = 100 if (nr, nc) == goal else -1

                new_value[r, c] = reward + gamma * value[nr, nc]

        value = new_value

    # Policy Improvement
    policy_stable = True

    for r in range(rows):
        for c in range(cols):

            if (r, c) in obstacles or (r, c) == goal:
                continue

            old_action = policy[r, c]

            action_values = []

            for action in range(4):

                dr, dc = actions[action]

                nr = r + dr
                nc = c + dc

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    nr, nc = r, c

                if (nr, nc) in obstacles:
                    nr, nc = r, c

                reward = 100 if (nr, nc) == goal else -1

                action_value = reward + gamma * value[nr, nc]

                action_values.append(action_value)

            best_action = np.argmax(action_values)

            policy[r, c] = best_action

            if old_action != best_action:
                policy_stable = False

    if policy_stable:
        break

print("Optimal Policy")
print("----------------")

for r in range(rows):
    row = []

    for c in range(cols):

        if (r, c) in obstacles:
            row.append("X")

        elif (r, c) == goal:
            row.append("G")

        else:
            row.append(symbols[policy[r, c]])

    print(" ".join(row))

print("\nState Values")
print(np.round(value, 2))

plt.imshow(value)

plt.colorbar(label="State Value")

plt.title("Policy Iteration - Drone Navigation")

plt.xlabel("Column")
plt.ylabel("Row")

plt.show()
