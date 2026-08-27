import numpy as np
import matplotlib.pyplot as plt

grid = [
    [0, 0, 0, 0, 10],
    [0, -1, 0, -1, 0],
    [0, 0, 0, 0, 0],
    [0, -1, 0, -1, 0],
    [0, 0, 0, 0, 0]
]

rows = len(grid)
cols = len(grid[0])

gamma = 0.9
threshold = 0.001

V = np.zeros((rows, cols))

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
symbols = ["↑", "↓", "←", "→"]

while True:
    new_V = V.copy()
    delta = 0

    for r in range(rows):
        for c in range(cols):

            if grid[r][c] == -1:
                continue

            if grid[r][c] == 10:
                continue

            values = []

            for dr, dc in actions:
                nr = r + dr
                nc = c + dc

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    values.append(-100)
                elif grid[nr][nc] == -1:
                    values.append(-100)
                else:
                    values.append(V[nr][nc])

            best_value = max(values)

            new_V[r][c] = -1 + gamma * best_value

            delta = max(delta, abs(new_V[r][c] - V[r][c]))

    V = new_V

    if delta < threshold:
        break

policy = []

for r in range(rows):
    row = []

    for c in range(cols):

        if grid[r][c] == -1:
            row.append("X")

        elif grid[r][c] == 10:
            row.append("G")

        else:
            values = []

            for dr, dc in actions:
                nr = r + dr
                nc = c + dc

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    values.append(-100)
                elif grid[nr][nc] == -1:
                    values.append(-100)
                else:
                    values.append(V[nr][nc])

            best_action = np.argmax(values)
            row.append(symbols[best_action])

    policy.append(row)

print("Optimal Value Function:")
print(np.round(V, 2))

print("\nOptimal Policy:")

for row in policy:
    print(" ".join(row))

plt.imshow(V)

plt.colorbar(label="State Value")
plt.title("Value Iteration - Robot Path Planning")
plt.xlabel("Column")
plt.ylabel("Row")

plt.show()
