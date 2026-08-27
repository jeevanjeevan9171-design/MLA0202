import random
import matplotlib.pyplot as plt

movies = ["Movie A", "Movie B", "Movie C", "Movie D"]

epsilon = 0.1

reward_estimates = [0, 0, 0, 0]

counts = [0, 0, 0, 0]

actual_rewards = [0.3, 0.5, 0.7, 0.4]

num_interactions = 1000

history = [[], [], [], []]

for i in range(num_interactions):

    if random.random() < epsilon:
        selected = random.randint(0, 3)
    else:
        selected = reward_estimates.index(max(reward_estimates))

    if random.random() < actual_rewards[selected]:
        reward = 1
    else:
        reward = 0

    counts[selected] += 1

    reward_estimates[selected] += (
        reward - reward_estimates[selected]
    ) / counts[selected]

    for j in range(4):
        history[j].append(reward_estimates[j])

best_movie = reward_estimates.index(max(reward_estimates))

print("Movie Recommendation Results")
print("----------------------------")

for i in range(4):
    print(
        movies[i],
        "-> Selections:", counts[i],
        ", Estimated Reward:",
        round(reward_estimates[i], 3)
    )

print("\nBest Movie:", movies[best_movie])
print("Highest Expected Reward:",
      round(reward_estimates[best_movie], 3))

plt.figure(figsize=(10, 6))

for i in range(4):
    plt.plot(history[i], label=movies[i])

plt.xlabel("User Interactions")
plt.ylabel("Estimated Reward")
plt.title("Epsilon-Greedy Movie Recommendation")
plt.legend()
plt.grid(True)

plt.show()
