import random
import matplotlib.pyplot as plt

states = ["Start", "Level 1", "Level 2", "Level 3", "Final"]

value = {
    "Start": 0,
    "Level 1": 0,
    "Level 2": 0,
    "Level 3": 0,
    "Final": 0
}

alpha = 0.1
gamma = 0.9
episodes = 100

history = []

for episode in range(episodes):

    current_state = "Start"

    while current_state != "Final":

        if current_state == "Start":
            next_state = "Level 1"
            reward = random.randint(0, 2)

        elif current_state == "Level 1":
            next_state = "Level 2"
            reward = random.randint(1, 3)

        elif current_state == "Level 2":
            next_state = "Level 3"
            reward = random.randint(2, 4)

        elif current_state == "Level 3":
            next_state = "Final"
            reward = random.randint(5, 10)

        td_target = reward + gamma * value[next_state]

        td_error = td_target - value[current_state]

        value[current_state] += alpha * td_error

        current_state = next_state

    history.append(value["Start"])

print("Final State Values")
print("------------------")

for state in states:
    print(state, ":", round(value[state], 2))

plt.plot(history)

plt.xlabel("Episode")
plt.ylabel("Estimated Value")
plt.title("TD Learning - Game Score Prediction")
plt.grid(True)

plt.show()
