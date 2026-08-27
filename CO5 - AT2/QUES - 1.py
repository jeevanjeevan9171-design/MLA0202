import random
import matplotlib.pyplot as plt

num_simulations = 10000
on_time_count = 0

simulation_points = []
probability_values = []

for i in range(1, num_simulations + 1):

    traffic = random.uniform(0, 100)

    delivery_time = 30 + (traffic * 0.5) + random.uniform(-5, 5)

    if delivery_time <= 60:
        on_time_count += 1

    if i % 100 == 0:
        probability = on_time_count / i
        simulation_points.append(i)
        probability_values.append(probability * 100)

final_probability = on_time_count / num_simulations

print("Total Simulations:", num_simulations)
print("On-Time Deliveries:", on_time_count)
print("Estimated Probability:", round(final_probability * 100, 2), "%")

plt.plot(simulation_points, probability_values)

plt.xlabel("Number of Simulations")
plt.ylabel("Estimated Probability (%)")
plt.title("Monte Carlo Simulation - On-Time Delivery Probability")
plt.grid(True)

plt.show()
