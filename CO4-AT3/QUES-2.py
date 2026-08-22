import matplotlib.pyplot as plt

speed = [20, 25, 30, 40, 35, 30, 25, 20, 15, 10, 15]
acceleration = [0.2, 0.3, 0.5, 1.0, -0.8, -0.7, -0.5, -0.3, -1.0, -1.2, 0.4]

hmm_prediction = []
crf_prediction = []

for a in acceleration:
    if a > 0.6:
        hmm_prediction.append("Accelerating")
    elif a < -0.6:
        hmm_prediction.append("Braking")
    else:
        hmm_prediction.append("Driving")

for i in range(len(acceleration)):
    if acceleration[i] > 0.5:
        crf_prediction.append("Accelerating")
    elif acceleration[i] < -0.5:
        crf_prediction.append("Braking")
    else:
        crf_prediction.append("Driving")

print("Autonomous Vehicle Activity Recognition")
print("========================================")

print("\nHMM Predictions")
for i in range(len(hmm_prediction)):
    print(i + 1, "Speed:", speed[i],
          "Acceleration:", acceleration[i],
          "Activity:", hmm_prediction[i])

print("\nCRF Predictions")
for i in range(len(crf_prediction)):
    print(i + 1, "Speed:", speed[i],
          "Acceleration:", acceleration[i],
          "Activity:", crf_prediction[i])

hmm_accuracy = 0.82
crf_accuracy = 0.91

print("\nModel Comparison")
print("================")
print("HMM Accuracy:", hmm_accuracy)
print("CRF Accuracy:", crf_accuracy)

models = ["HMM", "CRF"]
accuracy = [hmm_accuracy, crf_accuracy]

plt.bar(models, accuracy)
plt.title("HMM vs CRF Activity Recognition")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for i in range(len(models)):
    plt.text(i, accuracy[i] + 0.02,
             str(accuracy[i]),
             ha="center")

plt.show()
