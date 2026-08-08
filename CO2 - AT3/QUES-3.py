import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_iris()

X = data.data
y = data.target

X1, X2, y1, y2 = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X1, y1)

prediction = model.predict(X2)

accuracy = accuracy_score(y2, prediction)
cm = confusion_matrix(y2, prediction)

print("NAIVE BAYES CLASSIFICATION")
print("--------------------------")
print("Accuracy:", round(accuracy, 3))

print("\nConfusion Matrix")
print(cm)

print("\nCLASSIFICATION RESULTS")
for i in range(3):
    correct = cm[i][i]
    total = sum(cm[i])
    print(data.target_names[i], ":", correct, "correct out of", total)

sample = [[5.1, 3.5, 1.4, 0.2]]
result = model.predict(sample)

print("\nNEW PREDICTION")
print("Sepal Length:", sample[0][0])
print("Sepal Width :", sample[0][1])
print("Petal Length:", sample[0][2])
print("Petal Width :", sample[0][3])
print("Predicted Flower:", data.target_names[result[0]])

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    [0, 1, 2],
    data.target_names
)

plt.yticks(
    [0, 1, 2],
    data.target_names
)

for i in range(3):
    for j in range(3):
        plt.text(
            j, i, cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar(label="Number of Samples")
plt.tight_layout()
plt.show()

