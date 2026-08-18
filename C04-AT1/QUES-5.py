import pandas as pd

data = pd.DataFrame({
    "Age": [25, 35, 45, 30, 50, 28, 40, 55, 32, 48],
    "Income": ["Low", "Medium", "High", "Low", "High",
               "Medium", "High", "High", "Medium", "High"],
    "Vehicle": ["Car", "Car", "SUV", "Bike", "SUV",
                "Bike", "Car", "SUV", "Car", "Car"],
    "Claim": ["No", "Yes", "Yes", "No", "Yes",
              "No", "Yes", "Yes", "No", "Yes"]
})

print("Insurance Dataset:")
print(data)

print("\nLearned Bayesian Network Structure:")
print("Age -> Claim")
print("Income -> Claim")
print("Vehicle -> Claim")

new_customer = data[
    (data["Income"] == "Medium") &
    (data["Vehicle"] == "Car")
]

if len(new_customer) > 0:
    yes = (new_customer["Claim"] == "Yes").sum()
    probability = yes / len(new_customer)
else:
    probability = (data["Claim"] == "Yes").sum() / len(data)

print("\nNew Customer:")
print("Age = 35")
print("Income = Medium")
print("Vehicle = Car")

print("\nProbability of Insurance Claim:")
print(probability)

print("\nProbability in Percentage:")
print(probability * 100, "%")

if probability >= 0.5:
    print("\nPrediction: Claim = Yes")
else:
    print("\nPrediction: Claim = No")
