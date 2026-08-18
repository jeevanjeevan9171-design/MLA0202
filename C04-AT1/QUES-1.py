print("Bayesian Network for Disease Prediction")
P_Obesity = {
    "No": 0.7,
    "Yes": 0.3
}

P_HighSugar = {
    "No": 0.6,
    "Yes": 0.4
}
P_Diabetes = {
    ("No", "No"): 0.01,
    ("No", "Yes"): 0.20,
    ("Yes", "No"): 0.10,
    ("Yes", "Yes"): 0.70
}
obesity = "Yes"
high_sugar = "Yes"
probability = P_Diabetes[(obesity, high_sugar)]

print("\nBayesian Network:")
print("Obesity -> Diabetes")
print("High Blood Sugar -> Diabetes")

print("\nEvidence:")
print("Obesity =", obesity)
print("High Blood Sugar =", high_sugar)

print("\nPredicted Probability of Diabetes:")
print(probability)

print("Predicted Probability of Diabetes:", probability * 100, "%")
