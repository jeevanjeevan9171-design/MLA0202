import matplotlib.pyplot as plt

income = [25000, 40000, 30000, 60000, 20000, 50000, 35000, 70000]
credit_score = [580, 720, 650, 780, 550, 740, 620, 800]
transaction_risk = [0.8, 0.2, 0.5, 0.1, 0.9, 0.2, 0.6, 0.1]

bn_risk = []
mrf_risk = []

for i in range(len(income)):
    if credit_score[i] < 600 and transaction_risk[i] > 0.7:
        bn_risk.append(0.90)
    elif credit_score[i] < 650 or transaction_risk[i] > 0.5:
        bn_risk.append(0.65)
    else:
        bn_risk.append(0.20)

for i in range(len(income)):
    score = 0

    if credit_score[i] < 600:
        score += 2

    if transaction_risk[i] > 0.7:
        score += 2

    if income[i] < 25000:
        score += 1

    mrf_risk.append(score / 5)

print("Financial Risk Prediction")
print("=========================")

print("\nCustomer Details and Predictions")

for i in range(len(income)):
    print(
        "Customer", i + 1,
        "| Income:", income[i],
        "| Credit Score:", credit_score[i],
        "| Transaction Risk:", transaction_risk[i],
        "| BN Risk:", round(bn_risk[i], 2),
        "| MRF Risk:", round(mrf_risk[i], 2)
    )

bn_average = sum(bn_risk) / len(bn_risk)
mrf_average = sum(mrf_risk) / len(mrf_risk)

print("\nModel Comparison")
print("================")
print("Bayesian Network Average Risk:", round(bn_average, 3))
print("MRF Average Risk:", round(mrf_average, 3))

models = ["Bayesian Network", "Markov Random Field"]
risk = [bn_average, mrf_average]

plt.bar(models, risk)
plt.title("Financial Risk Prediction Comparison")
plt.xlabel("Models")
plt.ylabel("Average Risk Probability")
plt.ylim(0, 1)

for i in range(len(models)):
    plt.text(i, risk[i] + 0.02, round(risk[i], 3), ha="center")

plt.show()
