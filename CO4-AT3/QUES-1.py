import matplotlib.pyplot as plt

p_disease = 0.30
p_no_disease = 0.70

p_fever_given_disease = 0.80
p_cough_given_disease = 0.70

p_fever_given_no_disease = 0.20
p_cough_given_no_disease = 0.10

disease_probability = (
    p_disease *
    p_fever_given_disease *
    p_cough_given_disease
)

no_disease_probability = (
    p_no_disease *
    p_fever_given_no_disease *
    p_cough_given_no_disease
)

total = disease_probability + no_disease_probability

bn_result = disease_probability / total

fever = 1
cough = 1

disease_score = 0
no_disease_score = 0

if fever == 1:
    disease_score += 4
    no_disease_score += 1

if cough == 1:
    disease_score += 3
    no_disease_score += 1

total_score = disease_score + no_disease_score

mrf_result = disease_score / total_score

print("Healthcare Disease Diagnosis")
print("============================")

print("\nBayesian Network")
print("Probability of Disease:", round(bn_result, 3))

if bn_result >= 0.5:
    print("Diagnosis: Disease is likely")
else:
    print("Diagnosis: Disease is unlikely")

print("\nMarkov Random Field")
print("Disease Score:", disease_score)
print("No Disease Score:", no_disease_score)
print("Probability of Disease:", round(mrf_result, 3))

if mrf_result >= 0.5:
    print("Diagnosis: Disease is likely")
else:
    print("Diagnosis: Disease is unlikely")

print("\nComparison")
print("Bayesian Network:", round(bn_result, 3))
print("Markov Random Field:", round(mrf_result, 3))

models = ["Bayesian Network", "Markov Random Field"]
probabilities = [bn_result, mrf_result]

plt.bar(models, probabilities)
plt.title("Disease Probability Comparison")
plt.xlabel("Models")
plt.ylabel("Probability of Disease")
plt.ylim(0, 1)

for i, value in enumerate(probabilities):
    plt.text(i, value + 0.02, round(value, 3), ha="center")

plt.show()
