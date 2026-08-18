print("Hidden Markov Model for Weather Prediction")
states = ["Sunny", "Cloudy", "Rainy"]
observations = ["Sunny", "Cloudy", "Rainy", "Rainy", "Cloudy"]
initial_probability = {
    "Sunny": 0.5,
    "Cloudy": 0.3,
    "Rainy": 0.2
}
transition_probability = {
    "Sunny": {
        "Sunny": 0.6,
        "Cloudy": 0.3,
        "Rainy": 0.1
    },

    "Cloudy": {
        "Sunny": 0.3,
        "Cloudy": 0.4,
        "Rainy": 0.3
    },

    "Rainy": {
        "Sunny": 0.1,
        "Cloudy": 0.3,
        "Rainy": 0.6
    }
}
emission_probability = {
    "Sunny": {
        "Sunny": 0.8,
        "Cloudy": 0.15,
        "Rainy": 0.05
    },

    "Cloudy": {
        "Sunny": 0.2,
        "Cloudy": 0.6,
        "Rainy": 0.2
    },

    "Rainy": {
        "Sunny": 0.05,
        "Cloudy": 0.25,
        "Rainy": 0.70
    }
}
predicted_states = []

previous_state = None

for observation in observations:

    best_state = None
    best_probability = 0

    for state in states:

        emission = emission_probability[state][observation]

        if previous_state is None:
            probability = initial_probability[state] * emission
        else:
            transition = transition_probability[previous_state][state]
            probability = transition * emission

        if probability > best_probability:
            best_probability = probability
            best_state = state

    predicted_states.append(best_state)
    previous_state = best_state

print("\nObservation Sequence:")
print(observations)

print("\nPredicted Hidden Weather States:")

for i in range(len(predicted_states)):
    print(
        "Observation:",
        observations[i],
        "-> Hidden State:",
        predicted_states[i]
    )

print("\nFinal Predicted Sequence:")
print(predicted_states)
