print("Conditional Random Field for Named Entity Recognition")
train_sentences = [
    ["John", "ordered", "Laptop", "Order123"],
    ["Alice", "bought", "Phone", "Order456"],
    ["Robert", "purchased", "Tablet", "Order789"],
    ["David", "ordered", "Headphones", "Order321"]
]
train_labels = [
    ["CUSTOMER", "O", "PRODUCT", "ORDER"],
    ["CUSTOMER", "O", "PRODUCT", "ORDER"],
    ["CUSTOMER", "O", "PRODUCT", "ORDER"],
    ["CUSTOMER", "O", "PRODUCT", "ORDER"]
]
def get_features(word):
    features = {}

    features["word"] = word
    features["lower"] = word.lower()
    features["length"] = len(word)
    features["digit"] = word.isdigit()
    features["capital"] = word[0].isupper()

    return features
feature_data = []
for sentence, labels in zip(train_sentences, train_labels):
    for word, label in zip(sentence, labels):

        features = get_features(word)

        feature_data.append({
            "word": word,
            "label": label,
            "features": features
        })


def predict(word):

    if word.startswith("Order"):
        return "ORDER"

    if word in ["Laptop", "Phone", "Tablet", "Headphones"]:
        return "PRODUCT"

    if word[0].isupper():
        return "CUSTOMER"

    return "O"


test_sentence = [
    "Sai",
    "ordered",
    "Laptop",
    "Order999"
]

print("\nTest Sentence:")
print(test_sentence)
print("\nPredicted Labels:")
for word in test_sentence:
    label = predict(word)
    print(word, "->", label)
