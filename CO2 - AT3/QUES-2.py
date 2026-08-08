import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.DataFrame({
    "message": [
        "Win a free lottery prize now",
        "Congratulations you won a cash reward",
        "Claim your free gift today",
        "You have won a free mobile phone",
        "Get free money by clicking this link",
        "Urgent claim your lottery prize",
        "Free entry to win cash prize",
        "You are selected for a reward",
        "Win a shopping voucher now",
        "Congratulations claim your reward",
        "Are you coming to college today",
        "Please send me the assignment",
        "Can we meet tomorrow",
        "I will call you later",
        "Please bring the project file",
        "What time is the class",
        "The meeting starts at ten",
        "Can you send the notes",
        "I reached home safely",
        "See you tomorrow"
    ],
    "label": [
        "spam","spam","spam","spam","spam",
        "spam","spam","spam","spam","spam",
        "ham","ham","ham","ham","ham",
        "ham","ham","ham","ham","ham"
    ]
})

X_train, X_test, y_train, y_test = train_test_split(
    data.message,
    data.label,
    test_size=0.3,
    random_state=42,
    stratify=data.label
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1,2)
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("SPAM EMAIL CLASSIFICATION")
print("-------------------------")
print("Initial Accuracy:",
      round(accuracy_score(y_test, prediction), 2))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("\nOPTIMIZED MODEL")
print("Accuracy:",
      round(accuracy_score(y_test, prediction), 2))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

messages = [
    "Congratulations you won a free cash prize",
    "Please send me the assignment tomorrow",
    "Claim your free reward now",
    "Can we meet after class"
]

new_messages = vectorizer.transform(messages)
results = model.predict(new_messages)

print("\nNEW MESSAGE PREDICTIONS")

for message, result in zip(messages, results):
    print("\nMessage:", message)
    print("Prediction:", result.upper())
