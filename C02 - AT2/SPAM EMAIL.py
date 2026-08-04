import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = {
    'Free': [10, 0, 8, 1, 12, 0, 9, 2, 15, 1],
    'Offer': [8, 1, 7, 0, 10, 1, 8, 1, 12, 0],
    'Money': [7, 0, 6, 1, 8, 0, 7, 1, 9, 0],
    'Spam': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
X = df[['Free', 'Offer', 'Money']]
y = df['Spam']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Actual Labels:", list(y_test))
print("Predicted Labels:", list(y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
new_email = pd.DataFrame({
    'Free': [11],
    'Offer': [9],
    'Money': [8]
})

prediction = model.predict(new_email)

if prediction[0] == 1:
    print("\nPrediction: Spam Email")
else:
    print("\nPrediction: Not Spam")
