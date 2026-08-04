import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
data = {
    'Fever':    [1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    'Cough':    [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    'Headache': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
    'Flu':      [1, 1, 1, 0, 0, 1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)
X = df[['Fever', 'Cough', 'Headache']]
y = df['Flu']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Actual Labels:", list(y_test))
print("Predicted Labels:", list(y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
new_patient = pd.DataFrame({
    'Fever': [1],
    'Cough': [1],
    'Headache': [1]
})
prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("\nPrediction: Patient Has Flu")
else:
    print("\nPrediction: Patient Does Not Have Flu")
