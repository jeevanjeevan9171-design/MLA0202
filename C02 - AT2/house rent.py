import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
data = {
    'Floor_Area': [600, 750, 900, 1200, 1500, 1800, 850, 950, 1300, 1600],
    'Rooms': [1, 2, 2, 3, 3, 4, 2, 2, 3, 4],
    'Location': ['Downtown', 'Suburb', 'Downtown', 'Suburb', 'Downtown',
                 'Suburb', 'Downtown', 'Suburb', 'Downtown', 'Suburb'],
    'Rent': [15000, 18000, 22000, 25000, 32000,
             35000, 21000, 23000, 30000, 37000]
}

df = pd.DataFrame(data)

X = df[['Floor_Area', 'Rooms', 'Location']]
y = df['Rent']

categorical_features = ['Location']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)


model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Actual Rent:", list(y_test))
print("Predicted Rent:", y_pred)

print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


new_apartment = pd.DataFrame({
    'Floor_Area': [1100],
    'Rooms': [3],
    'Location': ['Downtown']
})

predicted_rent = model.predict(new_apartment)

print("\nPredicted Rent for New Apartment: ₹{:.2f}".format(predicted_rent[0]))
