import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_excel("AmesHousing.xlsx")

# Select relevant features (customize as needed)
features = ["LotArea", "YearBuilt", "OverallQual", "GarageCars", "TotalBsmtSF", "GrLivArea"]
target = "SalePrice"

# Drop missing values
df = df[features + [target]].dropna()

# Split into features (X) and target variable (y)
X = df[features]
y = df[target]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open("model/housing_model.pkl", "wb"))

# Evaluate model
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

print("Model training complete. Model saved in /model directory.")
