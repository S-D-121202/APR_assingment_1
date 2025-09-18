import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load dataset
file_path = "/kaggle/input/train-dataset/train.csv"  # replace with your path
data = pd.read_csv(file_path)

# Drop the ID column (not useful for prediction)
data_clean = data.drop(columns=['Id'])

# Separate features (X) and target (y)
X = data_clean.drop(columns=['SalePrice'])
y = data_clean['SalePrice']

# Select only numeric features for linear regression
X_numeric = X.select_dtypes(include=[np.number])

# Handle missing values: fill with median
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_numeric)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# Train linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions
y_pred = lin_reg.predict(X_test)

# Evaluate performance
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


train_predictions = lin_reg.predict(X_train)
trainr2 = r2_score(y_train, train_predictions)
trainrmse = np.sqrt(mean_squared_error(y_train, train_predictions))

print("R2 score:(train)", trainr2)
print("RMSE:(train)", trainrmse)

print("R2 score:(validation)", r2)
print("RMSE:(validation)", rmse)

test_data = pd.read_csv("/kaggle/input/test-dataset/test.csv")   # replace with your actual test file path

# Save the Ids for submission (if needed)
test_ids = test_data['Id']

# Drop Id column
test_features = test_data.drop(columns=['Id'])

# Select numeric features (same as training)
test_numeric = test_features.select_dtypes(include=[np.number])

# Apply the same imputer from training
X_test_final = imputer.transform(test_numeric)

# Predict using trained model
test_predictions = lin_reg.predict(X_test_final)

# Put results into a DataFrame
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_predictions
})

# Save to CSV (for Kaggle submission, if that's the use case)
submission.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")