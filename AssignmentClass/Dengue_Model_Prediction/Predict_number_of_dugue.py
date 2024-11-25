import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('AssignmentClass/Dengue_Model_Prediction/pivot_output.csv')

# Features and target variable
X = data[['datesick_month', 'AvgRain', 'province', 'Tmax', 'Tmin', 'datesick_date', 'datesick_year']]
y = data['Total']

# Collect user input
input_monthsick = int(input("Enter the month of sickness: "))
input_avgRain = int(input("Enter the average rainfall (in mm): "))
input_province = int(input(
    "Select your province:\n"
    "[1] Ang Thong\n"
    "[2] Lopburi\n"
    "[3] Nakhon Nayok\n"
    "[4] Nonthaburi\n"
    "[5] Pathum Thani\n"
    "[6] Phra Nakhon Si Ayutthaya\n"
    "[7] Saraburi\n"
    "[8] Sing Buri\n"
    "Your province (choose 1-8): "
))
input_Tmax = float(input("Enter the maximum temperature (°C): "))
input_Tmin = float(input("Enter the minimum temperature (°C): "))

# Province map
province_map = {
    1: "Ang Thong",
    2: "Lopburi",
    3: "Nakhon Nayok",
    4: "Nonthaburi",
    5: "Pathum Thani",
    6: "Phra Nakhon Si Ayutthaya",
    7: "Saraburi",
    8: "Sing Buri"
}

# Validate province input
if input_province in province_map:
    selected_province = province_map[input_province]
else:
    raise ValueError("Invalid province selection. Please choose a number between 1 and 8.")

# Encode categorical variables (province)
encoder = OneHotEncoder()
encoded_provinces = encoder.fit_transform(data[['province']]).toarray()

# Combine features for the model
X_encoded = np.hstack((data[['datesick_month', 'AvgRain', 'Tmax', 'Tmin', 'datesick_date', 'datesick_year']].values, encoded_provinces))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Hyperparameter tuning (optional)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_regressor = grid_search.best_estimator_

# Prepare user input data
new_data = pd.DataFrame({
    'datesick_month': [input_monthsick],
    'AvgRain': [input_avgRain],
    'province': [selected_province],
    'Tmax': [input_Tmax],
    'Tmin': [input_Tmin],
    'datesick_date':[1],
    'datesick_year': [2018]
})

# Encode the user's province
new_encoded_province = encoder.transform(new_data[['province']]).toarray()
new_X = np.hstack((new_data[['datesick_month', 'AvgRain', 'Tmax', 'Tmin', 'datesick_date', 'datesick_year']].values, new_encoded_province))

# Predict the total dengue cases
predicted_cases = best_rf_regressor.predict(new_X)
print(f"\nPredicted Dengue Cases in {new_data['province'].iloc[0]} for Month {new_data['datesick_month'].iloc[0]}: {predicted_cases[0]:.2f}")

# Evaluate the model
y_pred = best_rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Feature Importance (Optional)
feature_names = ['datesick_month', 'AvgRain', 'Tmax', 'Tmin', 'datesick_date', 'datesick_year'] + list(encoder.get_feature_names_out(['province']))
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': best_rf_regressor.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importances)