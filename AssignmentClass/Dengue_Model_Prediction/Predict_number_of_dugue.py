import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('AssignmentClass/Dengue_Model_Prediction/pivot_output.csv')

X = data[['datesick_month', 'AvgRain', 'province', 'Tmax', 'Tmin']]
y = data['Total']

# Encoding categorical variables
encoder = OneHotEncoder()
province_encoded = encoder.fit_transform(data[['province']]).toarray()

# Combine all features
X = np.hstack((data[['datesick_month', 'AvgRain', 'Tmax', 'Tmin']].values, province_encoded))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

new_data = pd.DataFrame({
    'month_sick': [12],                
    'AvgRain': [120],    
    'province': ['Nakhon Nayok'],
    'Tmax': [32],
    'Tmin': [26]             
})

province_encoded_new = encoder.transform(new_data[['province']]).toarray()

# Prepare feature matrix
new_X = np.hstack((new_data[['month_sick', 'AvgRain', 'Tmax', 'Tmin']].values, province_encoded_new))

# Predict
predicted_cases = rf_regressor.predict(new_X)
print(f"Predicted Dengue Cases in {new_data['province'].iloc[0]} for Month {new_data['month_sick'].iloc[0]}: {predicted_cases[0]}")

y_pred = rf_regressor.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")