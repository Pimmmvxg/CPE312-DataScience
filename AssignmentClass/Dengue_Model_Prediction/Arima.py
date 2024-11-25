import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = 'AssignmentClass/Dengue_Model_Prediction/pivot_output.csv'
df = pd.read_csv(data)

df = df.sort_values(by='datesick_month')
# Split data into training and testing
train = df.iloc[:-12]  # Train on all but the last 12 months
test = df.iloc[-12:]   # Test on the last 12 months

# Separate the target variable (Total) and exogenous variables (Tmax, Tmin, AvgRain)
y_train = train['Total']
X_train = train[['Tmax', 'Tmin', 'AvgRain']]
y_test = test['Total']
X_test = test[['Tmax', 'Tmin', 'AvgRain']]

# Fit the SARIMAX model
model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_model = model.fit(disp=False)

# Print model summary
print(sarimax_model.summary())

# Forecast on the test set
y_pred = sarimax_model.predict(start=len(y_train), end=len(y_train)+len(y_test)-1, exog=X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='--')

plt.title('SARIMAX Prediction of Dengue Cases')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.grid()
plt.show()
