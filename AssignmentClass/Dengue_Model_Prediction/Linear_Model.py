import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = 'AssignmentClass/Dengue_Model_Prediction/pivot_output.csv'
df = pd.read_csv(data)

df_Male = df['Tmax'].values.reshape(-1, 1) 
df_Female = df['Total'].values

# LinearRegression
model = LinearRegression()
# train
model.fit(df_Male, df_Female)

# test
male_fit = np.linspace(df_Male.min(), df_Male.max(), 100).reshape(-1, 1)

# Predict Female values using the trained model
female_fit = model.predict(male_fit)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(df_Male, df_Female, label="Actual Data", color="blue")
plt.plot(male_fit, female_fit, label="Linear Fit", color="red")
plt.title("Number of Degue Tmax vs Total")
plt.xlabel("Tmax")
plt.ylabel("Total")
plt.legend()
plt.grid(True)
plt.show()