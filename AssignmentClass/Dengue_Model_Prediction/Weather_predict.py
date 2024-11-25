import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = 'AssignmentClass/Dengue_Model_Prediction/pivot_output.csv'
dataset = pd.read_csv(data)

AvgRain_set = dataset['AvgRain'].values.reshape(-1, 1)
Total_set = dataset['Total'].values.reshape(-1, 1)

print('AvgRain = ',AvgRain_set.shape)
print('Total = ',Total_set.shape)

# 80% 20%
AvgRain_train, AvgRain_test, Total_train, Total_test = train_test_split(AvgRain_set, Total_set, test_size=0.5, random_state=0)

# training
model = LinearRegression()
model.fit(AvgRain_train, Total_train)

# test
Total_pred = model.predict(AvgRain_test)

# compare true data and predict data
df = pd.DataFrame({'Actually': Total_test.flatten(),'Predicted':Total_pred.flatten()})

print(df)
print(df.shape)

r2_score = model.score(AvgRain_test, Total_test)
print("R-squared (R²) score:", r2_score)

plt.scatter(AvgRain_test, Total_test, label="Actual Data", color="blue")  # Add label and color for scatter plot
plt.plot(AvgRain_test, Total_pred, color="red", linewidth=2, label="Predicted Line")  # Add label for the line
plt.title("Relationship between AvgRain and Total")  # Add a title
plt.xlabel("AvgRain")  # Label for x-axis
plt.ylabel("Total")  # Label for y-axis
plt.legend()  # Add a legend to the plot
plt.grid(True)  # Add grid for better readability
plt.show()
