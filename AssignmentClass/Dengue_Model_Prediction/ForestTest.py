# นำเข้าไลบรารีที่จำเป็น
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# อ่านข้อมูล
data = 'AssignmentClass/Dengue_Model_Prediction/pivot_output.csv'
df = pd.read_csv(data)

# เลือกฟีเจอร์ (X) และเป้าหมาย (y)
X = df['AvgRain'].values.reshape(-1, 1)  # AvgRain เป็นตัวแปรอิสระ
y = df['Total'].values  # Total เป็นตัวแปรเป้าหมาย

# แบ่งข้อมูลเป็น Train และ Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและเทรนโมเดล Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# ประเมินผลโมเดล
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# ตัวอย่างการทำนาย
example_input = np.array([[50]])
example_prediction = model.predict(example_input)
print(f"Prediction for AvgRain={example_input[0][0]}: {example_prediction[0]:.2f}")

# สร้าง DataFrame สำหรับแสดงผล
results_df = pd.DataFrame({
    "True Value": y_test,
    "Predicted Value": y_pred
})

# เพิ่มคอลัมน์ความแตกต่าง (Error)
results_df["Error"] = results_df["True Value"] - results_df["Predicted Value"]

# แสดงผลตาราง
print(results_df.head(10))  # แสดง 10 แถวแรก

# สร้างกราฟเปรียบเทียบค่าจริงและค่าทำนาย
plt.figure(figsize=(8, 6))

# Plot ค่าจริง vs ค่าทำนาย
plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predicted vs True')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')

# ปรับแต่งกราฟ
plt.title("Random Forest Regressor: Predicted vs True Values", fontsize=14)
plt.xlabel("True Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# แสดงกราฟ
plt.show()
