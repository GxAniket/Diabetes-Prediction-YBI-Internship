# =============================
# Diabetes Prediction Program
# =============================

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
df = pd.read_csv("diabetes.csv")

# Step 2: Display first few rows (optional)
print("Sample Data:\n", df.head())

# Step 3: Separate features (X) and target label (y)
X = df.drop("Outcome", axis=1)  # All input features
y = df["Outcome"]               # Target label (0 = No Diabetes, 1 = Diabetes)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize the feature values for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 7: Evaluate the model on test data
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model trained. Accuracy on test data:", accuracy)

# =============================
# Step 8: Display Correlation Heatmap
# =============================
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()  # REQUIRED to display the graph

# =============================
# Step 9: Real-Time User Input
# =============================

print("\n--- Enter Patient Info Below ---")
try:
    pregnancies = float(input("Pregnancies: "))
    glucose = float(input("Glucose Level: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))
except ValueError:
    print("\n❌ Invalid input! Please enter numeric values only.")
    exit()

# Step 10: Prepare user input in correct format with column names
user_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]],
                         columns=X.columns)  # Matching training columns

# Step 11: Scale user input using the same scaler
user_data_scaled = scaler.transform(user_data)

# Step 12: Make prediction
prediction = model.predict(user_data_scaled)[0]

# Step 13: Display result
if prediction == 1:
    print("\n⚠️ The patient is likely **Diabetic**.")
else:
    print("\n✅ The patient is likely **Not Diabetic**.")
