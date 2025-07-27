# 🩺 Diabetes Prediction using Machine Learning – YBI Foundation Internship Project

This project was developed as part of the **Artificial Intelligence & Machine Learning Internship** offered by the **YBI Foundation**. The objective is to build a predictive model using machine learning to determine whether a patient is diabetic based on medical attributes.

---

## 📌 Objective

To use a classification algorithm to predict diabetes based on input features like glucose level, BMI, insulin, and age. The model is trained using the **Pima Indians Diabetes Dataset**.

---

## 🧠 Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Logistic Regression**
- **Joblib** (for model persistence)

---

## 📂 Files & Structure
<pre>
Diabetes-Prediction-YBI/
├── project/
│   ├── predict_diabetes.py       # Main script for training and prediction
│   ├── diabetes.csv              # CSV dataset (Pima Indians Diabetes Dataset)
├── README.md                     # Project documentation
</pre>

---
## 🖼️ Screenshot – VS Code Project View

![Project Screenshot]<img width="1233" height="834" alt="Screenshot 1" src="https://github.com/user-attachments/assets/d88b3650-d874-4cd2-9943-3a2f888c7a0d" />

----

![Project Screenshot]<img width="1012" height="243" alt="Screenshot 2" src="https://github.com/user-attachments/assets/cdc43cc9-696f-483b-afe1-e280c1e4e647" />

----

## 🎯 2. Train the Model

python predict_diabetes.py --train diabetes.csv

---

## 🔍 3. Predict Using New Data
python predict_diabetes.py --predict 6 148 72 35 0 33.6 0.627 50

----

##Output:
sql
Prediction result: Diabetic

---

## 📊 Sample Dataset Fields

<pre>
| Field                    | Description                    |
| ------------------------ | ------------------------------ |
| Pregnancies              | Number of times pregnant       |
| Glucose                  | Plasma glucose level           |
| BloodPressure            | Diastolic blood pressure       |
| SkinThickness            | Triceps skin fold (mm)         |
| Insulin                  | 2-Hour serum insulin           |
| BMI                      | Body mass index                |
| DiabetesPedigreeFunction | Diabetes pedigree function     |
| Age                      | Age (in years)                 |
| Outcome                  | 0 = Non-diabetic, 1 = Diabetic |
</pre>

---

## 🧑‍🎓 Developed By
Name: Aniket Sundriyal
Internship: YBI Foundation – AI & ML Internship
Month: July 2025


