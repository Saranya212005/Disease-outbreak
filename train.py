import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

diabetes_data = pd.read_csv("C:/Disease outbreak/diabetes.csv")


print(diabetes_data.columns)  

X_diabetes = diabetes_data.drop(columns=["Outcome"])
y_diabetes = diabetes_data["Outcome"]

scaler = StandardScaler()
X_diabetes = scaler.fit_transform(X_diabetes)

X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

diabetes_model = LogisticRegression()
diabetes_model.fit(X_train, y_train)

y_pred = diabetes_model.predict(X_test)
print(f"Diabetes Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

joblib.dump(diabetes_model, "C:/Disease outbreak/diabetes_model.pkl")

heart_data = pd.read_csv("C:/Disease outbreak/Heart_Disease_Prediction.csv")

print(heart_data.columns)

# Use correct target column name
X_heart = heart_data.drop(columns=["Heart Disease"]) 
y_heart = heart_data["Heart Disease"]

X_heart = scaler.fit_transform(X_heart)


X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

heart_disease_model = LogisticRegression()
heart_disease_model.fit(X_train, y_train)

y_pred = heart_disease_model.predict(X_test)
print(f"Heart Disease Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

joblib.dump(heart_disease_model, "C:/Disease outbreak/heart_disease_model.pkl")

parkinsons_data = pd.read_csv("C:/Disease outbreak/parkinsons.csv")

X_parkinsons = parkinsons_data.drop(columns=["name", "status"])  # Drop non-numeric column
y_parkinsons = parkinsons_data["status"]

scaler = StandardScaler()
X_parkinsons = scaler.fit_transform(X_parkinsons)

X_train, X_test, y_train, y_test = train_test_split(X_parkinsons, y_parkinsons, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, r"C:\Disease outbreak\parkinsons_model.pkl")

print("Model trained and saved successfully!")