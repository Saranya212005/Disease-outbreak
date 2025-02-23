import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE  # Handle imbalanced data

# Load datasets
diabetes_data = pd.read_csv("diabetes.csv")
heart_data = pd.read_csv("Heart_Disease_Prediction.csv")
parkinsons_data = pd.read_csv("parkinsons.csv")

# Fix target column for Heart Disease dataset
heart_data["Heart Disease"] = heart_data["Heart Disease"].map({'Absence': 0, 'Presence': 1})

# Prepare datasets
X_diabetes = diabetes_data.drop(columns=["Outcome"])
y_diabetes = diabetes_data["Outcome"]

X_heart = heart_data.drop(columns=["Heart Disease"])
y_heart = heart_data["Heart Disease"]

X_parkinsons = parkinsons_data.drop(columns=["name", "status"])
y_parkinsons = parkinsons_data["status"]

# Standardize features
scaler_d = StandardScaler()
X_diabetes = scaler_d.fit_transform(X_diabetes)

scaler_h = StandardScaler()
X_heart = scaler_h.fit_transform(X_heart)

scaler_p = StandardScaler()
X_parkinsons = scaler_p.fit_transform(X_parkinsons)  # ✅ Fixed: No reassignment after scaling!

# Split into training and testing sets
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_parkinsons, y_parkinsons, test_size=0.2, random_state=42)

# Handle class imbalance for heart disease
smote = SMOTE(random_state=42)
X_train_h, y_train_h = smote.fit_resample(X_train_h, y_train_h)

# Train models using RandomForest for better accuracy
diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
diabetes_model.fit(X_train_d, y_train_d)

heart_disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
heart_disease_model.fit(X_train_h, y_train_h)

parkinsons_model = RandomForestClassifier(n_estimators=100, random_state=42)
parkinsons_model.fit(X_train_p, y_train_p)

# Evaluate models
diabetes_acc = accuracy_score(y_test_d, diabetes_model.predict(X_test_d))
heart_acc = accuracy_score(y_test_h, heart_disease_model.predict(X_test_h))
parkinsons_acc = accuracy_score(y_test_p, parkinsons_model.predict(X_test_p))

print(f"Diabetes Model Accuracy: {diabetes_acc * 100:.2f}%")
print(f"Heart Disease Model Accuracy: {heart_acc * 100:.2f}%")
print(f"Parkinson's Model Accuracy: {parkinsons_acc * 100:.2f}%")

# Save models and scalers
joblib.dump(diabetes_model, "diabetes_model.pkl")
joblib.dump(heart_disease_model, "heart_disease_model.pkl")
joblib.dump(parkinsons_model, "parkinsons_model.pkl")

joblib.dump(scaler_d, "scaler_diabetes.pkl")
joblib.dump(scaler_h, "scaler_heart.pkl")
joblib.dump(scaler_p, "scaler_parkinsons.pkl")

print("Models and scalers trained and saved successfully!")
