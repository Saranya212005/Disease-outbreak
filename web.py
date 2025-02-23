import joblib
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Disease Prediction", layout="wide", page_icon="🏥")

# Load trained models and scalers
diabetes_model = joblib.load("diabetes_model.pkl")
heart_disease_model = joblib.load("heart_disease_model.pkl")
parkinsons_model = joblib.load("parkinsons_model.pkl")

scaler_d = joblib.load("scaler_diabetes.pkl")
scaler_h = joblib.load("scaler_heart.pkl")
scaler_p = joblib.load("scaler_parkinsons.pkl")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
        menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    inputs = []
    
    col1, col2, col3 = st.columns(3)
    with col1: inputs.append(st.number_input('Number of Pregnancies', min_value=0))
    with col2: inputs.append(st.number_input('Glucose Level', min_value=0))
    with col3: inputs.append(st.number_input('Blood Pressure Value', min_value=0))
    with col1: inputs.append(st.number_input('Skin Thickness Value', min_value=0))
    with col2: inputs.append(st.number_input('Insulin Value', min_value=0))
    with col3: inputs.append(st.number_input('BMI Value', min_value=0.0, format="%.2f"))
    with col1: inputs.append(st.number_input('Diabetes Pedigree Function Value', min_value=0.0, format="%.2f"))
    with col2: inputs.append(st.number_input('Age of the Person', min_value=0))
    
    if st.button('Predict Diabetes'):
        inputs = np.array(inputs).reshape(1, -1)
        inputs = scaler_d.transform(inputs)
        prediction = diabetes_model.predict(inputs)
        st.success("The person is diabetic" if prediction[0] == 1 else "The person is not diabetic")

# Heart Disease Prediction
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    inputs = []

    col1, col2, col3 = st.columns(3)
    with col1: inputs.append(st.number_input('Age', min_value=0))
    with col2: inputs.append(st.radio('Sex', [0, 1], format_func=lambda x: "Female" if x == 0 else "Male"))
    with col3: inputs.append(st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3))
    with col1: inputs.append(st.number_input('Blood Pressure', min_value=0))
    with col2: inputs.append(st.number_input('Cholesterol Level', min_value=0))
    with col3: inputs.append(st.radio('Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"))
    with col1: inputs.append(st.number_input('ECG Results (0-2)', min_value=0, max_value=2))
    with col2: inputs.append(st.number_input('Max Heart Rate Achieved', min_value=0))
    with col3: inputs.append(st.radio('Exercise-Induced Angina', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"))
    with col1: inputs.append(st.number_input('ST Depression Induced by Exercise', min_value=0.0, format="%.2f"))
    with col2: inputs.append(st.number_input('Slope of Peak Exercise ST Segment (0-2)', min_value=0, max_value=2))
    with col3: inputs.append(st.number_input('Number of Major Vessels (0-3)', min_value=0, max_value=3))
    with col1: inputs.append(st.number_input('Thallium Stress Test Result (0-3)', min_value=0, max_value=3))

    if st.button('Predict Heart Disease'):
        inputs = np.array(inputs).reshape(1, -1)
        inputs = scaler_h.transform(inputs)  # Ensure proper scaling
        prediction = heart_disease_model.predict(inputs)
        st.success("The person has heart disease" if prediction[0] == 1 else "The person does not have heart disease")

# Parkinson's Prediction
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction")
    inputs = []

    col1, col2, col3 = st.columns(3)
    with col1: inputs.append(st.number_input('MDVP:Fo(Hz)', min_value=0.0, format="%.2f"))
    with col2: inputs.append(st.number_input('MDVP:Fhi(Hz)', min_value=0.0, format="%.2f"))
    with col3: inputs.append(st.number_input('MDVP:Flo(Hz)', min_value=0.0, format="%.2f"))
    with col1: inputs.append(st.number_input('MDVP:Jitter(%)', min_value=0.0, format="%.2f"))
    with col2: inputs.append(st.number_input('MDVP:Jitter(Abs)', min_value=0.0, format="%.5f"))
    with col3: inputs.append(st.number_input('MDVP:RAP', min_value=0.0, format="%.5f"))
    with col1: inputs.append(st.number_input('MDVP:PPQ', min_value=0.0, format="%.5f"))
    with col2: inputs.append(st.number_input('Jitter:DDP', min_value=0.0, format="%.5f"))
    with col3: inputs.append(st.number_input('MDVP:Shimmer', min_value=0.0, format="%.2f"))
    with col1: inputs.append(st.number_input('MDVP:Shimmer(dB)', min_value=0.0, format="%.2f"))
    with col2: inputs.append(st.number_input('Shimmer:APQ3', min_value=0.0, format="%.5f"))
    with col3: inputs.append(st.number_input('Shimmer:APQ5', min_value=0.0, format="%.5f"))
    with col1: inputs.append(st.number_input('MDVP:APQ', min_value=0.0, format="%.5f"))
    with col2: inputs.append(st.number_input('Shimmer:DDA', min_value=0.0, format="%.5f"))
    with col3: inputs.append(st.number_input('NHR', min_value=0.0, format="%.5f"))
    with col1: inputs.append(st.number_input('HNR', min_value=0.0, format="%.2f"))
    with col2: inputs.append(st.number_input('RPDE', min_value=0.0, format="%.5f"))
    with col3: inputs.append(st.number_input('DFA', min_value=0.0, format="%.5f"))
    with col1: inputs.append(st.number_input('spread1', min_value=0.0, format="%.5f"))
    with col2: inputs.append(st.number_input('spread2', min_value=0.0, format="%.5f"))
    with col3: inputs.append(st.number_input('D2', min_value=0.0, format="%.5f"))
    with col1: inputs.append(st.number_input('PPE', min_value=0.0, format="%.5f"))

    if st.button("Predict Parkinson's Disease"):
        inputs = np.array(inputs).reshape(1, -1)
        inputs = scaler_p.transform(inputs)  # Ensure proper scaling
        prediction = parkinsons_model.predict(inputs)
        st.success("The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease")
