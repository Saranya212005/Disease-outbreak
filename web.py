import joblib
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Prediction of Disease Outbreaks",
                   layout="wide",
                   page_icon="doctor")

# Load trained models
diabetes_model = joblib.load(r"C:\Disease outbreak\diabetes_model.pkl")  
heart_disease_model = joblib.load(r"C:\Disease outbreak\heart_disease_model.pkl")  
parkinsons_model = joblib.load(r"C:\Disease outbreak\parkinsons_model.pkl")  

# Sidebar menu
with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreak System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)


if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction Using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        glucose = st.text_input('Glucose Level')
    with col3:
        blood_pressure = st.text_input('Blood Pressure Value')
    with col1:
        skin_thickness = st.text_input('Skin Thickness Value')
    with col2:
        insulin = st.text_input('Insulin Value')
    with col3:
        bmi = st.text_input('BMI Value')
    with col1:
        pedigree_function = st.text_input('Diabetes Pedigree Function Value')
    with col2:
        age = st.text_input('Age of the Person')

    diab_diagnosis = ""

    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(pregnancies), float(glucose), float(blood_pressure),
                          float(skin_thickness), float(insulin), float(bmi),
                          float(pedigree_function), float(age)]
            
            diab_prediction = diabetes_model.predict([user_input])
            
            if diab_prediction[0] == 1:
                diab_diagnosis = "The person is diabetic"
            else:
                diab_diagnosis = "The person is not diabetic"
        except ValueError:
            diab_diagnosis = "Please enter valid numerical values!"

    st.success(diab_diagnosis)

if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction Using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (1 = Male, 0 = Female)')
    with col3:
        cp = st.text_input('Chest Pain Type (0-3)')
    with col1:
        bp = st.text_input('Blood Pressure')
    with col2:
        cholesterol = st.text_input('Cholesterol Level')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)')
    with col1:
        ekg = st.text_input('ECG Results (0-2)')
    with col2:
        max_hr = st.text_input('Max Heart Rate Achieved')
    with col3:
        exercise_angina = st.text_input('Exercise-Induced Angina (1 = Yes, 0 = No)')
    with col1:
        st_depression = st.text_input('ST Depression Induced by Exercise')
    with col2:
        slope = st.text_input('Slope of the Peak Exercise ST Segment')
    with col3:
        vessels = st.text_input('Number of Major Vessels (0-3)')
    with col1:
        thallium = st.text_input('Thallium Stress Test Result (0-3)')

    heart_diagnosis = ""

    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(age), float(sex), float(cp), float(bp), float(cholesterol),
                          float(fbs), float(ekg), float(max_hr), float(exercise_angina),
                          float(st_depression), float(slope), float(vessels), float(thallium)]
            
            heart_prediction = heart_disease_model.predict([user_input])
            
            if heart_prediction[0] == 1:
                heart_diagnosis = "The person has heart disease"
            else:
                heart_diagnosis = "The person does not have heart disease"
        except ValueError:
            heart_diagnosis = "Please enter valid numerical values!"

    st.success(heart_diagnosis)

if selected == 'Parkinsons Prediction':
    st.title('Parkinson\'s Disease Prediction Using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz) - Average vocal frequency')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz) - Maximum vocal frequency')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz) - Minimum vocal frequency')
    with col1:
        jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col2:
        jitter_abs = st.text_input('MDVP:Jitter(Abs)')
    with col3:
        rap = st.text_input('MDVP:RAP')
    with col1:
        ppq = st.text_input('MDVP:PPQ')
    with col2:
        ddp = st.text_input('Jitter:DDP')
    with col3:
        shimmer = st.text_input('MDVP:Shimmer')
    with col1:
        shimmer_db = st.text_input('MDVP:Shimmer(dB)')
    with col2:
        hnr = st.text_input('HNR - Harmonic-to-Noise Ratio')

    parkinsons_diagnosis = ""

    if st.button('Parkinson\'s Test Result'):
        try:
            user_input = [float(fo), float(fhi), float(flo), float(jitter_percent),
                          float(jitter_abs), float(rap), float(ppq), float(ddp),
                          float(shimmer), float(shimmer_db), float(hnr)]
            
            parkinsons_prediction = parkinsons_model.predict([user_input])
            
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
        except ValueError:
            parkinsons_diagnosis = "Please enter valid numerical values!"

    st.success(parkinsons_diagnosis)