import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the trained Pipeline model
MODEL_PATH = r"D:\Full projet\Predicting_chronic_kidney_disease\Src\model_pipeline.pkl"
model = joblib.load(MODEL_PATH)

# Set Page Configuration
st.set_page_config(page_title="CKD Prediction System", layout="wide")

st.title(" Chronic Kidney Disease (CKD) Prediction ")
st.markdown("---")
st.markdown("### Patient Clinical Information")
st.info("Please fill in the medical parameters below to predict the risk of CKD.")

# 2. Organize inputs into columns for a professional UI
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("General & Vital Signs")
    age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
    bp = st.number_input("Blood Pressure (mm/Hg)", min_value=40, max_value=200, value=80, step=1)
    sg = st.selectbox("Specific Gravity", options=[1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.selectbox("Albumin", options=[0, 1, 2, 3, 4, 5])
    su = st.selectbox("Sugar", options=[0, 1, 2, 3, 4, 5])

with col2:
    st.subheader("Laboratory Tests")
    rbc = st.selectbox("Red Blood Cells", options=["normal", "abnormal"])
    pc = st.selectbox("Pus Cell", options=["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", options=["notpresent", "present"])
    ba = st.selectbox("Bacteria", options=["notpresent", "present"])
    bgr = st.number_input("Blood Glucose Random (mgs/dl)", min_value=20.0, max_value=500.0, value=120.0)
    bu = st.number_input("Blood Urea (mgs/dl)", min_value=1.0, max_value=400.0, value=40.0)
    sc = st.number_input("Serum Creatinine (mgs/dl)", min_value=0.1, max_value=20.0, value=1.2)

with col3:
    st.subheader("Blood & Systemic Info")
    sod = st.number_input("Sodium (mEq/L)", min_value=100.0, max_value=200.0, value=135.0)
    pot = st.number_input("Potassium (mEq/L)", min_value=1.0, max_value=10.0, value=4.5)
    hemo = st.number_input("Hemoglobin (gms)", min_value=3.0, max_value=20.0, value=15.0)
    pcv = st.number_input("Packed Cell Volume", min_value=10.0, max_value=60.0, value=40.0)
    wbcc = st.number_input("White Blood Cell Count", min_value=1000.0, max_value=30000.0, value=8000.0)
    rbcc = st.number_input("Red Blood Cell Count", min_value=2.0, max_value=8.0, value=5.0)

st.markdown("### Medical History & Symptoms")
col4, col5, col6 = st.columns(3)

with col4:
    htn = st.selectbox("Hypertension", options=["no", "yes"])
    dm = st.selectbox("Diabetes Mellitus", options=["no", "yes"])

with col5:
    cad = st.selectbox("Coronary Artery Disease", options=["no", "yes"])
    appet = st.selectbox("Appetite", options=["good", "poor"])

with col6:
    pe = st.selectbox("Pedal Edema", options=["no", "yes"])
    ane = st.selectbox("Anemia", options=["no", "yes"])

# 3. Data Preparation
# We create a DataFrame because the Pipeline requires column names to match the training data
data = {
    'age': [age],
    'blood pressure': [bp],
    'specific gravity': [sg],
    'albumin': [al],
    'sugar': [su],
    'red blood cells': [rbc],
    'pus cell': [pc],
    'pus cell clumps': [pcc],
    'bacteria': [ba],
    'blood glucose random': [bgr],
    'blood urea': [bu],
    'serum creatinine': [sc],
    'sodium': [sod],
    'potassium': [pot],
    'hemoglobin': [hemo],
    'packed cell volume': [pcv],
    'white blood cell count': [wbcc],
    'red blood cell count': [rbcc],
    'hypertension': [htn],
    'diabetes mellitus': [dm],
    'coronary artery disease': [cad],
    'appetite': [appet],
    'pedal edema': [pe],
    'anemia': [ane]
}

input_df = pd.DataFrame(data)

# 4. Prediction Logic
st.markdown("---")
if st.button("Predict CKD Status", use_container_width=True):
    try:
        # The pipeline automatically handles Imputation, Scaling, and One-Hot Encoding
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        if prediction[0] == 1:
            st.error(f"### Prediction: Positive (CKD Risk Identified)")
            st.write(f"The model is **{prediction_proba[0][1]*100:.2f}%** confident in this prediction.")
        else:
            st.success(f"### Prediction: Negative (No CKD Detected)")
            st.write(f"The model is **{prediction_proba[0][0]*100:.2f}%** confident in this prediction.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Ensure that column names in 'data' match exactly with the ones used during model training.")

st.markdown("---")
st.caption("Note: This tool is for educational/academic purposes only and should not replace professional medical advice.")