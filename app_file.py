import streamlit as st
import joblib
import numpy as np

# Load mô hình đã huấn luyện
model = joblib.load("model.pkl")

st.title("🧬 Dự đoán bệnh thận mãn tính (CKD)")

st.markdown("### 📋 Nhập thông tin bệnh nhân")

# Các input số
age = st.number_input("Số tuổi (age)",min_value=0,max_value= 150,step=1)
bp = st.number_input("Huyết áp (blood pressure)", min_value=40, max_value=200, step=1)
sg = st.selectbox("Trọng lượng riêng (specific gravity)", options=[1.005, 1.010, 1.015, 1.020, 1.025])
al = st.selectbox("Albumin", options=[0, 1, 2, 3, 4, 5])
su = st.selectbox("Đường (sugar)", options=[0, 1, 2, 3, 4, 5])

# Các input phân loại
rbc = st.selectbox("Hồng cầu (red blood cells)", options=["normal", "abnormal"])
pc = st.selectbox("Tế bào mủ (pus cell)", options=["normal", "abnormal"])
pcc = st.selectbox("Tế bào mủ vón cục (pus cell clumps)", options=["notpresent", "present"])
ba = st.selectbox("Vi khuẩn (bacteria)", options=["notpresent", "present"])

# Các input số tiếp
bgr = st.number_input("Đường huyết random (blood glucose random)", min_value=20.0, max_value=500.0)
bu = st.number_input("Urê máu (blood urea)", min_value=1.0, max_value=400.0)
sc = st.number_input("Creatinine", min_value=0.1, max_value=20.0)
sod = st.number_input("Natri (sodium)", min_value=100.0, max_value=200.0)
pot = st.number_input("Kali (potassium)", min_value=1.0, max_value=10.0)
hemo = st.number_input("Hemoglobin", min_value=3.0, max_value=20.0)
pcv = st.number_input("Thể tích hồng cầu (packed cell volume)", min_value=10.0, max_value=60.0)
wbcc = st.number_input("Số lượng bạch cầu (white blood cell count)", min_value=1000.0, max_value=30000.0)
rbcc = st.number_input("Số lượng hồng cầu (red blood cell count)", min_value=2.0, max_value=8.0)

# Các input yes/no
htn = st.selectbox("Tăng huyết áp (hypertension)", options=["no", "yes"])
dm = st.selectbox("Đái tháo đường (diabetes mellitus)", options=["no", "yes"])
cad = st.selectbox("Bệnh mạch vành (coronary artery disease)", options=["no", "yes"])
appet = st.selectbox("Thèm ăn (appetite)", options=["good", "poor"])
pe = st.selectbox("Phù chân (pedal edema)", options=["no", "yes"])
ane = st.selectbox("Thiếu máu (anemia)", options=["no", "yes"])

# Xử lý input: ánh xạ nhãn phân loại sang số
label_map = {
    "normal": 0, "abnormal": 1,
    "notpresent": 0, "present": 1,
    "no": 0, "yes": 1,
    "good": 0, "poor": 1
}

input_data = [
    age,bp, sg, al, su,
    label_map[rbc], label_map[pc], label_map[pcc], label_map[ba],
    bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc,
    label_map[htn], label_map[dm], label_map[cad],
    label_map[appet], label_map[pe], label_map[ane]
]

input_array = np.array([input_data])  # reshape cho đúng định dạng

# Nút Dự đoán
if st.button("Dự đoán"):
    result = model.predict(input_array)
    if result[0] == 1:
        st.error("⚠️ Bệnh nhân có khả năng bị bệnh thận mãn tính (CKD).")
    else:
        st.success("✅ Bệnh nhân không bị bệnh thận mãn tính.")

