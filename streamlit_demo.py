import streamlit as st
import pandas as pd
import pickle
import os
from dotenv import load_dotenv

# Import các module tùy chỉnh
from integrate_llm import chat_llm
from analysis import analyze_user_vs_population
from make_inference import make_inference

# 1. Load biến môi trường
load_dotenv()

# Cấu hình trang
st.set_page_config(page_title="Dự đoán & Tư vấn Sức khỏe Tinh thần", layout="wide")


# --- LOAD DATASET (Giả lập) ---
# Trong thực tế, bạn cần file 'data/clean_df.csv' ở cùng thư mục
@st.cache_data
def load_data():
    try:
        # Thay thế bằng đường dẫn thực tế file csv của bạn
        df = pd.read_csv("data/clean_df.csv")
        # Thực hiện một số bước clean cơ bản nếu cần để khớp tên cột
        return df
    except FileNotFoundError:
        st.error("Không tìm thấy file dữ liệu 'data/clean_df.csv'. Vui lòng kiểm tra lại.")
        return pd.DataFrame()  # Trả về DF rỗng để tránh crash


df = load_data()

# --- SIDEBAR: CẤU HÌNH ---
st.sidebar.title("Cấu hình Mô hình")

# 2. Dropdown chọn Model
model_options = {
    "K-Nearest Neighbors (KNN)": "models/KNN.pkl",
    "Logistic Regression (LR)": "models/LR.pkl",
    "Random Forest (RF)": "models/RF.pkl"
}

selected_model_name = st.sidebar.selectbox("Chọn mô hình dự đoán:", list(model_options.keys()))
model_path = model_options[selected_model_name]

# Load model được chọn
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.sidebar.success(f"Đã tải mô hình: {selected_model_name}")
except FileNotFoundError:
    st.sidebar.error(f"Không tìm thấy file model tại: {model_path}")
    model = None

# --- GIAO DIỆN CHÍNH ---
st.title("🧠 Ứng dụng Tư vấn Sức khỏe Tinh thần Sinh viên")
st.markdown("Nhập thông tin của bạn để nhận dự đoán Depression và lời khuyên từ AI.")

# Danh sách Degree (Bằng cấp)
degree_options = [
    "Class 12", "B.Ed", "B.Com", "B.Arch", "BCA", "MSc", "B.Tech", "MCA", "M.Tech",
    "BHM", "BSc", "M.Ed", "B.Pharm", "M.Com", "BBA", "MBBS", "LLB", "BE", "BA",
    "M.Pharm", "MD", "MBA", "MA", "PhD", "LLM", "MHM", "ME", "Others"
]

# Form nhập liệu
with st.form("user_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Giới tính", ["Female", "Male"])
        age = st.number_input("Tuổi", min_value=17, max_value=35, value=20)
        # Thay thế text input bằng selectbox cho Degree
        degree = st.selectbox("Học vị / Bằng cấp", degree_options)
        year_study = st.selectbox("Năm học", ["Year 1", "Year 2", "Year 3", "Year 4"])
        cgpa = st.number_input("Điểm CGPA (thang 10)", 0.0, 10.0, 8.0)

    with col2:
        sleep_dur = st.selectbox("Thời gian ngủ", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
        diet = st.selectbox("Thói quen ăn uống", ["Healthy", "Moderate", "Unhealthy"])
        suicidal = st.selectbox("Từng có suy nghĩ tự tử?", ["Yes", "No"])
        fam_history = st.selectbox("Gia đình có tiền sử bệnh tâm lý?", ["Yes", "No"])
        financial_stress = st.slider("Áp lực tài chính (1-5)", 1, 5, 3)
        academic_pressure = st.slider("Áp lực học tập (1-5)", 1, 5, 3)
        study_hours = st.slider("Giờ học/làm việc mỗi ngày", 0, 16, 6)
        study_sat = st.slider("Mức độ hài lòng việc học (1-5)", 1, 5, 3)

    submitted = st.form_submit_button("🚀 Phân tích & Nhận lời khuyên")

if submitted and not df.empty:
    # 1. Chuẩn bị dữ liệu cho hàm Phân Tích (Analysis)
    # Mapping dữ liệu nhập vào khớp với tên cột trong CSV gốc để so sánh
    user_data_analysis = {
        'Gender': gender,
        'Age': age,
        'Academic Pressure': academic_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_sat,
        'Sleep Duration': sleep_dur,
        'Dietary Habits': diet,
        'Degree': degree,  # Cập nhật giá trị Degree từ dropdown
        'Suicidal Thoughts': suicidal,
        'Work/Study Hours': study_hours,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': fam_history
    }

    # 2. Gọi hàm phân tích & Vẽ biểu đồ
    st.subheader("📊 Kết quả Phân tích So sánh")

    # 4. Call function in analyze_data
    report_text, fig = analyze_user_vs_population(user_data_analysis, df)

    col_chart, col_text = st.columns([1, 1])

    with col_chart:
        st.pyplot(fig)  # Hiển thị biểu đồ Radar

    with col_text:
        st.markdown("### Tóm tắt chỉ số")
        # Hiển thị một phần report dạng text thô nếu muốn
        st.text(report_text)
        st.info("Biểu đồ bên trái cho thấy sự khác biệt giữa bạn (Xanh) và trung bình cộng đồng (Đỏ).")

    # 3. Dự đoán với Model ML (Optional - demo logic)
    if model:
        st.subheader("🔮 Dự đoán từ AI (Machine Learning)")
        # LƯU Ý: Ở đây bạn cần code tiền xử lý (Encoder/Scaler) để biến đổi user_data_analysis
        # thành dạng số mà model.predict() hiểu được.
        # Ví dụ: prediction = model.predict([processed_data])
        st.warning("Đang chờ module Preprocessing để chạy dự đoán chính xác...")
        # st.write(f"Kết quả dự đoán: {prediction}")

    # 4. Gọi LLM để tư vấn
    st.subheader("💬 Lời khuyên từ AI Buddy")
    with st.spinner("AI đang suy nghĩ và viết thư cho bạn..."):
        # 3. Call integrate_llm
        advice = chat_llm(report_text)

    st.markdown(advice)

elif submitted and df.empty:
    st.error("Chưa load được dữ liệu để so sánh.")