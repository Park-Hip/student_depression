import streamlit as st
import pandas as pd
import pickle
import os
from dotenv import load_dotenv
import time

# Import custom modules
from integrate_llm import chat_llm
from analysis import analyze_user_vs_population
from make_inference import make_inference

# 1. Load environment variables
load_dotenv()

# Page Config - Must be the first Streamlit command
st.set_page_config(
    page_title="Student Mental Health Advisor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR CLEAN LOOK ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1E88E5;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- LOAD DATASET ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/clean_df.csv")
        return df
    except FileNotFoundError:
        st.error("Data file 'data/clean_df.csv' not found.")
        return pd.DataFrame()


df = load_data()

# --- SESSION STATE INITIALIZATION ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'report_text' not in st.session_state:
    st.session_state.report_text = ""
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None
if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'advice' not in st.session_state:
    st.session_state.advice = ""

# --- TEXT RESOURCES ---
TEXT = {
    "English": {
        "title": "Student Mental Health Advisor",
        "subtitle": "AI-Powered Analysis & Prediction",
        "tabs": ["📝 Input Profile", "📊 Analysis Dashboard", "💬 AI Consultant"],
        "submit": "Analyze Profile",
        "success": "Analysis Complete!",
        "metric_label": "Your Value",
        "avg_label": "Community Avg",
        "percentile": "Higher than",
        "prediction_card": "Depression Risk Prediction",
        "advice_card": "Personalized Advice",
        "loading_advice": "AI is generating advice...",
        "loading_pred": "Running prediction model...",
        "model_select": "Select Model",
        "models": {"Random Forest": "RF", "Logistic Regression": "LR", "K-Nearest Neighbors": "KNN"}
    },
    "Tiếng Việt": {
        "title": "Tư vấn Sức khỏe Tinh thần Sinh viên",
        "subtitle": "Phân tích & Dự đoán bằng AI",
        "tabs": ["📝 Nhập Hồ sơ", "📊 Bảng Phân tích", "💬 Tư vấn AI"],
        "submit": "Phân tích Hồ sơ",
        "success": "Hoàn tất phân tích!",
        "metric_label": "Chỉ số của bạn",
        "avg_label": "Trung bình cộng đồng",
        "percentile": "Cao hơn",
        "prediction_card": "Dự đoán Nguy cơ Trầm cảm",
        "advice_card": "Lời khuyên Cá nhân hóa",
        "loading_advice": "AI đang soạn lời khuyên...",
        "loading_pred": "Đang chạy mô hình dự đoán...",
        "model_select": "Chọn Mô hình",
        "models": {"Random Forest": "RF", "Logistic Regression": "LR", "K-Nearest Neighbors": "KNN"}
    }
}

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=100)
    st.title("Student Mental Health Advisor")
    st.markdown("---")

    language = st.selectbox("Language / Ngôn ngữ", ["English", "Tiếng Việt"])
    t = TEXT[language]

    # Model Selection
    model_name = st.selectbox(t["model_select"], list(t["models"].keys()))
    model_code = t["models"][model_name]
    st.session_state.model_code = model_code

    st.info("This application uses AI to analyze your mental health status based on academic and lifestyle factors.")
    st.markdown("---")
    st.caption("© 2025 Student Health Project")

# --- MAIN CONTENT ---
# Remove the custom header from the body since we are moving it to the top bar area via CSS/Hack or just keeping it clean.
# The user wants "Student Mental Health Advisor" on top aligned with "Deploy".
# We can't easily inject into the Streamlit toolbar, but we can make the title look like a header bar.

st.markdown("""
    <style>
        /* Adjust the default Streamlit header to be less intrusive but visible */
        header[data-testid="stHeader"] {
            background-color: var(--primary-background-color);
            z-index: 99999;
        }

        /* Adjust main content padding */
        .block-container {
            padding-top: 3rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Use session state for navigation
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = t["tabs"][0]

# Use st.tabs as requested
tab1, tab2, tab3 = st.tabs(t["tabs"])

# --- TAB 1: INPUT FORM ---
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Personal Info")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        age = st.number_input("Age", min_value=16, max_value=40, value=20, key="age")
        degree_options = [
            "Class 12", "B.Ed", "B.Com", "B.Arch", "BCA", "MSc", "B.Tech", "MCA", "M.Tech",
            "BHM", "BSc", "M.Ed", "B.Pharm", "M.Com", "BBA", "MBBS", "LLB", "BE", "BA",
            "M.Pharm", "MD", "MBA", "MA", "PhD", "LLM", "MHM", "ME", "Others"
        ]
        degree = st.selectbox("Degree", degree_options, key="degree")
        year = st.selectbox("Year of Study", [1, 2, 3, 4, 5, 6], key="year")

        st.markdown("### Lifestyle")
        sleep = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],
                             key="sleep")
        diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"], key="diet")

    with col2:
        st.markdown("### Academic & Financial")
        cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=8.0, step=0.1, key="cgpa")
        study_hours = st.number_input("Work/Study Hours per day", min_value=0, max_value=24, value=6, key="study_hours")
        academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3, key="academic_pressure")
        financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3, key="financial_stress")
        study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3, key="study_satisfaction")

        st.markdown("### History")
        suicidal = st.selectbox("Ever had suicidal thoughts?", ["Yes", "No"], key="suicidal")
        fam_hist = st.selectbox("Family history of mental illness?", ["Yes", "No"], key="fam_hist")

    with col1:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(t["submit"], type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                # Prepare Input Dict
                user_input = {
                    'Gender': gender,
                    'Age': age,
                    'City': 'City',  # Dummy
                    'Profession': 'Student',  # Dummy
                    'Academic Pressure': academic_pressure,
                    'Work Pressure': 0,  # Dummy
                    'CGPA': cgpa,
                    'Study Satisfaction': study_satisfaction,
                    'Job Satisfaction': 0,  # Dummy
                    'Sleep Duration': sleep,
                    'Dietary Habits': diet,
                    'Degree': degree,
                    'Have you ever had suicidal thoughts ?': suicidal,
                    # Note: Check exact column name in CSV later if needed
                    'Work/Study Hours': study_hours,
                    'Financial Stress': financial_stress,
                    'Family History of Mental Illness': fam_hist
                }

                # 1. Analysis
                # Note: We need to map keys to match what analysis.py expects if they differ.
                # Based on reading analysis.py, it expects keys like 'Suicidal Thoughts' but input might be different.
                # Let's normalize keys for analysis.py
                analysis_input = user_input.copy()
                analysis_input['Suicidal Thoughts'] = suicidal

                lang_code = "en" if language == "English" else "vi"
                report_text, fig, comparison_data = analyze_user_vs_population(analysis_input, df, language=lang_code)

                st.session_state.report_text = report_text
                st.session_state.fig = fig
                st.session_state.comparison_data = comparison_data

                # 2. Prediction
                # Map inputs to model expected format
                # Note: make_inference expects specific feature names.
                # We'll assume make_inference handles the mapping or we pass the raw dict.
                # Let's look at make_inference usage in streamlit_demo.py... it just passes user_input.
                # We need to ensure keys match what the model expects.
                # For now, we pass user_input.

            # Fix key for suicidal thoughts which often has weird spacing in datasets
            user_input['Have you ever had suicidal thoughts ?'] = suicidal

            # Use selected model
            selected_model = st.session_state.get('model_code', 'RF')
            pred_result = make_inference(selected_model, user_input)
            st.session_state.prediction_result = pred_result  # 3. LLM Advice
            advice = chat_llm(report_text, str(pred_result), language=lang_code)
            st.session_state.advice = advice

            st.session_state.analysis_done = True
            # st.session_state.active_tab = t["tabs"][1] # Cannot switch tabs programmatically with st.tabs
            st.success(t["success"])
            st.info("Please switch to the 'Analysis Dashboard' tab to view results.")
            # time.sleep(1)
            # st.rerun()

# --- TAB 2: ANALYSIS DASHBOARD ---
with tab2:
    if not st.session_state.analysis_done:
        st.info("Please submit your profile in the 'Input Profile' tab first.")
    else:
        st.markdown("### 📊 Comparative Analysis")

        # Radar Chart
        col_chart, col_stats = st.columns([1, 1])

        with col_chart:
            st.pyplot(st.session_state.fig)

        with col_stats:
            st.markdown("#### Key Metrics vs Community")
            data = st.session_state.comparison_data

            if data:
                # Use a grid layout for metrics to save space
                metric_cols = st.columns(2)
                for i, item in enumerate(data['numerical']):
                    with metric_cols[i % 2]:
                        delta = item['user'] - item['avg']
                        st.metric(
                            label=item['metric'],
                            value=f"{item['user']}",
                            delta=f"{delta:.2f} vs Avg",
                            delta_color="inverse" if item['metric'] in ['Financial Stress',
                                                                        'Academic Pressure'] else "normal"
                        )
                        st.progress(min(item['percentile'] / 100, 1.0))
                        st.caption(f"{t['percentile']} {item['percentile']:.1f}%")
                        st.markdown("---")

        st.markdown("### 🧬 Categorical Insights")
        if data:
            cat_cols = st.columns(3)
            for i, item in enumerate(data['categorical']):
                with cat_cols[i % 3]:
                    st.markdown(f"**{item['feature']}**")
                    st.info(f"{item['value']}")
                    st.caption(f"Shared by {item['percentage']:.1f}%")

# --- TAB 3: AI CONSULTANT ---
with tab3:
    if not st.session_state.analysis_done:
        st.info("Please submit your profile first to get AI advice.")
    else:
        col_pred, col_advice = st.columns([1, 2])

        with col_pred:
            st.markdown(f"### {t['prediction_card']}")
            result = st.session_state.prediction_result

            if result == "Yes":
                st.error("⚠️ High Risk of Depression")
                st.markdown("The model predicts a potential risk. Please consult a professional.")
            else:
                st.success("✅ Low Risk")
                st.markdown("The model predicts you are likely doing well.")

        with col_advice:
            st.markdown(f"### {t['advice_card']}")
            st.markdown(st.session_state.advice)

            st.download_button(
                label="Download Report",
                data=st.session_state.advice,
                file_name="mental_health_report.txt",
                mime="text/plain"
            )

