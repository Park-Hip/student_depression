import pandas as pd

def analyze_user_vs_population(user_input, df):
    """
    So sánh input của user với dataset và vẽ biểu đồ.
    Trả về:
        - report_text (str): Văn bản báo cáo để gửi cho LLM.
        - fig (matplotlib.figure): Biểu đồ radar để hiển thị trên UI.
    """
    report_text = ""
    report_text += "\n" + "=" * 40 + "\n"
    report_text += " BÁO CÁO PHÂN TÍCH NGƯỜI DÙNG VS CỘNG ĐỒNG\n"
    report_text += "=" * 40 + "\n\n"

    # ---------------------------------------------------------
    # PHẦN 1: SO SÁNH SỐ HỌC (NUMERICAL) - Dùng Percentile
    # ---------------------------------------------------------
    numeric_cols = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction',
                    'Work/Study Hours', 'Financial Stress']

    report_text += "--- 1. CHỈ SỐ ĐỊNH LƯỢNG (NUMERICAL) ---\n"
    for col in numeric_cols:
        if col in df.columns and col in user_input:
            user_val = float(user_input[col])
            pop_mean = df[col].mean()

            # Tính Percentile
            percentile = (df[col] < user_val).mean() * 100

            report_text += f"- {col}:\n"
            report_text += f"  + Bạn: {user_val} | Trung bình cộng đồng: {pop_mean:.2f}\n"
            report_text += f"  + Bạn cao hơn {percentile:.1f}% sinh viên khác.\n"
            report_text += "-" * 30 + "\n"

    # ---------------------------------------------------------
    # PHẦN 2: SO SÁNH ĐỊNH DANH (CATEGORICAL)
    # ---------------------------------------------------------
    categorical_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Degree',
                        'Suicidal Thoughts', 'Family History of Mental Illness']

    report_text += "\n--- 2. ĐẶC ĐIỂM ĐỊNH DANH (CATEGORICAL) ---\n"
    for col in categorical_cols:
        if col in df.columns and col in user_input:
            user_val = user_input[col]

            count = df[df[col] == user_val].shape[0]
            total = df.shape[0]
            percentage = (count / total) * 100

            report_text += f"- {col}: '{user_val}'\n"
            report_text += f"  + Có {percentage:.1f}% sinh viên có cùng đặc điểm này với bạn.\n"

            if percentage < 10:
                report_text += "  => (Đây là một đặc điểm hiếm gặp/thiểu số)\n"

    return report_text

if "__main__" == __name__:
    user_input = {
        'Gender': "Male",
        'Age': 18,
        'Academic Pressure': 2,
        'CGPA': 5,
        'Study Satisfaction': 3,
        'Sleep Duration': "5-6 hours",
        'Dietary Habits': "Healthy",
        'Degree': "BCA",
        'Have you ever had suicidal thoughts ?': "Yes",
        'Work/Study Hours': 9,
        'Financial Stress': 3,
        'Family History of Mental Illness': "Yes"
    }
    df = pd.read_csv('data/clean_df.csv')
    report_text = analyze_user_vs_population(user_input, df)
    print(report_text)

