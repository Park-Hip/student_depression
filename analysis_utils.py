from math import pi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

    comparison_data = {'Metric': [], 'User': [], 'Average': []}

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

            comparison_data['Metric'].append(col)
            comparison_data['User'].append(user_val)
            comparison_data['Average'].append(pop_mean)

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

    # ---------------------------------------------------------
    # PHẦN 3: VẼ BIỂU ĐỒ (RADAR CHART)
    # ---------------------------------------------------------
    categories = comparison_data['Metric']
    N = len(categories)

    def scale_values(values, metrics, data_frame):
        scaled = []
        for v, m in zip(values, metrics):
            min_v = data_frame[m].min()
            max_v = data_frame[m].max()
            if max_v - min_v == 0:
                scaled.append(0)
            else:
                scaled.append((v - min_v) / (max_v - min_v))
        return scaled

    user_values_scaled = scale_values(comparison_data['User'], categories, df)
    avg_values_scaled = scale_values(comparison_data['Average'], categories, df)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    user_values_scaled += user_values_scaled[:1]
    avg_values_scaled += avg_values_scaled[:1]

    # Tạo figure mới thay vì plt.show() trực tiếp
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    plt.xticks(angles[:-1], categories, color='grey', size=8)

    ax.plot(angles, user_values_scaled, linewidth=2, linestyle='solid', label='Bạn', color='blue')
    ax.fill(angles, user_values_scaled, 'blue', alpha=0.1)

    ax.plot(angles, avg_values_scaled, linewidth=2, linestyle='solid', label='Trung bình', color='red')
    ax.fill(angles, avg_values_scaled, 'red', alpha=0.1)

    plt.title('So sánh Tổng quan: Bạn vs Cộng đồng', size=12, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Trả về cả text report và figure
    return report_text, fig