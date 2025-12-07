import pandas as pd
import matplotlib.pyplot as plt


def analyze_user_vs_population(user_input, df, language="vi"):
    """
    So sánh input của user với dataset và vẽ biểu đồ.
    Trả về:
        - report_text (str): Văn bản báo cáo để gửi cho LLM.
        - fig (matplotlib.figure): Biểu đồ radar để hiển thị trên UI.
        - comparison_data (dict): Dữ liệu so sánh để hiển thị UI tùy chỉnh.
    """

    # Define text templates based on language
    texts = {
        "vi": {
            "header": "BÁO CÁO PHÂN TÍCH NGƯỜI DÙNG VS CỘNG ĐỒNG",
            "section1": "1. CHỈ SỐ ĐỊNH LƯỢNG (NUMERICAL)",
            "section2": "2. ĐẶC ĐIỂM ĐỊNH DANH (CATEGORICAL)",
            "you": "Bạn",
            "avg": "Trung bình cộng đồng",
            "higher": "Bạn cao hơn",
            "students": "sinh viên khác",
            "same_trait": "sinh viên có cùng đặc điểm này với bạn",
            "rare": "(Đây là một đặc điểm hiếm gặp/thiểu số)",
            "chart_you": "Bạn",
            "chart_avg": "Trung bình cộng đồng"
        },
        "en": {
            "header": "USER VS POPULATION ANALYSIS REPORT",
            "section1": "1. NUMERICAL METRICS",
            "section2": "2. CATEGORICAL CHARACTERISTICS",
            "you": "You",
            "avg": "Community Average",
            "higher": "You are higher than",
            "students": "of other students",
            "same_trait": "of students share this trait with you",
            "rare": "(This is a rare/minority trait)",
            "chart_you": "You",
            "chart_avg": "Community Avg"
        }
    }

    t = texts.get(language, texts["en"])

    report_text = ""
    report_text += "\n" + "=" * 40 + "\n"
    report_text += f" {t['header']}\n"
    report_text += "=" * 40 + "\n\n"

    # ---------------------------------------------------------
    # PHẦN 1: SO SÁNH SỐ HỌC (NUMERICAL) - Dùng Percentile
    # ---------------------------------------------------------
    numeric_cols = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction',
                    'Work/Study Hours', 'Financial Stress']

    comparison_data = {
        "numerical": [],
        "categorical": []
    }

    report_text += f"--- {t['section1']} ---\n"
    for col in numeric_cols:
        if col in df.columns and col in user_input:
            user_val = float(user_input[col])
            pop_mean = df[col].mean()

            # Tính Percentile
            percentile = (df[col] < user_val).mean() * 100

            report_text += f"- {col}:\n"
            report_text += f"  + {t['you']}: {user_val} | {t['avg']}: {pop_mean:.2f}\n"
            report_text += f"  + {t['higher']} {percentile:.1f}% {t['students']}.\n"
            report_text += "-" * 30 + "\n"

            comparison_data["numerical"].append({
                "metric": col,
                "user": user_val,
                "avg": pop_mean,
                "percentile": percentile
            })

    # ---------------------------------------------------------
    # PHẦN 2: SO SÁNH ĐỊNH DANH (CATEGORICAL)
    # ---------------------------------------------------------
    categorical_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Degree',
                        'Suicidal Thoughts', 'Family History of Mental Illness']

    report_text += f"\n--- {t['section2']} ---\n"
    for col in categorical_cols:
        if col in df.columns and col in user_input:
            user_val = user_input[col]

            count = df[df[col] == user_val].shape[0]
            total = df.shape[0]
            percentage = (count / total) * 100

            report_text += f"- {col}: '{user_val}'\n"
            report_text += f"  + {percentage:.1f}% {t['same_trait']}.\n"

            if percentage < 10:
                report_text += f"  => {t['rare']}\n"

            comparison_data["categorical"].append({
                "feature": col,
                "value": user_val,
                "percentage": percentage
            })

    # Create a modern radar chart
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('#fafbfc')

    categories = numeric_cols
    user_values = [float(user_input[col]) for col in categories]
    pop_means = [df[col].mean() for col in categories]

    # Normalize values to 0-10 scale for better visualization
    max_vals = [df[col].max() for col in categories]
    user_values_norm = [user_values[i] / max_vals[i] * 10 if max_vals[i] > 0 else 0 for i in range(len(user_values))]
    pop_means_norm = [pop_means[i] / max_vals[i] * 10 if max_vals[i] > 0 else 0 for i in range(len(pop_means))]

    angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
    angles += angles[:1]
    user_values_norm += user_values_norm[:1]
    pop_means_norm += pop_means_norm[:1]

    # Plot with better styling
    ax.plot(angles, user_values_norm, linewidth=2.5, linestyle='solid', label=t['chart_you'], color='#3b82f6',
            marker='o', markersize=6)
    ax.plot(angles, pop_means_norm, linewidth=2.5, linestyle='solid', label=t['chart_avg'], color='#ef4444', marker='s',
            markersize=6)
    ax.fill(angles, user_values_norm, color='#3b82f6', alpha=0.15)
    ax.fill(angles, pop_means_norm, color='#ef4444', alpha=0.15)

    # Styling
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, weight='bold', color='#374151')
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], size=9, color='#6b7280')
    ax.grid(color='#d1d5db', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.spines['polar'].set_color('#9ca3af')
    ax.spines['polar'].set_linewidth(1.5)

    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), frameon=True, shadow=True, fontsize=11)
    plt.tight_layout()

    return report_text, fig, comparison_data


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
    report_text, fig, data = analyze_user_vs_population(user_input, df, language="en")
    print(report_text)

