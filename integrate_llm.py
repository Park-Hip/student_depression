from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


def chat_llm(report_text, prediction):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "Lỗi: Không tìm thấy GROQ_API_KEY trong file .env. Vui lòng cấu hình."

    prompt_template = """
        VAI TRÒ CỦA BẠN:
        Bạn là "Người Bạn Đồng Hành Tâm Lý" tại một trường đại học. Bạn không phải là bác sĩ khô khan, mà là 
        một người tư vấn tâm lý cực kỳ thân thiện, vui vẻ, tích cực và thấu hiểu nỗi lòng của Gen Z.

        NHIỆM VỤ:
        Dựa vào "BÁO CÁO PHÂN TÍCH NGƯỜI DÙNG VS CỘNG ĐỒNG" được cung cấp bên dưới, hãy phân tích tình trạng của sinh 
        viên và đưa ra lời khuyên.

        DỮ LIỆU ĐẦU VÀO:
        {report_text}

        HƯỚNG DẪN PHÂN TÍCH & TRẢ LỜI:
        1.  **Giọng điệu (Tone & Voice):**
            -   Vui vẻ, ấm áp, dùng ngôn ngữ tự nhiên, gần gũi (có thể dùng emoji 🌟, 💪, 😊).
            -   Tuyệt đối không phán xét hay dọa nạt.
            -   Xưng hô: "Mình" và "Bạn" (hoặc tên nếu có).

        2.  **Cấu trúc câu trả lời:**
            -   **Chào hỏi & "Wow" Moment:** Bắt đầu bằng một lời chào năng lượng. Tìm ngay điểm sáng trong báo cáo 
            (ví dụ: CGPA cao, sự chăm chỉ) để khen ngợi thật lòng. Hãy cho họ thấy họ giỏi thế nào so với mặt bằng chung.
            -   **Góc nhìn thấu cảm:** Nhìn vào các chỉ số báo động (Academic Pressure, Financial
             Stress, Sleep Duration, Diet). So sánh nhẹ nhàng với cộng đồng để họ thấy: "À, mình đang ép bản thân quá
              mức so với mọi người".
            -   *Ví dụ:* "Mình thấy bạn đang chịu áp lực học tập cao hơn tới 77% các bạn khác, thảo nào mà điểm GPA cao
            chót vót (top đầu luôn!). Nhưng mà đổi lại, giấc ngủ và ăn uống đang 'biểu tình' kìa!"
            -   **Lời khuyên "Nhỏ mà Có võ" (Actionable Tips):** Đưa ra 2-3 lời khuyên cụ thể, dễ làm ngay lập tức.
            -   Kết hợp giải quyết vấn đề (Ví dụ: Stress tài chính + Ăn uống unhealthy -> Gợi ý meal prep giá rẻ).
            -   Nếu có "Family History of Mental Illness" hoặc "Suicidal Thoughts", hãy nhắc nhở nhẹ nhàng nhưng
                kiên quyết về việc tìm kiếm sự hỗ trợ chuyên nghiệp hoặc chia sẻ với người thân, đừng ôm đồm một mình.
            -   **Lời kết (Closing):** Một câu chốt động viên tinh thần cực kỳ tích cực.

        LƯU Ý QUAN TRỌNG:
        -   Dữ liệu cho thấy bạn ấy ngủ ít (5-6h) và ăn uống Unhealthy, lại có áp lực tài chính. Hãy khéo léo lồng
        ghép việc "Yêu bản thân" vào lời khuyên.
        -   Đừng chỉ liệt kê số liệu, hãy biến số liệu thành câu chuyện.
        BẮT ĐẦU CÂU TRẢ LỜI NGAY DƯỚI ĐÂY:
    """

    prompt = ChatPromptTemplate.from_template(template=prompt_template)

    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024
    )
    chain = prompt | llm
    try:
        report_text = report_text + f"\n--- 3. KẾT QUẢ DỰ ĐOÁN TRẦM CẢM: {prediction} ---"
        response = chain.invoke({"report_text": report_text})
        return response.content
    except Exception as e:
        return f"Xin lỗi, hệ thống đang bận. Lỗi chi tiết: {str(e)}"


if __name__ == "__main__":
    sample_report = """
========================================
 BÁO CÁO PHÂN TÍCH NGƯỜI DÙNG VS CỘNG ĐỒNG
========================================

--- 1. CHỈ SỐ ĐỊNH LƯỢNG (NUMERICAL) ---
- Age:
  + Bạn: 18.0 | Trung bình cộng đồng: 25.82
  + Bạn cao hơn 0.0% sinh viên khác.
------------------------------
- Academic Pressure:
  + Bạn: 2.0 | Trung bình cộng đồng: 3.14
  + Bạn cao hơn 17.3% sinh viên khác.
------------------------------
- CGPA:
  + Bạn: 5.0 | Trung bình cộng đồng: 7.66
  + Bạn cao hơn 0.0% sinh viên khác.
------------------------------
- Study Satisfaction:
  + Bạn: 3.0 | Trung bình cộng đồng: 2.94
  + Bạn cao hơn 40.5% sinh viên khác.
------------------------------
- Work/Study Hours:
  + Bạn: 9.0 | Trung bình cộng đồng: 7.16
  + Bạn cao hơn 55.8% sinh viên khác.
------------------------------
- Financial Stress:
  + Bạn: 3.0 | Trung bình cộng đồng: 3.14
  + Bạn cao hơn 36.5% sinh viên khác.
------------------------------

--- 2. ĐẶC ĐIỂM ĐỊNH DANH (CATEGORICAL) ---
- Gender: 'Male'
  + Có 55.7% sinh viên có cùng đặc điểm này với bạn.
- Sleep Duration: '5-6 hours'
  + Có 22.2% sinh viên có cùng đặc điểm này với bạn.
- Dietary Habits: 'Healthy'
  + Có 27.4% sinh viên có cùng đặc điểm này với bạn.
- Degree: 'BCA'
  + Có 5.1% sinh viên có cùng đặc điểm này với bạn.
  => (Đây là một đặc điểm hiếm gặp/thiểu số)
- Family History of Mental Illness: 'Yes'
  + Có 48.4% sinh viên có cùng đặc điểm này với bạn.
"""
    prediction = "Yes"
    print(chat_llm(sample_report, prediction))