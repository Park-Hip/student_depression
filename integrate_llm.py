from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load biến môi trường ngay khi import module này
load_dotenv()


def chat_llm(report_text):
    """
    Hàm gọi LLM để nhận lời khuyên dựa trên báo cáo phân tích.
    """
    # Lấy API Key từ biến môi trường
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "Lỗi: Không tìm thấy GROQ_API_KEY trong file .env. Vui lòng cấu hình."

    prompt_template = """
        VAI TRÒ CỦA BẠN:
        Bạn là "Người Bạn Đồng Hành Tâm Lý" (AI Buddy) tại một trường đại học. Bạn không phải là bác sĩ khô khan, mà là 
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
            -   **Góc nhìn thấu cảm (The Reality Check):** Nhìn vào các chỉ số báo động (Academic Pressure, Financial
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
        response = chain.invoke({"report_text": report_text})
        return response.content
    except Exception as e:
        return f"Xin lỗi, hệ thống đang bận. Lỗi chi tiết: {str(e)}"


if __name__ == "__main__":
    sample_report = """
    - Academic Pressure: Bạn: 4.5 (Cao hơn 77% sv khác)
    - Sleep: 5-6 hours (Thiếu ngủ)
    """
    print(chat_llm(sample_report))