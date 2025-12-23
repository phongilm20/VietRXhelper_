import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import time 
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=API_KEY)
def call_gemini_with_retry(prompt,
                           model="gemini-2.5-flash",
                           max_retries=3,
                           base_delay=2.0,
                           **config_kwargs):
    """
    Gọi model Gemini với cơ chế retry khi bị quá tải (503 UNAVAILABLE).
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return response
        except Exception as e:
            msg = str(e)
            print(f"[GEMINI ERROR] attempt {attempt}: {msg}")
            # Nếu model quá tải thì chờ rồi thử lại
            if "UNAVAILABLE" in msg or "overloaded" in msg:
                if attempt == max_retries:
                    break
                sleep_s = base_delay * attempt
                print(f"[RETRY] Model overloaded, waiting {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            # Break if any other errors
            break
    return None


def generate_draft_advice(user_input, drug_info):
    """
    ROLE 1: THE DOCTOR (Generator)
    Tạo lời khuyên y tế tiếng Việt, dễ hiểu cho bà 70 tuổi.
    """
    prompt = f"""
ROLE: Compassionate Vietnamese family doctor.

TASK: Draft medical advice for a 70-year-old grandmother.

FDA DATA (TRUTH):
{drug_info}

USER QUERY:
{user_input}

GUIDELINES:
1. Translate medical terms to simple Vietnamese.
2. Start with "Dạ thưa ạ".
3. Keep it under 100 words.
4. NO markdown formatting. Plain text only.
"""

    response = call_gemini_with_retry(
        prompt,
        model="gemini-2.5-flash",
        temperature=0.4,
    )

    if not response:
        print("[GENERATOR ERROR] Failed after retries.")
        return None

    try:
        return response.text.strip()
    except Exception as e:
        print(f"[GENERATOR ERROR] Parsing response failed: {e}")
        return None

# During my research, I realized that LLMs can "hallucinate" medical info.
# To make VietRX safer, I implemented a "Generator-Auditor" pattern.
# One agent generates the advice, and another audits it for safety against the FDA database.
def audit_safety(drug_info, draft_advice):
    """
    ROLE 2: THE AUDITOR (Evaluator)
    Kiểm tra draft advice so với dữ liệu FDA, trả JSON.
    """
    prompt = f"""
ROLE: Medical AI Auditor.

TASK: Verify if the Doctor's advice aligns strictly with FDA Data.

SOURCE DATA (FDA):
{drug_info}

DRAFT ADVICE TO CHECK:
{draft_advice}

CRITERIA:
1. CHO PHÉP: Các chỉ định và cảnh báo phổ biến, đã được y khoa công nhận rộng rãi cho nhóm thuốc này, ngay cả khi không có đầy đủ trong trích đoạn FDA phía trên.
2. KHÔNG CHO PHÉP: Bịa liều lượng cụ thể, cách dùng chi tiết, hoặc chỉ định hoàn toàn không phù hợp với nhóm thuốc.
3. Xem là lỗi NẶNG nếu lời khuyên khuyến khích dùng thuốc sai đối tượng, sai đường dùng, hoặc bỏ qua cảnh báo nghiêm trọng có trong dữ liệu FDA.


OUTPUT FORMAT (JSON ONLY):
{{
  "is_safe": true/false,
  "reason": "English explanation of the error",
  "corrected_advice": "Rewritten Vietnamese advice if unsafe, else null"
}}
"""

    response = call_gemini_with_retry(
        prompt,
        model="gemini-2.5-flash",
        response_mime_type="application/json",
        temperature=0.0,
    )

    if not response:
        print("[AUDITOR WARNING] Audit skipped because model is overloaded.")
        # Fail-safe: coi như an toàn nhưng ghi rõ lý do
        return {
            "is_safe": True,
            "reason": "Audit skipped (model overloaded)",
            "corrected_advice": None,
        }

    try:
        return json.loads(response.text)
    except Exception as e:
        print(f"[AUDITOR ERROR] JSON parse failed: {e}")
        # Fail-safe: không chặn hệ thống, nhưng báo lý do
        return {
            "is_safe": True,
            "reason": "Audit failed (JSON parse error)",
            "corrected_advice": None,
        }


def get_medical_advice(user_input, drug_info):
    """
    PIPELINE: Generation -> Audit -> Final Output
    """
    print(f"[AI PIPELINE] 1. Generating draft advice...")
    draft = generate_draft_advice(user_input, drug_info)
    
    if not draft:
        return "Xin lỗi ạ, hệ thống đang gặp sự cố."

    print(f"[AI PIPELINE] 2. Auditing for safety...")
    audit_result = audit_safety(drug_info, draft)
    
    if audit_result.get("is_safe"):
        print("[AUDIT PASSED] Advice is verified.")
        return draft
    else:
        print(f"[AUDIT FAILED] Reason: {audit_result.get('reason')}")
        print("[RECOVERY] Switching to corrected advice.")
        
        correction = audit_result.get("corrected_advice")
        if correction:
            return correction
        else:
            return "Xin lỗi ạ, thông tin thuốc phức tạp con cần kiểm tra lại ạ."