from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from langdetect import detect
import re

# إنشاء تطبيق الـ FastAPI
app = FastAPI()

# إضافة نظام الـ CORS للسماح للفرونت إند بالتواصل مع السيرفر دون قيود أمنية متصفحية
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# تعريف الموديلات (Pipelines) وتحميلها في الذاكرة مرة واحدة عند بدء التشغيل
# موديل تحليل المشاعر
sentiment_model = pipeline("sentiment-analysis")
# موديل تصنيف المواضيع (Zero-shot) لتحديد نوع الخبر أو المقال
topic_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# تعريف شكل البيانات المستلمة من المستخدم
class UserRequest(BaseModel):
    text: str

@app.get("/")
def home():
    # رسالة بسيطة للتأكد من أن السيرفر يعمل
    return {"status": "Server is up and running successfully."}

@app.post("/analyze")
def analyze_content(request: UserRequest):
    # تنظيف النص من أي فراغات زائدة في البداية أو النهاية
    input_text = request.text.strip()

    # التحقق من أن النص يحتوي على أحرف حقيقية وليس رموزاً أو فراغات فقط
    if not input_text or not re.search('[a-zA-Zا-ي]', input_text):
        return {"error": "Invalid input. Please provide a clear and meaningful text for analysis."}

    # البدء في تحليل المشاعر
    sentiment_data = sentiment_model(input_text)[0]
    sentiment_label = sentiment_data['label']
    
    # تحويل المشاعر إلى رسالة نصية احترافية بناءً على اللوجيك الخاص بك
    if sentiment_label == "NEGATIVE":
        sentiment_feedback = "We detected a negative tone. Remember that challenges are just opportunities for growth."
    else:
        sentiment_feedback = "We detected a positive tone. Your optimism is truly inspiring and adds great value."

    # كشف لغة النص باستخدام مكتبة langdetect
    try:
        language_code = detect(input_text)
    except:
        language_code = "Unknown"

    # تصنيف موضوع النص بين قائمة من التصنيفات المقترحة
    possible_categories = ["Politics", "Sports", "Technology", "Economy", "Health"]
    classification_output = topic_model(input_text, candidate_labels=possible_categories)
    dominant_topic = classification_output['labels'][0]

    # تجميع النتائج في رد نصي احترافي ومنسق
    formatted_response = (
        f"Content Analysis: This text is classified under [{dominant_topic}]. "
        f"Detected Language: [{language_code}]. "
        f"AI Insight: {sentiment_feedback}"
    )

    return {
        "status": "success",
        "result": formatted_response,
        "raw_data": {
            "category": dominant_topic,
            "language": language_code,
            "sentiment": sentiment_label
        }
    }
