from fastapi import FastAPI
from fastapi.responses import HTMLResponse  # 1. تم إضافة هذا السطر
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from langdetect import detect
import re

# إنشاء تطبيق الـ FastAPI
app = FastAPI()

# إضافة نظام الـ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# تعريف الموديلات (Pipelines)
print("Loading models... please wait.") # رسالة عشان تعرف في اللوجز إنه بيحمل
sentiment_model = pipeline("sentiment-analysis")
topic_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# تعريف شكل البيانات
class UserRequest(BaseModel):
    text: str

# 2. تعديل الـ Home ليعرض ملف HTML بدلاً من رسالة JSON
@app.get("/", response_class=HTMLResponse)
def home():
    # هنا السيرفر بيقرأ ملف index.html اللي أنت رفعته وبيرجعه للمتصفح
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found. Please upload the file.</h1>"

@app.post("/analyze")
def analyze_content(request: UserRequest):
    # تنظيف النص
    input_text = request.text.strip()

    # التحقق من المدخلات
    if not input_text or not re.search('[a-zA-Zا-ي]', input_text):
        return {"error": "Invalid input. Please provide a clear and meaningful text for analysis."}

    # تحليل المشاعر
    sentiment_data = sentiment_model(input_text)[0]
    sentiment_label = sentiment_data['label']
    
    # صياغة الرد
    if sentiment_label == "NEGATIVE":
        sentiment_feedback = "We detected a negative tone. Remember that challenges are just opportunities for growth."
    else:
        sentiment_feedback = "We detected a positive tone. Your optimism is truly inspiring and adds great value."

    # كشف اللغة
    try:
        language_code = detect(input_text)
    except:
        language_code = "Unknown"

    # تصنيف الموضوع
    possible_categories = ["Politics", "Sports", "Technology", "Economy", "Health"]
    classification_output = topic_model(input_text, candidate_labels=possible_categories)
    dominant_topic = classification_output['labels'][0]

    # تجميع النتائج
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
    
