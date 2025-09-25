# analyzer.py

import torch
from transformers import pipeline
import streamlit as st

# --- 1. تحديد الجهاز تلقائياً (لا تغيير هنا) ---
if torch.cuda.is_available():
    DEVICE = 0
    print("✅ GPU detected.")
else:
    DEVICE = -1
    print("⚠️ No GPU detected, using CPU.")


# --- 2. دوال تحميل النماذج (مع إضافة النموذج الجديد) ---
@st.cache_resource
def load_gender_model():
    print(f"⏳ Loading Gender Model...")
    return pipeline("text-classification", model="./models/gender_classifier_final", device=DEVICE)

@st.cache_resource
def load_sentiment_model():
    print(f"⏳ Loading Sentiment Model...")
    return pipeline("text-classification", model="./models/sentiment_analyzer_final", device=DEVICE)

# --- ✨ إضافة دالة تحميل نموذج اللهجات ✨ ---
@st.cache_resource
def load_dialect_model():
    print(f"⏳ Loading Dialect Model...")
    return pipeline("text-classification", model="./models/dialect_classifier_final", device=DEVICE)


# --- 3. دوال التنبؤ (مع إضافة الدالة الجديدة) ---
def predict_gender(text_list: list) -> list:
    gender_model = load_gender_model()
    batch_size = 32 if DEVICE == 0 else 8
    results = gender_model(text_list, batch_size=batch_size, truncation=True, padding=True, top_k=1)
    return [result[0]['label'] for result in results]

def predict_sentiment(text_list: list) -> list:
    sentiment_model = load_sentiment_model()
    batch_size = 32 if DEVICE == 0 else 8
    results = sentiment_model(text_list, batch_size=batch_size, truncation=True, padding=True, top_k=1)
    return [result[0]['label'] for result in results]

# --- ✨ إضافة دالة التنبؤ باللهجة ✨ ---
def predict_dialect(text_list: list) -> list:
    dialect_model = load_dialect_model()
    batch_size = 32 if DEVICE == 0 else 8
    results = dialect_model(text_list, batch_size=batch_size, truncation=True, padding=True, top_k=1)
    return [result[0]['label'] for result in results]
