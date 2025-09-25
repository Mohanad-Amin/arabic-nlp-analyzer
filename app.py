# app.py

import streamlit as st
import pandas as pd
# استيراد الدالة الجديدة
from analyzer import predict_gender, predict_sentiment, predict_dialect

@st.cache_data
def convert_df_to_csv(df):
    # utf-8-sig يضمن دعم اللغة العربية عند فتح الملف في Excel
    return df.to_csv(index=False).encode('utf-8-sig')

st.set_page_config(page_title="أداة تحليل البيانات النصية", layout="wide", page_icon="📂")

st.title("📂 أداة تحليل البيانات النصية من ملفات Excel")
st.write("ارفع ملف Excel، اختر العمود الذي يحتوي على النص، وحدد نوع التحليل المطلوب.")

uploaded_file = st.file_uploader("ارفع ملف Excel هنا", type=["xlsx"])

if uploaded_file is not None:
    try:
        df_original = pd.read_excel(uploaded_file)
        df = df_original.copy()
        st.subheader("عينة من البيانات التي تم رفعها:")
        st.dataframe(df.head())
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text_column = st.selectbox(
                "**الخطوة 1: اختر العمود الذي يحتوي على النص**",
                df.columns
            )
        
        with col2:
            analysis_options = st.multiselect(
                "**الخطوة 2: اختر نوع التحليل**",
                ["تحليل الجنس", "تحليل المشاعر", "تصنيف اللهجة"],
                default=[]
            )

        st.markdown("")
        if st.button("🚀 ابدأ التحليل الآن", type="primary", use_container_width=True):
            if text_column and analysis_options:
                texts_to_analyze = df[text_column].dropna().astype(str).tolist()
                
                if texts_to_analyze:
                    st.info(f"سيتم تحليل عدد {len(texts_to_analyze)} نص...")
                    progress_bar = st.progress(0, text="...بدء التحليل")
                    
                    # ✨ تم تعديل منطق شريط التقدم ليكون أبسط وأوضح ✨
                    total_tasks = len(analysis_options)
                    completed_tasks = 0

                    if "تحليل الجنس" in analysis_options:
                        gender_predictions = predict_gender(texts_to_analyze)
                        df.loc[df[text_column].notna(), 'الجنس المتوقع'] = gender_predictions
                        completed_tasks += 1
                        progress_bar.progress(int((completed_tasks/total_tasks)*100), text="...تم تحليل الجنس")


                    if "تحليل المشاعر" in analysis_options:
                        sentiment_predictions = predict_sentiment(texts_to_analyze)
                        df.loc[df[text_column].notna(), 'المشاعر المتوقعة'] = sentiment_predictions
                        completed_tasks += 1
                        progress_bar.progress(int((completed_tasks/total_tasks)*100), text="...تم تحليل المشاعر")
                    
                    if "تصنيف اللهجة" in analysis_options:
                        dialect_predictions = predict_dialect(texts_to_analyze)
                        df.loc[df[text_column].notna(), 'اللهجة المتوقعة'] = dialect_predictions
                        completed_tasks += 1
                        progress_bar.progress(int((completed_tasks/total_tasks)*100), text="...تم تصنيف اللهجة")

                    st.balloons()
                    st.success("!اكتمل التحليل بنجاح")

                    st.subheader("نتائج التحليل:")
                    st.dataframe(df)

                    csv_data = convert_df_to_csv(df)
                    st.download_button(
                        label="📥 تحميل النتائج كملف CSV",
                        data=csv_data,
                        file_name=f"نتائج_تحليل_{uploaded_file.name}.csv",
                        mime='text/csv',
                        use_container_width=True
                    )
                else:
                    st.error("العمود المختار لا يحتوي على أي نصوص لتحليلها.")
            else:
                st.warning("الرجاء اختيار عمود النص ونوع تحليل واحد على الأقل.")

    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الملف: {e}")
