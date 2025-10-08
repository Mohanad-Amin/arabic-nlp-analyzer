# 📂 أداة تحليل النصوص العربية المتقدمة

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)](https://huggingface.co/UBC-NLP/MARBERTv2)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)

أداة ويب تفاعلية تمكن المستخدمين من رفع ملفات Excel تحتوي على نصوص عربية وتحليلها تلقائيًا باستخدام ثلاثة نماذج تعلم عميق مدربة خصيصًا لتحديد جنس الكاتب، تحليل مشاعر النص، وتصنيف اللهجة العربية.

**[رابط النسخة الحية من التطبيق ((https://mohanad-arabic-nlp-analyzer.streamlit.app/))]()**

---

### 🖼️ لقطة شاشة للتطبيق
<img width="1855" height="902" alt="Screenshot from 2025-09-25 14-24-17" src="https://github.com/user-attachments/assets/e81e1eb8-9ace-4e32-a6b0-db19c243dc56" />

<img width="1855" height="884" alt="Screenshot from 2025-09-25 14-24-32" src="https://github.com/user-attachments/assets/3e52ad36-00f2-40bb-a3fa-54813b1e51db" />


### ✨ الميزات الرئيسية

-   **رفع ملفات Excel:** واجهة سهلة لرفع البيانات مباشرة.
-   **تحليل ثلاثي المهام:** يمكن للمستخدم اختيار تحليل الجنس، المشاعر، أو تصنيف اللهجة.
-   **تحديد عمود النص:** مرونة في اختيار العمود المراد تحليله من الملف.
-   **أداء مرن:** يكتشف الكود تلقائيًا وجود GPU لتسريع الأداء، ويعمل بكفاءة على CPU.
-   **تنزيل النتائج:** إمكانية تحميل البيانات الأصلية مع النتائج الجديدة كملف CSV.

---

### 🛠️ التقنيات المستخدمة

-   **النموذج الأساسي:** [UBC-NLP/MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2) تم ضبطه (Fine-tuned) للمهام الثلاث.
-   **إطار العمل:** Streamlit لبناء واجهة المستخدم التفاعلية.
-   **المكتبات الرئيسية:** PyTorch, Transformers, Pandas.

---

### 🚀 كيفية تشغيل المشروع محليًا

1.  **استنسخ المستودع:**
    ```bash
    git clone [https://github.com/Mohanad-Amin/arabic_analyzer.git](https://github.com/Mohanad-Amin/arabic_analyzer.git)
    cd arabic_analyzer
    ```
2.  **أنشئ وشغّل البيئة الافتراضية:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **ثبّت المكتبات المطلوبة:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **شغّل تطبيق Streamlit:**
    ```bash
    streamlit run app.py
    ```

---
### 📊 أداء النماذج

تم تدريب ثلاثة نماذج متخصصة مبنية على **MARBERTv2** لتحقيق أداء عالي الدقة في مهام مختلفة:

#### 1. نموذج تصنيف اللهجات (Dialect Classification)
-   **المهمة:** تصنيف النص إلى واحدة من 4 لهجات رئيسية (خليجية، فصحى، شامية، مصرية).
-   **بيانات التدريب:** تم تدريبه على مجموعة بيانات تحتوي على **20,299** نصًا.
-   **الأداء:** حقق النموذج دقة ممتازة وصلت إلى **Weighted F1-Score: 0.998**.

#### 2. نموذج تحليل المشاعر (Sentiment Analysis)
-   **المهمة:** تصنيف مشاعر النص إلى (إيجابي، سلبي، محايد).
-   **بيانات التدريب:** تم تدريبه على مجموعة بيانات مكونة من **45,948** تغريدة، تمت موازنتها لتصل إلى **47,100** عينة لضمان العدالة بين الفئات.
-   **الأداء:** حقق النموذج أداءً قويًا جدًا مع **Weighted F1-Score: 0.927**.

#### 3. نموذج تحديد الجنس (Gender Classification)
-   **المهمة:** تحديد جنس كاتب النص (ذكر، أنثى، غير معروف) بناءً على أسلوب الكتابة في الاسم والوصف.
-   **بيانات التدريب:** تم تدريبه على بيانات **3,995** مستخدم.
-   **الأداء:** أظهر النموذج قدرة جيدة على التمييز محققًا **Weighted F1-Score: 0.836**.
