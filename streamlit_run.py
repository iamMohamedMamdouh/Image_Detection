import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
import pygame
import tempfile
import os
from collections import Counter

st.set_page_config(page_title="Image Detection Model", layout="centered")
st.title("Image Detection Model")
st.markdown("Upload an image and see the result after the model analyzes it")

model = YOLO("yolov8n.pt")

translate = {
    "person": "شخص",
    "car": "سيارة",
    "truck": "شاحنة",
    "bus": "أتوبيس",
    "bicycle": "دراجة",
    "motorcycle": "دراجة نارية",
    "cat": "قطة",
    "dog": "كلب",
    "traffic light": "إشارة مرور",
    "handbag": "حقيبة يد",
    "chair": "كرسي",
    "bird": "طائر",
    "boat": "قارب",
    "backpack": "حقيبة ظهر",
}

uploaded_file = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("The picture:")
    st.image(image, use_container_width=True)

    result = model.predict(img_array, conf=0.3)[0]
    boxes = result.boxes.data
    class_names = result.names

    st.subheader("The image after Detection:")
    st.image(result.plot(), caption="✅ Detected Image", use_container_width=True)


    detected_labels = [class_names[int(cls)] for *_, cls in boxes.tolist()]
    if detected_labels:
        counts = Counter(detected_labels)

        col1, col2 = st.columns([1, 3])
        with col1:
            lang_choice = st.selectbox("🌐", ["English", "عربي"], index=0, label_visibility="collapsed")
        with col2:
            speak = st.button("🔊 Announcing the results")

        # تجهيز النص
        lines = []
        if lang_choice == "عربي":
            for k, v in counts.items():
                arabic = translate.get(k, k)
                if v == 1:
                    lines.append(f"{arabic} واحد")
                else:
                    lines.append(f"{v} {arabic}")
            speak_text = "تم اكتشاف: " + "، ".join(lines)
        else:
            for k, v in counts.items():
                label = k if v == 1 else f"{v} {k}s"
                lines.append(label)
            speak_text = "Detected: " + ", ".join(lines)

        display_text = "\n".join([f"- {line}" for line in lines])
        st.success("📊 Results:\n" + display_text)

        if speak:
            try:
                tts = gTTS(text=speak_text, lang='ar' if lang_choice == "عربي" else 'en')
                tmp_path = os.path.join(tempfile.gettempdir(), "speech.mp3")
                tts.save(tmp_path)

                pygame.mixer.init()
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                pygame.mixer.quit()
                os.remove(tmp_path)
            except Exception as e:
                st.error(f"⚠ حصلت مشكلة في النطق: {str(e)}")
    else:
        st.warning("❌ لا يوجد كائنات تم اكتشافها.")
#///////////////////////HTML///////////////////////////
# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO
# import numpy as np
# import cv2
# import base64
# import streamlit.components.v1 as components
#
# st.set_page_config(page_title="🧠 YOLO Interactive Voice", layout="centered")
# st.title("🧠 YOLO Interactive Detection with Voice")
#
# model = YOLO("yolov8n.pt")  # أو الموديل اللي دربته
#
# uploaded_file = st.file_uploader("📸 ارفع صورة", type=["jpg", "jpeg", "png"])
#
# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     img_array = np.array(image)
#     w, h = image.size
#
#     st.subheader("📷 الصورة الأصلية:")
#     st.image(image, caption="Original Image", use_column_width=False)
#
#     # 🔍 YOLO Prediction
#     results = model.predict(img_array, conf=0.3)[0]
#
#     # 🎨 رسم النتائج
#     annotated_img = results.plot()
#     _, buffer = cv2.imencode(".jpg", annotated_img)
#     img_base64 = base64.b64encode(buffer).decode()
#
#     st.subheader("🧠 الصورة بعد التحليل:")
#
#     # 📦 بناء HTML تفاعلي
#     html = f"""
#     <html>
#     <head>
#     <style>
#         .container {{
#             position: relative;
#             width: {w}px;
#             height: {h}px;
#             background-image: url("data:image/jpg;base64,{img_base64}");
#             background-size: contain;
#             background-repeat: no-repeat;
#             background-position: top left;
#             border: 2px solid #999;
#         }}
#         .box {{
#             position: absolute;
#             border: 2px solid red;
#             cursor: pointer;
#             z-index: 10;
#         }}
#     </style>
#     </head>
#     <body>
#     <div class="container">
#     """
#
#     # 🧠 إضافة بوكسات و onclick ناطق
#     for box in results.boxes.data:
#         x1, y1, x2, y2, conf, cls = box.tolist()
#         label = results.names[int(cls)]
#         box_width = x2 - x1
#         box_height = y2 - y1
#
#         html += f"""
#         <div class="box" onclick="speak('{label}')"
#             style="left:{x1}px; top:{y1}px; width:{box_width}px; height:{box_height}px;"
#             title="{label}">
#         </div>
#         """
#
#     # 🔊 JavaScript نطق عند الضغط
#     html += """
#     </div>
#     <script>
#     function speak(text) {
#         const msg = new SpeechSynthesisUtterance(text);
#         msg.lang = 'en';
#         window.speechSynthesis.cancel();
#         window.speechSynthesis.speak(msg);
#     }
#     </script>
#     </body>
#     </html>
#     """
#
#     components.html(html, height=h+60, width=w+30)
#////////////////////////////////First//////////////////////////
# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
#
# # إعداد الصفحة
# st.set_page_config(page_title="🔍 Image Detection with YOLOv8", page_icon="📸")
#
# st.title("🎯 YOLOv8 Image Detection App")
# st.markdown("ارفع صورة، وشوف النتيجة بعد ما الموديل يحللها 👁️‍🗨️")
#
# # تحميل الموديل
# @st.cache_resource
# def load_model():
#     return YOLO("yolov8n.pt")  # أو حط الموديل المدرب الخاص بيك هنا
#
# model = load_model()
#
# # رفع صورة
# uploaded_file = st.file_uploader("📤 ارفع صورة", type=["jpg", "png", "jpeg"])
#
# if uploaded_file is not None:
#     # عرض الصورة الأصلية
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="🖼️ الصورة الأصلية", use_container_width=True)
#
#     # تحويل الصورة لـ NumPy Array
#     img_array = np.array(image)
#
#     # معالجة الصورة
#     with st.spinner("🤖 جاري تحليل الصورة..."):
#         results = model.predict(source=img_array, conf=0.4)
#         result_img = results[0].plot()
#
#         # عرض النتيجة
#         st.image(result_img, caption="✅ الصورة بعد الـ Detection", use_container_width=True)
