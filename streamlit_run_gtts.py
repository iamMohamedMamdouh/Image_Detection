import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
import tempfile
import os
from collections import Counter
import base64

st.set_page_config(page_title="Image Detection Model", layout="centered")
st.title("Image Detection Model")
st.markdown("Upload an image and see the result after the model analyzes it")

model = YOLO("yolov8n.pt")  # تأكد إن الموديل مرفوع مع المشروع أو استخدم رابطه من Hugging Face

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

    st.subheader("📷 Original Image:")
    st.image(image, use_container_width=True)

    result = model.predict(img_array, conf=0.3)[0]
    boxes = result.boxes.data
    class_names = result.names

    st.subheader("🧠 Detected Image:")
    st.image(result.plot(), caption="✅ Detected", use_container_width=True)

    detected_labels = [class_names[int(cls)] for *_, cls in boxes.tolist()]
    if detected_labels:
        counts = Counter(detected_labels)

        col1, col2 = st.columns([1, 3])
        with col1:
            lang_choice = st.selectbox("🌐", ["English", "عربي"], index=0, label_visibility="collapsed")
        with col2:
            speak = st.button("🔊 Announce Results")

        # إعداد النص
        lines = []
        if lang_choice == "عربي":
            for k, v in counts.items():
                arabic = translate.get(k, k)
                lines.append(f"{arabic} واحد" if v == 1 else f"{v} {arabic}")
            speak_text = "تم اكتشاف: " + "، ".join(lines)
        else:
            for k, v in counts.items():
                lines.append(f"{k}" if v == 1 else f"{v} {k}s")
            speak_text = "Detected: " + ", ".join(lines)

        display_text = "\n".join(f"- {line}" for line in lines)
        st.success("📊 Results:\n" + display_text)

        if speak:
            try:
                tts = gTTS(text=speak_text, lang='ar' if lang_choice == "عربي" else 'en')
                tmp_path = os.path.join(tempfile.gettempdir(), "speech.mp3")
                tts.save(tmp_path)

                with open(tmp_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    b64 = base64.b64encode(audio_bytes).decode()
                    audio_html = f"""
                        <audio autoplay style="display: none">
                        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                        </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)

                os.remove(tmp_path)
            except Exception as e:
                st.error(f"⚠ حصلت مشكلة في النطق: {str(e)}")
    else:
        st.warning("❌ No objects detected.")
#/////////////////////////////////////////////////////////////////
# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO
# import numpy as np
# from gtts import gTTS
# import tempfile
# import os
# from collections import Counter
# import base64
#
# st.set_page_config(page_title="Image Detection Model", layout="centered")
# st.title("Image Detection Model")
# st.markdown("Upload an image and see the result after the model analyzes it")
#
# model = YOLO("yolov8n.pt")
#
# translate = {
#     "person": "شخص",
#     "car": "سيارة",
#     "truck": "شاحنة",
#     "bus": "أتوبيس",
#     "bicycle": "دراجة",
#     "motorcycle": "دراجة نارية",
#     "cat": "قطة",
#     "dog": "كلب",
#     "traffic light": "إشارة مرور",
#     "handbag": "حقيبة يد",
#     "chair": "كرسي",
#     "bird": "طائر",
#     "boat": "قارب",
#     "backpack": "حقيبة ظهر",
# }
#
# uploaded_file = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])
#
# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     img_array = np.array(image)
#
#     st.subheader("📷 Original Image:")
#     st.image(image, use_container_width=True)
#
#     result = model.predict(img_array, conf=0.3)[0]
#     boxes = result.boxes.data
#     class_names = result.names
#
#     st.subheader("🧠 Detected Image:")
#     st.image(result.plot(), caption="✅ Detected", use_container_width=True)
#
#     detected_labels = [class_names[int(cls)] for *_, cls in boxes.tolist()]
#     if detected_labels:
#         counts = Counter(detected_labels)
#
#         col1, col2 = st.columns([1, 3])
#         with col1:
#             lang_choice = st.selectbox("🌐", ["English", "عربي"], index=0, label_visibility="collapsed")
#         with col2:
#             speak = st.button("🔊 Announce Results")
#
#         # إعداد النص
#         lines = []
#         if lang_choice == "عربي":
#             for k, v in counts.items():
#                 arabic = translate.get(k, k)
#                 lines.append(f"{arabic} واحد" if v == 1 else f"{v} {arabic}")
#             speak_text = "تم اكتشاف: " + "، ".join(lines)
#         else:
#             for k, v in counts.items():
#                 lines.append(f"{k}" if v == 1 else f"{v} {k}s")
#             speak_text = "Detected: " + ", ".join(lines)
#
#         display_text = "\n".join(f"- {line}" for line in lines)
#         st.success("📊 Results:\n" + display_text)
#
#         if speak:
#             try:
#                 tts = gTTS(text=speak_text, lang='ar' if lang_choice == "عربي" else 'en')
#
#                 # نحفظ الملف ونقفله قبل استخدامه
#                 tmp_path = os.path.join(tempfile.gettempdir(), "speech.mp3")
#                 tts.save(tmp_path)
#
#                 # تأكد من إغلاق الملف قبل الفتح مرة تانية
#                 with open(tmp_path, "rb") as audio_file:
#                     audio_bytes = audio_file.read()
#                     b64 = base64.b64encode(audio_bytes).decode()
#                     audio_html = f"""
#                         <audio autoplay
#                         style="display: none">
#                         <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
#                         </audio>
#                     """
#                     st.markdown(audio_html, unsafe_allow_html=True)
#
#                 # احذف الملف بعد ما Streamlit يستخدمه
#                 os.remove(tmp_path)
#
#             except Exception as e:
#                 st.error(f"⚠ حصلت مشكلة في النطق: {str(e)}")
#     else:
#         st.warning("❌ No objects detected.")