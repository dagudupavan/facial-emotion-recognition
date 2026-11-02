
import json
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from streamlit_webrtc import webrtc_streamer, WebRtcMode
from src.preprocessing import preprocess_face


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Real-Time Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Real-Time Facial Emotion Recognition")

st.caption(
    "Live facial emotion detection using CNN, "
    "OpenCV and the FER-2013 dataset."
)

st.divider()


# ==========================================
# 2. MODEL CONFIGURATION
# ==========================================

MODEL_PATH = Path("models/best_emotion_model.keras")

DEFAULT_CLASSES = [
    "angry", "disgust", "fear",
    "happy", "sad", "surprise", "neutral"
]

if not MODEL_PATH.exists():
    st.error("Trained model not found.")
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_detector():

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Face detector could not be loaded.")

    return detector


model = load_model()
detector = load_detector()

class_file = MODEL_PATH.parent / "class_names.json"

if class_file.exists():
    classes = json.loads(class_file.read_text())
else:
    classes = DEFAULT_CLASSES

if len(classes) != 7:
    st.error("Expected seven emotion classes.")
    st.stop()


# ==========================================
# 3. LIVE VIDEO PROCESSOR
# ==========================================

class EmotionProcessor:

    def __init__(self):
        self.lock = threading.Lock()

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(70, 70),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:

            face = img[y:y+h, x:x+w]

            try:
                processed_face = preprocess_face(face)

                with self.lock:
                    probabilities = model.predict(
                        processed_face,
                        verbose=0
                    )[0]

                index = int(np.argmax(probabilities))

                emotion = classes[index].title()

                confidence = float(
                    probabilities[index] * 100
                )

                label = f"{emotion}: {confidence:.1f}%"

                # Draw face rectangle
                cv2.rectangle(
                    img,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                # Draw emotion label
                cv2.rectangle(
                    img,
                    (x, max(0, y-35)),
                    (x+w, y),
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    img,
                    label,
                    (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

            except Exception:
                cv2.putText(
                    img,
                    "Prediction error",
                    (x, max(25, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # Display face count
        cv2.putText(
            img,
            f"Faces detected: {len(faces)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ==========================================
# 4. LIVE CAMERA INTERFACE
# ==========================================

st.header("Live Camera Detection")

st.write(
    "Click START to activate your webcam. "
    "The system will continuously detect faces "
    "and predict their facial expressions."
)

left, right = st.columns(2)

with left:
    st.info("Camera: Live webcam")

with right:
    st.info("Model: CNN | 7 Emotion Classes")


# ==========================================
# 5. WEBRTC STREAM
# ==========================================

webrtc_ctx = webrtc_streamer(
    key="facial-emotion-recognition",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15}
        },
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    async_processing=True
)


# ==========================================
# 6. EMOTION CLASS INFORMATION
# ==========================================

st.divider()

st.header("Available Emotion Categories")

emotion_columns = st.columns(7)

emotion_icons = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

for column, emotion in zip(emotion_columns, classes):

    with column:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<h2>{emotion_icons.get(emotion, '🙂')}</h2>"
            f"<b>{emotion.title()}</b>"
            f"</div>",
            unsafe_allow_html=True
        )


# ==========================================
# 7. PROJECT INFORMATION
# ==========================================

st.divider()

st.subheader("Project Information")

st.write("**Dataset:** FER-2013")
st.write("**Model:** Convolutional Neural Network (CNN)")
st.write("**Face Detection:** OpenCV Haar Cascade")
st.write("**Video Streaming:** Streamlit-WebRTC")
st.write("**Input:** Live Webcam")
st.write("**Output:** Seven facial expression predictions")

st.caption(
    "B.Tech CSE Deep Learning Project | "
    "Real-Time Facial Emotion Recognition"
)

st.caption(
    "Predictions estimate visible facial expressions "
    "and do not establish a person's actual emotional state."
)
