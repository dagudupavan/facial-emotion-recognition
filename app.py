
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.preprocessing import preprocess_face


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Real-Time Facial Emotion Recognition")

st.caption(
    "Deep Learning based facial expression recognition "
    "using the FER-2013 dataset."
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
    st.error("Trained model not found in the models folder.")
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


# ==========================================
# 3. LOAD EMOTION CLASSES
# ==========================================

class_file = MODEL_PATH.parent / "class_names.json"

if class_file.exists():
    classes = json.loads(class_file.read_text())
else:
    classes = DEFAULT_CLASSES

if len(classes) != 7:
    st.error("Model must have seven emotion classes.")
    st.stop()


# ==========================================
# 4. IMAGE UPLOAD AND CAMERA
# ==========================================

st.header("Upload or Capture Face Image")

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

with col2:
    camera_image = st.camera_input(
        "Capture Using Camera"
    )

source = uploaded if uploaded is not None else camera_image


# ==========================================
# 5. FACE DETECTION
# ==========================================

if source is not None:

    try:

        image = Image.open(source).convert("RGB")

        rgb = np.array(image)

        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

        gray = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY
        )

        # Improve image contrast for detection
        gray = cv2.equalizeHist(gray)

        # Stricter face detection settings
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        preview = bgr.copy()

        if len(faces) == 0:

            st.warning(
                "No face detected. Please upload a clear, "
                "front-facing image with good lighting."
            )

        else:

            st.success(
                f"Successfully detected {len(faces)} face(s)."
            )

            results = []

            # ==========================================
            # 6. PREDICT EACH DETECTED FACE
            # ==========================================

            for number, (x, y, w, h) in enumerate(
                faces, start=1
            ):

                face = bgr[y:y+h, x:x+w]

                processed_face = preprocess_face(face)

                probabilities = model.predict(
                    processed_face,
                    verbose=0
                )[0]

                index = int(np.argmax(probabilities))

                emotion = classes[index].title()

                confidence = (
                    float(probabilities[index]) * 100
                )

                results.append({
                    "number": number,
                    "emotion": emotion,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "face": face
                })

                # Draw face rectangle
                cv2.rectangle(
                    preview,
                    (x, y),
                    (x+w, y+h),
                    (40, 220, 90),
                    3
                )

                label = f"{emotion} {confidence:.1f}%"

                cv2.putText(
                    preview,
                    label,
                    (x, max(30, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (40, 220, 90),
                    2,
                    cv2.LINE_AA
                )

            # ==========================================
            # 7. DISPLAY DETECTED IMAGE
            # ==========================================

            st.divider()

            st.header("Face Detection Result")

            preview_rgb = cv2.cvtColor(
                preview,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                preview_rgb,
                caption="Detected faces",
                use_container_width=True
            )

            # ==========================================
            # 8. DISPLAY EMOTION RESULTS
            # ==========================================

            st.divider()

            st.header("Emotion Prediction Results")

            for result in results:

                st.subheader(
                    f"Face {result['number']}"
                )

                left, right = st.columns([1, 2])

                with left:

                    face_rgb = cv2.cvtColor(
                        result["face"],
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        face_rgb,
                        caption="Detected face",
                        width=200
                    )

                with right:

                    st.success(
                        f"Predicted Emotion: {result['emotion']}"
                    )

                    st.metric(
                        "Model Confidence",
                        f"{result['confidence']:.2f}%"
                    )

                # ==========================================
                # 9. ALL SEVEN EMOTION PROBABILITIES
                # ==========================================

                st.subheader(
                    "All Seven Emotion Probabilities"
                )

                probabilities = result["probabilities"]

                for name, probability in zip(
                    classes,
                    probabilities
                ):

                    percentage = (
                        float(probability) * 100
                    )

                    st.write(
                        f"{name.title()}: {percentage:.2f}%"
                    )

                    st.progress(
                        float(probability)
                    )

                # ==========================================
                # 10. EMOTION COMPARISON CHART
                # ==========================================

                st.subheader(
                    "Emotion Comparison Chart"
                )

                chart_data = pd.DataFrame({
                    "Emotion": [
                        name.title() for name in classes
                    ],
                    "Probability": [
                        float(value) for value in probabilities
                    ]
                })

                st.bar_chart(
                    chart_data.set_index("Emotion")
                )

                st.divider()

    except Exception as error:

        st.error(
            f"Error while processing image: {error}"
        )


# ==========================================
# 11. FOOTER
# ==========================================

st.caption(
    "B.Tech CSE Project | FER-2013 | CNN Deep Learning"
)

st.caption(
    "Predictions estimate visible facial expressions; "
    "they do not establish a person's actual emotional state."
)
