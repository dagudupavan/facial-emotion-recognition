import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from src.preprocessing import preprocess_face

DEFAULT_CLASSES = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]

def main():
    parser = argparse.ArgumentParser(description="Real-time facial-expression recognition.")
    parser.add_argument("--model", default="models/best_emotion_model.keras")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Hide label below this confidence threshold, e.g. 0.35")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}. Train it first with train.py.")
    model = tf.keras.models.load_model(model_path)
    class_path = model_path.parent / "class_names.json"
    classes = json.loads(class_path.read_text(encoding="utf-8")) if class_path.exists() else DEFAULT_CLASSES

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise SystemExit("Could not load OpenCV Haar cascade.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}. Try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev = time.perf_counter()
    fps = 0.0
    Path("reports").mkdir(exist_ok=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))

            for (x, y, w, h) in faces:
                # Slight padding reduces overly tight face crops.
                pad_x, pad_y = int(0.08 * w), int(0.10 * h)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(frame.shape[1], x + w + pad_x), min(frame.shape[0], y + h + pad_y)
                crop = frame[y1:y2, x1:x2]
                model_input = preprocess_face(crop)
                probs = model.predict(model_input, verbose=0)[0]
                idx = int(np.argmax(probs))
                confidence = float(probs[idx])
                label = classes[idx] if confidence >= args.min_confidence else "uncertain"
                text = f"{label.title()}  {confidence:.0%}"
                color = (50, 210, 80) if label != "uncertain" else (0, 180, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 32)), (x1 + 220, y1), color, -1)
                cv2.putText(frame, text, (x1 + 6, max(20, y1 - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - prev, 1e-6)
            fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f} | Press q to quit | s to save",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Real-Time Facial Emotion Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out = Path("reports") / f"webcam_capture_{int(time.time())}.jpg"
                cv2.imwrite(str(out), frame)
                print(f"Saved {out}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
