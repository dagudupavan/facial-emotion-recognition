# Real-Time Facial Emotion Recognition Using Deep Learning

A B.Tech CSE project that trains a CNN on FER-2013 and predicts visible facial-expression categories from webcam frames.

## Features
- CNN training with augmentation and early stopping
- Validation and test evaluation: accuracy, precision, recall, macro F1, confusion matrix
- OpenCV webcam inference using a Haar cascade face detector
- Optional Streamlit image/webcam-frame demo
- Model and label order saved with the trained network

> Important: the system estimates visible facial-expression patterns; it does not determine a person's true internal emotional state. Use consent-based demos and avoid consequential decisions based on predictions.

## 1. Environment
Recommended: Python 3.10 or 3.11. A GPU is optional.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If TensorFlow installation is not compatible with your platform/Python version, create an environment using a TensorFlow-supported Python version and consult the official TensorFlow installation guide.

## 2. Obtain FER-2013 data
Download FER-2013 from a legitimate dataset source and arrange it in one of these formats.

### Option A: folder layout (recommended)
```text
data/fer2013/
  train/
    angry/
    disgusted/
    fearful/
    happy/
    sad/
    surprised/
    neutral/
  validation/
    angry/
    disgusted/
    fearful/
    happy/
    sad/
    surprised/
    neutral/
  test/
    angry/
    disgusted/
    fearful/
    happy/
    sad/
    surprised/
    neutral/
```
Images can be grayscale or RGB; they will be converted to grayscale and resized to 48x48. The folder names must match the seven class names above. Validation/test folders are optional; if missing, the script creates a validation split from `train`, but a separately held-out test set is preferable.

### Option B: original FER-2013 CSV
The CSV must have columns `emotion`, `pixels`, and `Usage` (common Kaggle layout). `pixels` contains 2304 space-separated pixel values and `Usage` contains `Training`, `PublicTest`, or `PrivateTest`.
```bash
python train.py --csv data/fer2013.csv --epochs 40
```

For folder layout:
```bash
python train.py --data-dir data/fer2013 --epochs 40
```

Training logs are written to `reports/training_log.csv` and `reports/tensorboard/`.
To view the TensorBoard logs, run `tensorboard --logdir reports/tensorboard`
in a separate terminal and open the URL it prints.

The model is saved to `models/best_emotion_model.keras`; class names are saved to `models/class_names.json`.

## 3. Evaluate
```bash
python evaluate.py --data-dir data/fer2013
```
or
```bash
python evaluate.py --csv data/fer2013.csv
```
Evaluation prints metrics and writes `reports/classification_report.txt` and `reports/confusion_matrix.png`.

## 4. Run webcam application
After training:
```bash
python webcam.py --model models/best_emotion_model.keras
```
Press `q` to quit. Press `s` to save a screenshot to `reports/`.

If the camera is not found, try `--camera 1`. On macOS, grant camera permission to the terminal/IDE. On Linux, verify `/dev/video0` permissions.

## 5. Optional Streamlit demo
```bash
streamlit run app.py
```
Upload a face image or use Streamlit's camera input. This demo is for still images/camera snapshots, not continuous video streaming.

## 6. Project structure
```text
real_time_facial_emotion_recognition/
  src/
    data_utils.py
    model.py
    preprocessing.py
  models/                  # generated model files go here
  reports/                 # generated metrics and screenshots go here
  train.py
  evaluate.py
  webcam.py
  app.py
  requirements.txt
```

## 7. Methodology
1. Load labeled FER-2013 images.
2. Convert images to grayscale, resize to 48x48, and scale pixels to [0, 1].
3. Apply augmentation only to training images.
4. Train a CNN with categorical cross-entropy and Adam.
5. Use validation loss for early stopping and learning-rate reduction.
6. Evaluate once on the held-out test set, if provided.
7. In the webcam loop, detect face regions, preprocess each crop identically, and classify with the trained model.

## 8. Research/report guidance
Record the actual environment, dataset version, split, epoch count, training time, and metrics from your run. Do not invent accuracy figures. Compare models using macro F1 and per-class recall because class imbalance can make accuracy misleading. Discuss lighting, pose, occlusion, dataset bias, domain shift, and the limitations of inferring emotion from appearance.
