import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.data_utils import CLASS_NAMES, load_csv_data, load_folder_data

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained facial-expression model.")
    parser.add_argument("--data-dir", default="data/fer2013")
    parser.add_argument("--csv")
    parser.add_argument("--model", default="models/best_emotion_model.keras")
    args = parser.parse_args()
    _, _, test_ds = load_csv_data(args.csv) if args.csv else load_folder_data(args.data_dir)
    if test_ds is None:
        raise SystemExit("No test split found. Add a held-out test folder or CSV PrivateTest split.")
    model = tf.keras.models.load_model(args.model)
    y_true, y_prob = [], []
    for x, y in test_ds:
        y_prob.append(model.predict(x, verbose=0))
        y_true.append(np.argmax(y.numpy(), axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.argmax(np.concatenate(y_prob), axis=1)

    report = classification_report(y_true, y_pred, labels=range(7), target_names=CLASS_NAMES, digits=4, zero_division=0)
    print(report)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=range(7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False)
    ax.set_title("FER Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=160)
    print("Saved reports/classification_report.txt and reports/confusion_matrix.png")

if __name__ == "__main__":
    main()
