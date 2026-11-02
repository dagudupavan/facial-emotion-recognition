
"""Experiment 01: Original CNN with improved data augmentation."""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.data_utils_v2 import (
    CLASS_NAMES,
    load_folder_data
)
from src.model_v1 import build_emotion_cnn


def calculate_class_weights(data_dir):
    """Calculate balanced class weights from training images."""

    train_dir = Path(data_dir) / "train"

    counts = {}

    for class_name in CLASS_NAMES:
        class_dir = train_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(class_dir)

        counts[class_name] = sum(
            1 for file in class_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

    total = sum(counts.values())
    number_of_classes = len(CLASS_NAMES)

    weights = {
        index: total / (number_of_classes * counts[name])
        for index, name in enumerate(CLASS_NAMES)
    }

    print("\nClass weights:")

    for index, name in enumerate(CLASS_NAMES):
        print(
            f"{name:10s}: {counts[name]:5d} images | "
            f"weight: {weights[index]:.3f}"
        )

    return weights


def main():

    parser = argparse.ArgumentParser(
        description="Experiment 01: Improved FER-2013 training"
    )

    parser.add_argument(
        "--data-dir",
        default="data/fer2013"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40
    )

    parser.add_argument(
        "--experiment-dir",
        default="experiments/experiment_01"
    )

    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)

    experiment_dir = Path(args.experiment_dir)
    model_dir = experiment_dir / "models"
    report_dir = experiment_dir / "reports"

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load improved datasets
    train_ds, val_ds, test_ds = load_folder_data(
        args.data_dir
    )

    if test_ds is None:
        raise ValueError("Separate test dataset is required.")

    # Calculate class weights
    class_weights = calculate_class_weights(
        args.data_dir
    )

    weight_vector = tf.constant(
        [class_weights[i] for i in range(len(CLASS_NAMES))],
        dtype=tf.float32
    )

    # Apply per-example sample weights
    train_ds = train_ds.map(
        lambda images, labels: (
            images,
            labels,
            tf.reduce_sum(
                labels * weight_vector,
                axis=-1
            )
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Use original CNN architecture
    model = build_emotion_cnn()

    model.summary()

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_emotion_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            str(report_dir / "training_log.csv")
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=str(report_dir / "tensorboard")
        )
    ]

    print("\nStarting Experiment 01...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False
    )

    # Save final model
    model.save(
        model_dir / "final_emotion_model.keras"
    )

    # Save class names
    with open(
        model_dir / "class_names.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(CLASS_NAMES, file, indent=2)

    best_epoch = (
        max(
            range(len(history.history["val_accuracy"])),
            key=lambda i: history.history["val_accuracy"][i]
        ) + 1
    )

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    print("\nExperiment 01 completed!")

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    print(f"Best model: {model_dir / 'best_emotion_model.keras'}")
    print(f"Training log: {report_dir / 'training_log.csv'}")


if __name__ == "__main__":
    main()
