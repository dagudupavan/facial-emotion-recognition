
import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.data_utils import CLASS_NAMES, load_csv_data, load_folder_data
from src.model import build_emotion_cnn


def calculate_class_weights(data_dir):
    """Calculate balanced weights from the training image counts."""
    train_dir = Path(data_dir) / "train"

    counts = {}

    for class_name in CLASS_NAMES:
        class_dir = train_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        counts[class_name] = sum(
            1 for file in class_dir.iterdir() if file.is_file()
        )

    total = sum(counts.values())
    num_classes = len(CLASS_NAMES)

    weights = {
        index: total / (num_classes * counts[class_name])
        for index, class_name in enumerate(CLASS_NAMES)
    }

    print("\nClass distribution and weights:")
    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name:10s}: "
            f"{counts[class_name]:5d} images | "
            f"weight: {weights[index]:.3f}"
        )

    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Train the FER facial-expression CNN."
    )

    parser.add_argument(
        "--data-dir",
        default="data/fer2013",
        help="Folder-layout dataset root"
    )
    parser.add_argument(
        "--csv",
        help="FER-2013 CSV file; if provided, takes precedence"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--model-dir", default="models")

    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # Load dataset
    if args.csv:
        train_ds, val_ds, _ = load_csv_data(args.csv)
        class_weights = None
        print("CSV dataset: automatic folder-based class weights disabled.")
    else:
        train_ds, val_ds, _ = load_folder_data(args.data_dir)
        class_weights = calculate_class_weights(args.data_dir)

    # Apply sample weights to the one-hot labels.
    if class_weights is not None:
        weight_vector = tf.constant(
            [class_weights[i] for i in range(len(CLASS_NAMES))],
            dtype=tf.float32
        )

        train_ds = train_ds.map(
            lambda images, labels: (
                images,
                labels,
                tf.reduce_sum(labels * weight_vector, axis=-1)
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Build model
    model = build_emotion_cnn()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_dir / "best_emotion_model.keras",
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
            "reports/training_log.csv"
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir="reports/tensorboard"
        ),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False
    )

    # Save model
    model.save(model_dir / "final_emotion_model.keras")

    with open(
        model_dir / "class_names.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(CLASS_NAMES, file, indent=2)

    print("\nTraining completed!")
    print(
        f"Best validation accuracy: "
        f"{max(history.history['val_accuracy']):.4f}"
    )
    print(
        f"Saved best model to: "
        f"{model_dir / 'best_emotion_model.keras'}"
    )
    print(
        f"Saved final model to: "
        f"{model_dir / 'final_emotion_model.keras'}"
    )
    print(
        f"Completed epochs: "
        f"{len(history.history.get('loss', []))}"
    )


if __name__ == "__main__":
    main()
