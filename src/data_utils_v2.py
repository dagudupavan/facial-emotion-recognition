
"""Improved FER-2013 dataset loader with controlled augmentation."""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split


CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

CLASS_TO_ID = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# Mild geometric augmentation
rotation = tf.keras.layers.RandomRotation(
    factor=0.04,
    fill_mode="reflect",
    seed=42
)

translation = tf.keras.layers.RandomTranslation(
    height_factor=0.05,
    width_factor=0.05,
    fill_mode="reflect",
    seed=43
)

zoom = tf.keras.layers.RandomZoom(
    height_factor=(-0.08, 0.08),
    width_factor=(-0.08, 0.08),
    fill_mode="reflect",
    seed=44
)


def collect_images(split_dir):

    split_dir = Path(split_dir)

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {split_dir}"
        )

    paths = []
    labels = []

    for class_name in CLASS_NAMES:

        class_dir = split_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing emotion folder: {class_dir}"
            )

        images = [
            p for p in class_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            raise ValueError(
                f"No images found in {class_dir}"
            )

        for image_path in images:
            paths.append(str(image_path))
            labels.append(CLASS_TO_ID[class_name])

    return paths, labels


def decode_image(path, label):

    raw = tf.io.read_file(path)

    image = tf.io.decode_image(
        raw,
        channels=1,
        expand_animations=False
    )

    image.set_shape([None, None, 1])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    label = tf.one_hot(
        label,
        depth=len(CLASS_NAMES)
    )

    return image, label


def augment_image(image, label):
    """Apply mild augmentation without changing emotion labels."""

    image = tf.image.random_flip_left_right(image)

    image = rotation(image, training=True)
    image = translation(image, training=True)
    image = zoom(image, training=True)

    image = tf.image.random_brightness(
        image,
        max_delta=0.06
    )

    image = tf.image.random_contrast(
        image,
        lower=0.92,
        upper=1.08
    )

    image = tf.clip_by_value(
        image,
        0.0,
        1.0
    )

    return image, label


def create_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        decode_image,
        num_parallel_calls=AUTOTUNE
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 4000),
            seed=42,
            reshuffle_each_iteration=True
        )

        dataset = dataset.map(
            augment_image,
            num_parallel_calls=AUTOTUNE
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def load_folder_data(data_dir):

    root = Path(data_dir)

    train_dir = root / "train"
    validation_dir = root / "validation"
    test_dir = root / "test"

    train_paths, train_labels = collect_images(train_dir)

    if validation_dir.exists():

        val_paths, val_labels = collect_images(validation_dir)

    else:

        (
            train_paths,
            val_paths,
            train_labels,
            val_labels
        ) = train_test_split(
            train_paths,
            train_labels,
            test_size=0.15,
            random_state=42,
            stratify=train_labels
        )

    train_ds = create_dataset(
        train_paths,
        train_labels,
        training=True
    )

    val_ds = create_dataset(
        val_paths,
        val_labels,
        training=False
    )

    test_ds = None

    if test_dir.exists():

        test_paths, test_labels = collect_images(test_dir)

        test_ds = create_dataset(
            test_paths,
            test_labels,
            training=False
        )

    print("\nImproved FER-2013 dataset loaded!")
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")

    if test_ds is not None:
        print(f"Testing images: {len(test_paths)}")

    print("\nEmotion classes:")

    for idx, name in enumerate(CLASS_NAMES):
        print(f"{idx}: {name}")

    return train_ds, val_ds, test_ds


def load_csv_data(csv_path):

    df = pd.read_csv(csv_path)

    required_columns = {
        "emotion",
        "pixels",
        "Usage"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain: {required_columns}"
        )

    usage_map = {
        "Training": "train",
        "PublicTest": "validation",
        "PrivateTest": "test",
        "train": "train",
        "validation": "validation",
        "test": "test"
    }

    df["split"] = df["Usage"].map(usage_map)
    df = df[df["split"].notna()].copy()

    def make_dataset(split_name, training=False):

        rows = df[df["split"] == split_name]

        if rows.empty:
            return None

        pixels = np.stack([
            np.fromstring(
                str(pixel_string),
                sep=" ",
                dtype=np.float32
            ).reshape(48, 48, 1) / 255.0
            for pixel_string in rows["pixels"]
        ])

        labels = rows["emotion"].astype(int).to_numpy()

        if labels.min() < 0 or labels.max() >= len(CLASS_NAMES):
            raise ValueError("Invalid emotion labels.")

        dataset = tf.data.Dataset.from_tensor_slices(
            (pixels, labels)
        )

        dataset = dataset.map(
            lambda x, y: (
                x,
                tf.one_hot(y, len(CLASS_NAMES))
            ),
            num_parallel_calls=AUTOTUNE
        )

        if training:

            dataset = dataset.shuffle(
                4000,
                seed=42,
                reshuffle_each_iteration=True
            )

            dataset = dataset.map(
                augment_image,
                num_parallel_calls=AUTOTUNE
            )

        return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    train_ds = make_dataset("train", training=True)
    val_ds = make_dataset("validation")
    test_ds = make_dataset("test")

    if train_ds is None or val_ds is None:
        raise ValueError("Training or validation data is missing.")

    print("\nFER-2013 CSV loaded successfully!")

    return train_ds, val_ds, test_ds