
"""Residual CNN for seven-class FER-2013 emotion recognition."""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def residual_block(x, filters, stride=1, dropout_rate=0.0):
    shortcut = x

    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, padding="same",
            use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)

    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    return x


def build_emotion_cnn(input_shape=(48, 48, 1), num_classes=7):

    inputs = layers.Input(shape=input_shape, name="face_image")

    x = layers.Conv2D(
        32, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = residual_block(x, 32, dropout_rate=0.10)
    x = residual_block(x, 64, stride=2, dropout_rate=0.15)
    x = residual_block(x, 64, dropout_rate=0.15)
    x = residual_block(x, 128, stride=2, dropout_rate=0.20)
    x = residual_block(x, 128, dropout_rate=0.20)
    x = residual_block(x, 192, stride=2, dropout_rate=0.25)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_probabilities"
    )(x)

    model = models.Model(inputs, outputs, name="fer_residual_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-4
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
