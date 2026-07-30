"""Deterministic Iris classifier using TensorFlow/Keras."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class IrisData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    target_names: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    accuracy: float
    predictions: np.ndarray
    expected: np.ndarray
    target_names: np.ndarray


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_data(test_size: float = 0.30, seed: int = 42) -> IrisData:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=test_size,
        random_state=seed,
        stratify=iris.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return IrisData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        target_names=iris.target_names,
    )


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(12, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(epochs: int = 60, seed: int = 42, verbose: int = 0) -> TrainingResult:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    configure_determinism(seed)
    data = load_data(seed=seed)
    model = build_model()
    model.fit(
        data.x_train,
        data.y_train,
        epochs=epochs,
        batch_size=8,
        validation_split=0.20,
        verbose=verbose,
    )
    _, accuracy = model.evaluate(data.x_test, data.y_test, verbose=0)
    probabilities = model.predict(data.x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    return TrainingResult(
        accuracy=float(accuracy),
        predictions=predictions,
        expected=data.y_test,
        target_names=data.target_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = train(args.epochs, args.seed, int(args.verbose))
    print(f"Test accuracy: {result.accuracy:.2%}")
    for index in range(min(5, len(result.predictions))):
        expected = result.target_names[result.expected[index]]
        predicted = result.target_names[result.predictions[index]]
        print(f"Sample {index + 1}: expected={expected}, predicted={predicted}")


if __name__ == "__main__":
    main()
