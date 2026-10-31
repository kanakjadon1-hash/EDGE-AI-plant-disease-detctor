
import os
import time
import csv
import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------
# Experiment configuration
# -----------------------------
DATASET_PATH = "dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

os.makedirs("ablation_results", exist_ok=True)

# Use identical class order and validation split in both runs
def create_generators(use_augmentation):
    if use_augmentation:
        train_generator = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
    else:
        train_generator = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2
        )

    train_data = train_generator.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED
    )

    val_data = train_generator.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED
    )

    return train_data, val_data


def build_model(num_classes):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_experiment(name, use_augmentation):
    print(f"\nStarting experiment: {name}")

    tf.keras.backend.clear_session()

    train_data, val_data = create_generators(use_augmentation)
    model = build_model(train_data.num_classes)

    start = time.time()

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        verbose=1
    )

    training_seconds = time.time() - start

    # Evaluate on the validation split, not the test set.
    val_data.reset()
    probabilities = model.predict(val_data, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    y_true = val_data.classes

    results = {
        "Experiment": name,
        "Augmentation": use_augmentation,
        "Validation Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall (macro)": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1-score (macro)": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Training Time (seconds)": round(training_seconds, 2)
    }

    model.save(f"ablation_results/{name}.keras")
    print(results)
    return results


if __name__ == "__main__":
    results = []

    results.append(
        run_experiment("baseline_augmentation", True)
    )

    results.append(
        run_experiment("no_augmentation", False)
    )

    df = pd.DataFrame(results)
    df.to_csv("ablation_results/results.csv", index=False)

    print("\nABLATlON EXPERIMENT RESULTS")
    print(df.to_string(index=False))
    print("\nResults saved in ablation_results/results.csv")
