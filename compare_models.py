
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Project paths
MODEL_DIR = "model"
KERAS_PATH = os.path.join(
    MODEL_DIR, "plant_disease_model.keras"
)
TFLITE_PATH = os.path.join(
    MODEL_DIR, "plant_disease_model.tflite"
)
TEST_PATH = "test_dataset_new"

IMG_SIZE = 224
BATCH_SIZE = 32

# Check that both models exist
for path in [KERAS_PATH, TFLITE_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

# Load the same test images for both models
generator = ImageDataGenerator(rescale=1.0 / 255)

test_data = generator.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_data.class_indices.keys())
y_true = test_data.classes

print("\nClass mapping:", test_data.class_indices)
print("Test images:", len(y_true))

# Load Keras model
keras_model = tf.keras.models.load_model(KERAS_PATH)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

input_index = input_info["index"]
output_index = output_info["index"]

def predict_tflite(batch):
    predictions = []

    for image in batch:
        image = np.expand_dims(image, axis=0)
        dtype = input_info["dtype"]

        if np.issubdtype(dtype, np.integer):
            scale, zero = input_info["quantization"]
            if scale > 0:
                image = np.round(image / scale + zero)
            limits = np.iinfo(dtype)
            image = np.clip(image, limits.min, limits.max)

        image = image.astype(dtype)

        interpreter.set_tensor(input_index, image)
        interpreter.invoke()

        output = interpreter.get_tensor(output_index)[0]
        output_dtype = output_info["dtype"]

        # Dequantize integer output if necessary
        if np.issubdtype(output_dtype, np.integer):
            scale, zero = output_info["quantization"]
            if scale > 0:
                output = (output.astype(np.float32) - zero) * scale

        predictions.append(output)

    return np.asarray(predictions)

def evaluate_model(name, predict_function):
    test_data.reset()
    all_predictions = []
    inference_seconds = 0.0

    for batch_index in range(len(test_data)):
        images, _ = test_data[batch_index]

        start = time.perf_counter()
        probabilities = predict_function(images)
        inference_seconds += time.perf_counter() - start

        all_predictions.extend(
            np.argmax(probabilities, axis=1)
        )

    y_pred = np.asarray(all_predictions)

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Inference seconds": inference_seconds,
        "Milliseconds per image": (
            inference_seconds / len(y_true) * 1000
        )
    }

    print(f"\n--- {name} ---")
    for key, value in result.items():
        if key != "Model":
            print(f"{key}: {value:.4f}")

    return result

# Run both evaluations
results = []

results.append(
    evaluate_model(
        "Keras MobileNetV2",
        lambda images: keras_model.predict(
            images, verbose=0
        )
    )
)

results.append(
    evaluate_model(
        "TFLite MobileNetV2",
        predict_tflite
    )
)

# Save comparison table
os.makedirs("comparison_results", exist_ok=True)

df = pd.DataFrame(results)
df.to_csv(
    "comparison_results/model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(df.to_string(index=False))
print(
    "\nSaved: comparison_results/model_comparison.csv"
)
