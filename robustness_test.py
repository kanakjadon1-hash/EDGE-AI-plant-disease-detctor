
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path

# Paths
MODEL_PATH = "model/plant_disease_model.keras"
TEST_PATH = "test_dataset_new"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_leaf_mold"
]

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Image transformations
def apply_perturbation(image, condition):

    if condition == "low_light":
        image = ImageEnhance.Brightness(image).enhance(0.4)

    elif condition == "bright":
        image = ImageEnhance.Brightness(image).enhance(1.6)

    elif condition == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=2))

    elif condition == "rotation":
        image = image.rotate(15)

    elif condition == "noise":
        arr = np.array(image).astype(np.float32)
        noise = np.random.normal(0, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)

    return image


conditions = [
    "original",
    "low_light",
    "bright",
    "blur",
    "rotation",
    "noise"
]

results = {}

# Test each image under different conditions
for condition in conditions:

    correct = 0
    total = 0

    for class_name in CLASS_NAMES:

        folder = Path(TEST_PATH) / class_name

        if not folder.exists():
            print("Missing folder:", folder)
            continue

        for file in folder.iterdir():

            if file.suffix.lower() not in [
                ".jpg", ".jpeg", ".png"
            ]:
                continue

            image = Image.open(file).convert("RGB")
            image = apply_perturbation(image, condition)
            image = image.resize(IMG_SIZE)

            img_array = np.array(image).astype(
                np.float32
            ) / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(
                img_array, verbose=0
            )

            predicted_class = CLASS_NAMES[
                np.argmax(prediction[0])
            ]

            if predicted_class == class_name:
                correct += 1

            total += 1

    accuracy = (
        (correct / total) * 100 if total > 0 else 0
    )

    results[condition] = accuracy

# Display results
print("\nROBUSTNESS TEST RESULTS")
print("-----------------------")

for condition, accuracy in results.items():
    print(f"{condition}: {accuracy:.2f}%")

# Save results
with open("robustness_results.txt", "w") as f:
    for condition, accuracy in results.items():
        f.write(f"{condition}: {accuracy:.2f}%\n")

print("\nResults saved to robustness_results.txt")
