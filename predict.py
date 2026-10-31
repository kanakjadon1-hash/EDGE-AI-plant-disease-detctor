print("Prediction program started!")
import os
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
MODEL_PATH = "model/plant_disease_model.keras"
DATASET_PATH = "dataset"

model = tf.keras.models.load_model(MODEL_PATH)

# Get class names from dataset folders
class_names = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

# Enter the image path you want to test
IMAGE_PATH = "test images/leaf.jpg"

# Read the leaf image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: Image not found!")
    print("Check the image path and filename.")
else:
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to 224 x 224
    image = cv2.resize(image, (224, 224))

    # Normalize exactly as in training
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(image, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index]) * 100

    print("\n--- Plant Disease Detection ---")
    print("Prediction:", class_names[predicted_index])
    print(f"Confidence: {confidence:.2f}%")
