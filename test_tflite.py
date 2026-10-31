

import os
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model/plant_disease_model.tflite"
IMAGE_PATH = "test images/leaf.jpg"
DATASET_PATH = "dataset"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH)

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(IMAGE_PATH)

# Recover the class order from the training dataset
class_names = sorted(
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Load and preprocess the test image
image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((224, 224))

input_dtype = input_details["dtype"]
image_array = np.asarray(image, dtype=np.float32) / 255.0

if input_dtype != np.float32:
    scale, zero_point = input_details["quantization"]
    if scale > 0:
        image_array = np.round(image_array / scale + zero_point)
    image_array = image_array.astype(input_dtype)

image_array = np.expand_dims(image_array, axis=0)

# Predict
interpreter.set_tensor(input_details["index"], image_array)
interpreter.invoke()

scores = interpreter.get_tensor(output_details["index"])[0]
predicted_index = int(np.argmax(scores))

if predicted_index < len(class_names):
    print("\n--- TFLite Plant Disease Detection ---")
    print("Predicted disease:", class_names[predicted_index])
    print(f"Output score: {scores[predicted_index] * 100:.2f}%")
else:
    print("Class count mismatch. Check the training class order.")
