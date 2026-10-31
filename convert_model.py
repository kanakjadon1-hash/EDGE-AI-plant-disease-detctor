
import os
import tensorflow as tf

# Paths
keras_path = "model/plant_disease_model.keras"
tflite_path = "model/plant_disease_model.tflite"

# Check that the trained model exists
if not os.path.exists(keras_path):
    raise FileNotFoundError(
        f"Model not found: {keras_path}"
    )

print("Loading trained model...")
model = tf.keras.models.load_model(keras_path)

print("Converting model to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply lightweight dynamic-range optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the converted model
with open(tflite_path, "wb") as file:
    file.write(tflite_model)

size_kb = os.path.getsize(tflite_path) / 1024

print("\nConversion successful!")
print(f"Saved to: {tflite_path}")
print(f"TFLite model size: {size_kb:.2f} KB")
