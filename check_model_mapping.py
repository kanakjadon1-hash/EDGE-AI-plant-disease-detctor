
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model(
    "model/plant_disease_model.keras"
)

test_data = ImageDataGenerator(
    rescale=1.0 / 255
).flow_from_directory(
    "test_dataset_new",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

print("\nTest folder mapping:")
for name, index in test_data.class_indices.items():
    print(index, "->", name)

print("\nModel output shape:", model.output_shape)
print("Model input shape:", model.input_shape)

print("\nModel final layers:")
for layer in model.layers[-5:]:
    print(layer.name, layer.output.shape)


print("\nModel output class count:", model.output_shape[-1])
print("Test class count:", len(test_data.class_indices))

print("\nTest classes in model output order:")
for index, name in sorted(
    (index, name)
    for name, index in test_data.class_indices.items()
):
    print(index, "->", name)

