
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_PATH = "dataset"
TEST_PATH = "test_dataset_new"

for path in [TRAIN_PATH, TEST_PATH]:
    print(f"\n{'=' * 45}")
    print("FOLDER:", path)

    if not os.path.isdir(path):
        print("ERROR: Folder not found!")
        continue

    generator = ImageDataGenerator(rescale=1.0 / 255)

    data = generator.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    print("Class mapping:", data.class_indices)
    print("Total images:", data.samples)
    print("Images per class:")

    for name, index in data.class_indices.items():
        count = int((data.classes == index).sum())
        print(f"  {name}: {count}")
