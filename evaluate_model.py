
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Model and test dataset paths
MODEL_PATH = "model/plant_disease_model.keras"
TEST_PATH = "test_dataset_new"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Keep this order consistent with your model's training classes
CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___Healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold"
]

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load test images without shuffling
test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)
print("Test class mapping:", test_data.class_names)
print("Number of classes:", len(test_data.class_names))
print("Model output classes:", model.output_shape[-1])
# Verify the folder-to-class mapping
print("Dataset class mapping:", test_data.class_names)

# Ensure the dataset order matches the model's class order
CLASS_NAMES = test_data.class_names

print("Test class mapping:", CLASS_NAMES)
print("Number of classes:", len(CLASS_NAMES))
print("Model output classes:", model.output_shape[-1])

if len(CLASS_NAMES) != model.output_shape[-1]:
    raise ValueError("The number of test classes does not match the model.")

# Collect true labels and model predictions
y_true = np.concatenate([
    labels.numpy().astype(int)
    for _, labels in test_data
])

probabilities = model.predict(test_data, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)
f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

print("\n--- Model Performance ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Macro F1:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=range(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    zero_division=0
))

# Plot confusion matrix
cm = confusion_matrix(
    y_true, y_pred, labels=range(len(CLASS_NAMES))
)

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Tomato Disease Detection - Confusion Matrix")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("\nSaved confusion_matrix.png")
