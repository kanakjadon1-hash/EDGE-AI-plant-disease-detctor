import cv2
import os

folder = "dataset/Tomato___Healthy"
filename = os.listdir(folder)[0]
image_path = os.path.join(folder, filename)

image = cv2.imread(image_path)

# Resize image to 224x224
resized = cv2.resize(image, (224, 224))

cv2.imshow("Original", image)
cv2.imshow("Resized 224x224", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()