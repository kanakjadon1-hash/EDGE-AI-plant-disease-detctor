import cv2
import os

# Open the Healthy tomato folder
folder = "dataset/Tomato___Healthy"

# Take the first image automatically
filename = os.listdir(folder)[0]
image_path = os.path.join(folder, filename)

# Read image
image = cv2.imread(image_path)

# Show image
cv2.imshow("Healthy Tomato Leaf", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
