"""
Created on Sat Sep 23 14:30:31 2023

@author: onurc
Version: V0.1
"""
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to count brown pixels in an image
def count_brown_pixels(image):
    # Define lower and upper bounds for brown color in HSV color space
    lower_brown = np.array([0, 70, 0])
    upper_brown = np.array([20, 255, 200])

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for brown pixels
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Count the number of brown pixels
    brown_pixel_count = cv2.countNonZero(brown_mask)

    return brown_pixel_count

# Define a function to load and preprocess images
def load_and_preprocess_images(folder_path, label):
    images = []
    labels = []

    counter = 0 
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            brown_pixel_count = count_brown_pixels(image)
            images.append(brown_pixel_count)
            labels.append(label)
            print("Banaan", counter, label)
            counter+=1

    return images, labels

# Define paths to your dataset folders (e.g., unripe and ripe bananas)
fase1_path = 'Banaanfase1'
fase2_path = 'Banaanfase2'
fase3_path = 'Banaanfase3'
fase4_path = 'Banaanfase4'

# Load and preprocess images for both classes
fase1_images, fase1_labels = load_and_preprocess_images(fase1_path, 1)
fase2_images, fase2_labels = load_and_preprocess_images(fase2_path, 2)
fase3_images, fase3_labels = load_and_preprocess_images(fase3_path, 3)
fase4_images, fase4_labels = load_and_preprocess_images(fase4_path, 4)

# Combine data from both classes
X = fase1_images + fase2_images + fase3_images + fase4_images
y = fase1_labels + fase2_labels + fase3_labels + fase4_labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(np.array(X_train).reshape(-1, 1), y_train)

# Make predictions on the test data
y_pred = model.predict(np.array(X_test).reshape(-1, 1))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')