"""
Created on Mon Sep 25 12:15:50 2023

@author: onurc
Version: V0.1
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Function to compute and normalize histograms for grayscale images
def compute_and_normalize_histogram(image, num_bins, hist_range):
    hist, bins = np.histogram(image, bins=num_bins, range=hist_range)
    normalized_hist = hist / hist.sum()
    return normalized_hist

# Function to count the percentage of brown pixels in an image
def compute_brown_percentage(image):
    # Define a threshold for brown color (adjust as needed)
    lower_brown = np.array([10, 60, 20], dtype=np.uint8)
    upper_brown = np.array([60, 160, 90], dtype=np.uint8)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask to select brown pixels
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

    # Calculate the percentage of brown pixels
    total_pixels = image.shape[0] * image.shape[1]
    brown_pixels = np.count_nonzero(brown_mask)
    brown_percentage = (brown_pixels / total_pixels) * 100

    return brown_percentage

# Function to calculate the percentage of white pixels in a Canny edge image
def compute_white_percentage_canny(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)

    # Calculate the percentage of white pixels in the Canny edge image
    total_pixels = edges.shape[0] * edges.shape[1]
    white_pixels = np.count_nonzero(edges)
    white_percentage = (white_pixels / total_pixels) * 100

    return white_percentage

# Function to load and preprocess images with multiple feature extraction methods
def load_and_preprocess_images(folder_path, label, num_bins, hist_range):
    features = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Compute and normalize histogram feature
            histogram_feature = compute_and_normalize_histogram(image, num_bins, hist_range)

            # Compute brown percentage feature
            brown_percentage_feature = compute_brown_percentage(image)

            # Compute white percentage in Canny edge feature
            white_percentage_canny_feature = compute_white_percentage_canny(image)

            # Combine features
            combined_features = np.concatenate((histogram_feature, [brown_percentage_feature, white_percentage_canny_feature]))

            features.append(combined_features)
            labels.append(label)

    return features, labels

# Function to predict the ripeness phase of an individual image
def predict_ripeness(image, knn_classifier, num_bins, hist_range):

    # Compute and normalize histogram feature
    histogram_feature = compute_and_normalize_histogram(image, num_bins, hist_range)

    # Compute brown percentage feature
    brown_percentage_feature = compute_brown_percentage(image)

    # Compute white percentage in Canny edge feature
    white_percentage_canny_feature = compute_white_percentage_canny(image)

    # Combine features into a single feature vector
    feature_vector = np.concatenate((histogram_feature, [brown_percentage_feature, white_percentage_canny_feature]))

    # Make a prediction using the KNN classifier
    prediction = knn_classifier.predict([feature_vector])

    return prediction[0]


# Define paths to your dataset folders for each ripeness phase
fase1_path = 'Banaanfase1'
fase2_path = 'Banaanfase2'
fase3_path = 'Banaanfase3'
#fase4_path = 'Banaanfase4'

# Define labels for each ripeness phase (0, 1, 2, and 3 for unripe, semi-ripe, ripe, and over-ripe)
labels = [1, 2, 3]

# Define the number of bins and histogram range
num_bins = 256
hist_range = (0, 256)

# Load and preprocess images for all ripeness phases with multiple features
unripe_features, unripe_labels = load_and_preprocess_images(fase1_path, labels[0], num_bins, hist_range)
semi_ripe_features, semi_ripe_labels = load_and_preprocess_images(fase2_path, labels[1], num_bins, hist_range)
ripe_features, ripe_labels = load_and_preprocess_images(fase3_path, labels[2], num_bins, hist_range)
#over_ripe_features, over_ripe_labels = load_and_preprocess_images(fase4_path, labels[3], num_bins, hist_range)

# Combine data from all ripeness phases
X = unripe_features + semi_ripe_features + ripe_features
y = unripe_labels + semi_ripe_labels + ripe_labels
#X = unripe_features + semi_ripe_features + ripe_features + over_ripe_features
#y = unripe_labels + semi_ripe_labels + ripe_labels + over_ripe_labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Predict individual image
image = cv2.imread('Banaanfase2\Banaan2_13.jpg') # read image
# Predict the ripeness phase of the individual image
predicted_ripeness = predict_ripeness(image, knn_classifier, num_bins, hist_range)

# Print the predicted ripeness phase (1, 2, 3, or 4 for unripe, semi-ripe, ripe, or over-ripe)
print(f'Predicted Ripeness: {predicted_ripeness}')


# Plot n for KNN
# Define a range of n_neighbors values to test
n_neighbors_values = range(1, 21)  # Vary from 1 to 20

# Initialize lists to store accuracy values
accuracy_values = []


# Loop through different n_neighbors values and compute accuracy
for n_neighbors in n_neighbors_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)

# Create a plot to analyze the impact of n_neighbors on accuracy
plt.figure(figsize=(10, 6))
plt.plot(n_neighbors_values, accuracy_values, marker='o', linestyle='-', color='b')
plt.title('KNN Classifier Accuracy vs. n_neighbors')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(n_neighbors_values)
plt.show()

