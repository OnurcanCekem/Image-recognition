"""
Created on Sat Sep 23 14:30:31 2023

@author: onurc
Version: V0.2
"""
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

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

# Function to count white pixels in a Canny edge image
def count_white_pixels_canny(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # Count the number of white pixels (edges)
    white_pixel_count = cv2.countNonZero(edges)

    return white_pixel_count

# Function to compute the histogram of a grayscale in an image
def compute_histogram(image):
    # Convert the image to gray color space
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the histogram of grayscale in the masked region
    #histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    #histogram /= histogram.sum()

    hist,bins = np.histogram(gray_image.ravel(),256,[0,256])
    # Normalize the histogram
    normalized_hist = hist / hist.sum()

    return normalized_hist

# Define a function to load and preprocess images
def load_and_preprocess_images(folder_path, label, number_images):
    brown_pixel_counts = []
    white_pixel_counts = []
    histograms = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            brown_pixel_count = count_brown_pixels(image)
            white_pixel_count = count_white_pixels_canny(image)
            histogram = compute_histogram(image)
            brown_pixel_counts.append(brown_pixel_count)
            white_pixel_counts.append(white_pixel_count)
            histograms.append(histogram)
            labels.append(label)
            number_images +=1
    print("Amount of images: ", number_images," with label: ", label)
    
    return brown_pixel_counts, white_pixel_counts, histograms, labels


counter = 0
normalized_histograms = []
# Define paths to your dataset folders (e.g., unripe and ripe bananas)
fase1_path = 'Banaanfase1'
fase2_path = 'Banaanfase2'
fase3_path = 'Banaanfase3'
fase4_path = 'Banaanfase4'

# Load and preprocess images for both classes
fase1_brown_pixel, fase1_white_pixel, fase1_histograms, fase1_labels = load_and_preprocess_images(fase1_path, 1, counter)
fase2_brown_pixel, fase2_white_pixel, fase2_histograms, fase2_labels = load_and_preprocess_images(fase2_path, 2, counter)
fase3_brown_pixel, fase3_white_pixel, fase3_histograms, fase3_labels = load_and_preprocess_images(fase3_path, 3, counter)
fase4_brown_pixel, fase4_white_pixel, fase4_histograms, fase4_labels = load_and_preprocess_images(fase4_path, 4, counter)
#fase1_images, fase1_labels = load_and_preprocess_images(fase1_path, 1)
#fase2_images, fase2_labels = load_and_preprocess_images(fase2_path, 2)
#fase3_images, fase3_labels = load_and_preprocess_images(fase3_path, 3)
#fase4_images, fase4_labels = load_and_preprocess_images(fase4_path, 4)
# Load and preprocess images for both classes

# Combine data from both classes
#X = fase1_images + fase2_images + fase3_images + fase4_images
#y = fase1_labels + fase2_labels + fase3_labels + fase4_labels

X_brown = fase1_brown_pixel + fase2_brown_pixel + fase3_brown_pixel + fase4_brown_pixel
X_white = fase1_white_pixel + fase2_white_pixel + fase3_white_pixel + fase4_white_pixel
X_histograms = fase1_histograms + fase2_histograms + fase3_histograms + fase4_histograms
y = fase1_labels + fase2_labels + fase3_labels + fase4_labels
#X_brown = fase2_brown_pixel + fase3_brown_pixel
#X_white = fase2_white_pixel + fase3_white_pixel
#X_histograms = fase2_histograms + fase3_histograms
#y = fase2_labels + fase3_labels

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X_histograms, y, test_size=0.2, random_state=42)
X_brown_train, X_brown_test, X_white_train, X_white_test, X_histograms_train, X_histograms_test, y_train, y_test = train_test_split(
    X_brown, X_white, X_histograms, y, test_size=0.2, random_state=42
)


#X_train = np.column_stack((X_brown_train, X_white_train, X_histograms_train))
X_train = np.column_stack((X_brown_train, X_white_train))
X_train = np.concatenate((X_train, X_histograms_train), axis=1)
#X_test = np.column_stack((X_brown_test, X_white_test, X_histograms_test))
X_test = np.column_stack((X_brown_test, X_white_test))
X_test = np.concatenate((X_test, X_histograms_test), axis=1)

#X_train = np.column_stack((X_brown_train, X_white_train))
#X_train = np.concatenate((X_train, X_histograms_train), axis=1)
#X_test = np.column_stack((X_brown_test, X_white_test))
#X_test = np.concatenate((X_test, X_histograms_test), axis=1)


# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("phases: 1,2,3,4")
print("methods: brown count, white pixel canny, histogram")
print(f'Accuracy: {accuracy * 100:.2f}%')

# Attempt to make read image and predict
#image = cv2.imread('Banaanfase1\Banaan1_1.jpg') # read image
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#y_pred = model.predict(gray)
#accuracy2 = accuracy_score(y_test, y_pred)
#print(f'Accuracy: {accuracy2 * 100:.2f}%')
