"""
Created on Mon Sep 25 12:15:50 2023

@author: onurc
Version: V0.4
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Methods
HISTOGRAM_RGB = True # Method 1: Compute and normalize histogram feature
BROWN_PERCENTAGE = True # Method 2: Compute brown percentage feature
CANNY_WHITE_PIXEL = True # Method 3: Compute white percentage in Canny edge feature

PERCENTAGE_GRAPH = False
KNN_SCATTER = True
KNN_NEIGHBOR_GRAPH = False

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
    

    return white_pixels

def generate_combined_features(image):
    # Method 1: Compute and normalize histogram feature
    if HISTOGRAM_RGB:
        histogram_feature = compute_and_normalize_histogram(image, num_bins, hist_range)
    else:
        histogram_feature = []
    
    # Method 2: Compute brown percentage feature
    if BROWN_PERCENTAGE:
        brown_percentage_feature = compute_brown_percentage(image)
    else:
        brown_percentage_feature = 0

    # Method 3: Compute white percentage in Canny edge feature
    if CANNY_WHITE_PIXEL:
        white_percentage_canny_feature = compute_white_percentage_canny(image)
    else:
        white_percentage_canny_feature = 0

    # Combine features
    combined_features = np.concatenate((histogram_feature, [brown_percentage_feature, white_percentage_canny_feature]))

    return combined_features

def preprocess_image(image_path=0):
    image = cv2.imread(image_path)
    image_bilateralblur = cv2.bilateralFilter(image,9,75,75)
    image_grayscaledbilateralblur = cv2.cvtColor(image_bilateralblur, cv2.COLOR_BGR2GRAY)
    _, binary_image_yellow = cv2.threshold(image_grayscaledbilateralblur, 100, 255, cv2.THRESH_BINARY)
    _, binary_image_brown = cv2.threshold(image_grayscaledbilateralblur, 100, 255, cv2.THRESH_BINARY)
    
    # HSV 
    hsv_image = cv2.cvtColor(image_bilateralblur, cv2.COLOR_BGR2HSV)

    # Define thresholds
    lower_yellow = np.array([15, 50, 70])  # lower threshold for yellow (example: [10, 50, 70])
    upper_yellow = np.array([40, 255, 255])  # upper threshold for yellow (example:  [30, 255, 255])
    lower_brown = np.array([0, 70, 0])  # lower threshold for brown (example: [10, 100, 20])
    upper_brown = np.array([20, 255, 200])  # upper threshold for brown (example: [20, 255, 200])
    
    # Yellow mask (Filter with color range)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow) # Create mask
    segmented_image_yellow = cv2.bitwise_and(image, image, mask=yellow_mask) # Apply yellow mask on original image
    binary_image_yellow = cv2.bitwise_and(binary_image_yellow, binary_image_yellow, mask=yellow_mask) # Apply mask on binary image

    # Brown mask (Filter with color range)
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown) # Create mask
    segmented_image_brown = cv2.bitwise_and(image, image, mask=brown_mask) # Apply brown mask on original image
    binary_image_brown = cv2.bitwise_and(binary_image_brown, binary_image_brown, mask=brown_mask) # Apply mask on binary image

    # Combine yellow and brown
    banana_mask = cv2.add(binary_image_yellow, binary_image_brown)
    kernel = np.ones([3,3])
    banana_mask_dilate = cv2.dilate(banana_mask,kernel,1)
    preprocessed_banana = cv2.bitwise_and(image, image, mask=banana_mask_dilate)

    return preprocessed_banana

# Function to load and preprocess images with multiple feature extraction methods
def load_and_preprocess_images(folder_path, label, num_bins, hist_range):
    features = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            #image = preprocess_image(image_path)

            combined_features = generate_combined_features(image)

            features.append(combined_features)
            labels.append(label)

    return features, labels

# Function to predict the ripeness phase of an individual image
def predict_ripeness(image, knn_classifier, num_bins, hist_range):
    combined_features = generate_combined_features(image)
    
    # Make a prediction using the KNN classifier
    prediction = knn_classifier.predict([combined_features])

    return prediction[0]

# Function to go through a folder and return a accuracy percentage
def get_accuracy_percentage(image_paths, label_phase):
    correct = 0
    wrong = 0
    for filename in os.listdir(image_paths):
            if filename.endswith('.jpg'):
                image_path = os.path.join(image_paths, filename)
                image = cv2.imread(image_path)
                predicted_ripeness = predict_ripeness(image, knn_classifier, num_bins, hist_range) # Predict the ripeness phase of the individual image
                if label_phase != predicted_ripeness:
                    print(f"{predicted_ripeness} Wrong {image_path}")
                    wrong+=1
                else:
                    print(f"{predicted_ripeness} Correct {image_path}")
                    correct+=1
    #print(f"wrong: {wrong} and correct: {correct}")
    return (correct /(correct+wrong)*100)

# Plot scatter for KNN
def Single_knn_scatter(feature_vector, color):
    for i in range(len(feature_vector)):
        test1_outcome = feature_vector[1][i]*100 # Brown percentage
        test2_outcome = feature_vector[2][i] # Canny white pixel count

        print(test1_outcome, test2_outcome)
        #plt.scatter (unripe_features[1][i],unripe_features[2][i],c='b')
        plt.scatter(test1_outcome,test2_outcome,c=color)
        plt.xlabel("outcome 1, brown percentage")
        plt.ylabel("outcome 2, white canny pixel")
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.show()
    for i in labels:
        print (i,colors[(i)])

    

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# ============================================
# Predict individual image
image = cv2.imread('Banaanfase3\Banaan3_41.jpg') # read image
combined_features_image = generate_combined_features(image)
print("Image results: ", combined_features_image[1]*100, combined_features_image[2])
predicted_ripeness = predict_ripeness(image, knn_classifier, num_bins, hist_range) # Predict the ripeness phase of the individual image
print(f'Predicted Ripeness: {predicted_ripeness}') # Print the predicted ripeness phase (1, 2, 3, or 4 for unripe, semi-ripe, ripe, or over-ripe)

# ============================================
# Graph accuracy for each file path
if PERCENTAGE_GRAPH:
    fase1_percentage_correct = get_accuracy_percentage(fase1_path, 1)
    fase2_percentage_correct = get_accuracy_percentage(fase2_path, 2)
    fase3_percentage_correct = get_accuracy_percentage(fase3_path, 3)

    percentage_correct = [fase1_percentage_correct, fase2_percentage_correct, fase3_percentage_correct]
    percentage_incorrect = [100-fase1_percentage_correct, 100-fase2_percentage_correct, 100-fase3_percentage_correct]

    x = range(len(labels))
    #plt.ion()
    plt.subplot(1,2,1)
    #plt.figure(figsize=(10, 6))
    plt.bar(x, percentage_correct, width=0.4, label='Correct Predictions')
    plt.bar(x, percentage_incorrect, width=0.4, label='Incorrect Predictions', bottom=percentage_correct)
    plt.xlabel('Phase')
    plt.ylabel('Percentage')
    plt.title('Percentage graph by phases')
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()

# ============================================


#Knn_scatter(unripe_features, 'b')


# Plot scatter for KNN
# unripe_features, semi_ripe_features, ripe_features
if KNN_SCATTER:
    colors=['b','c','g','k,','m','r','w','y']
    length = max(len(unripe_features), len(ripe_features), len(semi_ripe_features))
    x = [None] * length
    y = [None] * length
    for i in range(len(labels)):
        for j in range(length):
            if i == 0:
                test1_outcome = unripe_features[1][j]*100
                test2_outcome = unripe_features[2][j]
                x[j] = test1_outcome
                y[j] = test2_outcome
                color = 'b'
            if i == 1:
                test1_outcome = ripe_features[1][j]*100
                test2_outcome = ripe_features[2][j]
                x[j] = test1_outcome
                y[j] = test2_outcome
                color = 'r'
            if i == 2:
                test1_outcome = semi_ripe_features[1][j]*100
                test2_outcome = semi_ripe_features[2][j]
                x[j] = test1_outcome
                y[j] = test2_outcome
                color = 'g'
            #print(test1_outcome, test2_outcome)
            #plt.scatter (unripe_features[1][i],unripe_features[2][i],c='b')
        plt.scatter (x,y,c=color, label=f"phase {i+1}")
        plt.xlabel("outcome 1, brown percentage")
        plt.ylabel("outcome 2, white canny pixel")

    #plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================
# Plot n for KNN
# Define a range of n_neighbors values to test
if KNN_NEIGHBOR_GRAPH:

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
"""
# ============================================
# Plot scatter for KNN
# unripe_features, semi_ripe_features, ripe_features
if KNN_SCATTER:
    colors=['b','c','g','k,','m','r','w','y']
    for i in range(len(semi_ripe_features)):
        test1_outcome = semi_ripe_features[1][i]*100
        test2_outcome = semi_ripe_features[2][i]

        print(test1_outcome, test2_outcome)
        #plt.scatter (unripe_features[1][i],unripe_features[2][i],c='b')
        plt.scatter (test1_outcome,test2_outcome,c='b')
        plt.xlabel("outcome 1, brown percentage")
        plt.ylabel("outcome 2, white canny pixel")
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.show()
    for i in labels:
        print (i,colors[(i)])
"""