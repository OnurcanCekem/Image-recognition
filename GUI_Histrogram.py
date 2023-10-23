"""
Created on Mon Oct 02 10:22:39 2023

@author: onurc
Version: V0.8
Description: 
"""
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import Predictknn

global image_index
global image_global
image_index = 0
global mean_image
global dominant_frequency

test1_outcome=[]
test2_outcome=[]

# Function to close the program
def close_program():
    root.quit()

# Function to load and display the next image
def load_next_image():
    global image_index, image_paths
    if image_index < len(image_paths) - 1:
        image_index += 1
        load_image(image_index)

# Function to load and display the previous image
def load_previous_image():
    global image_index, image_paths
    if image_index > 0:
        image_index -= 1
        load_image(image_index)

# Function to load and display a specific image, automated
def load_image(index):
    global image_global
    cv2.destroyAllWindows() 
    image_path = image_paths[index]
    image_global = cv2.imread(image_path) # set global image
    img = Image.open(image_path)
    img.thumbnail((400, 400))  # Resize the image if needed
    photo = ImageTk.PhotoImage(img)

    # Update the label with the new image
    image_label.config(image=photo)
    image_label.image = photo

    #run the tests
    predicted_ripeness = predict_ripeness(image_global, knn_classifier, num_bins, hist_range) # Predict the ripeness phase of the individual image
    brown_percentage, white_pixel_canny = generate_combined_features(image_global)
    test1_outcome_label.config(text = f"Brown percentage image: {brown_percentage}") #shows as text in the window
    test2_outcome_label.config(text = f"White pixel canny image: {white_pixel_canny}") #shows as text in the window
    predicted_ripeness_label.config(text = f"Predicted ripeness: {predicted_ripeness}") #shows as text in the window
    print(f'Predicted Ripeness: {predicted_ripeness}') # Print the predicted ripeness phase (1, 2, 3, or 4 for unripe, semi-ripe, ripe, or over-ripe)

    test1_mean_img()
    display_unpreprocessed_histogram(image_global, index)
    
    preprocessed_image = preprocess_image(image_path)
    display_rgb_histogram(preprocessed_image, index) # Display the RGB histogram of the current image
    print("Image path: ",image_path)

# Function to load and display a specific image, manually
def select_image_manually():
    global image_global
    cv2.destroyAllWindows() 
    image_path = filedialog.askopenfilename(title="Select an Image", filetype=(('image files','*.jpg'),('all files','*.*')))
    image_global = cv2.imread(image_path)
    #cv2.imshow("Yee",image)

    img = Image.open(image_path)
    img.thumbnail((400, 400))  # Resize the image if needed
    photo = ImageTk.PhotoImage(img)
    

    # Update the label with the new image
    image_label.config(image=photo)
    image_label.image = photo

    #run the tests
    predicted_ripeness = predict_ripeness(image_global, knn_classifier, num_bins, hist_range) # Predict the ripeness phase of the individual image
    brown_percentage, white_pixel_canny = generate_combined_features(image_global)
    test1_outcome_label.config(text = f"Brown percentage image: {brown_percentage}") #shows as text in the window
    test2_outcome_label.config(text = f"White pixel canny image: {white_pixel_canny}") #shows as text in the window
    predicted_ripeness_label.config(text = f"Predicted ripeness: {predicted_ripeness}") #shows as text in the window

    print(f'Predicted Ripeness: {predicted_ripeness}') # Print the predicted ripeness phase (1, 2, 3, or 4 for unripe, semi-ripe, ripe, or over-ripe)

    test1_mean_img()
    display_unpreprocessed_histogram(image_global)
    preprocessed_image = preprocess_image(image_path)
    display_rgb_histogram(preprocessed_image) # Display the RGB histogram of the current image
    
    file_name = os.path.basename(image_path)
    selected_ripeness = file_name[6]
    selected_ripeness_label.config(text = f"Selected ripeness: {selected_ripeness}") #shows as text in the window
    #print("Selected ripeness: ", selected_ripeness)
    #print(os.path.splitext(file_name)[0])
    #phase = image_path
    
    #print("Image path: ",image_path)


    #image = preprocess_image(image_path)
    # Display the RGB histogram of the current image

# Function to load and preprocess images with multiple feature extraction methods
def preprocess_image(image_path=0):
    global image_global
    if image_path != 0:
        image = cv2.imread(image_path)
    else:
        image = image_global
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

# Function to display the RGB histogram of an image
def display_rgb_histogram(image, index=0):
    #plt.ion()
    b, g, r = cv2.split(image)
    
    plt.subplot(2,2,2)
    
    # Calculate the histograms for each channel
    hist_b = cv2.calcHist([b], [0], None, [256], [1, 255])
    hist_g = cv2.calcHist([g], [0], None, [256], [1, 255])
    hist_r = cv2.calcHist([r], [0], None, [256], [1, 255])

    # Create a Matplotlib figure for the RGB histogram

    plt.title(f'Preprocessed RGB #{index}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Plot the histograms for each channel
    plt.plot(hist_b, color='b', label='Blue')
    plt.plot(hist_g, color='g', label='Green')
    plt.plot(hist_r, color='r', label='Red')

    #plt.plot(hist_r, color='r', label='Red')
    #plt.title(title)
    # Add a legend to distinguish the channels
    plt.show()
    plt.legend()
    # Find the dominant color and its frequency

    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [1, 255, 1, 255, 1, 255])
    dominant_color = np.unravel_index(hist.argmax(), hist.shape)
    dominant_frequency = hist[dominant_color]
    #print("Dominant Color (RGB):", dominant_color)
    #print("Dominant Frequency:", dominant_frequency)
    dominant_frequency_label.config(text = f"Dominant frequency image: {dominant_frequency}") #shows as text in the window

def display_unpreprocessed_histogram(image, index=0):    
    plt.clf()
    plt.ion()
    b, g, r = cv2.split(image)
    
    plt.subplot(2,2,1)
    
    # Calculate the histograms for each channel
    hist_b = cv2.calcHist([b], [0], None, [256], [1, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [1, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [1, 256])
    
    # Create a Matplotlib figure for the RGB histogram

    plt.title(f'Unpreprocessed RGB #{index}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Plot the histograms for each channel
    plt.plot(hist_b, color='b', label='Blue')
    plt.plot(hist_g, color='g', label='Green')
    plt.plot(hist_r, color='r', label='Red')

    #plt.plot(hist_r, color='r', label='Red')
    #plt.title(title)
    # Add a legend to distinguish the channels
    plt.legend()
    plt.show()


# Function to detect yellow pixels
def detect_yellow():
    global image_paths, image_index
    global image_global
    
    img = image_global
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for yellow color in HSV
    lower_yellow = np.array([15, 50, 70], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

    # Create a mask to extract yellow pixels
    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    # Display the image with yellow pixels highlighted
    yellow_highlighted = cv2.bitwise_and(img, img, mask=yellow_mask)
    cv2.imshow('Yellow Detection', yellow_highlighted)
    #detect_escape_key()
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

# Function to detect brown pixels
def detect_brown():
    global image_paths, image_index

    global image_global
    img = image_global
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for brown color in HSV
    lower_brown = np.array([0, 70, 0], dtype=np.uint8)
    upper_brown = np.array([20, 255, 200], dtype=np.uint8)

    # Create a mask to extract brown pixels
    brown_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)

    # Display the image with brown pixels highlighted
    brown_highlighted = cv2.bitwise_and(img, img, mask=brown_mask)
    cv2.imshow('Brown Detection', brown_highlighted)
        #detect_escape_key()
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

# Function to detect brown pixels
def detect_canny():

    global image_global
    img = image_global

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)

    cv2.imshow("Canny edge",edges)

# Function to browse and select a folder of images
def browse_folder():
    global image_paths, image_index
    folder_path = filedialog.askdirectory()
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]
    image_index = -1  # Start from the first image (index 0) when a new folder is selected
    load_next_image()

# Function to sort folder locations for buttons
def select_phase(index):
    #cv2.destroyAllWindows() 
    #plt.clf()
    global image_paths, image_index
    #folder_path = 'Banaanfase2'
    if index == 1:
        folder_path = 'Banaanfase1'
    elif index == 2:
        folder_path = 'Banaanfase2'
    elif index == 3:
        folder_path = 'Banaanfase3'
    elif index == 4:
        folder_path = 'Banaanfase4'
    
    selected_ripeness_label.config(text = f"Selected ripeness: {index}") #shows as text in the window

    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]
    image_index = -1  # Start from the first image (index 0) when a new folder is selected
    load_next_image()

def test1_mean_img():
    global image_global
    global mean_image
    mean_image = np.mean(image_global)
    #print("Mean Image: ", mean_image)
    mean_image_label.config(text = f"Mean image: {mean_image}") #shows as text in the window

def detect_escape_key():
    while(True):
        c = cv2.waitKey(0)
        if c == 27 & 0xFF:
            cv2.destroyAllWindows()

# Create the main application window
root = tk.Tk()
root.title("Image Viewer")
#fig = plt.figure(figsize=(6, 4))

# Create a label to display the images
image_label = tk.Label(root)
image_label.pack(padx=20, pady=20)

# Create buttons to browse for a folder, load the next image, and load the previous image
browse_button = tk.Button(root, bg='green', text="Browse Folder", command=browse_folder)
next_button = tk.Button(root, bg='green', text="Next Image", command=load_next_image)
previous_button = tk.Button(root, bg='green', text="Previous Image", command=load_previous_image)
close_button = tk.Button(root, bg='red', text="Close Program", command=close_program)
yellow_button = tk.Button(root, bg='yellow', text="Detect Yellow", command=detect_yellow)
brown_button = tk.Button(root, bg='brown', text="Detect Brown", command=detect_brown)
canny_button = tk.Button(root, bg='gray', text="Detect canny", command=detect_canny)
individual_image = tk.Button(root, bg='green', text="Select image", command=select_image_manually)
fase1_path = tk.Button(root, bg='green', text="Select phase 1", command=lambda: select_phase(1))
fase2_path = tk.Button(root, bg='green', text="Select phase 2", command=lambda: select_phase(2))
fase3_path = tk.Button(root, bg='green', text="Select phase 3", command=lambda: select_phase(3))
fase4_path = tk.Button(root, bg='green', text="Select phase 4", command=lambda: select_phase(4))
mean_image_label = tk.Label(root, text="Waiting image...") #shows as text in the window
dominant_frequency_label = tk.Label(root, text="Waiting image...") #shows as text in the window
selected_ripeness_label = tk.Label(root, text="Waiting image...") #shows as text in the window
predicted_ripeness_label = tk.Label(root, text="Waiting image...") #shows as text in the window
test1_outcome_label = tk.Label(root, text="Waiting image...") #shows as text in the window
test2_outcome_label = tk.Label(root, text="Waiting image...") #shows as text in the window

#mean_label = tk.Label(root, text=f"Mean: {mean_image}") #shows as text in the window
browse_button.pack(pady=10)
individual_image.pack(padx=10)
fase1_path.pack(padx=10)
fase2_path.pack(padx=10)
fase3_path.pack(padx=10)
fase4_path.pack(padx=10)
previous_button.pack(side='left', pady=10)
next_button.pack(side='right', padx=10)
yellow_button.pack(pady=0)
brown_button.pack(pady=0)
canny_button.pack(pady=0)
close_button.pack(padx=20, pady=10)
mean_image_label.pack(padx=0, pady=0)
dominant_frequency_label.pack(padx=0, pady=0)
selected_ripeness_label.pack(padx=0, pady=0)
predicted_ripeness_label.pack(padx=0, pady=0)
test1_outcome_label.pack(padx=0, pady=0)
test2_outcome_label.pack(padx=0, pady=0)

# Initialize global variables
image_paths = []
image_index = -1


# ==================================================================================
# ==================================================================================
# Predictknn.py code
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

BROWN_PERCENTAGE = True # Method 1: Compute brown percentage feature
CANNY_WHITE_PIXEL = True # Method 2: Compute white percentage in Canny edge feature
HISTOGRAM_RGB = True # Method 3: Compute and normalize histogram feature

# Function to count the percentage of brown pixels in an image
def test2_compute_brown_percentage(image):
    # Define a threshold for brown color (adjust as needed)
    #lower_brown = np.array([10, 60, 20], dtype=np.uint8)
    #upper_brown = np.array([60, 160, 90], dtype=np.uint8)
    lower_brown = np.array([0, 70, 0], dtype=np.uint8)
    upper_brown = np.array([20, 255, 200], dtype=np.uint8)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask to select brown pixels
    brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
    # Calculate the percentage of brown pixels
    max_contour = get_contour(image)
    total_pixels = max_contour
    brown_pixels = np.count_nonzero(brown_mask)
    brown_percentage = (brown_pixels / total_pixels)

    return brown_percentage

# Function to calculate the amount of white pixels in a Canny edge image
def test3_compute_white_pixel_canny(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)

    # Calculate the percentage of white pixels in the Canny edge image
    max_contour = get_contour(image)
    total_pixels = max_contour
    white_pixels = np.count_nonzero(edges)
    white_percentage = (white_pixels / total_pixels) * 100

    return white_pixels

# input: image
# return: total pixels from the highest contour
def get_contour(image):
    correct_banana = 0
    fake_banana = 0
    max_contour = 0
    percentage_area = 0.08 # How much % of the image should be the banana (1.00 is 100%)
    area = image.shape[0]*image.shape[1]*percentage_area # Amount of pixels required for area to identify banana

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < area:
            continue

        current_contour = cv2.contourArea(contour)
        if current_contour > max_contour:
            max_contour = current_contour
            x, y, w, h = cv2.boundingRect(contour)

        # THIS DOES NOTHING AT THE MOMENT
        # Check ratio , if rectangle it's likely a banana (ratio is roughly 2:1 but depends on image dimensions)
        if (w*1.2 < h) or (h*1.2 < w): # If banana is found (condition is met)
            correct_banana +=1
        else:
            fake_banana +=1

    #print(w+x,h+y)
    return max_contour
    #print(image.shape[0], image.shape[1])

def generate_combined_features(image):
    combined_features = [None] * 2
    # Method 1: Compute and normalize histogram feature
    #if HISTOGRAM_RGB:
    #    histogram_feature = compute_and_normalize_histogram(image, num_bins, hist_range)
    #else:
    #    histogram_feature = []
    
    # Method 2: Compute brown percentage feature
    if BROWN_PERCENTAGE:
        brown_percentage_feature = test2_compute_brown_percentage(image)
    else:
        brown_percentage_feature = 0

    # Method 3: Compute white percentage in Canny edge feature
    if CANNY_WHITE_PIXEL:
        white_percentage_canny_feature = test3_compute_white_pixel_canny(image)
    else:
        white_percentage_canny_feature = 0

    #print("canny ", white_percentage_canny_feature)

    # Combine features
    #combined_features = np.concatenate((histogram_feature, [brown_percentage_feature, white_percentage_canny_feature])) # Something goes wrong here
    combined_features[0] = brown_percentage_feature # Something goes wrong here
    combined_features[1] = white_percentage_canny_feature # Something goes wrong here

    #print("combined ",combined_features[1])
    return combined_features

# Function to predict the ripeness phase of an individual image
def predict_ripeness(image, knn_classifier, num_bins, hist_range):
    combined_features = generate_combined_features(image)
    
    # Make a prediction using the KNN classifier
    prediction = knn_classifier.predict([combined_features])

    return prediction[0]

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
            # debug
            #if combined_features[0] < 0.10 and label == 3:
            #    print(image_path, combined_features)
            features.append(combined_features)
            labels.append(label)

    return features, labels

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
Y = unripe_labels + semi_ripe_labels + ripe_labels
#X = unripe_features + semi_ripe_features + ripe_features + over_ripe_features
#Y = unripe_labels + semi_ripe_labels + ripe_labels + over_ripe_labels
#print(unripe_features)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy at 5 n_neighbors: {accuracy * 100:.2f}%')

# End of Predictknn.py
#===============================
# Start the GUI main loop
root.mainloop()