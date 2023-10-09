"""
Created on Mon Oct 02 10:22:39 2023

@author: onurc
Version: V0.5
Description: 
"""
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

global image_index
global image_global
image_index = 0
global mean_image

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
    test1_mean_img()
    image = preprocess_image(image_path)
    display_rgb_histogram(image, index) # Display the RGB histogram of the current image
    
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
    test1_mean_img()
    image = preprocess_image(image_path)
    display_rgb_histogram(image) # Display the RGB histogram of the current image
    

    print("Image path: ",image_path)

    #image = preprocess_image(image_path)
    # Display the RGB histogram of the current image
    #display_rgb_histogram(image)

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
    #h,s,v = cv2.split(hsv_image)

    # define thresholds
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
    banana_color_image = cv2.add(segmented_image_yellow, segmented_image_brown)
    binary_image_combined = cv2.add(binary_image_yellow, binary_image_brown)
    
    # Dilate
    kernel = np.ones([3,3])
    dilated_binary_image = cv2.dilate(banana_color_image,kernel,1)
    #cv2.imshow("Title ",banana_color_image)
    return dilated_binary_image

# Function to display the RGB histogram of an image
def display_rgb_histogram(image, index=0):
    plt.clf()
    plt.ion()
    b, g, r = cv2.split(image)
    
    plt.subplot(1,2,1)
    
    # Calculate the histograms for each channel
    hist_b = cv2.calcHist([b], [0], None, [256], [1, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [1, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [1, 256])
    
    # Create a Matplotlib figure for the RGB histogram

    plt.title(f'RGB Histogram {index}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Plot the histograms for each channel
    plt.plot(hist_b, color='b', label='Blue')
    plt.plot(hist_g, color='g', label='Green')
    plt.plot(hist_r, color='r', label='Red')

    #plt.plot(hist_r, color='r', label='Red')
    plt.title('Test1 mean')
    # Add a legend to distinguish the channels
    plt.legend()

    # 2nd plot, still available for something
    plt.subplot(1,2,2)

    mean_histogram = hist_b.mean() + hist_g.mean() + hist_r.mean()
    mean_histogram_label.config(text = f"Mean_histogram: {mean_histogram}") #shows as text in the window
    print("Mean hist: ",mean_histogram)
    plt.plot(mean_histogram, color='y', label='Yellow')

    plt.title(f'Mean Histogram {index}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Display the RGB histogram
    plt.show()

    # Find the dominant color and its frequency
    #dominant_color = np.unravel_index(hist.argmax(), hist.shape)
    #dominant_frequency = hist[dominant_color]
    # Convert dominant_color to BGR format (OpenCV)
    #dominant_color_bgr = (dominant_color[2], dominant_color[1], dominant_color[0])

    # Calculate the mean of the RGB histogram
    #total_pixels = np.sum(hist)
    #mean_rgb = [np.sum(hist[:, :, i] * np.arange(256)) / total_pixels for i in range(3)]
    #print("Dominant Color (RGB):", dominant_color)
    #print("Dominant Color (BGR):", dominant_color_bgr)
    #print("Dominant Frequency:", dominant_frequency)
    #print("Mean RGB:", mean_rgb)


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

# Function to browse and select a folder of images
def browse_folder():
    global image_paths, image_index
    folder_path = filedialog.askdirectory()
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]
    image_index = -1  # Start from the first image (index 0) when a new folder is selected
    load_next_image()

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

    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]
    image_index = -1  # Start from the first image (index 0) when a new folder is selected
    load_next_image()

def test1_mean_img():
    global image_global
    global mean_image
    mean_image = np.mean(image_global)
    print("Mean Image: ",mean_image)
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
individual_image = tk.Button(root, bg='green', text="Select image", command=select_image_manually)
fase1_path = tk.Button(root, bg='green', text="Select phase 1", command=lambda: select_phase(1))
fase2_path = tk.Button(root, bg='green', text="Select phase 2", command=lambda: select_phase(2))
fase3_path = tk.Button(root, bg='green', text="Select phase 3", command=lambda: select_phase(3))
fase4_path = tk.Button(root, bg='green', text="Select phase 4", command=lambda: select_phase(4))
mean_image_label = tk.Label(root, text="Waiting image...") #shows as text in the window
mean_histogram_label = tk.Label(root, text="Waiting image...") #shows as text in the window

#mean_label = tk.Label(root, text=f"Mean: {mean_image}") #shows as text in the window
browse_button.pack(pady=10)
individual_image.pack(padx=10)
fase1_path.pack(padx=10)
fase2_path.pack(padx=10)
fase3_path.pack(padx=10)
fase4_path.pack(padx=10)
previous_button.pack(side='left', padx=10)
next_button.pack(side='right', padx=10)
yellow_button.pack(pady=10)
brown_button.pack(pady=10)
close_button.pack(padx=20, pady=10)
mean_image_label.pack(padx=20, pady=10)
mean_histogram_label.pack(padx=20, pady=10)

# Initialize global variables
image_paths = []
image_index = -1



# Start the GUI main loop
root.mainloop()

