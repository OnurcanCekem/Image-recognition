# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:30:31 2023

@author: onurc
Version: V0.7
Description: Image manipulation now used for testing with a trackbar
Filters: blur, eroding, dilating, Circle and template matching.

"""

import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

# Resize image. Designed to be similar as cv2.imshow()
# variable name: Name of the end result of the image
# variable scale: Scale the image
# variable img: image input
def resize_image(name, scale, img):
        
    # Grab dimensions and scale the image. 
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height) # Dimensions
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # HSV split, seperately and resized
    cv2.imshow(name,resized)

# Function to resize and concatenate. Basically concatenate with some freedom
# variable img: image input
# variable img2: image input
# variable name: Name of the end result of the image
# variable w: Desired width
# variable h: Desired height
def resized_concat(img, img2, name, w, h):
    
    img = cv2.resize(img, (w, h))
    img2 = cv2.resize(img2, (w, h))
    combined_img = cv2.hconcat([img, img2])
    cv2.imshow(name, combined_img)

def empty(a):
    pass

def close_button(*args):
    cv2.destroyAllWindows() 
    pass

cv2.namedWindow("parameters")
cv2.resizeWindow("parameters",320,80)
cv2.createTrackbar("threshold1", "parameters", 150,255,empty)
cv2.createTrackbar("threshold2", "parameters", 50,255,empty)
cv2.createTrackbar("threshold_tm", "parameters", 95,100,empty)
cv2.createTrackbar("threshold_circle_max", "parameters", 100,100,empty)
cv2.createTrackbar("threshold_circle_min", "parameters", 13,100,empty)

#cv2.createTrackbar("area", "parameters", 5000, 90000, empty)

# Template matching example, currently unused as it's a bit wonky.
# All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
def template_matching_example(image, template_image):
    template_image = cv2.imread('Banaan_template3.jpg')
    threshold_tm = (cv2.getTrackbarPos("threshold_tm", "parameters")/100)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    #result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED) # This one was the example, works at 0.6 threshold
    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCORR_NORMED) # This one works at 0.9 threshold
    #threshold = 0.8 # Adjust this threshold as needed
    locations = np.where(result >= threshold_tm)
    width = template_image.shape[1]
    height = template_image.shape[0]
    for pt in zip(*locations[::-1]):
        cv2.rectangle(image, pt, (pt[0] + width, pt[1] + height), (0, 255, 0), 2)
    cv2.imshow('Detected Bananas', image)

# Template matching attempt #2
# variable image: image input
# variable template_img: image input to compare with    
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
def template_matching(image, template_image):
    # Template image attempt 2
    # All the 6 methods for comparison in a list


    # Apply template Matching
    res = cv2.matchTemplate(image,template_image, cv2.TM_CCOEFF_NORMED) # Result, locate matches
    
    # Locate and grab dimensions
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 
    top_left = max_loc
    width = template_image.shape[1] # Grab dimensions
    height = template_image.shape[0]
    bottom_right = (top_left[0] + width, top_left[1] + height)
    
    # Draw rectangle
    image = cv2.rectangle(image,top_left, bottom_right, 255, 3)
    resized_concat(image, template_image, "Template matching: input (left) and template (right)", 550, 400)

# Create histogram
# variable image: image input
# variable title: title of plot
def histogram(image, title):
    # Calculate the histogram
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(histogram)
    plt.xlim([0, 256])  # Set the x-axis range from 0 to 255 (pixel values)
    plt.grid(True)
    plt.show()


while(True):
    # Create variables
    image =  cv2.imread('Banaanfase3\Banaan3_2.jpg')
    image2 =  cv2.imread('Banaanfase3\Banaan3_1.jpg')
    constant_image = image.copy()
    template_image = cv2.imread('Banaan_template3_650x360.jpg')



    threshold1 = cv2.getTrackbarPos("threshold1", "parameters")
    threshold2 = cv2.getTrackbarPos("threshold2", "parameters")
    threshold_circle_max = cv2.getTrackbarPos("threshold_circle_max", "parameters")
    threshold_circle_min = cv2.getTrackbarPos("threshold_circle_min", "parameters")
    #threshold_circle_max = 100
    #threshold_circle_min = 15
    #threshold1 = 35  # (Used for image: 35, 43)
    #threshold2 = 78 # (Used for image: 78, 104)

    # Blur
    #blur = cv2.blur(image,(7,7))
    blur = cv2.GaussianBlur(image,(9,9),2)
    blur2 = cv2.GaussianBlur(image2,(9,9),2)

    # Convert the blurred image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    grayblur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grayblur2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)

    # Binary image
    binary_image = cv2.threshold(grayblur, 100, 255, cv2.THRESH_BINARY)

    # Gaussian blur
    gaussian = cv2.GaussianBlur(grayblur,(7,7),1)
    # Median blur

    median = cv2.medianBlur(grayblur,7)
    # Bilateral blur
    bilateral = cv2.bilateralFilter(grayblur,9,75,75)

    # Canny edge
    canny = cv2.Canny(grayblur, threshold1, threshold2)
    cannygaussian = cv2.Canny(gaussian, threshold1, threshold2)
    cannymedian = cv2.Canny(median, threshold1, threshold2)
    cannybilateral = cv2.Canny(bilateral, threshold1, threshold2)
    kernel = np.ones([5,5])

    # Erosion
    #eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    erode = cv2.erode(canny,kernel,1)

    # Dilating
    #dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    imdil = cv2.dilate(canny,kernel,1)

    # Gamma / Brightness
    alpha = 1.8 # Contrast threshold
    beta = 20 # Brightness threshold
    bright_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta) # Create brightness image
    bilateral_bright = cv2.bilateralFilter(bright_image,9,75,75)
    gray_bright = cv2.cvtColor(bilateral_bright, cv2.COLOR_BGR2GRAY)
    canny_bright = cv2.Canny(gray_bright, threshold1, threshold2)



    # Use the Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        grayblur,               # Input grayscale image
        cv2.HOUGH_GRADIENT,    # Detection method
        dp=1,                  # Inverse ratio of accumulator resolution
        minDist=10,            # Minimum distance between detected centers
        param1=threshold_circle_max,             # Upper threshold for edge detection 50
        param2=threshold_circle_min,             # Threshold for center detection 30
        minRadius=1,          # Minimum radius of the circle
        maxRadius=25          # Maximum radius of the circle
    )    
    
    # If circles are found, draw them on the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            # Draw the circle outline
            cv2.circle(image, center, radius, (0, 255, 0), 2)


    #plt.figure(figsize=(8, 6))
    #plt.title('Circle Detection in Banana Image')
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.axis('off')
    #plt.show()

    resize_image("Image",50, image)

    #histogram(gray, "Histogram image")
    #histogram(gray, "Histogram image2")
    

    # Calculate the histograms for both images
    hist1 = cv2.calcHist([grayblur], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([grayblur2], [0], None, [256], [0, 256])



    circles = cv2.HoughCircles(
            grayblur,               # Input grayscale image
            cv2.HOUGH_GRADIENT,    # Detection method
            dp=1,                  # Inverse ratio of accumulator resolution
            minDist=10,            # Minimum distance between detected centers
            param1=threshold_circle_max,             # Upper threshold for edge detection 50
            param2=threshold_circle_min,             # Threshold for center detection 30
            minRadius=1,          # Minimum radius of the circle
            maxRadius=25          # Maximum radius of the circle
        )
    # Plot both histograms in a single figure
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Histogram of Banana Image 1')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(hist1)
    plt.xlim([0, 256])  # Set the x-axis range from 0 to 255 (pixel values)
    plt.grid(True)

    plt.subplot(122)
    plt.title('Histogram of Banana Image 2')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(hist2)
    plt.xlim([0, 256])  # Set the x-axis range from 0 to 255 (pixel values)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Press escape to clear all windows    
    c = cv2.waitKey(1)
    if c == 27 & 0xFF:
        break
#cv2.imshow("Dilate", dilated_image)
#cv2.imshow("Binary image", binary_image)

