# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:30:31 2023

@author: onurc
Version: V0.1
Description: Color manipulation. Shows the color dimensions: RGB, HSV
Outputs all the blur, eroding and dilating filters.
"""

import cv2
import numpy as np
#import sys

image = cv2.imread('Banaan4.png')

# RGB color
r,g,b = cv2.split(image)
rgb_split = np.concatenate((r,g,b),axis=1)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# HSV color
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv_image)
hsv_split = np.concatenate((h,s,v),axis=1)
#cv2.imshow("Split HSV",hsv_split)

# Scale the image. 
# This is not used for function, but rather vanity (I wanted to make a better screenshot for the essay). 
scale_percent = 80 # percent of original size
width = int(image.shape[1] * scale_percent / 34)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height) # Dimensions

# resize image
resized_hsv = cv2.resize(hsv_split, dim, interpolation = cv2.INTER_AREA) # HSV split, seperately and resized
resized_rgb = cv2.resize(rgb_split, dim, interpolation = cv2.INTER_AREA) # HSV split, seperately and resized

# Color threshold
lower_yellow = np.array([20, 100, 100])  # Example lower threshold for yellow
upper_yellow = np.array([30, 255, 255])  # Example upper threshold for yellow

# Filter with color range
yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow) # Create mask
segmented_image = cv2.bitwise_and(image, image, mask=yellow_mask) # Apply mask on original image
binary_image = cv2.bitwise_and(binary_image, binary_image, mask=yellow_mask) # Apply mask on binary image

# Show other images
cv2.imshow("Display image", image) # Original image
cv2.imshow("Gray", gray) # Grayscaled image
cv2.imshow("segmented_image", segmented_image) # Segmented image with hsv color after yellow mask
cv2.imshow("HSV", hsv_image) # HSV of original image
cv2.imshow("Binary image", binary_image) # Binary image
cv2.imshow("H (left) S (middle) V (right)", resized_hsv) # HSV split, seperately and resized
cv2.imshow("R (left) G (middle) B (right)", resized_rgb) # HSV split, seperately and resized
#cv2.imshow("yellow mask", yellow_mask)

# Press backspace to clear all windows
cv2.waitKey(0)
cv2.destroyAllWindows() 