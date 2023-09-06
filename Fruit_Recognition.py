# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:30:31 2023

@author: onurc
Version: V0.10
"""

import cv2
import numpy as np
#import sys
from matplotlib import pyplot as plt
import os


# Print pixel color and set pixel color to red
# variable x: x-coordinate
# variable y: y-coordinate
# variable thickness: thickness of square
def printcolor(x, y, thickness):
    print(image[y,x])
    red_color = (0,0,255) # color (Blue, Green, Red)

    # Draw a rectangle around the pixel (fill)
    cv2.rectangle(image, (x, y), (x + thickness - 1, y + thickness - 1), red_color, -1)
    image[y,x] = red_color

# Concatenate images with a scale
# variable img1: first image input
# variable img2: second image input
# variable img3: third image input
# variable scale: Scale the image
# variable name: Name of the end result of the image
def concatenate(img1, img2, img3, scale, name):
    combined_image = np.concatenate((img1,img2,img3),axis=1)
    
    # Scale the image. 
    # This is not used for function, but rather vanity (I wanted to make a better screenshot for the essay). 
    scale_percent = scale # percent of original size
    width = int(img1.shape[1] * scale_percent / 34)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height) # Dimensions

    # resize image
    resized_color = cv2.resize(combined_image, dim, interpolation = cv2.INTER_AREA) # HSV split, seperately and resized
    cv2.imshow(name, resized_color)


# Resize image
# variable img: image input
# variable scale: Scale the image
# variable name: Name of the end result of the image
def resize_image(img, scale, name):
        
    # Grab dimensions and scale the image. 
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height) # Dimensions
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # HSV split, seperately and resized
    cv2.imshow(name,resized)

# Draw text in a position
# variable x: x-coordinate
# variable y: y-coordinate
# variable text: Text to be drawn
def drawtext(x, y, text, fontScale=1):
    cv2.putText(image,text, 
    (x,y-10), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)

# Write some Text
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 2
lineType               = 1


# Read the image
image = cv2.imread('Banaan4_2.jpg') # read image
frame_image = image.copy()

# Dimensions 
# height, width, number of channels in image
height = image.shape[0] # y
width = image.shape[1] # x
percentage_area = 0.10 # How much % of the image should be the banana (1.00 is 100%)
area = height*width*percentage_area # Amount of pixels required for area to identify banana
#print("Area:", area) # debug

# Blur
blur = cv2.GaussianBlur(image,(7,7),1)
bilateral = cv2.bilateralFilter(image,9,75,75)
#cv2.imshow("Blurred image", blur)

# Convert the blur to grayscale
gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
_, binary_image_yellow = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
_, binary_image_brown = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Canny edge detection
threshold1 = 42
threshold2 = 104
canny = cv2.Canny(gray, threshold1, threshold2)
 

# Grayscale van blur
grayblur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# HSV 
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv_image)
hsv_split = np.concatenate((h,s,v),axis=1)

# Saturation
sat_mask = cv2.inRange(s,70,255)
s = cv2.bitwise_and(s, s, mask=sat_mask)
blur_s = cv2.GaussianBlur(s,(7,7),1)
bilateral_s = cv2.bilateralFilter(s,9,75,75)
gray_s = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny_s = cv2.Canny(bilateral_s, threshold1, threshold2)
#cv2.imshow("Canny Sat", canny_s)
cv2.imshow("s", s)

# Contour
imgContour = image.copy()
contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Color threshold
#lower_yellow = np.array([10, 50, 70])  # Example lower threshold for yellow
#upper_yellow = np.array([30, 255, 255])  # Example upper threshold for yellow
lower_yellow = np.array([10, 50, 70])  # lower threshold for yellow (example: [10, 50, 70])
upper_yellow = np.array([30, 255, 255])  # upper threshold for yellow (example:  [30, 255, 255])
lower_brown = np.array([0, 70, 0])  # lower threshold for brown (example: [10, 100, 20])
upper_brown = np.array([20, 255, 200])  # upper threshold for brown (example: [20, 255, 200])
lower = np.array([22, 93, 0])
upper = np.array([45, 255, 255])
# Yellow mask (Filter with color range)
yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow) # Create mask
segmented_image_yellow = cv2.bitwise_and(image, image, mask=yellow_mask) # Apply yellow mask on original image
binary_image_yellow = cv2.bitwise_and(binary_image_yellow, binary_image_yellow, mask=yellow_mask) # Apply mask on binary image

# Yellow mask (Filter with color range)
yellow_mask2 = cv2.inRange(hsv_image, lower, upper) # Create mask
segmented_image_yellow2 = cv2.bitwise_and(image, image, mask=yellow_mask2) # Apply yellow mask on original image
binary_image_yellow2 = cv2.bitwise_and(binary_image_yellow, binary_image_yellow, mask=yellow_mask2) # Apply mask on binary image

# Brown mask (Filter with color range)
brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown) # Create mask
segmented_image_brown = cv2.bitwise_and(image, image, mask=brown_mask) # Apply brown mask on original image
binary_image_brown = cv2.bitwise_and(binary_image_brown, binary_image_brown, mask=brown_mask) # Apply mask on binary image

# Combine yellow and brown
color_image = cv2.add(segmented_image_yellow, segmented_image_brown)
binary_image_combined = cv2.add(binary_image_yellow, binary_image_brown)

# Eroding and dilating
kernel = np.ones([3,3])
yellow_dil = cv2.dilate(binary_image_yellow,kernel,1)
#yellow_erode = cv2.erode(yellow_mask,kernel,1)
color_dil = cv2.dilate(color_image,kernel,1)
binary_image_combined = cv2.dilate(binary_image_combined,kernel,1)

# Filter contours based on area
contours, _ = cv2.findContours(canny_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


"""
# Orb detector

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(gray,None)

# compute the descriptors with ORB
kp, des = orb.compute(image, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()


keypoints_list = []
descriptors_list = []

for image in dataset:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)
"""

# debug
counter = 0
# Iterate over the contours and draw bounding boxes around bananas
for contour in contours:

    # Ignore small contours
   #if cv2.contourArea(contour) < 4000:
    if cv2.contourArea(contour) < area:
        continue
    
    # Compute the bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)
    #print(counter, x, w, y, h)

    # Check ratio, if rectangle it's likely a banana (ratio is roughly 2:1 but depends on image dimensions)
    if (w*1.5 < h) or (h*1.5 < w): # If banana is found (condition is met)
        counter2 = 0
        
        # Calculate the area of the contour
        contour_area = cv2.contourArea(contour)
        
        # Create a mask for the current contour
        contour_mask = np.zeros_like(brown_mask)
        cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)
        
        # Calculate the brown pixel count within the contour
        brown_pixel_count = np.sum(np.logical_and(brown_mask, contour_mask))
        # Calculate the brown percentage within the contour
        brown_percentage = (brown_pixel_count / contour_area) * 100
   
        # Draw the bounding box on the image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Filter out the already found bananas from frame_image (useful for multiple bananas)
        # This currently does nothing important
        inverted_binary = cv2.bitwise_not(binary_image_combined)
        frame_image = cv2.bitwise_and(frame_image, frame_image, mask=inverted_binary) 
        
        # debug locations
        counter +=1
        print(counter, x, w, y, h)

        # Decide which phase banana is based on brown
        if brown_percentage > 50:
            print("Phase 4")
        elif brown_percentage > 22:
            print("Phase 3")
        elif brown_percentage > 8:
            print("Phase 2")
        elif brown_percentage < 2:
            print("Not a banana.")
            continue
        else:
            print("Phase 1")

        # Draw the contour on the original image
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        #binary_image[y:y+h, x:x+w] = (0) # clear already found bananas

        # Display the brown percentage
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(image, f"{brown_percentage:.2f}%", (contour[0][0][0], contour[0][0][1]), font, 0.5, (255, 0, 0), 2) # original text code
        drawtext(x,y+20, f"{brown_percentage:.2f}% brown", 0.5) # Text brown percentage
        drawtext(x,y+40, f"{w/h:.2f}:1 width:height ratio", 0.5) # Text width:height ratio
        drawtext(x,y-10,"Found banana!") # Text banana is found

    #else: # If banana is not found
        # Fill the found square with tuplet's color to make sure it doesn't find it again (50,250,50)
        #frame_image[y:y+h, x:x+w] = (50,250,50)

# A list of possible images
#print(filtered_contours)
cv2.imshow("original image", image)
#cv2.imshow("frame image (copy of original)", frame_image)
#cv2.imshow("gray", gray)
#cv2.imshow("HSV", hsv_image)
#cv2.imshow("Binary image", binary_image)
#cv2.imshow("Binary Combined", binary_image_combined)
#cv2.imshow("yellow mask", yellow_mask)
#cv2.imshow("yellow dil image", yellow_dil)
#cv2.imshow("brown mask", brown_mask)
#cv2.imshow("yellow (color range)", segmented_image_yellow)
#cv2.imshow("brown (color range)", segmented_image_brown)
#cv2.imshow("color (yellow and brown) dilated", color_dil)
#cv2.imshow("color (yellow and brown)", color_image)
#cv2.imshow("Canny", canny)
resize_image(image, 50, "original")
resize_image(segmented_image_yellow, 50, "Tweaked yellow filter")
resize_image(segmented_image_yellow2, 50, "Original yellow filter")
#resize_image(color_image, 50, "Color combined")


color_split = np.concatenate((segmented_image_yellow2,segmented_image_yellow,image),axis=1)
# Scale the image. 
# This is not used for function, but rather vanity (I wanted to make a better screenshot for the essay). 
scale_percent = 60 # percent of original size
width = int(image.shape[1] * scale_percent / 34)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height) # Dimensions

# resize image
resized_color = cv2.resize(color_split, dim, interpolation = cv2.INTER_AREA) # HSV split, seperately and resized
cv2.imshow("Color", resized_color)

#concatenate(segmented_image_yellow, segmented_image_brown, color_image, 60, "Coooool")
# Press backspace to clear all windows
cv2.waitKey(0)
cv2.destroyAllWindows() 