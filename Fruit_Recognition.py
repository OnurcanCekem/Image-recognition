# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:30:31 2023

@author: onurc
Version: V0.7
"""

import cv2
import numpy as np
#import sys

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

# Draw text in a position
# variable x: x-coordinate
# variable y: y-coordinate
# variable text: Text to be drawn
def drawtext(x, y, text):
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
image = cv2.imread('Banaan3.png') # read image
frame_image = image.copy()

# Blur
blur = cv2.GaussianBlur(image,(7,7),1)
bilateral = cv2.bilateralFilter(image,9,75,75)
#cv2.imshow("Blurred image", blur)

# Convert the blur to grayscale
gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Canny edge detection
threshold1 = 42
threshold2 = 104
canny = cv2.Canny(gray, threshold1, threshold2)

# Grayscale van blur
grayblur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# HSV 
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Contour
imgContour = image.copy()
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Color threshold
#lower_yellow = np.array([10, 50, 70])  # Example lower threshold for yellow
#upper_yellow = np.array([30, 255, 255])  # Example upper threshold for yellow
lower_yellow = np.array([20, 70, 70])  # Example lower threshold for yellow
upper_yellow = np.array([30, 255, 255])  # Example upper threshold for yellow
lower_brown = np.array([10, 100, 20])  # Example lower threshold for brown
upper_brown = np.array([20, 255, 200])  # Example upper threshold for brown

# Yellow mask (Filter with color range)
yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow) # Create mask
segmented_image_yellow = cv2.bitwise_and(image, image, mask=yellow_mask) # Apply yellow mask on original image
binary_image = cv2.bitwise_and(binary_image, binary_image, mask=yellow_mask) # Apply mask on binary image

# Brown mask (Filter with color range)
brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown) # Create mask
segmented_image_brown = cv2.bitwise_and(image, image, mask=brown_mask) # Apply brown mask on original image
binary_image2 = cv2.bitwise_and(binary_image, binary_image, mask=brown_mask) # Apply mask on binary image

# Filter contours based on area
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# debug
counter = 0

# Iterate over the contours and draw bounding boxes around bananas
for contour in contours:

    # Ignore small contours
    if cv2.contourArea(contour) < 4000:
        continue
    
    # Compute the bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)
    #print(counter, x, w, y, h)

    # Check ratio, if rectangle it's likely a banana (ratio is 2:1)
    if (w*1.5 < h) or (h*1.5 < w): # If banana is found (condition is met)
        counter2 = 0
        # Check which phase banana is in
        
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
        inverted_binary = cv2.bitwise_not(binary_image)
        frame_image = cv2.bitwise_and(frame_image, frame_image, mask=inverted_binary) 
        drawtext(x,y-10,"Found!")
        
        # Draw the contour on the original image
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        #binary_image[y:y+h, x:x+w] = (0) # clear already found bananas

        # debug locations
        counter +=1
        print(counter, x, w, y, h)

        # Decide which phase based on brown
        if brown_percentage > 20:
            print("Phase 3")
        elif brown_percentage > 12:
            print("Phase 2")
        else:
            print("Phase 1")

        # Display the brown percentage
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(image, f"{brown_percentage:.2f}%", (contour[0][0][0], contour[0][0][1]), font, 0.5, (255, 0, 0), 2)
        drawtext(x,y+20, f"{brown_percentage:.2f}%")
    
    #else: # If banana is not found
        # Fill the found square with tuplet's color to make sure it doesn't find it again (50,250,50)
        #frame_image[y:y+h, x:x+w] = (50,250,50)

#print(filtered_contours)
cv2.imshow("original image", image)
#cv2.imshow("gray", gray)
#cv2.imshow("HSV", hsv_image)
#cv2.imshow("Binary image", binary_image)
cv2.imshow("frame image", frame_image)

#cv2.imshow("yellow (color range)", segmented_image_yellow)
#cv2.imshow("brown (color range)", segmented_image_brown)
#cv2.imshow("Canny", canny)
cv2.imshow("yellow mask", yellow_mask)

# Press backspace to clear all windows
cv2.waitKey(0)
cv2.destroyAllWindows() 