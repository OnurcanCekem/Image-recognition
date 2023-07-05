# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:30:31 2023

@author: onurc
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

image = cv2.imread('Banaantros1.png')





# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

#blur
blur = cv2.GaussianBlur(image,(7,7),1)

#grayscale van blur
grayblur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)


imgContour = image.copy()
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#cv2.imshow("HSV", hsv_image)

# Original starting values
#lower_yellow = np.array([20, 100, 100])  # Example lower threshold for yellow
#upper_yellow = np.array([30, 255, 255])  # Example upper threshold for yellow

# Values I want to keep
#lower_yellow = np.array([10, 50, 70])  # Example lower threshold for yellow
#upper_yellow = np.array([30, 255, 255])  # Example upper threshold for yellow

# Testing purposes
lower_yellow = np.array([10, 40, 50])  # Example lower threshold for yellow (B,G,R)
upper_yellow = np.array([30, 255, 255])  # Example upper threshold for yellow (B,G,R)

# Filter with color range
yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
segmented_image = cv2.bitwise_and(image, image, mask=yellow_mask)
binary_image = cv2.bitwise_and(binary_image, binary_image, mask=yellow_mask)
cv2.imshow("segmented_image", segmented_image)


# Filter contours based on area
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# debug
counter = 0

# Iterate over the contours and draw bounding boxes around bananas
for contour in contours:
    # Ignore small contours
    if cv2.contourArea(contour) < 1000:
        continue
    
    # Compute the bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)

    #Check ratio, if rectangle it's likely a banana (ratio is 2:1)
    if (w*2 < h) or (h*2 < w):
        # Draw the bounding box on the image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        drawtext(x,y-10,"Found!")
        binary_image[y:y+h, x:x+w] = (0) # clear already found bananas

        # debug locations
        counter +=1
        print(counter, x, w, y, h)
    else:
        # fill found banana's with tuplet's color (50,250,50)
        image[y:y+h, x:x+w] = (50,250,50)

#print(filtered_contours)
cv2.imshow("Binary image", binary_image)

"""
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    #if min_area <= area <= max_area:
    cv2.drawContours(yellow_mask, contour, -1, (0,255,0), 3 )
    peri = cv2.arcLength( contour, True)
    rect = cv2.approxPolyDP( contour, 0.02 * peri, True )
    print (len(rect))
    x,y,w,h = cv2.boundingRect(rect)
    cv2.rectangle(yellow_mask, (x, y), (x + w , y + h), (0, 255, 0), 3)
    filtered_contours.append(contour)
    cv2.putText(yellow_mask, "points: " + str(len(rect)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
    cv2.putText(yellow_mask, "area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)



# Contour
binary_image = cv2.imread('Banaantros1.png', cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
min_area = 1000  # Minimum area threshold
max_area = 10000  # Maximum area threshold

filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if min_area <= area <= max_area:
        filtered_contours.append(contour)
"""

# Code to draw a few pixels
# Set the thickness of the pixel (in pixels)
#printcolor(80, 370, 5) # x = 80, y = 370, thickness = 5
#printcolor(310, 320, 5) # x = 310, y = 320, thickness = 5

cv2.imshow("Display window", image)
#cv2.imshow("yellow mask", yellow_mask)

# Press backspace to clear all windows
cv2.waitKey(0)
cv2.destroyAllWindows() 


#image2 = cv2.imread('Banaantros1.png')
#cv2.imshow("Display window2", image2)