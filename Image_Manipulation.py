# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:30:31 2023

@author: onurc
Version: V0.1
Description: Image manipulation in the form of blurring, eroding and dilating images.
Outputs all the blur, eroding and dilating filters.
"""

import cv2
import numpy as np
import copy
#import sys

def empty(a):
    pass


# Create variables
image = cv2.imread('Banaan4.png')



# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# grayscale
grayblur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(grayblur, 100, 255, cv2.THRESH_BINARY)

# blur
blur = cv2.blur(image,(7,7))

# Gaussian blur
gaussian = cv2.GaussianBlur(image,(7,7),1)

# Median blur
median = cv2.medianBlur(image,7)

# Bilateral blur
bilateral = cv2.bilateralFilter(image,9,75,75)



#Erosion

cv2.namedWindow("parameters")
cv2.resizeWindow("parameters",320,80)
cv2.createTrackbar("threshold1", "parameters", 150,255,empty)
cv2.createTrackbar("threshold2", "parameters", 50,255,empty)
cv2.createTrackbar("area", "parameters", 5000, 90000, empty)

while(True):
    image = cv2.imread('Banaan4.png')
    blur = cv2.blur(image,(7,7))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("threshold1", "parameters")
    threshold2 = cv2.getTrackbarPos("threshold2", "parameters")

    canny = cv2.Canny(gray, threshold1, threshold2)
    kernel = np.ones([5,5])
    imdil = cv2.dilate(canny,kernel,1)
    contours, hierarchy = cv2.findContours(imdil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.imshow("thresh", canny)
    cv2.imshow("dilate", imdil)

    #Dilating

    #cv2.imshow("Original", image)
#cv2.imshow("Grayscale", gray)
#cv2.imshow("Blur", blur)
#cv2.imshow("Gaussian blur", gaussian)
#cv2.imshow("Median blur", median)
#cv2.imshow("Binary image", binary_image)
#Bilateral
#Erosion
#dilatation

# Press backspace to clear all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
