# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:30:31 2023

@author: onurc
Version: V0.2
Description: Image manipulation in the form of blurring, eroding and dilating images.
Outputs all the blur, eroding and dilating filters.
"""

import cv2
import numpy as np
import copy

def empty(a):
    pass

cv2.namedWindow("parameters")
cv2.resizeWindow("parameters",320,80)
cv2.createTrackbar("threshold1", "parameters", 150,255,empty)
cv2.createTrackbar("threshold2", "parameters", 50,255,empty)
#cv2.createTrackbar("area", "parameters", 5000, 90000, empty)

while(True):
    # Create variables
    image = cv2.imread('Banaan4.png')
    copy_image = image.copy()
    threshold1 = cv2.getTrackbarPos("threshold1", "parameters")
    threshold2 = cv2.getTrackbarPos("threshold2", "parameters")
    #threshold1 = 35
    #threshold2 = 78
    
    # Blur
    blur = cv2.blur(image,(7,7))
    # Convert the blurred image to grayscale
    
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # Binary image
    
    binary_image = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # Gaussian blur
    
    gaussian = cv2.GaussianBlur(gray,(7,7),1)
    # Median blur
    
    median = cv2.medianBlur(gray,7)
    # Bilateral blur
    bilateral = cv2.bilateralFilter(gray,9,75,75)

    # Canny edge
    canny = cv2.Canny(gray, threshold1, threshold2)
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

    #cv2.imshow("Grayscale", gray)
    #cv2.imshow("Blur", blur)
    #cv2.imshow("Gaussian blur", gaussian)
    #cv2.imshow("Median blur", median)
    cv2.imshow("Canny", canny)
    cv2.imshow("Canny_gaussian", cannygaussian)
    cv2.imshow("Canny_median", cannymedian)
    cv2.imshow("Canny_bilateral", cannybilateral)
    #cv2.imshow("dilate", imdil)
    #cv2.imshow("Erode", erode)
    #cv2.imshow("Image", copy_image)

    # Press backspace to clear all windows    
    c = cv2.waitKey(1)
    if c == 27 & 0xFF:
        break

#cv2.imshow("Dilate", dilated_image)
#cv2.imshow("Binary image", binary_image)

