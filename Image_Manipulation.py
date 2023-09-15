# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:30:31 2023

@author: onurc
Version: V0.4
Description: Image manipulation in the form of blurring, eroding and dilating images.
Outputs all the blur, eroding and dilating filters.
"""

import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

# Resize image
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

def empty(a):
    pass

cv2.namedWindow("parameters")
cv2.resizeWindow("parameters",320,80)
cv2.createTrackbar("threshold1", "parameters", 150,255,empty)
cv2.createTrackbar("threshold2", "parameters", 50,255,empty)
cv2.createTrackbar("threshold_tm", "parameters", 95,100,empty)
#cv2.createTrackbar("area", "parameters", 5000, 90000, empty)

# Template matching example
# All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
def template_matching():
    template_image = cv2.imread('Banaan_template.jpg')
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


while(True):
    # Create variables
    image = cv2.imread('Banaan3_11.jpg')
    image2 = image.copy()
    #template_matching()
    

    # Template image attempt 2
    template_image = cv2.imread('Banaan_template.jpg')
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    
    width = template_image.shape[1]
    height = template_image.shape[0]

    image = image2.copy()
    # Apply template Matching
    res = cv2.matchTemplate(image,template_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(image,top_left, bottom_right, 255, 2)
    cv2.imshow("Yoink", image)



    copy_image = image.copy()
    threshold1 = cv2.getTrackbarPos("threshold1", "parameters")
    threshold2 = cv2.getTrackbarPos("threshold2", "parameters")
    #threshold1 = 43  # (Used for image: 35)
    #threshold2 = 104 # (Used for image: 78)
    
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

    # Gamma / Brightness
    alpha = 1.8 # Contrast threshold
    beta = 20 # Brightness threshold
    bright_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta) # Create brightness image
    bilateral_bright = cv2.bilateralFilter(bright_image,9,75,75)
    gray_bright = cv2.cvtColor(bilateral_bright, cv2.COLOR_BGR2GRAY)
    canny_bright = cv2.Canny(gray_bright, threshold1, threshold2)

    #cv2.imshow("Grayscale", gray)
    #cv2.imshow("Blur", blur)
    #cv2.imshow("Gaussian blur", gaussian)
    #cv2.imshow("Median blur", median)
    #cv2.imshow("Original", image)
    #cv2.imshow("Canny", canny)
    #cv2.imshow("Canny_Bright", canny_bright)
    #resize_image("Original", 50, image)
    #resize_image("Original Bright", 50, bright_image)
    #resize_image("Canny", 50, canny)
    #resize_image("Canny_Bright", 50, canny_bright)
    #cv2.imshow("Canny_gaussian", cannygaussian)
    #cv2.imshow("Canny_median", cannymedian)
    #cv2.imshow("Canny_bilateral", cannybilateral)
    #cv2.imshow("dilate", imdil)
    #cv2.imshow("Erode", erode)
    #cv2.imshow("Image", copy_image)

    # Press backspace to clear all windows    
    c = cv2.waitKey(1)
    if c == 27 & 0xFF:
        break

#cv2.imshow("Dilate", dilated_image)
#cv2.imshow("Binary image", binary_image)

