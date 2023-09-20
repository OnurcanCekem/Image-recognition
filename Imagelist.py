# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:30:31 2023

@author: onurc
Version: V0.1
"""

import cv2
import numpy as np
#import sys
from matplotlib import pyplot as plt
import os

# Define the directory containing your images
image_dir = r"C:\Users\onurc\Dropbox\HBO Elektrotechniek\3rde Jaar\Periode A - Beeldherkenning\Py bestanden"

# Get a list of image file paths in the directory
image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.png'))]
images = []

# Loop through the image files and preprocess each one
for image_path in image_files:
    # Load an image
    image = cv2.imread(image_path)
    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Image not found at {image_path}")
        continue
    # Append image
    images.append(image)
    # Orb detector


print("Hi. ")


# Initiate ORB detector
orb = cv2.ORB_create()


train_image = cv2.imread('Banaan3_5.jpg') # read image
train_image_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_image_gray,None)

keypoints_list = []
descriptors_list = []

#for image in images:
image = images[0]
# find the keypoints with ORB
kp = orb.detect(image,None)

# compute the descriptors with ORB
kp, des = orb.compute(image, kp)

# draw only keypoints location,not size and orientation
#img2 = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2 = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
#plt.imshow(img2), plt.show()


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
keypoints, descriptors = orb.detectAndCompute(gray_image, None)
keypoints_list.append(keypoints)
descriptors_list.append(descriptors)

# Initialize the Matcher for matching
# the keypoints and then match the
# keypoints
matcher = cv2.BFMatcher()
matches = matcher.match(descriptors,trainDescriptors)
#cv2.imshow("yeet", image)

# draw the matches to the final image
# containing both the images the drawMatches()
# function takes both images and keypoints
# and outputs the matched query image with
# its train image
final_img = cv2.drawMatches(img2, keypoints, 
train_image_gray, trainKeypoints, matches[:20],None)

final_img = cv2.resize(final_img, (1000,650))
cv2.imshow("Yoink", final_img)

for n in range(len(images)):
    n+=1
    #cv2.imshow("Yeet",images)
    print(n)



# Press backspace to clear all windows
cv2.waitKey(0)
cv2.destroyAllWindows() 