from typing import Counter
import cv2 as cv
import numpy as np
import random as rng
import matplotlib.pyplot as plt
import math
import sys
from numpy.core.fromnumeric import shape 
# from scipy.optimize import fsolve
# from scipy import ndimage



PATH = 'D:\\Python Image Processing\\cutting-inserts-detection\\tensor_flow_samples\\'
PATH2 = 'cutting-inserts-detection\\samples\\'
SERIES='3_'

#  Get an image
for img_index in range(1,34):
    img_path= PATH + SERIES + str(img_index) +'.png'
    img = cv.imread(img_path,-1)
    try:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    except:
        print("Image not found")
        sys.exit(1)

  
 
    # Prepare image by finding conturs
    adaptive = cv.adaptiveThreshold(img,100,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,51,15)

    edged = cv.Canny(adaptive, 100, 125)

    '''cv.namedWindow(str(img_index)+'apaptive', cv.WINDOW_FREERATIO)
    cv.imshow(str(img_index)+'apaptive', edged)
    cv.resizeWindow(str(img_index)+'apaptive', int(img.shape[1]/2),int(img.shape[0]/2)) '''
   
    kernel = np.ones((3, 3), np.uint8)
    edged = cv.dilate(edged, kernel) 

    '''cv.namedWindow(str(img_index)+'edges', cv.WINDOW_FREERATIO)
    cv.imshow(str(img_index)+'edges', edged)
    cv.resizeWindow(str(img_index)+'edges', int(img.shape[1]/2),int(img.shape[0]/2))'''

    # Reject small non-signifficant conturs
    contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cont2 = []
    for i in range(len(contours)):
        if(cv.contourArea(contours[i])>80):
            cont2.append(contours[i])

    # Find bounding box        
    minX = 9999
    minY = 9999
    maxX = 0
    maxY = 0
    for c in cont2:
        for p in c:
            if(minX>p[0][0]):minX=p[0][0]
            if(maxX<p[0][0]):maxX=p[0][0]
            if(minY>p[0][1]):minY=p[0][1]
            if(maxY<p[0][1]):maxY=p[0][1]

    XC = int((maxX + minX)/2)
    YC = int((maxY + minY)/2)
    #cv.drawContours(img, cont2, -1, (0, 255, 255), 3)
    
    
    # Define ROI as 1/4 of the cutting insert
    sep = 100
    start_point = (XC, minY-sep)
    end_point = (maxX+sep, YC)
    img = cv.rectangle(img, start_point, end_point, 255, 2)
    start_point = (XC, YC)
    end_point = (maxX+sep, maxY+sep)
    img = cv.rectangle(img, start_point, end_point, 255, 2)
    start_point = (minX-sep, minY-sep)
    end_point = (XC, YC)
    img = cv.rectangle(img, start_point, end_point, 255, 2)
    start_point = (minX-sep, YC)
    end_point = (XC, maxY+sep)
    img = cv.rectangle(img, start_point, end_point, 255, 2)
    
    '''cv.namedWindow(str(img_index), cv.WINDOW_FREERATIO)
    cv.imshow(str(img_index), img)
    cv.resizeWindow(str(img_index), int(img.shape[1]/2),int(img.shape[0]/2))'''


    # Get 4 images from 1x(1/4) by filping by x and y axis
    roi = img.copy()[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    cv.imwrite(PATH2+SERIES+str(img_index)+'_1.png', roi)

    
    flipVertical = cv.flip(roi, 0)
    cv.imwrite(PATH2+SERIES+str(img_index)+'_2.png', flipVertical)
    
    flipHorizontal = cv.flip(roi, 1)
    cv.imwrite(PATH2+SERIES+str(img_index)+'_3.png', flipHorizontal)
    
    flipVerticalHorizontal = cv.flip(flipVertical, 1)
    cv.imwrite(PATH2+SERIES+str(img_index)+'_4.png', flipVerticalHorizontal)
    


    cv.waitKey(100)
 
