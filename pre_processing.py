import cv2 as cv
import numpy as np
import random as rng
import matplotlib.pyplot as plt
import math 
import sys
from itertools import combinations
import time
from skimage.filters import threshold_otsu


PATH = r'C:\Users\Konrad\cutting-inserts-detection-master\tensor_flow_samples'
PATH2 = r'C:\Users\Konrad\cutting-inserts-detection-master\samples2'
SERIES='1_'


#Kapur threshold
def kapur_threshold(image):

    hist, _ = np.histogram(image, bins=range(256), density=True)
    c_hist = hist.cumsum()
    c_hist_i = 1.0 - c_hist

    # To avoid invalid operations regarding 0 and negative values.
    c_hist[c_hist <= 0] = 1
    c_hist_i[c_hist_i <= 0] = 1

    c_entropy = (hist * np.log(hist + (hist <= 0))).cumsum()
    b_entropy = -c_entropy / c_hist + np.log(c_hist)

    c_entropy_i = c_entropy[-1] - c_entropy
    f_entropy = -c_entropy_i / c_hist_i + np.log(c_hist_i)

    return np.argmax(b_entropy + f_entropy)

#Otsu threshold
def otsu_threshold(image=None, hist=None):

    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either''the input image or its histogram')

    # Calculating histogram
    if not hist: hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    cdf_backg = np.cumsum(np.arange(len(hist)) * hist)
    w_backg = np.cumsum(hist)  # The number of background pixels
    w_backg[w_backg == 0] = 1  # To avoid divisions by zero
    m_backg = cdf_backg / w_backg  # The means

    cdf_foreg = cdf_backg[-1] - cdf_backg
    w_foreg = w_backg[-1] - w_backg  # The number of foreground pixels
    w_foreg[w_foreg == 0] = 1  # To avoid divisions by zero
    m_foreg = cdf_foreg / w_foreg  # The means

    var_between_classes = w_backg * w_foreg * (m_backg - m_foreg) ** 2

    return np.argmax(var_between_classes)



#  Get an image
for img_index in range(1,34):
    img_path= PATH +'\\' +SERIES + str(img_index) +'.png'
    print(img_path)
    img = cv.imread(img_path,-1)
    try:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    except:
        print("Image not found")
        sys.exit(1)

    start_time = time.time()
    
    # Prepare image by finding conturs - kapur threshold
    thresh_val = threshold_otsu(img)
    ret,img2 = cv.threshold(img,thresh_val,255,cv.THRESH_TOZERO)

    elapsed_time = time.time() - start_time
    print(elapsed_time)

    '''cv.namedWindow('prog', cv.WINDOW_FREERATIO)
    cv.imshow('prog', img2)
    cv.resizeWindow('prog', int(img.shape[1]/2),int(img.shape[0]/2)) '''
    
    kernel = np.ones((5, 5), np.uint8)
    edged = cv.erode(img2, kernel) 
    kernel2 = np.ones((9, 9), np.uint8)
    edged = cv.dilate(edged, kernel) 

    '''cv.namedWindow(str(img_index)+'edges', cv.WINDOW_FREERATIO)
    cv.imshow(str(img_index)+'edges', edged)
    cv.resizeWindow(str(img_index)+'edges', int(img.shape[1]/2),int(img.shape[0]/2))'''
    
    # Reject small non-signifficant conturs
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cont2 = []
    for i in range(len(contours)):
        if(cv.contourArea(contours[i])>1000):
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
    
    # Centre of the cutting insert
    Xdim = 1100
    Ydim = 650
    
    # Save all 4 parts to the file
    start_point = (XC, int(YC-Ydim))
    end_point = (XC+int(Xdim), YC)
    img = cv.rectangle(img, start_point, end_point, 255, 2)
    img2 = img.copy()[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    cv.imwrite(PATH2+"\\"+SERIES+str(img_index)+'_1.png', img2)

    start_point = (XC, YC)
    end_point = (int(XC+Xdim), int(YC+Ydim))
    img = cv.rectangle(img, start_point, end_point, 255, 2)
    img2 = img.copy()[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    cv.imwrite(PATH2+"\\"+SERIES+str(img_index)+'_2.png', img2)
    
    start_point = (int(XC-Xdim), int(YC-Ydim))
    end_point = (XC, YC)
    img = cv.rectangle(img, start_point, end_point, 255, 2)
    img2 = img.copy()[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    cv.imwrite(PATH2+"\\"+SERIES+str(img_index)+'_3.png', img2)

    start_point = (int(XC-Xdim), YC)
    end_point = (XC, int(YC+Ydim))
    img = cv.rectangle(img, start_point, end_point, 255, 2)
    img2 = img.copy()[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    cv.imwrite(PATH2+"\\"+SERIES+str(img_index)+'_4.png', img2)
    
    cv.namedWindow("out", cv.WINDOW_FREERATIO)
    cv.imshow("out", img)
    cv.resizeWindow("out", int(img.shape[1]/2),int(img.shape[0]/2))


    cv.waitKey(0)
