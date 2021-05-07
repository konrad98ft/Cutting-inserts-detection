import cv2 as cv
import numpy as np
import random as rng
import matplotlib.pyplot as plt
import math 
PATH = 'D:\\python Image Processing\\cutting_inserts\\oswietlacz_pierscieniowy_backlight\\3.bmp'

def searchingBox(image, points=(300,650,400,500), direction=(0,1)):
    #points = (x1,x2,y1,y2) direction = (x_dir,y_dir)

    dst = image[points[2]:points[3],points[0]:points[1]]
    cv.rectangle(image,(points[0],points[2]),(points[1],points[3]),(255,255,255),2)
    cv.namedWindow('ROI', cv.WINDOW_NORMAL)
    cv.imshow('ROI',image)
    cv.resizeWindow('ROI',800,600 ) 

    cv.namedWindow('Searching Box', cv.WINDOW_NORMAL)
    cv.imshow('Searching Box',dst)
    cv.resizeWindow('Searching Box',points[1]-points[0],points[3]-points[2] ) 
    


# Get image
img = cv.imread(PATH,-1)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

searchingBox(img.copy(),(300,650,400,500),(0,0))
cv.waitKey(0)
searchingBox(img.copy(),(730,800,240,350),(0,0))  
cv.waitKey(0)

plt.imshow(img)    
plt.show()




cv.waitKey(0)
cv.destroyAllWindows()
 


