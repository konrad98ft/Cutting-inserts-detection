
import cv2 as cv
import numpy as np
import sys
import os



PATH = 'cutting-inserts-detection\\samples\\'
PATH_GOOD = 'cutting-inserts-detection\\samples_good\\'
PATH_FAULTY = 'cutting-inserts-detection\\samples_faulty\\'

g_number = 0
f_number = 0
k = 0
l = 0

for filename in os.listdir(PATH):
    img = cv.imread(os.path.join(PATH,filename))
    cv.imshow('Classify sample',img)
    l+=1

    if(l%4==1): k = cv.waitKey(0)
    if  k == ord('g'):
        print("Good")
        cv.imwrite(PATH_GOOD+str(g_number)+'.png', img)
        g_number+=1
    elif k == ord('f'):
        print("Faulty")
        cv.imwrite(PATH_FAULTY+str(f_number)+'.png', img)
        f_number+=1
 

  
