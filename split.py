
import cv2 as cv
import numpy as np
import sys
import os
import random



#source path
PATH = 'cutting-inserts-detection\\samples\\'

#output path
PATH_GOOD = 'cutting-inserts-detection\\traning\\samples_good\\'
PATH_FAULTY = 'cutting-inserts-detection\\traning\\samples_faulty\\'

LEARN_PATH_GOOD = 'cutting-inserts-detection\\traning\\samples_good\\'
LEARN_PATH_FAULTY = 'cutting-inserts-detection\\traning\\samples_faulty\\'

VAL_PATH_GOOD = 'cutting-inserts-detection\\validation\\samples_good\\'
VAL_PATH_FAULTY = 'cutting-inserts-detection\\validation\\samples_faulty\\'

TEST_PATH_GOOD = 'cutting-inserts-detection\\test\\samples_good\\'
TEST_PATH_FAULTY = 'cutting-inserts-detection\\test\\samples_faulty\\'

#path list
PATHS = (PATH_GOOD,PATH_FAULTY,LEARN_PATH_GOOD,LEARN_PATH_FAULTY,VAL_PATH_GOOD,VAL_PATH_FAULTY,TEST_PATH_GOOD,TEST_PATH_FAULTY)
print(PATHS)

#claer all prev images
for dir in PATHS:
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

g_number = 0
f_number = 0
k = 0
l = 0

for filename in os.listdir(PATH):
    img = cv.imread(os.path.join(PATH,filename))
    cv.imshow('Classify sample',img)
    resized = cv.resize(img, (150,150), interpolation = cv.INTER_AREA)
    cv.namedWindow('Resized',cv.WINDOW_FREERATIO)
    cv.imshow('Resized',resized)
    l+=1
    
    chose_path=random.randint(0, 100)
    if chose_path<60:
        PATH_GOOD = LEARN_PATH_GOOD
        PATH_FAULTY = LEARN_PATH_FAULTY
    elif chose_path<80:
        PATH_GOOD = VAL_PATH_GOOD
        PATH_FAULTY = VAL_PATH_FAULTY
    else:
        PATH_GOOD = TEST_PATH_GOOD
        PATH_FAULTY = TEST_PATH_FAULTY     

    if(l%4==1): k = cv.waitKey(0)
    if  k == ord('g'):
        print("Good")
        cv.imwrite(PATH_GOOD+'g'+str(g_number)+'.png', resized)
        g_number+=1
    elif k == ord('f'):
        print("Faulty")
        cv.imwrite(PATH_FAULTY+'f'+str(f_number)+'.png', resized)
        f_number+=1
 

  
