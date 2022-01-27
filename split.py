import cv2 as cv
import numpy as np
import sys
import os
import random
import zipfile
from os.path import basename

'''
Split images for learining to: validation-20% test-20% and train-60% directories.
As an input require cropped images. Use pre_processing.py to prepare it.
User should assign displayed images for 2 categories: "good" or "faulty".
Use keys "g" and "f".
'''


# Source images path
PATH = 'samples\\'

# Output images path
PATH_GOOD = 'traning\\samples_good\\'
PATH_FAULTY = 'traning\\samples_faulty\\'

LEARN_PATH_GOOD = 'traning\\samples_good\\'
LEARN_PATH_FAULTY = 'traning\\samples_faulty\\'

VAL_PATH_GOOD = 'validation\\samples_good\\'
VAL_PATH_FAULTY = 'validation\\samples_faulty\\'

TEST_PATH_GOOD = 'test\\samples_good\\'
TEST_PATH_FAULTY = 'test\\samples_faulty\\'

# Path list
PATHS = (PATH_GOOD,PATH_FAULTY,LEARN_PATH_GOOD,LEARN_PATH_FAULTY,VAL_PATH_GOOD,VAL_PATH_FAULTY,TEST_PATH_GOOD,TEST_PATH_FAULTY)
print(PATHS)

# Claer all prev images
for dir in PATHS:
    try: 
        os.mkdir(dir) 
    except OSError as error:    
        print(error)  
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

g_number = 0
f_number = 0
k = 0 # user key input 
l = 0 # used to putting 4 simillar images to the 1 directory by only 1 decision

for filename in os.listdir(PATH):
    
    # Show orginal and resized image
    img = cv.imread(os.path.join(PATH,filename))
    cv.imshow('Classify sample',img)
    resized = cv.resize(img, (500,500), interpolation = cv.INTER_AREA)
    cv.namedWindow('Resized',cv.WINDOW_FREERATIO)
    cv.imshow('Resized',resized)
    l+=1
    k = cv.waitKey(0)
        
    # Split images for learining~60% validation~20% testing~20%  
    chose_path=random.randint(0, 100)
        
    if chose_path<60:
        PATH_GOOD = LEARN_PATH_GOOD
        PATH_FAULTY = LEARN_PATH_FAULTY
        print("Traning")
    elif chose_path<80:
        PATH_GOOD = VAL_PATH_GOOD
        PATH_FAULTY = VAL_PATH_FAULTY
        print("Validation")
    else:
        PATH_GOOD = TEST_PATH_GOOD
        PATH_FAULTY = TEST_PATH_FAULTY
        print("Test")  

    if  k == ord('g'):
        cv.imwrite(PATH_GOOD+'g'+str(g_number)+'.png', resized)
        g_number+=1
    elif k == ord('f'):
        cv.imwrite(PATH_FAULTY+'f'+str(f_number)+'.png', resized)
        f_number+=1
 

