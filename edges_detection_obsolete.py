import cv2 as cv
import numpy as np
import random as rng
import matplotlib.pyplot as plt
import math 
PATH = 'D:\\python Image Processing\\cutting_inserts\\oswietlacz_pierscieniowy_backlight\\3.bmp'
PATH2 = '1.bmp'

font = cv.FONT_HERSHEY_SIMPLEX # font
org = (50, 50) # org
fontScale = 1 # fontScale
color = (255, 0, 0) # Blue color in BGR
thickness = 2 # Line thickness of 2 px

def noisesFiltration(img_in):
    kernel = np.ones((3,3),np.uint8)
    img_out = cv.dilate(img_in,kernel,iterations = 1)
    temp = cv.erode(img_in,kernel,iterations = 1)
    img_out=img_out-temp
    #kernel = np.ones((5,5),np.uint8)
    #img_out = cv.erode(img_out,kernel,iterations = 1)    
    img_out=cv.adaptiveThreshold(img_out,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,15,-7)
    
    #img_out=cv.morphologyEx(img_out,cv.MORPH_OPEN,kernel)


    cv.namedWindow('Erode Image', cv.WINDOW_NORMAL)
    cv.imshow('Erode Image',img_out)
    cv.resizeWindow('Erode Image',600,400)

    return img_out

def thresh_callback(val,img_in):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(img_in, threshold, 255)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    # Show in a window
    n_of_cont = "Ilosc konturow: " + str(len(contours))
    image = cv.putText(drawing, n_of_cont, org, font, fontScale, color, thickness, cv.LINE_AA)
    cv.namedWindow('All conturs', cv.WINDOW_NORMAL)
    cv.imshow('All conturs', drawing)  
    cv.resizeWindow('All conturs',600,400)    
    # Use only the biggest contours
    cont2 = []
    minContSize=300
    index = 0
    for i in range(len(contours)):
        if cv.contourArea(contours[i]) > minContSize:
            #print("Pole konturu nr. " ,i, "to ", cv.contourArea(contours[i]))
            cont2.append(contours[i])
    
    drawing2 = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(cont2)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing2, cont2, i, color,2  , cv.LINE_AA , hierarchy, 0)  
    drawing = drawing2.copy()

   # Show in a window
    n_of_cont = "Ilosc konturow: " + str(len(cont2))
    image = cv.putText(drawing2, n_of_cont, org, font, fontScale, color, thickness, cv.LINE_AA)
    cv.namedWindow('Biggest conturs', cv.WINDOW_NORMAL)
    cv.imshow('Biggest conturs', drawing2) 
    cv.resizeWindow('Biggest conturs',600,400)
    return drawing   

def canny_callback(val,img_in):
    threshold = val
    hough_lines=img_in.copy()
    height, width, channels = img_in.shape
    canny_output = cv.Canny(img_in, threshold, 255)
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    mask = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    
    #Create bounding box
    cnt = contours[0]
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    #Find ROI
    s,d,ang=rect
    xs = s[0]
    ys = s[1]
    points = cv.boxPoints(rect)
    for p in points:    #offest points to create ROI
        p[0]= p[0] -50 if p[0] < ys else p[0] + 50
        p[1]= p[1] -50 if p[1] < ys else p[1] + 50
    
    ROI = [1,points[0],points[1],s]
    for  p in points:
        cv.circle(drawing,(int(p[0]),int(p[1])),5, (0,0,255), -1)

    #Create mask
    maskThinckess=7
    cv.drawContours(mask,[box],0,(255,255,255,50),maskThinckess)
    result = cv.bitwise_or(drawing, mask)
    cv.namedWindow('Bouding box', cv.WINDOW_NORMAL)
    cv.imshow('Bouding box',result)
    cv.resizeWindow('Bouding box',600,400)  
    result2 = cv.bitwise_and(drawing, mask)
    
    #Generate bunch of ROI
    ROI=[]
    p3 = int(s[0]),int(s[1]) 
    for i in range(3): 
        p1 = int(points[i][0]),int(points[i][1])
        p2 = int(points[i+1][0]),int(points[i+1][1]) 
        roi = np.array([[p1,p2,p3]],np.int32)
        ROI.append(roi)
    p1 = int(points[3][0]),int(points[3][1])
    p2 = int(points[0][0]),int(points[0][1])
    roi = np.array([[p1,p2,p3]],np.int32)
    ROI.append(roi)
    pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    trg_msk = [np.zeros((result2.shape[0], result2.shape[1], 3), dtype=np.uint8) for i in range(4)]


    for i,pts in enumerate(ROI):
        cv.fillPoly(trg_msk[i], [pts], (255,255,255))

        trg_msk[i]= cv.bitwise_and(result2, trg_msk[i])
        trg_msk[i]= cv.cvtColor(trg_msk[i], cv.COLOR_BGR2GRAY)
        '''
        title="trg"+str(i)
        cv.namedWindow(title, cv.WINDOW_NORMAL)
        cv.imshow(title,trg_msk[i])
        cv.resizeWindow(title,600,400) '''

    lines=[]
    min_max=[]
    for i,find_line in enumerate(trg_msk):
        
        ####Find best fitting lines
        contours, hierarchy = cv.findContours(find_line, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
        cnt = contours[0]
        vx,vy,x,y= cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01); 
        lines.append([vx,vy,x,y])
        #print("Linia ",i," ",vx,vy,x,y)
        lineColor = (255, 0, 0)
        height, width = find_line.shape
        lefty = int((-x * vy / vx) + y)
        righty = int(((width - x) * vy / vx) + y)
        point1 = (width-1, righty)
        point2 = (0, lefty)
 
        ###Find x0,y0 and xk yk of the lines from previous step
        a=vy/vx
        b=y-a*x
        min_y = 9999.0
        min_x = 9999.0
        max_x = 0.0
        max_y = 0.0
        if a >= 0.5:
            for x in range(find_line.shape[0]):
                for y in range(find_line.shape[1]):
                    if find_line[x][y] > 0:
                        if y < min_x : min_x = y  
                        if y > max_x: max_x = y 
            min_y = a*min_x+b
            max_y = a*max_x+b
            min_max.append([min_x,min_y,max_x,max_y])  
            
            P1 = (int(min_x),int(min_y))
            P2 = (int(max_x),int(max_y))
            print(i,'P1:',min_x,min_y,'P2:',max_x, max_y )
            cv.line(find_line,P1,P2, (255,255,255), 5)
            
        if a < 0.5:
            for x in range(find_line.shape[0]):
                for y in range(find_line.shape[1]):
                    if find_line[x][y] > 0:
                        if x < min_y : min_y = x 
                        if x > max_y: max_y = x 
            min_x=((min_y-b)/a)
            max_x=((max_y-b)/a)
            min_max.append([min_x,min_y,max_x,max_y])  
            
            P1 = (int(min_x),int(min_y))
            P2 = (int(max_x),int(max_y))
            print(i,'P1:',min_x,min_y,'P2:',max_x, max_y )
            cv.line(find_line,P1,P2, (255,255,255), 5)
        


        title="line "+str(i)
        cv.namedWindow(title, cv.WINDOW_NORMAL)
        cv.imshow(title,find_line)
        cv.resizeWindow(title,600,400) 


        
            
    
    #find lines
    plt.imshow(result2)
    plt.title("Lines")
    plt.show()




    return drawing

def processImg(val):
    cont = cv.getTrackbarPos('Contours', 'Erode Image')
    edges = cv.getTrackbarPos('Canny', 'Erode Image')
    img2 = thresh_callback(cont,img)
    img3 = canny_callback(edges,img2)

 
# Get image
img = cv.imread(PATH,-1)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.namedWindow('Source Image', cv.WINDOW_NORMAL)
cv.imshow('Source Image',img)
cv.resizeWindow('Source Image',600,400)  

# Preapre imagr
img = noisesFiltration(img)

#Find contours
thresh = 30
maxTresh = 200
cv.createTrackbar('Contours', 'Erode Image', thresh, maxTresh, processImg)
cv.createTrackbar('Canny', 'Erode Image', thresh, maxTresh, processImg)
#processImg(thresh)

cv.waitKey(0)



cv.destroyAllWindows()


