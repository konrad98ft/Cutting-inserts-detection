import cv2 as cv
import numpy as np
import random as rng
import matplotlib.pyplot as plt
import math 
from scipy.optimize import fsolve
PATH = 'D:\\python Image Processing\\cutting_inserts\\oswietlacz_pierscieniowy_backlight\\'
#R = 4mm R = 170px 
PX2MM = 170/4

def findLinesPoints(roi,direction):   
    pts = []
    searching_msk = 3
    msk_val = (30,50,30)
    # X direction searching
    if(direction[1] == 1):              
        x_range = (0,roi.shape[0]-searching_msk,1)
        y_range = (0,roi.shape[1]-1,1) 
    if(direction[1] == -1): 
        x_range = (roi.shape[0]-1-searching_msk,0,-1)  
        y_range = (0,roi.shape[1]-1,1)              
    # Y direction searching
    if(direction[0] == 1): 
        x_range = (0,roi.shape[1]-searching_msk,1)
        y_range = (roi.shape[0]-1,0,-1) 
    if(direction[0] == -1):            
        x_range = (roi.shape[1]-1-searching_msk,0,-1)
        y_range = (roi.shape[0]-1,0,-1)  

    drawing = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
    for y in range(y_range[0],y_range[1],y_range[2]): #y_range
        for x in range(x_range[0],x_range[1],x_range[2]): #x_range
            box = 0
            if(direction[1] != 0):
                for i in range(searching_msk):
                    if(roi[x+i,y] > msk_val[i]): box+=1              
                if(box >= searching_msk-1):
                    xl = x + int(searching_msk/2)      
                    pts.append([y,xl])
                    for j in range(searching_msk): drawing[x+j,y]=(200,0,0)
                    drawing[xl,y]=(0,255,0)
                    break

            if(direction[0] != 0): 
                for i in range(searching_msk):
                    if(roi[y,x+i] > msk_val[i]): box+=1               
                if(box >= searching_msk-1):
                    xl = x + int(searching_msk/2)    
                    pts.append([xl,y])
                    for j in range(searching_msk): drawing[y,x+j]=(200,0,0)
                    drawing[y,xl]=(0,0,255)                    
                    break
    #Show searching effect                
    roi = cv.cvtColor(roi,cv.COLOR_GRAY2BGR)
    '''### Visualization
    #drawing = cv.bitwise_or(drawing, roi)
    cv.namedWindow('Drawing', cv.WINDOW_NORMAL)
    cv.imshow('Drawing',drawing)'''
 

    return pts
 
def linesFiltration(roi,direction):
    kernel1 = np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[4,4,4,4,4],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]])
    kernel2 = np.array([[-1,-1,4,-1,-1],[-1,-1,4,-1,-1],[-1,-1,4,-1,-1],[-1,-1,4,-1,-1],[-1,-1,4,-1,-1]])
   
    if(direction[1]!=0): kernel = kernel1
    if(direction[0]!=0): kernel = kernel2
    
    roi2 = cv.filter2D(roi,-1,kernel)
    
    #Show filter effect
    '''cv.namedWindow('Drawing2', cv.WINDOW_NORMAL)
    cv.imshow('Drawing2',roi2) '''
    return roi2

def searchingBox(image, points=(300,650,400,500), direction=(0,1)):
    #points = (x1,x2,y1,y2) direction = (x_dir,y_dir)

    #Apply ROI
    roi = image.copy()[points[2]:points[3],points[0]:points[1]]
    #Treshold
    ret,roi = cv.threshold(roi,80,255,cv.THRESH_TOZERO)
    #Show ROI
    '''cv.namedWindow('Searching Box', cv.WINDOW_NORMAL)
    cv.imshow('Searching Box',  roi)
    cv.resizeWindow('Searching Box', points[1]-points[0], points[3]-points[2] )'''

    #Distract various searching directions
    roi =  linesFiltration(roi,direction)
    pts = findLinesPoints(roi,direction)
   
    #Fit line
    vector = np.array(pts)
    vx,vy,x,y = cv.fitLine(vector,cv.DIST_HUBER, 0, 0.01, 0.05) 
    
    #Show ROI and fitted line on the orgnial image  
    x = x + points[0]   #go back to the global coordinate system
    y = y + points[2]
    line = vx,vy,x,y

    k = 1000
    p1 = (int(x - k*vx), int(y - k * vy))
    p2 = (int(x + k*vx), int(y + k * vy))
    cv.line(image, p1,p2 , (255,255,255), 1, cv.LINE_AA, 0)
    cv.rectangle(image,(points[0],points[2]),(points[1],points[3]),(255,255,255),2)
    cv.namedWindow('ROI', cv.WINDOW_NORMAL)
    cv.imshow('ROI',image)
    cv.resizeWindow('ROI',800,600)

    return line 
     
def findArcPoint(image,line1,line2):
    # Solving linear equation to find lines crossing point
    vx1,vy1,x1,y1 = line1
    vx2,vy2,x2,y2 = line2
    A = np.array([[vx1, 0, -1,0], [vy1, 0, 0,-1], [0, vx2, -1,0], [0, vy2, 0,-1]], dtype='float')
    B = np.array([-x1,-y1,-x2,-y2], dtype='float')
    R = np.linalg.inv(A).dot(B)
    xs,ys = R[2:]
    vx = vx1 + vx2
    vy = vy1 + vy2

    # Solving non linear equation to find arc centre
    def f(z):   
        xc = z[0]
        yc = z[1]
        k = (yc-ys)/vy
        F = np.empty((2))
        F[0] =  math.sqrt((xs-xc)**2 +(ys-yc)**2) - PX2MM*4 
        F[1] =  xc - vx*k - xs
        return F
    xc,yc = fsolve(f, (0.1,0.1) ) 

    ### Visualization ###
    cv.circle(img,(int(R[2]),int(R[3])),10,(255,255,255),2) #Lines intersection
    cv.circle(img,(int(xc),int(yc)),1,(255,255,255),2) #Arc centre
    cv.circle(img,(int(xc),int(yc)),int(PX2MM*4/math.sqrt(2)),(255,255,255),1) #Arc radius
    
    # ROI boundaries
    inc = 50
    rx0 = int(xc) if xc < xs else int(xs) 
    ry0 = int(yc) if yc < ys else int(ys)
    rxk = int(xc+inc) if xc > xs else int(xs+inc) 
    ryk = int(yc+inc) if yc > ys else int(ys+inc)
    roi = image.copy()[ry0:ryk,rx0:rxk]
    
    ### Visualization ###
    cv.namedWindow('Arc ROI', cv.WINDOW_NORMAL)    
    cv.imshow('Arc ROI', roi)
    cv.resizeWindow('Arc ROI', (rxk-rx0)*3,(ryk-ry0)*3) 
    print(roi.shape)

    # Polar transform and filtration
    roi = polarTransform(roi,start_point=(0,0),r=(int(PX2MM*0.75),int(PX2MM*3)),theta=90,theta_inc=0.25)
    ret,roi2 = cv.threshold(roi,80,255,cv.THRESH_TOZERO)
    roi2 = linesFiltration(roi2,(0,-1))
    pts = findLinesPoints(roi2,(0,-1))
    pts_y = []
    for i in range(len(pts)): pts_y.append(pts[i][1])

    s = srednia(pts_y) 
    m = mediana(pts_y)  
    o = odchylenie(pts_y, s)  
    print("Srednia: {:.2f}   mediana: {:.2f}   odchylenie standardowe: {:.2f}".format(s,m,o))
   
    ### Visualization ###
    cv.namedWindow('orginal ROI', cv.WINDOW_NORMAL)
    cv.imshow('orginal ROI', roi)
    cv.resizeWindow('orginal ROI', (rxk-rx0)*3,(ryk-ry0)*3)
    print(roi.shape)
    cv.namedWindow('binary ROI', cv.WINDOW_NORMAL)
    cv.imshow('binary ROI', roi2)
    cv.resizeWindow('binary ROI', (rxk-rx0)*3,(ryk-ry0)*3) 
    

def polarTransform(roi,start_point,r,theta,theta_inc):
    drawing = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
    roi2 = np.zeros((int(r[1]-r[0]),int(theta/theta_inc)+1, 1), dtype=np.uint8)
    theta_range = np.arange(0, theta, theta_inc)

    for alpha in theta_range:
        x0 = int(math.sin(math.radians(alpha))*r[0])
        y0 = int(math.cos(math.radians(alpha))*r[0])
        xk = int(math.sin(math.radians(alpha))*r[1])
        yk = int(math.cos(math.radians(alpha))*r[1])
  
        roid = cv.cvtColor(roi,cv.COLOR_GRAY2BGR)
        #cv.waitKey(1) ### Visualization ###
        for R in range(r[0],r[1]):
            x = int(math.sin(math.radians(alpha))*R)+x0
            y = int(math.cos(math.radians(alpha))*R)+y0
            cv.circle(drawing,(x,y),1,(0,0,255),1)
              
            roi2[R-r[0],int(alpha/theta_inc)] = roi[x,y]

            '''### Visualization ###
            drawing = cv.bitwise_or(drawing, roid)
            cv.namedWindow('polar lines', cv.WINDOW_NORMAL)
            cv.imshow('polar lines', drawing)
            cv.resizeWindow('polar lines',drawing.shape[0]*3,drawing.shape[1]*3)

            cv.namedWindow('polar roi', cv.WINDOW_NORMAL)
            cv.imshow('polar roi', roi2)
            cv.resizeWindow('polar roi',roi2.shape[1]*3,roi2.shape[0]*3)'''
    return roi2  

def srednia(pts):
    suma = sum(pts)
    return suma / float(len(pts))

def mediana(pts):
    pts.sort()
    if len(pts) % 2 == 0:  
        half = int(len(pts) / 2)
        return float(sum(pts[half - 1:half + 1])) / 2.0
    else: 
        return pts[int(len(pts) / 2)]

def wariancja(pts, srednia):
    sigma = 0.0
    for ocena in pts:
        sigma += (ocena - srednia)**2
    return sigma / len(pts)

def odchylenie(pts, srednia): 
    w = wariancja(pts, srednia)
    return math.sqrt(w)


# Get image
for img_index in range(1,11):
    img_path= PATH + str(img_index) +'.bmp'
    img = cv.imread(img_path,-1)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img2 = img.copy()

    line1 = searchingBox(img,(300,650,400,500),(0,-1))
    line2 = searchingBox(img,(730,800,240,350),(-1,0))
    findArcPoint(img2,line1,line2)
    cv.waitKey(0)


'''
searchingBox(img,(300,650,100,200),(0,1))
cv.waitKey(0)
searchingBox(img,(150,220,240,350),(1,0))
cv.waitKey(0) 
'''





cv.waitKey(0)
cv.destroyAllWindows()
 


