
import numpy as np
import cv2 as cv

import math
from numpy.core.fromnumeric import shape
from numpy.core.numeric import rollaxis 
from scipy import ndimage
import time
import warnings
import sys


from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
'''
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.preprocessing import image
'''
warnings.filterwarnings("ignore", category=DeprecationWarning) 

PATH = 'D:\\Python Image Processing\\cutting-inserts-detection\\cutting_inserts\\oswietlacz_pierscieniowy_backlight\\'
PATH2 = 'C:\\Users\Konrad\\cutting-inserts-detection-master\\tensor_flow_samples\\'
PATH3 = 'C:\\Users\Konrad\\cutting-inserts-detection-master\\standSamples\\'

#model = keras.models.load_model('C:\\Users\\Konrad\\cutting-inserts-detection-master\\model_sztuczne_wadliwe2')
PX2MM = 580/4 #R = 4mm R = 620px 
 
def linesFiltration(roi,direction):
    # Define kernels
    a = [-2,-1,1,6,1,-1,-2]
    kernel1 = np.array([[a[0],a[0],a[0],a[0],a[0],a[0],a[0]],[a[1],a[1],a[1],a[1],a[1],a[1],a[1]],[a[2],a[2],a[2],a[2],a[2],a[2],a[2]],
                       [a[3],a[3],a[3],a[3],a[3],a[3],a[3]],
                       [a[2],a[2],a[2],a[2],a[2],a[2],a[2]],[a[1],a[1],a[1],a[1],a[1],a[1],a[1]],[a[0],a[0],a[0],a[0],a[0],a[0],a[0]]])
    kernel2 = np.array([a,a,a,a,a])
    
    # Chose kernel proper for defined direction
    if(direction[1]!=0): kernel = kernel1
    if(direction[0]!=0): kernel = kernel2
    
    roi2 = cv.filter2D(roi,-1,kernel)
    
    #Show filter effect
    cv.namedWindow('linesFiltration', cv.WINDOW_NORMAL)
    cv.imshow('linesFiltration',roi2)

    return roi2

def findLinesPoints(roi,direction):   
    
    # Some preprocessing
    kernel = np.ones((7,7),np.uint8)
    roi = cv.morphologyEx(roi, cv.MORPH_OPEN, kernel)
    roi = cv.Canny(roi,100,30)
    ### Visualization
    cv.namedWindow('findLinesPoints', cv.WINDOW_NORMAL)
    cv.imshow('findLinesPoints',roi)  
    
    # Rotate image to ensure proper searching direction
    if(direction[0] == -1): roi = cv.flip(roi, 1) 
    if(direction[1] != 0): 
        roi =  cv.rotate(roi, cv.ROTATE_90_CLOCKWISE)
        if(direction[1] == -1): roi = cv.flip(roi, 1)  

    # Find min non zero val in each row
    rows,cols = roi.shape
    drawing = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        non_zero_values=np.flatnonzero(roi[r])
        if any(non_zero_values): 
            drawing.itemset((r,non_zero_values[0]),255)

    # Rotate image to go back to the base coordinates
    if(direction[0] == -1): drawing = cv.flip(drawing, 1) 
    if(direction[1] != 0): 
        if(direction[1] == -1): drawing = cv.flip(drawing, 1) 
        drawing =  cv.rotate(drawing, cv.ROTATE_90_COUNTERCLOCKWISE)
         
    # Find outer line points 
    pts = []
    pts = cv.findNonZero(drawing)

    ### Visualization
    cv.namedWindow('findLinesPoints4', cv.WINDOW_NORMAL)
    cv.imshow('findLinesPoints4',drawing)         
    return pts
 
def searchingBox(image, points, direction=(0,1)): 
    
    ### points = (x1,y1,dx1,dy2) direction = (x_dir,y_dir) ###
    pts = (points[0],(points[0]+points[2]),points[1],(points[1]+points[3]))

    # Apply ROI
    roi = image.copy()[pts[2]:pts[3],pts[0]:pts[1]]

    # Treshold
    ret,roi = cv.threshold(roi,150,255,cv.THRESH_TOZERO)
    
    
    # Find points with belongs to the edge
    roi = linesFiltration(roi,direction)
    pts = findLinesPoints(roi,direction)
   
    # Break in case of faulty input image
    if(pts is None):
        print("Any line found")
        return -1,-1,-1,-1  

    # Fit line
    vector = np.array(pts)
    vx,vy,x,y = cv.fitLine(vector,cv.DIST_HUBER, 0, 0.01, 0.05) 
    
    # Show ROI and fitted line on the orgnial image  
    x = x + points[0]   # Go back to the global coordinate system
    y = y + points[1]
    line = vx,vy,x,y

    k = 10000
    p1 = (int(x - k*vx), int(y - k * vy))
    p2 = (int(x + k*vx), int(y + k * vy))
    cv.line(img, p1,p2 , (255,255,255), 3, cv.LINE_AA, 0)
    cv.rectangle(img,(points[0],points[1],points[2],points[3]),(255,255,255),2)
    cv.namedWindow(str(img_index), cv.WINDOW_FREERATIO)
    cv.imshow(str(img_index), img)
    cv.resizeWindow(str(img_index), int(img.shape[1]/2),int(img.shape[0]/2)) 
    return line 
     
def findArcPoint(image,line1,line2):
    # Solving linear equation to find lines crossing point
    vx1,vy1,x1,y1 = line1
    vx2,vy2,x2,y2 = line2
    A = np.array([[vx1, 0, -1,0], [vy1, 0, 0,-1], [0, vx2, -1,0], [0, vy2, 0,-1]], dtype='float')
    B = np.array([-x1,-y1,-x2,-y2], dtype='float')
    R = np.linalg.inv(A).dot(B)
    xs,ys = R[2:]
    rot_ang = math.atan2(vy2,vx2) 
    vy =  abs( vx1 +  vx2 ) if vy2 < 0 else abs( vy1 +  vy2 )
    vx = abs( vy1 +  vy2 ) if vy2 < 0 else abs( vx1 +  vx2 )

    l = math.sqrt(vx**2 + vy**2) # lenght of those vectors
    k = (PX2MM*4)/l # how many vectors is between line crossing point and cutting insert arc centre
    p1 = (int(xs + k*vx), int(ys + k * vy))
    p2 = (int(xs ), int(ys ))
    cv.line(img, p1,p2 , (255,255,255), 2, cv.LINE_AA, 0)

    
    # Find 4 possible arc centres of the cutting insert
    C = [] # coortinates of the 4 possible arc centres
    v = np.array([[vx,vy],[-vx,vy],[-vx,-vy],[vx,-vy]], dtype='float')  # all possible direction of the vectors

    for i in range(len(v)): # all possible configurations
        pom = xs + v[i][0]*k  , ys + v[i][1]*k
        cv.circle(img,(int(xs + v[i][0]*k),int(ys + v[i][1]*k)),1,(255,255,255),4) ### Visualization ###
        C.append(pom)
 
    # Chose ROI with contains cutting insert arc - closest to the centre of the image
    min_dist = 9999
    img_cy,img_cx=img.shape[:2]
    for i in range(len(v)):
        dist = math.sqrt( (C[i][0]-img_cx/2)**2 +  (C[i][1]-img_cy/2)**2 )
        if(  dist < min_dist):
            min_dist = dist
            properArc = i       
    xc,yc=C[properArc] #proper arc centre coordinates

    # Build roi between arc centre(xc,yc) and lines crossing point (xs,ys) in dependece on their location 
    inc = 100 #offset outer boundaries by some offset to avoid cutting the arc
    rx0 = int(xc) if xc < xs else int(xs-inc) 
    ry0 = int(yc) if yc < ys else int(ys-inc)
    rxk = int(xc) if xc > xs else int(xs+inc) 
    ryk = int(yc) if yc > ys else int(ys+inc)
    roi = image.copy()[ry0:ryk,rx0:rxk]

    # Rotate roi
    ang =0
    if(xc>xs and yc<ys): ang = 90 
    if(xc>xs and yc>ys): ang = 180 
    if(xc<xs and yc>ys): ang = 270  
    roi = ndimage.rotate(roi, ang)

    ### Visualization ###
    cv.circle(img,(int(R[2]),int(R[3])),int(PX2MM*4),(255,255,255),3) #Lines intersection
    cv.circle(img,(int(xc),int(yc)),5,(255,255,255),3) #Arc centre
    cv.circle(img,(int(xc),int(yc)),int(PX2MM*4/math.sqrt(2)),(255,255,255),2) #Arc radius

    ### Visualization ###
    cv.namedWindow('Arc ROI', cv.WINDOW_NORMAL)    
    cv.imshow('Arc ROI', roi)
    cv.resizeWindow('Arc ROI', roi.shape[1],roi.shape[0]) 
    cv.namedWindow(str(img_index), cv.WINDOW_FREERATIO)
    cv.imshow(str(img_index), img)
    cv.resizeWindow(str(img_index), int(img.shape[1]/2),int(img.shape[0]/2)) 

    # Polar transform and filtration
    try:
        roi = polarTransform(roi,start_point=(0,0),r=(int(PX2MM*1),int(PX2MM*2.25)),theta=90,theta_inc=0.25)
    except:
        roi = roi
        print("Can't find cutting insert arc")
        return -1

    ### Visualization ###
    cv.namedWindow('Arc ROI2', cv.WINDOW_NORMAL)    
    cv.imshow('Arc ROI2', roi)
    cv.resizeWindow('Arc ROI2', roi.shape[1],roi.shape[0]) 

    
    ret,roi2 = cv.threshold(roi,150,255,cv.THRESH_TOZERO)
    roi2 = linesFiltration(roi2,(0,-1))
    pts = findLinesPoints(roi2,(0,1))

    if(pts is None):
        print("Any line found")
        return -1
    else: 
        pts_y = []
        for i in range(len(pts)): pts_y.append(pts[i][0][1])

        s = srednia(pts_y) 
        m = mediana(pts_y)  
        o = odchylenie(pts_y, s)  
        print("Średnia: {:.2f}\nMediana: {:.2f}\nOdchylenie standardowe: {:.2f}".format(s,m,o))
        if(s < 137 and s > 129) and o < 1.5:
            cv.putText(img,('OK    '+'srednia: {:.2f} odchylenie: {:.2f}').format(s,o),(100,100), cv.FONT_HERSHEY_PLAIN, 5,255,2)
        else:
            cv.putText(img,('N_OK   '+'srednia: {:.2f} odchylenie: {:.2f}').format(s,o),(100,100), cv.FONT_HERSHEY_PLAIN, 5,255,2)
    
    ### Visualization ###
    '''cv.namedWindow('orginal ROI', cv.WINDOW_NORMAL)
    cv.imshow('orginal ROI', roi)
    cv.resizeWindow('orginal ROI', (rxk-rx0)*3,(ryk-ry0)*3)

    cv.namedWindow('binary ROI', cv.WINDOW_NORMAL)
    cv.imshow('binary ROI', roi2)
    cv.resizeWindow('binary ROI', (rxk-rx0)*3,(ryk-ry0)*3) '''
    
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
        
        for R in range(r[0],r[1]):
            x = int(math.sin(math.radians(alpha))*R)+x0
            y = int(math.cos(math.radians(alpha))*R)+y0
            cv.circle(drawing,(x,y),1,(0,0,255),1)
              
            roi2[R-r[0],int(alpha/theta_inc)] = roi[x,y]

            ### Visualization ###
            '''drawing = cv.bitwise_or(drawing, roid)
            cv.namedWindow('polar lines', cv.WINDOW_NORMAL)
            cv.imshow('polar lines', drawing)
            cv.resizeWindow('polar lines',drawing.shape[0],drawing.shape[1])

            cv.namedWindow('polar roi', cv.WINDOW_NORMAL)
            cv.imshow('polar roi', roi2)
            cv.resizeWindow('polar roi',roi2.shape[1],roi2.shape[0])
            cv.waitKey(1)'''
        

    return roi2  

# Output analyze
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

def findInsertCentreOtsu(img):
       
    # Prepare image by finding conturs - otsu threshold
    thresh_val = threshold_otsu(img)
    ret,img2 = cv.threshold(img,thresh_val,255,cv.THRESH_TOZERO)

    kernel = np.ones((5, 5), np.uint8)
    edged = cv.erode(img2, kernel) 
    kernel2 = np.ones((9, 9), np.uint8)
    edged = cv.dilate(edged, kernel) 

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

    # Centre of the cutting insert
    Xdim = 1100
    Ydim = 650
    
    return XC,YC



 
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))


    # remove artifacts connected to image border
    cleared = clear_border(bw)


    # label image regions
    label_image = label(cleared)


    printTime("Alt-2")
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
  
    max_region = regionprops(label_image)[0]
    printTime("Alt-2.2")
    for region in regionprops(label_image):
        # find the largest region
        if region.area >= max_region.area:
            max_region = region
    printTime("Alt-3")
    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = max_region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    print("Angle:",rect)
    printTime("Alt-4")

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    return image

def printTime(str='time'):
    elapsed_time = time.time() - start_time
    print("{}: \t {:.3f}s".format(str,elapsed_time))

def deepL(image):
   
    # Reshape image for deepL clasification
    XC,YC = 1480,1220
    Xdim, Ydim = 1000, 600
    end_point = (XC, YC)
    start_point = (int(XC-Xdim), int(YC-Ydim))
    deepL_img = img3[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    deepL_img = cv.resize(deepL_img, (150,150), interpolation = cv.INTER_AREA)
    ### Visualization
    cv.namedWindow("deepL_img", cv.WINDOW_FREERATIO)
    cv.imshow("deepL_img", deepL_img)
    cv.resizeWindow("deepL_img", int(deepL_img.shape[1]),int(deepL_img.shape[0])) 

    # Clasification
    classification = []
    x = image.img_to_array(deepL_img)
    x = np.expand_dims(x, axis=0)
    x/=255
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)

    if classes > 0.5:
        title =  "is good  " + str(round(float((classes)*100),2)) + "%"
    else:
        title =  "is faulty  " + str(round(float((1-classes)*100),2)) + "%"
    print(title)
    cv.putText(img,title,(100,300), cv.FONT_HERSHEY_PLAIN, 5,255,2)




for img_index in range(1,15):
    # Get an image
    img_path= PATH3 + str(img_index) +'.png'
    img = cv.imread(img_path,-1)
    img3 = img.copy()
    try:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    except:
        print("Image not found")
        sys.exit(1)
    start_time = time.time()

    # Find lines to define arc centre
    img2 = img.copy()
    cv.rectangle(img,(1000,600,800,250),(255,255,255),2)
    cv.rectangle(img,(375,1050,300,250),(255,255,255),2)
    # Show effects
    cv.namedWindow(str(img_index), cv.WINDOW_FREERATIO)
    cv.imshow(str(img_index), img)
    cv.resizeWindow(str(img_index), int(img.shape[1]/2),int(img.shape[0]/2)) 
    cv.waitKey(1)
    printTime("Not important time")

    line1 = searchingBox(img2,(1000,600,800,250),(0,-1))
    line2 = searchingBox(img2,(375,1050,300,250),(1,0)) 
    printTime("Detecting lines")

    # Find breaches
    findArcPoint(img2,line1,line2) 
    printTime("Examine edge")

    # DeepL clacification
    #deepL(img3)
    

    # Show effects
    cv.namedWindow(str(img_index), cv.WINDOW_FREERATIO)
    cv.imshow(str(img_index), img)
    cv.resizeWindow(str(img_index), int(img.shape[1]/2),int(img.shape[0]/2)) 


    
    cv.waitKey(0)
    cv.destroyAllWindows()




