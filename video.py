#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import cv2
import os
import sys
import os.path
import numpy as np


def webcam_image(img):
    """
    ->>> !!!! CLOSE WINDOW BY PRESSING ESC !!!! <<<<-
        this should receive the webcam/video frame
    """

    # uncomment if red and blue values are inverted
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


cv2.namedWindow("preview")

# using video
vc = cv2.VideoCapture("video.mp4")

# using webcam
#vc = cv2.VideoCapture(0)


if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:

    #defining image and masks

    img = webcam_image(frame)

    imgH, imgW, imgC = img.shape

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # mask 1 lower value
    image_lower_hsv1 = np.array([60, 70, 100])
    # mask 1 higher value
    image_upper_hsv1 = np.array([92, 200, 180])

    # mask 2 lower value
    image_lower_hsv2 = np.array([130, 130, 80])
    # mask 2 higher value
    image_upper_hsv2 = np.array([200, 255, 255])

    # getting mask result value and combining masks
    mask_hsv1 = cv2.inRange(img_hsv, image_lower_hsv1, image_upper_hsv1)
    mask_hsv2 = cv2.inRange(img_hsv, image_lower_hsv2, image_upper_hsv2)
    mask = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    # combining mask results to image
    target = cv2.bitwise_and(img,img, mask=mask)

    # finding shapes for masks 1 and 2
    contours1, _ = cv2.findContours(mask_hsv1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contours2, _ = cv2.findContours(mask_hsv2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    # defining font for displaying text
    font = cv2.FONT_HERSHEY_SIMPLEX

    #defining size and color for text (cross) 
    size = 15
    color = (0,0,0)

    try: #treating errors
        bothOk = 1
        try:
            #selecting shape with biggest area value
            max_contorno1 = max(contours1, key = cv2.contourArea)

            #drawing edges on shape
            cv2.drawContours(img, max_contorno1, -1, [255, 0, 0], 5);

            M1 = cv2.moments(max_contorno1)
        
            # getting coordinates for center of mass
            cx1 = int(M1['m10']/M1['m00'])
            cy1 = int(M1['m01']/M1['m00'])

            # drawing cross at center of mass
            cv2.line(img,(cx1 - size,cy1),(cx1 + size,cy1),color,5)
            cv2.line(img,(cx1,cy1 - size),(cx1, cy1 + size),color,5)

            # getting and displaying are value for shape 1 at top left
            text1 = (M1['m00'])
            origin1 = (50, 40)
            cv2.putText(img, str(text1), origin1, font,1,(200,50,0),2,cv2.LINE_AA)

        except:
            bothOk = 0

        try:
            
            max_contorno2 =max(contours2, key = cv2.contourArea)

            cv2.drawContours(img, max_contorno2, -1, [255, 0, 0], 5);

            M2 = cv2.moments(max_contorno2)

            cx2 = int(M2['m10']/M2['m00'])
            cy2 = int(M2['m01']/M2['m00'])

            cv2.line(img,(cx2 - size,cy2),(cx2 + size,cy2),color,5)
            cv2.line(img,(cx2,cy2 - size),(cx2, cy2 + size),color,5)

            text2 = (M2['m00'])
            origem2 = (imgW - 200, (imgH - 30))


            cv2.putText(img, str(text2), origem2, font,1,(200,50,0),2,cv2.LINE_AA)
        except:
            bothOk = 0

        if bothOk == 1: #if an error occurs line between shapes is not displayed
            # drawing line from shape 1 to shape 2 and displaying angle value
            tan = math.degrees(math.atan2(abs(cy1-cy2),abs(cx1-cx2)))
            cv2.putText(img, str(round(tan)), (round(imgW/2),round(imgH/2 - 20)), font,1,(200,50,0),2,cv2.LINE_AA)
            cv2.line(img, (cx1,cy1) ,( cx2,cy2), (255,0,0), 5)
            
    except:
        print("An exception occurred") 
    finally:
        # previewing results
        cv2.imshow("preview", img)
    
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

# killing program on while break
cv2.destroyWindow("preview")
vc.release()
