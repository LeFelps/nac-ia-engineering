#!/usr/bin/python
# -*- coding: utf-8 -*-

# Programa simples com camera webcam e opencv

import math
import cv2
import os
import sys
import os.path
import numpy as np


def image_da_webcam(img):
    """
    ->>> !!!! FECHE A JANELA COM A TECLA ESC !!!! <<<<-
        deve receber a imagem da camera e retornar uma imagems filtrada.
    """
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


cv2.namedWindow("preview")
vc = cv2.VideoCapture("video.mp4")


if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:

    img = image_da_webcam(frame)

    imgH, imgW, imgC = img.shape

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    image_lower_hsv1 = np.array([60, 70, 100])
    image_upper_hsv1 = np.array([92, 200, 180])

    image_lower_hsv2 = np.array([130, 130, 80])
    image_upper_hsv2 = np.array([200, 255, 255])

    mask_hsv1 = cv2.inRange(img_hsv, image_lower_hsv1, image_upper_hsv1)

    mask_hsv2 = cv2.inRange(img_hsv, image_lower_hsv2, image_upper_hsv2)

    mask = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    target = cv2.bitwise_and(img,img, mask=mask)

    contornos1, _ = cv2.findContours(mask_hsv1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    contornos2, _ = cv2.findContours(mask_hsv2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    max_contorno1 = max(contornos1, key = cv2.contourArea)

    max_contorno2 =max(contornos2, key = cv2.contourArea)

    cv2.drawContours(img, max_contorno1, -1, [255, 0, 0], 5);

    cv2.drawContours(img, max_contorno2, -1, [255, 0, 0], 5);

    M1 = cv2.moments(max_contorno1)

    M2 = cv2.moments(max_contorno2)

    cx1 = int(M1['m10']/M1['m00'])
    cy1 = int(M1['m01']/M1['m00'])

    cx2 = int(M2['m10']/M2['m00'])
    cy2 = int(M2['m01']/M2['m00'])

    font = cv2.FONT_HERSHEY_SIMPLEX

    size = 15
    color = (0,0,0)

    cv2.line(img,(cx1 - size,cy1),(cx1 + size,cy1),color,5)
    cv2.line(img,(cx1,cy1 - size),(cx1, cy1 + size),color,5)

    cv2.line(img,(cx2 - size,cy2),(cx2 + size,cy2),color,5)
    cv2.line(img,(cx2,cy2 - size),(cx2, cy2 + size),color,5)


    text1 = (M1['m00'])
    origem1 = (50, 40)

    text2 = (M2['m00'])
    origem2 = (imgW - 200, (imgH - 30))

    cv2.putText(img, str(text1), origem1, font,1,(200,50,0),2,cv2.LINE_AA)

    cv2.putText(img, str(text2), origem2, font,1,(200,50,0),2,cv2.LINE_AA)

    tan = math.degrees(math.atan2(abs(cy1-cy2),abs(cx1-cx2)))
    cv2.putText(img, str(round(tan)), (round(imgW/2),round(imgH/2 - 20)), font,1,(200,50,0),2,cv2.LINE_AA)

    #contornos_img = cv2.bitwise_or(contornos_img1, contornos_img2)

    cv2.line(img, (cx1,cy1) ,( cx2,cy2), (255,0,0), 5)


    cv2.imshow("preview", img)

    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
