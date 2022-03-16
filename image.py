import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

img = cv2.imread('circulo.PNG')

# image heigth width and channels
imgH, imgW, imgC = img.shape

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

image_lower_hsv1 = np.array([86, 166, 226])  
image_upper_hsv1 = np.array([86, 166, 226])

image_lower_hsv2 = np.array([0, 239, 177])  
image_upper_hsv2 = np.array([0, 239, 177])

mask_hsv1 = cv2.inRange(img_hsv, image_lower_hsv1, image_upper_hsv1)

mask_hsv2 = cv2.inRange(img_hsv, image_lower_hsv2, image_upper_hsv2)

mask = cv2.bitwise_or(mask_hsv1, mask_hsv2)


contornos1, _ = cv2.findContours(mask_hsv1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
contornos2, _ = cv2.findContours(mask_hsv2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

mask_rgb1 = cv2.cvtColor(mask_hsv1, cv2.COLOR_GRAY2RGB) 
contornos_img1 = mask_rgb1.copy() 

mask_rgb2 = cv2.cvtColor(mask_hsv2, cv2.COLOR_GRAY2RGB) 
contornos_img2 = mask_rgb2.copy()


cv2.drawContours(contornos_img1, contornos1, -1, [255, 0, 0], 5);

cv2.drawContours(contornos_img2, contornos2, -1, [255, 0, 0], 5);

cnt1 = contornos1[0]

cnt2 = contornos2[0]

M1 = cv2.moments(cnt1)

M2 = cv2.moments(cnt2)

cx1 = int(M1['m10']/M1['m00'])
cy1 = int(M1['m01']/M1['m00'])

cx2 = int(M2['m10']/M2['m00'])
cy2 = int(M2['m01']/M2['m00'])

font = cv2.FONT_HERSHEY_SIMPLEX

size = 15
color = (0,0,0)

cv2.line(contornos_img1,(cx1 - size,cy1),(cx1 + size,cy1),color,5)
cv2.line(contornos_img1,(cx1,cy1 - size),(cx1, cy1 + size),color,5)

cv2.line(contornos_img2,(cx2 - size,cy2),(cx2 + size,cy2),color,5)
cv2.line(contornos_img2,(cx2,cy2 - size),(cx2, cy2 + size),color,5)


text1 = (M1['m00'])
origem1 = (50, 40)

text2 = (M2['m00'])
origem2 = (imgW - 200, (imgH - 30))

cv2.putText(contornos_img1, str(text1), origem1, font,1,(200,50,0),2,cv2.LINE_AA)

cv2.putText(contornos_img2, str(text2), origem2, font,1,(200,50,0),2,cv2.LINE_AA)

tan = math.degrees(math.atan2(abs(cy1-cy2),abs(cx1-cx2)))
cv2.putText(contornos_img2, str(round(tan)), (round(imgW/2),round(imgH/2 - 20)), font,1,(200,50,0),2,cv2.LINE_AA)

contornos_img = cv2.bitwise_or(contornos_img1, contornos_img2)

cv2.line(contornos_img, (cx1,cy1) ,( cx2,cy2), (255,0,0), 5)


fig = plt.figure(figsize=(20,20))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.subplot(1, 2, 2)
plt.imshow(contornos_img)
plt.show()