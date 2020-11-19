import cv2
import pygame
import math
from time import sleep
import numpy as np

def rescale(pic, scale):
    img = np.copy(pic)
    for row in range(0,img.shape[0]-scale+1,scale):
        for col in range(0, img.shape[1]-scale+1,scale):
            tot = 0
            for index in range(scale):
                tot += sum(img[row + index][col:col + scale])
            if tot//scale**2 > 125:
                val = 255
            else:
                val = 0
            for index in range(scale):
                img[row+index][col:col+scale] = [val]*scale
    return img

photo = cv2.imread("cw3a1Photo.png")
photo = cv2.resize(photo, (photo.shape[1]*2, photo.shape[0]*2))
photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
for row in range(0,photo.shape[0]):
    for col in range(0, photo.shape[1]):
        if photo[row][col] > 125:
            photo[row][col] = 255
        else:
            photo[row][col] = 0
for i in range(8):
    new_img = rescale(photo, 2**i)
    cv2.imshow("scale:"+str(i), new_img)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
