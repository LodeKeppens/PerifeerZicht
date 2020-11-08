import cv2
import pygame
import math
from time import sleep
import numpy as np
from sympy import symbols, solve

input = cv2.imread("test123.jpg")
input = cv2.resize(input, (960, 540))
amount_of_colours = 4
div = 256 / amount_of_colours
result = div
quantized = input // div * div + div // 2
#Bron: https://stackoverflow.com/questions/5906693/how-to-reduce-the-number-of-colors-in-an-image-with-opencv

nb_red = []
nb_green = []
nb_blue = []
for i in range(len(quantized)):
    for j in range(len(quantized[0])):
        for k in range(3):
            element = quantized[i][j][k]
            if k == 0 and element not in nb_red:
                nb_red.append(element)
            elif k == 1 and element not in nb_green:
                nb_green.append(element)
            elif k == 2 and element not in nb_blue:
                nb_blue.append(element)

print("red: ", len(nb_red))
print("green: ", len(nb_green))
print("blue: ", len(nb_blue))


cv2.imwrite('testfoto4.jpg', quantized)
# cv2.imshow("output", img)
# cv2.waitKey(0)

img = cv2.imread('testfoto4.jpg')
cv2.imshow("result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


