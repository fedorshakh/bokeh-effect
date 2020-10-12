# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:23:14 2020

@author: User
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

def boke(image, kernel):
    n1, n2 = image.shape
    m1, m2 = kernel.shape
    layer = np.zeros((n1 + m1*2 - 2, n2 + m2*2 - 2))
    layer[m1-1:n1+m1-1,m2-1:n2+m2-1] = image
    layer[:m1,:m2] = layer[m1-1,m2-1]  # upper left
    layer[:m1,n2+m2-2:] = layer[m1-1,n2+m2-2]  # upper right
    layer[n1+m1-2:,n2+m2-2:] = layer[n1+m1-2,n2+m2-2]  # lower right
    layer[n1+m1-2:,:m2] = layer[n1+m1-2,m2-1]  # lower left
    
    cell = layer[:m2,m1:n2+m2-2]  # upper
    for i in range(len(cell)):
        cell[i] = cell[-1]
        
    cell = layer[m1:n2+m2-2,n1+m1-2:]  # right
    for i in range(len(cell)):
        cell[i] = cell[i][0]
        
    cell = layer[n1+m1-2:,m1:n2+m2-2]  # lower
    for i in range(len(cell)):
        cell[i] = cell[0][i]
        
    cell = layer[m1:n2+m2-2,:m2]  # left
    for i in range(len(cell)):
        cell[i] = cell[i][-1]
        
    for i in range(n1+m1-1):
        for j in range(n2+m2-1):
            layer[i,j] = sum(sum(layer[i:i+m1,j:j+m2]*kernel))/sum(sum(kernel))
    ret = layer[m1//2:n1+m1//2,m2//2:n2+m2//2]
    return ret

photo = cv2.imread('./images_2/lighthouse.jpg') #your path to the image
photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
ph = np.array(photo)

filter_ = cv2.imread('./images_2/hexagon_9.jpg',cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(filter_, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = 127
im_bw = cv2.threshold(filter_, thresh, 255, cv2.THRESH_BINARY)[1]/255

res = np.zeros_like(ph)

for i in range(3):
    channel = ph[:,:,i]
    rgb = boke(channel,im_bw)
    rgb = rgb.ravel()
    res = res.reshape(res.shape[0]*res.shape[1], 3)
    for j in range(rgb.shape[0]):
        res[j,i] = rgb[j]
        
    res = res.reshape(ph.shape)

res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
cv2.imwrite('result.jpg', res)
