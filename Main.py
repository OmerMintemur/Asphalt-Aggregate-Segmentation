# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:24:46 2024

@author: OMER
"""
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from skimage.filters import try_all_threshold,threshold_otsu,threshold_sauvola
import random as rng
import math
# from skimage.segmentation import random_walker,flood,flood_fill
# from skimage.morphology import disk, binary_dilation
from scipy.ndimage import binary_fill_holes
from scipy import ndimage
# https://pyimagesearch.com/2016/04/04/measuring-distance-between-objects-in-an-image-with-opencv/
# https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
# https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/20_image_segmentation/08_binary_mask_refinement.html
# https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_sharpen.html
# https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_sharpen.html
# https://stackoverflow.com/questions/60894593/how-to-sharpen-the-edges-in-opencv-python

def distance_between_two_points(points):
    average = 0
    
    for p in range(len(points)):
        temp = 0
        for c in range(p+1,len(points)):
            # Calculate distance with point c to all points
            temp+= math.hypot(points[p][0] - points[c][0], points[p][1] - points[c][1])
    
        average+=temp/len(points)
    return average

def imageshow(image,name, save=True):
    
    fig = plt.figure(1,figsize=(12,12))
    plt.imshow(image,cmap="gray")
    plt.axis("off")
    if save:
        cv2.imwrite(f"{name}.png",image)
    
def find_zero_pixels(image):
    mask = np.ones((image.shape[0],image.shape[1]),dtype=int)
    for x in range(0,image.shape[0]):
        for y in range(0,image.shape[1]):
            if image[x,y] == 0:
                mask[x,y]=0
    return mask


def check_neighbours(image,size=3):
    mask = np.zeros((image.shape[0],image.shape[1]),dtype=int)
    
    for x in range(0,image.shape[0]):
        for y in range(0,image.shape[1]):
            temp = image[x:x+size,y:y+size]
            temp = np.sum(temp)/255
            if temp > 0.8:
                mask[x,y] = 255
    return mask

image_,image = cv2.imread("Example.tif",0), cv2.imread("Example.tif")
cv2.imwrite("Results\\First.jpg", image_)
mask = find_zero_pixels(image_)


h_s_v_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(h_s_v_image)


temp = cv2.equalizeHist(s-v)
temp = cv2.threshold(temp,180,255,cv2.THRESH_BINARY)
temp = cv2.equalizeHist(temp[1])
imageshow(temp,"Results\\Second")


kernel = np.ones((7,7),np.float32)/49
dst = cv2.filter2D(temp,-1,kernel)
imageshow(dst,"Results\\Third")



ret, binary = cv2.threshold(dst,0,255,cv2.THRESH_OTSU)
binary = binary.astype(np.uint8)
imageshow(binary,"Results\\Fourth")


kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17,17))
temp = cv2.dilate(binary, kernel, iterations = 1)


kernel = np.ones((7,7),np.float32)/49
dst = cv2.filter2D(~temp,-1,kernel)

dst = np.multiply(mask,dst)
imageshow(dst,"Results\\Fifth")

kernel = np.ones((5,5),np.float32)/25
dst = binary_fill_holes(dst, structure=kernel).astype(np.uint8)
imageshow(dst,"Results\\Sixth") # Problem with this one TODO: Solve



threshold = 750


contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

count = 0
bigger = [count+1 for c in contours if cv2.contourArea(c) > threshold]
centers = []
for c in contours:
    if cv2.contourArea(c) > threshold:
        cv2.drawContours(image, [c], 0, (0, 255, 255), 3)
                
        x, y, w, h = cv2.boundingRect(c)
        center = [x+w/2, y+h/2]
        centers.append((x+w/2, y+h/2))
        
imageshow(image,"Results\\Seventh")







