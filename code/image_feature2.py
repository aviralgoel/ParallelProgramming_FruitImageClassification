# -*- coding: utf-8 -*-
"""
@author: Nasser
"""
import cv2
import scipy.stats as stats
import numpy as np
from skimage.feature import greycomatrix,greycoprops
import pandas as pd

def get_image_cf(image):
    #image = cv2.imread(image_file) #Read the image file

    red = image[:,:,0:1]#get Red Channel
    blue = image[:,:,1:2]#get blue channel
    green = image[:,:,2:3]#get the green channel

    red = np.reshape(red,(len(red)*len(red[1,:]),1))#Reshape the red channel
    blue = np.reshape(blue,(len(blue)*len(blue[1,:]),1))#reshape the blue channel
    green = np.reshape(green,(len(green)*len(green[1,:]),1))#reshape the grren channel

    #get the mean and standard deviation shape (1,3)
    (u, std) = cv2.meanStdDev(image)
    u = u[:,0]
    std = std[:,0]

    #calculate the skew for each channel
    skwr = stats.skew(red,axis = 0)
    skwb = stats.skew(blue,axis = 0)
    skwg = stats.skew(green,axis = 0)

    ##calculate the kitosis for each channel
    kirtr = stats.kurtosis(red)
    kirtb = stats.kurtosis(blue)
    kirtg = stats.kurtosis(green)

    skw = [skwr[0],skwb[0],skwg[0]]
    kirt = [kirtr[0],kirtb[0],kirtg[0]]

    return u,std,skw,kirt

###########
def get_boundry_image(image_gray):
    #image = cv2.imread(image_file,0)#open the image file as gray

    #convert the image to black and while
    (thresh, bw_img) = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)

    #get the complement of the image 1 to 0 and 0 to 1
    bw_img = abs(255-bw_img)

    #perform the erosion
    kernel = np.ones((3,3),np.uint8)
    eros_img = cv2.erode(bw_img,kernel,iterations = 1)

    #subtract the eroded image from the black and white image
    sbtr_img = bw_img-eros_img

    #perform Dilation
    dilat_img = cv2.dilate(sbtr_img,kernel,iterations = 1)

    #Apply median filter to remove the noise
    med_img = cv2.medianBlur(dilat_img,5)

    return med_img

############
def get_image_mf(image_gray):
    image = get_boundry_image(image_gray)

    (u, std) = cv2.meanStdDev(image)
    u = u[0,0]
    std = std[0,0]
    image = np.reshape(image,len(image)*len(image[1,:]))

    skw = stats.skew(image)
    kirt = stats.kurtosis(image)

    return u,std,skw,kirt

###########
def get_image_tf(image_gray):
    #image = cv2.imread(image_file,0)

    glcm_img = greycomatrix(image_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=256)

    Ct = greycoprops(glcm_img,'contrast')
    Cn = greycoprops(glcm_img,'correlation')
    Ey = greycoprops(glcm_img,'energy')
    Hy = greycoprops(glcm_img,'homogeneity')

    Ct = Ct[0,:]
    Cn = Cn[0,:]
    Ey = Ey[0,:]
    Hy = Hy[0,:]

    return Ct,Cn,Ey,Hy


def get_image_dataframe(image_file, label):
    image = cv2.imread(image_file) #Read the image file RGB
    image_gray = cv2.imread(image_file,0)#open the image file as gray

    (u,std,skw,kirt) = get_image_cf(image)
    (u_mf,std_mf,skw_mf,kirt_mf) = get_image_mf(image_gray)
    (Ct,Cn,Ey,Hy) = get_image_tf(image_gray)

    features = [[label,u[0],u[1],u[2],std[0],std[1],std[2],skw[0],skw[1],skw[2],kirt[0],kirt[1],kirt[2],
                 u_mf,std_mf,skw_mf,kirt_mf,
                 Ct[0],Ct[1],Ct[2],Ct[3],Cn[0],Cn[1],Cn[2],Cn[3],
                 Ey[0],Ey[1],Ey[2],Ey[3],Hy[0],Hy[1],Hy[2],Hy[3]]]

    return features
