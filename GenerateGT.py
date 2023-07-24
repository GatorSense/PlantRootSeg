#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:57:06 2018

@author: weihuangxu
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pdb
import cv2

# find the x range and y range of two given points
def linearFunction(P1, P2):
    
    jud = P2[0]-P1[0]
    if jud == 0:
        yRange = np.arange(min(P1[1], P2[1]),max(P1[1], P2[1])+1)
        xRange = P2[0]*np.ones((len(yRange)))
    else:
        k = (P2[1]-P1[1])/jud #slope
        xRange = np.arange(min(P1[0], P2[0]), max(P1[0], P2[0])+1)
        yRange = np.round(k*(xRange-P1[0]))+P1[1]
    
    xyRange = np.stack((xRange, yRange), axis=-1).astype(int)
    return xyRange

# find the area indexes for rectangle shape
def Rectangle(vertex):
    
    vertex = vertex.astype(int)
    A = vertex[0:2] 
    C = vertex[2:4] 
    B = vertex[4:6] 
    D = vertex[6:8] 
    
    ABRange = linearFunction(A,B)
    CDRange = linearFunction(C,D)
    CBRange = linearFunction(C,B)
    ADRange = linearFunction(A,D)
    
    allRange = np.vstack((ABRange, CDRange, CBRange, ADRange))
    xmin = min(allRange[:,0])
    xmax = max(allRange[:,0])
    
    allx = []
    ally = []
    
    for i in range(int(xmin), int(xmax+1)):

        yRange = allRange[np.where(allRange[:,0] == i), 1].squeeze()
        ymin = min(yRange)
        ymax = max(yRange)
        y = np.arange(ymin, ymax+1)
        x = i * np.ones(np.shape(y)).astype(int)
        allx += x.tolist()
        ally += y.tolist()
    
    #area = np.hstack(allx, ally)
    
    return allx, ally

def del_empty(path, print_result=False):
    """
    Delete the empty label files

    Parameters
    ----------
    path : str, the directory of all the data
    print_result : Boolean. print the progress or not

    """
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    
    count = 0
    totLen = len(listOfFiles)
    # import pdb; pdb.set_trace()
    for item in listOfFiles:
        label = np.loadtxt(item, dtype='str')
        Rlocation = np.core.defchararray.find(label, 'R')
        # find the idx of string starting with R
        Ridx = np.where(Rlocation == 0)
        if len(Ridx[0]) <= 0:
            os.remove(item)
            count += 1
            if print_result:
                print('%d/%d deleted: %s.' %(count, totLen, item.split('/')[-1]))
            

def generate_mask(label, img_size, black_root=False):
    '''
    Generate binary mask for Minirhizotron images using .pat label file.

    Parameters
    ----------
    label : np.array. array of string 

    img_size : tuple, the image size.

    Returns: the binary mask of root of 1 and background of 0
    '''

    h, w, c = img_size
    GT = np.zeros((h,w), dtype=(np.uint8))
    # find the string start with 'R', -1 is the string not start with R and 0 is the string start with R
    Rlocation = np.core.defchararray.find(label, 'R')
    # find the idx of string starting with R
    Ridx = np.where(Rlocation == 0)
    
    numRec = np.shape(Ridx)[1]
    vertex = np.zeros((numRec, 8))
    
    xidx = []
    yidx = []
    for i in range(numRec):
        idx = Ridx[0][i] #index of 'R1,2..."
        vertex[i,:] = label[idx+9:idx+17]
        tempx, tempy = Rectangle(vertex[i])
        xidx += tempx
        yidx += tempy

    yidx = np.asarray(yidx) 
    xidx = np.asarray(xidx)
    yidx[np.where(yidx >= h)] = h - 1
    xidx[np.where(xidx >= w)] = w - 1
    allindex = (yidx, xidx)
    
    GT[allindex] = 255
    if black_root:
        GT = 255 - GT
    return GT

            
            

    
