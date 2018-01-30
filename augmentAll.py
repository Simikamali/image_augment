"""
Created on Mon Jan 29 16:03:10 2018

@author: justin
"""

import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage import data, io, transform
from scipy import misc
import os

rootPath = os.getcwd()
augmentPath = os.path.join(rootPath, 'augmentedImages')
try:
    os.mkdir(augmentPath)
except:
    input('Augmented Images Folder Exists Ctrl-C While You Can')

def convert_yolo_2_tf(label):
    # Takes in text file of yolo coords and
    # converts [center x, center y, width height] in %
    # to  [y_min, x_min, y_max, x_max] in %
    # Since this will only be used for rotating the 
    # existing data I'm not super worried about maintaining
    # the order of class to coordinates, created boxesDict
    # in case this becomes an issue in the future
    text = open(label, 'r')
    oldBoxes = text.read().split('\n')[:-1]
    text.close()
    oldBoxes = [i.split() for i in oldBoxes]
    
    # Returns a list for each object with [0] index 
    # being the class and [1:5] the coordinates
    for i in oldBoxes:
        for ii in range(1,5):
            i[ii] = float(i[ii])

    # Create New Blank Boxes in format:   
    # [batch, number of bounding boxes, coords]
    numBoxes = len(oldBoxes)
    classes = [None]*numBoxes
    boxes = np.zeros([1,numBoxes,4])
    boxesDict = {}

    # Fill in new boxes
    for i in range(numBoxes):
        boxes[:,i,0] = (oldBoxes[i][2]-oldBoxes[i][4]/2)
        boxes[:,i,1] = (oldBoxes[i][1]-oldBoxes[i][3]/2)
        boxes[:,i,2] = (oldBoxes[i][2]+oldBoxes[i][4]/2)
        boxes[:,i,3] = (oldBoxes[i][1]+oldBoxes[i][3]/2)
        boxesDict[oldBoxes[i][0]] = boxes[:,i,:]
        classes[i] = int(oldBoxes[i][0])

    return numBoxes, boxes, oldBoxes, classes, boxesDict

def convert_tf_2_yolo(classes,labels,filename):
    # Takes in a list of classes and coordinates (assuming
    # that they are in the same order) and converts the array
    # [batch, number of bounding boxes, coords] for all 
    # from: [y_min, x_min, y_max, x_max] in % 
    # to: [center x, center y, width height] in %
    os.chdir(augmentPath)
    labels = open(filename, 'w')
    
    
    
    
num, boxes, oldBoxes, classes, boxesDict = convert_yolo_2_tf('labels.txt')