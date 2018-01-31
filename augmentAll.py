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
import shutil

class Augment:
    def __init__(self):
        self.rootPath = os.getcwd()
        self.augmentPath = os.path.join(self.rootPath, 'augmentedImages')
        self.sanityPath = os.path.join(self.augmentPath, 'sanityCheck')
        self.dataPath = os.path.join(self.rootPath, 'data')
        self.create_directory()
        self.session = tf.Session()
                
    def create_directory(self):
        try:
            os.mkdir(self.augmentPath)
        except:
            ask = input('Augmented Img. Folder Found: Delete? [Y/N] ').lower()
            if ask == 'y':
                shutil.rmtree(self.augmentPath)
                os.mkdir(self.augmentPath)
    
    def convert_yolo_2_tf(self,label):
        # Takes in text file of yolo coords and converts from: 
        # [center x, center y, width height] in perentages to:
        # [y_min, x_min, y_max, x_max] in percentages
        # Since I will only be rotating an image if 100% of the classes are in
        # the current screen I don't worry about removing bounding boxes from
        # the list or losing label order, added boxesDict just in case.
        text = open(label, 'r')
        oldBoxes = text.read().split('\n')[:-1]
        text.close()
        oldBoxes = [i.split() for i in oldBoxes]
        
        # Converts old boxes labels from strings to floats
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
            boxes[:,i,0] = oldBoxes[i][2]-oldBoxes[i][4]/2
            boxes[:,i,1] = oldBoxes[i][1]-oldBoxes[i][3]/2
            boxes[:,i,2] = oldBoxes[i][2]+oldBoxes[i][4]/2
            boxes[:,i,3] = oldBoxes[i][1]+oldBoxes[i][3]/2
            boxesDict[oldBoxes[i][0]] = boxes[:,i,:]
            classes[i] = int(oldBoxes[i][0])
    
        return numBoxes, boxes, oldBoxes, classes, boxesDict
    
    def convert_tf_2_yolo(self, classes, oldBoxes, filename):
        # Takes in a list of classes and coordinates (assuming
        # that they are in the same order) and converts the array
        # [batch, number of bounding boxes, coords] for all 
        # from: [y_min, x_min, y_max, x_max] in % 
        # to: [center x, center y, width height] in %
        os.chdir(self.augmentPath)
        output = open(filename, 'w')
        newBoxes = np.zeros([len(classes), 4])
        
        for i in range(len(classes)):
            newBoxes[i][0] = (oldBoxes[:,i,1] + oldBoxes[:,i,3])/2
            newBoxes[i][1] = (oldBoxes[:,i,0] + oldBoxes[:,i,2])/2
            newBoxes[i][2] = (oldBoxes[:,i,3] - oldBoxes[:,i,1])
            newBoxes[i][3] = (oldBoxes[:,i,2] - oldBoxes[:,i,0])
            output.write('{} {:.7f} {:.7f} {:.7f} {:.7f} \n'
                         .format(classes[i], 
                         newBoxes[i][0], newBoxes[i][1], 
                         newBoxes[i][2], newBoxes[i][3])) 
        output.close()
        os.chdir(self.rootPath)
        
    #TODO(JF): TF TO CSV (PIXEL VALUES & IMG SIZE), 
    def convert_tf_2_csv(self):
        pass
    
    def randomCrop(self, image, boxes, sanity):
        # Takes in image file name, boxes in a list format and performs rand.
        # crop exporting a new image with set of labels into augmentPath
        # if sanity is True will export image with drawn boxes in sanity folder
        min_object_perc = 0.06
        
        os.chdir(self.dataPath)
        img = mpimg.imread(image)
        shape = img.shape
        tf_img = tf.convert_to_tensor(np.expand_dims(img,0), np.float32)
        box = tf.convert_to_tensor(boxes, np.float32)
        test = tf.image.draw_bounding_boxes(tf_img, box)
        
        
        
            
        
aug = Augment()        
n, boxes, oldBoxes, classes, _ = aug.convert_yolo_2_tf('labels.txt')
aug.convert_tf_2_yolo(classes, boxes, 'test2.txt')