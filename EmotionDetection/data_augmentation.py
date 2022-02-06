import random
import copy

import numpy as np
import cv2

import config as C

def augment(imdb, augment=True):
    assert 'filepath' in imdb
    assert 'bboxes' in imdb
 
    augmented_imdb = copy.deepcopy(imdb) #make changes to copy so as to not affect the original dataframe
    img = cv2.imread(imdb['filepath']) 

    if not augment: #if augment is false
        return augmented_imdb
    
    height, width = img.shape[:2] #current width and height of image
    
    if C.horizontal_flip and random.randint(0,1): #flipping horizontally
        img = cv2.flip(img, 1) #flips along y-axis
        imdb['x1'] = width - imdb['x1']
        imdb['x2'] = width - imdb['x2']
        imdb['x1'], imdb['x2'] = imdb['x2'], imdb['x1'] 

    if C.vertical_flip and random.randint(0,1): #flipping vertically
        img = cv2.flip(img,0) #flips along x-axis
        imdb['y1'] = height - imdb['y1']
        imdb['y2'] = height - imdb['y2']
        imdb['y1'], imdb['y2'] = imdb['y2'], imdb['y1']

    if C.rotate:
        angle = random.choice([0,90,180,270])
        
        #rotate img
        if angle == 270:
            img = np.transpose(img,(1,0,2))
            img = cv2.flip(img,0)
        
        elif angle == 180:
            img = cv2.flip(img,-1)
        
        elif angle == 90:
            img = np.transpose(img,(1,0,2))
            img = cv2.flip(img,1)

        else:
            pass

        #rotate bounding boxes