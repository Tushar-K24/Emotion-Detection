import random
import copy

import numpy as np
import cv2

import config as C

def augment(img, imdb, augment=True):
    
    aug_img = copy.deepcopy(img)
    augmented_imdb = copy.deepcopy(imdb) #make changes to copy so as to not affect the original dataframe
    
    if not augment: #if augment is false
        return augmented_imdb, aug_img
    
    height, width = aug_img.shape[:2] #current width and height of image
    
    if C.horizontal_flip and random.randint(0,1): #flipping horizontally
        aug_img = cv2.flip(aug_img, 1) #flips along y-axis
        augmented_imdb['x1'] = width - imdb['x2']
        augmented_imdb['x2'] = width - imdb['x1'] 

    if C.vertical_flip and random.randint(0,1): #flipping vertically
        aug_img = cv2.flip(aug_img,0) #flips along x-axis
        augmented_imdb['y1'] = height - imdb['y2']
        augmented_imdb['y2'] = height - imdb['y1']

    if C.rotate:
        angle = random.choice([0,90,180,270]) #choosing random angle to rotate image
        x1 = copy.deepcopy(augmented_imdb['x1'])
        x2 = copy.deepcopy(augmented_imdb['x2'])
        y1 = copy.deepcopy(augmented_imdb['y1'])
        y2 = copy.deepcopy(augmented_imdb['y2'])
        #rotate img
        if angle == 270:
            aug_img = np.transpose(aug_img,(1,0,2))
            aug_img = cv2.flip(aug_img,0)
            augmented_imdb['x1'] = y1
            augmented_imdb['x2'] = y2
            augmented_imdb['y1'] = width - x2
            augmented_imdb['y2'] = width - x1    
        
        elif angle == 180:
            aug_img = cv2.flip(aug_img,-1)
            augmented_imdb['x1'] = width - x2
            augmented_imdb['x2'] = width - x1
            augmented_imdb['y1'] = height - y2
            augmented_imdb['y2'] = height - y1

        elif angle == 90:
            aug_img = np.transpose(aug_img,(1,0,2))
            aug_img = cv2.flip(aug_img,1)
            augmented_imdb['x1'] = height - y2
            augmented_imdb['x2'] = height - y1
            augmented_imdb['y1'] = x1
            augmented_imdb['y2'] = x2
        else:
            pass

    return augmented_imdb, aug_img