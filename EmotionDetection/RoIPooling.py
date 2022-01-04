import math

import tensorflow as tf
from keras.layers import Layer, MaxPooling2D, Concatenate

class RoIPoolingLayer(Layer):
    '''
    Arguments
        pool_size: int
            Size of the pooled region to be generated. pool_size = 7 will generate a 7x7 region

        num_rois: int
            Number of RoIs to be used

    Input
        list of two tensors [X_img, X_roi] with shape:
            X_img: (1, rows, cols, channels)

            X_roi: (1, num_rois, 4) where 4 represents the ordering (x,y,w,h)

    Output
        Output
            a tensor with shape: (1, num_rois, pool_size, pool_size, channels)
    '''

    def __init__(self, pool_size, num_rois, **kwargs):
        
        self.pool_size = pool_size
        self.num_rois = num_rois

        super().__init__(**kwargs)

    def call(self, X):
        '''
        call() 
            Returns the pooled output from the feature map for every RoI

        Arguments
            X: list of two tensors [image, rois]

        Output
            fixed size pooled output in form of a tensor: (1, num_rois, pool_size, pool_size, channels)
        '''

        assert(len(X)==2)

        img = X[0]  #dims = (1, rows, cols, channels)
        rois = X[1] #dims = (1,num_rois,4)
        
        num_channels = img.shape()[3] #number of channels in input image

        assert(rois.shape[1] == self.num_rois)

        outputs = []

        for roi_idx in range(self.num_rois):
            x, y, w, h = rois[0, roi_idx]
            
            W = math.floor(w/self.pool_size)
            H = math.floor(h/self.pool_size)

            pool = MaxPooling2D(pool_size = (W,H))
            rs = pool(img[:,x:x+w,y:y+h,:])
            
            #storing the pooled image in outputs
            outputs.append(rs)

        #converting the outputs list to a tensor
        pooled_rois = Concatenate(axis=0)(outputs) 
        
        #reshaping the pooled_rois layer to (1, num_rois, pool_size, pool_size, channels)
        pooled_rois = tf.reshape(pooled_rois, (1,self.num_rois,self.pool_size, self.pool_size, num_channels))

        return pooled_rois

    def get_config(self):
        config = {'pool_size': self.pool_size, 
                  'num_rois': self.num_rois}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))