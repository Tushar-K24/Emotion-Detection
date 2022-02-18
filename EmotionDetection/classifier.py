from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
from keras.objectives import categorical_crossentropy
import tensorflow as tf

import config as C

import roi_pooling

def clf_regr_loss(num_classes,beta = 1):
    '''
    Smooth L1 loss:
        0.5*x*x/beta, if abs(x)<beta
        abs(x) - 0.5*beta, else
    '''
    def loss_fn(y_true, y_pred):
        x = abs(y_true[:,:,4*num_classes:] - y_pred)
        x_bool = K.cast(x<beta, tf.float32)
        n_reg = K.sum(1e-4 + y_true[:,:,:4*num_classes])
        return C.lambda_clf_regr*K.sum(y_true[:,:,:4*num_classes]*(x_bool*0.5*x*x/beta + (1-x_bool)*(x-0.5*beta)))/n_reg

    return loss_fn

def clf_cls_loss(y_true, y_pred):
    return C.lambda_clf_cls*K.mean(categorical_crossentropy(y_true,y_pred))
    

def classifier_layer(fmap, input_rois, num_rois, num_classes=4, pool_size=7):
    '''
    classifier_layer()
        Returns the output classifier with both classification head and regression head

        Note: If needed, change the architecture of the classifier in classifier.py

    Arguments
        fmap: 
            an input tensor to pool the features

        input_rois: 
            a 3D input tensor of shape (1,num_rois,4) where 4 is for (x,y,w,h)

        num_rois: 
            number of rois for processing

        num_classes:
            number of classes to output(including background class)
            Default: 4

        pool_size:
            pre-defined pool size for fixed size feature vectors
            Default: 7

    Output
        a list of two Layers [out_class, out_reg]
        
        out_class: Classification head with num_classes classes as output
        out_reg: Regression head for region refinement
    '''
    pooled_rois = roi_pooling(pool_size, num_rois)([fmap, input_rois])

    out = Flatten(name='Flatten')(pooled_rois)
    out = Dense(4096, activation='relu', name='fc1')(out)
    out = Dropout(0.5)(out)
    out = Dense(4096, activation='relu', name='fc2')(out)
    out = Dropout(0.5)(out)

    #Classification Head 
    out_class = Dense(num_classes, activation='softmax', name='class_head')(out)

    #Regression Head
    out_reg = Dense(4*(num_classes-1), activation='linear', name='regression_head')(out)

    return [out_class, out_reg]
