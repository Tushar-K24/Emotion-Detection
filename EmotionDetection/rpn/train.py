import sys
sys.path.append('..') 

import tensorflow as tf
from keras import backend as K

import config as C

def rpn_regr_loss(num_anchors, beta=1):
    '''
    Smooth L1 loss:
        0.5*x*x/beta, if abs(x)<beta
        abs(x) - 0.5*beta, else
    '''
    def loss_fn(y_true, y_pred):
        x = abs(y_true[:,:,:,4*num_anchors:] - y_pred)
        x_bool = K.cast(x<beta, tf.float32)
        n_reg = K.sum(1e-4 + y_true[:,:,:,:4*num_anchors])
        return C.lambda_rpn_regr*K.sum(y_true[:,:,:,:4*num_anchors]*(x_bool*0.5*x*x/beta + (1-x_bool)*(x-0.5*beta)))/n_reg

    return loss_fn

def rpn_cls_loss(num_anchors):
    '''
    Log loss(binary cross-entropy)

    y_true[:,:,:,:num_anchors] = y_is_box_valid
    y_true[:,:,:,num_anchors:] = y_rpn_overlap 
    '''
    def loss_fn(y_true, y_pred):
        n_cls = K.sum(1e-4 + y_true[:,:,:,:num_anchors])
        return C.lambda_rpn_cls*K.sum(y_true[:,:,:,:num_anchors]*K.binary_crossentropy(y_true[:,:,:,num_anchors:],y_pred))/n_cls
    return loss_fn