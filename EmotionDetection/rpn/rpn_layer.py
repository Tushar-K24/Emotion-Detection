import tensorflow as tf
from keras.layers import Layer,Conv2D

class RPN(Layer):
    '''
    RPN()

    Arguments
        fmap: tensor
            feature map obtained from base_model.py

    Methods
        rpn_layer() 
            Transforms the input tensor to a feature map using a fully CNN architecture

        get_config()
            Returns the current configuration of the base layer in the form of dictionary
    '''
    def __init__(self, fmap, **kwargs):
        self.fmap = fmap
        super().__init__(**kwargs)

    def rpn_layer(self, num_anchors=9):
        '''
            rpn_layer()
                Generates a Region Proposal Network with classification and regression head

            Arguments
                num_anchors:
                    number of anchor boxes
                    Default = 9
            
            Output: [class_layer, reg_layer]
                A list with 2 tensors:
                    Classification Head: tensor of depth num_anchors for computing the
                    probabilities of the object being present in the anchor boxes.

                    Regression Head: tensor of depth 4*num_anchors for computing the 
                    bounding boxes.
        '''
        x = Conv2D(512, (3,3), padding='same', activation='relu', name='rpn_conv')(self.fmap)
        
        class_layer = Conv2D(num_anchors, (1,1), activation='sigmoid', name='rpn_class_layer')(x) #classification layer
        reg_layer = Conv2D(4*num_anchors, (1,1), activation='sigmoid', name='rpn_reg_layer')(x) #bbox regression layer

        return [class_layer, reg_layer]