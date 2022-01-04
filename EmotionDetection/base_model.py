import tensorflow as tf

from keras import Model
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D

class Base(Model):
    '''
    Arguments
        input_tensor:
            Input Image to be feeded into the CNN of shape (x,y,num_channels)
            Default = None
        
        num_channels:
            Number of channels in the input tensor (value passed only if input_tensor = None)
            Default = 3 (for an RGB image)
    
    Input
        trainable: 
            True if the model is trainable

    Output
        Base Model
    '''

    def __init__(self, input_tensor=None, num_channels=3, **kwargs):
        self.input_tensor = input_tensor
        self.num_channels = num_channels
        if input_tensor: #if input_tensor is given
            self.num_channels = input_tensor.shape()[2]
        super().__init__(**kwargs)
    
    def get_model(self, trainable=False):
        '''
        get_model() 
            Returns a CNN model with only convolutional layers (No dense layers)

            By Default, VGG16 is selected as the base model. If needed, change the architecture 
            of the base model here.

        Arguments
            trainable: bool
                True: Model is trainable
                False: Model is non-trainable

        Output: Base model
        '''

        input_shape = (None, None, self.num_channels)
        if self.input_tensor:
            input_shape = self.input_tensor.shape()

        if not K.is_keras_tensor(input_tensor):
            input_img = Input(tensor = input_tensor, shape = input_shape)
        else:
            input_img = input_tensor
        
        #Block 1
        vgg = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(input_img)
        vgg = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(vgg)
        vgg = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(vgg)

        #Block 2
        vgg = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(vgg)
        vgg = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(vgg)
        vgg = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(vgg)

        #Block 3
        vgg = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(vgg)
        vgg = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(vgg)
        vgg = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(vgg)
        vgg = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(vgg)

        #Block 4
        vgg = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(vgg)
        vgg = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(vgg)
        vgg = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(vgg)
        vgg = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(vgg)

        #Block 5
        vgg = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1')(vgg)
        vgg = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2')(vgg)
        vgg = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3')(vgg)
        #We won't consider the last pooling layer
        #vgg = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(vgg) 

        if not trainable:
            for layer in vgg:
                layer.trainable = False
        
        return vgg