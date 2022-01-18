from keras import backend as K
from keras.layers import Layer, Input, Conv2D, MaxPooling2D

class Base(Layer):
    '''
    Base()

    Arguments
        num_channels: int
            number of channels in the input tensor
            Default = 3

    Methods
        get_model() 
            Transforms the input tensor to a feature map using a fully CNN architecture

        get_config()
            Returns the current configuration of the base layer in the form of dictionary
    '''

    def __init__(self, num_channels=3, **kwargs):
        self.num_channels=num_channels  
        self.base_model = 'VGG16' #default CNN architecture used
        super().__init__(**kwargs)

    def get_output_dims(self, width, height):
        '''
        get_output_dims()
            Returns the width and height of the fmap for the given input width and height

        Arguments
            width: 
                Width of the input image
            height:
                Height of the input image

        Output: 
            Width of the feature map
            Height of the feature map
        '''
        rescale = lambda x: x//16 #16 because the VGG architecture used has 4 max pooling layers
        return rescale(width), rescale(height)

    def get_fmap(self, input_tensor=None, trainable=False):
        '''
        get_model() 
            Transforms the input tensor to a feature map using a fully CNN architecture

            By Default, VGG16 is selected as the base model. If needed, change the architecture 
            of the base model in base_mode.py.

        Arguments
            input_tensor:
                Input Image to be feeded into the CNN of shape (x,y,num_channels)
                Default = None

            trainable: bool
                True: Model is trainable
                False: Model is non-trainable

        Output: Feature Map
        '''
        
        #if input_tensor is None
        if not input_tensor:
            input_shape = (None, None, self.num_channels)
            input_tensor = Input(shape = input_shape)
        
        if not K.is_keras_tensor(input_tensor): #Converting input_tensor to appropriate dtype
            input_img = Input(tensor = input_tensor)
        else:
            input_img = input_tensor

        self.img_shape = input_img.get_shape() #input_shape

        #Block 1
        fmap = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(input_img)
        fmap = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(fmap)
        fmap = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(fmap)

        #Block 2
        fmap = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(fmap)
        fmap = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(fmap)
        fmap = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(fmap)

        #Block 3
        fmap = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(fmap)
        fmap = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(fmap)
        fmap = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(fmap)
        fmap = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(fmap)

        #Block 4
        fmap = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(fmap)
        fmap = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(fmap)
        fmap = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(fmap)
        fmap = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(fmap)

        #Block 5
        fmap = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1')(fmap)
        fmap = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2')(fmap)
        fmap = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3')(fmap)
        #We won't consider the last pooling layer
        #fmap = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(fmap) 
        
        self.fmap = fmap
        self.fmap_shape = fmap.get_shape() #output shape
        
        return fmap

    def get_config(self):
        config = {'base': self.base_model, 
                  'channels': self.num_channels,
                  'input_shape': self.img_shape,
                  'fmap_shape': self.fmap_shape}

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

