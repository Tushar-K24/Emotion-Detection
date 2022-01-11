from keras.layers import Layer, Flatten, Dense, Dropout
import roi_pooling

class Classifier(Layer):
    '''
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, fmap, input_rois, num_rois, num_classes=4, pool_size=7):
        
        input_shape = (num_rois, pool_size, pool_size, fmap.get_shape()[3])
        
        pooled_rois = roi_pooling(pool_size, num_rois)([fmap, input_rois])

        out = Flatten(name='Flatten')(pooled_rois)
        out = Dense(4096, activation='relu', name='fc1')(out)
        out = Dropout(0.5)(out)
        out = Dense(4096, activation='relu', name='fc2')(out)
        out = Dropout(0.5)(out)

        #Classification Head 
        out_class = Dense(num_classes + 1, activation='softmax', name='class_head')(out)

        #Regression Head
        out_reg = Dense(4*num_classes, activation='linear', name='regression_head')(out)

        return [out_class, out_reg]