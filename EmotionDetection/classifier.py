from keras.layers import Flatten, Dense, Dropout
import roi_pooling

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