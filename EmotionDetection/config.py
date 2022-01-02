import numpy as np

class Config:

    def __init__(self):

        #Whether to print the process or not
        self.verbose = True

        #base model
        self.base_model = 'vgg'

        #Anchor box scales
        self.anchor_box_scales = [128,256,512]

        #Anchor box ratios
        self.anchor_box_ratios = [[1,1],[1./np.sqrt(2),1],[1,1./np.sqrt(2)]]

        #image size (size of the smaller side of the image)
        self.im_size = 600

        #no. of RoI