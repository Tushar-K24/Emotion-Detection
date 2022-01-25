import math

'''
Default configurations are stored here
(to be used in dependencies)
'''
#default base model
base_model = 'vgg'

#anchor boxes
anchor_box_scales = [128, 256, 512] #anchor box scales
anchor_box_ratios = [[1,1], [1,math.sqrt(2)], [math.sqrt(2),1]]

#downscaling
rpn_stride = 16

#rpn overlaps
rpn_max_overlap = 0.7
rpn_min_overlap = 0.3

#total no. of anchor boxes to process in the mini batch
num_samples = 256