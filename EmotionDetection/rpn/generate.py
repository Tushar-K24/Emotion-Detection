import sys
sys.path.append('..') #system path is now the parent directory


import numpy as np
import config as C
from base_model import get_output_dims

def union(ex_roi, gt_roi, area_intersection):
    '''
    ex_roi,gt_roi: lists of ordering x1,y1,x2,y2
    (x1,y1)-> top left
    (x2,y2)-> bottom right
    '''
    area_ex = (ex_roi[2]-ex_roi[0])*(ex_roi[3]-ex_roi[1])
    area_gt = (gt_roi[2]-gt_roi[0])*(gt_roi[3]-gt_roi[0])
    return area_ex + area_gt - area_intersection

def intersection(ex_roi, gt_roi):
    x = max(ex_roi[0],gt_roi[0])
    y = max(ex_roi[1],gt_roi[1])
    w = min(ex_roi[2],gt_roi[2])-x
    h = min(ex_roi[3],gt_roi[3])-y
    if w<0 or h<0:
        return 0
    return w*h

def iou(ex_roi, gt_roi):
    if ex_roi[0]>ex_roi[2] or ex_roi[1]>ex_roi[3] or gt_roi[0]>gt_roi[2] or gt_roi[1]>gt_roi[3]:
        return 0
    area_i = intersection(ex_roi,gt_roi)
    area_u = union(ex_roi, gt_roi, area_i)
    return area_i/(area_u + 1e-6) #to avoid divisibility by 0

def generate_anchor_boxes(imdb, resized_width, resized_height, width, height):

    anchor_box_scales = C.anchor_box_scales
    anchor_box_ratios = C.anchor_box_ratios
    downscale = C.rpn_stride

    gt_bboxes = np.zeros((len(imdb['bboxes'],4)))

    out_width, out_height = get_output_dims(resized_width, resized_height)

    # calculating ground truth bounding boxes
    for idx, bbox in enumerate(imdb['bboxes']):
        gt_bboxes[idx,0] = bbox['x1']*(resized_width/width)
        gt_bboxes[idx,1] = bbox['y1']*(resized_height/height)
        gt_bboxes[idx,2] = bbox['x2']*(resized_width/width)
        gt_bboxes[idx,3] = bbox['y2']*(resized_height/height)
        
    for anchor_scale in anchor_box_scales:
        for anchor_ratio in anchor_box_ratios:
            anchor_x = anchor_scale*anchor_ratio[0]
            anchor_y = anchor_scale*anchor_ratio[1]

            for x in range(out_width):
                x1 = downscale*x - anchor_x/2
                x2 = downscale*x + anchor_x/2
                if x1<0 or x2>=resized_width: #points out of image
                    continue
                for y in range(out_height):
                    y1 = downscale*y - anchor_y/2
                    y2 = downscale*y + anchor_y/2
                    if y1<0 or y2>=resized_height: #points out of image
                        continue

                    #the anchor boxes are valid, so proceed accordingly    