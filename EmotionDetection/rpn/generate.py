import sys
import random
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

    num_bboxes = len(imdb['bboxes'])
    num_anchors = len(anchor_box_ratios)*len(anchor_box_scales)

    gt_bboxes = np.zeros((num_bboxes,4))
    out_width, out_height = get_output_dims(resized_width, resized_height)

    best_iou_for_bbox = np.zeros(num_bboxes)
    best_anchor_for_bbox = np.zeros((num_bboxes,4))
    best_dx_for_bbox = np.zeros((num_bboxes,4)) # tx, ty, tw, th (for loss)

    y_rpn_overlap = np.zeros((out_width, out_height, num_anchors)) #represents if that anchor overlaps with gt_bbox
    y_is_box_valid = np.zeros((out_width, out_height, num_anchors)) #represents if that anchor has an object 
    y_rpn_regr = np.zeros((out_width, out_height, 4*num_anchors)) #tx,ty,tw,th for every positive class

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
                    
                    anchor_box = [x1,y1,x2,y2] 
                    
                    bbox_type = 'neg' #initialize with negative
                    
                    best_iou_for_anchor = 0 #for setting y_rpn_regr

                    for bbox_num in range(num_bboxes): #iterate for every gt_bbox

                        curr_iou = iou(anchor_box, gt_bboxes[bbox_num]) #iou of anchor_box and current gt
                        
                        if curr_iou > C.rpn_max_overlap or curr_iou>best_iou_for_bbox[bbox_num]:
                            #center of gt bbox
                            cx = (gt_bboxes[bbox_num,0] + gt_bboxes[bbox_num,2])/2
                            cy = (gt_bboxes[bbox_num,1] + gt_bboxes[bbox_num,3])/2
                            #center of anchor box
                            cxa = (x1 + x2)/2
                            cya = (y1 + y2)/2

                            #calculating dx (from the original paper)
                            tx = (cx - cxa)/(x2 - x1)
                            ty = (cy - cya)/(y2 - y1)
                            tw = np.log((gt_bboxes[bbox_num,2]-gt_bboxes[bbox_num,0])/(x2-x1))
                            th = np.log((gt_bboxes[bbox_num,3]-gt_bboxes[bbox_num,1])/(y2-y1))
                        
                        if imdb['class'][bbox_num]!='bg':
                            #mapping every gt_bbox with an anchor box to see which one's the best
                            if curr_iou>best_iou_for_bbox[bbox_num]:
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_anchor_for_bbox[bbox_num,:] = anchor_box
                                best_dx_for_bbox[bbox_num,:] = [tx,ty,tw,th]
                                
                                if curr_iou>best_iou_for_anchor:
                                    best_iou_for_anchor = curr_iou
                                    best_rpn_regr = [tx,ty,tw,th]

                                if curr_iou>C.rpn_max_overlap:
                                    bbox_type = 'pos'
                                elif C.rpn_min_overlap<curr_iou<C.rpn_max_overlap and bbox_type == 'neg':
                                    bbox_type = 'neutral'
                    
                    idx = len(anchor_box_ratios)*anchor_scale + anchor_ratio

                    if bbox_type == 'neg':
                        y_is_box_valid[x,y,idx] = 1
                        y_rpn_overlap[x,y,idx] = 0

                    elif bbox_type == 'neutral':
                        y_is_box_valid[x,y,idx] = 0
                        y_rpn_overlap[x,y,idx] = 0
                    
                    else:
                        y_is_box_valid[x,y,idx] = 1
                        y_rpn_overlap[x,y,idx] = 1
                        y_rpn_regr[x,y,4*idx:4*idx+4] = best_rpn_regr
                    
    #converting the numpy arrays from 3D to 4D
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0) #dims: (1,out_width, out_height, num_anchors)
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0) #dims: (1,out_width, out_height, num_anchors)
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0) #dims: (1,out_width, out_height, 4*num_anchors)

    #total count of samples should not exceed 256, 128 for both ideally
    positive_anchors = np.where((y_is_box_valid==1)&(y_rpn_overlap==1)) #stores the indices of positive anchors
    negative_anchors = np.where((y_is_box_valid==1)&(y_rpn_overlap==0)) #stores the indices of negative anchors

    num_pos, num_neg = len(positive_anchors[0]), len(negative_anchors[0])

    if num_pos>C.num_samples/2: #turning extra positive anchors off
        non_selected = random.sample(range(num_pos), num_pos-C.num_samples/2)
        y_is_box_valid[num_pos[0][non_selected],num_pos[1][non_selected], num_pos[2][non_selected]]=0 
        num_pos = C.num_samples/2
    
    if num_neg>C.num_samples - num_pos: #turning extra negative anchors off
        non_selected = random.sample(range(num_neg), num_neg + num_pos - C.num_samples)
        y_is_box_valid[num_neg[0][non_selected], num_neg[1][non_selected], num_neg[2][non_selected]]=0 
        num_neg = C.num_samples - num_pos

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=3) #dims: (1,out_width, out_height, 2*num_anchors)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap,4,axis=3),y_rpn_regr],axis=3) #dims: (1,out_width, out_height, 4*num_anchors)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr), num_pos