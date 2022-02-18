import random
import numpy as np

import config as C

def union(ex_roi, gt_roi, ar_i):
    #rois are in format (x1,y1,x2,y2)
    ex_ar = (ex_roi[2]-ex_roi[0])*(ex_roi[3]-ex_roi[1])
    gt_ar = (gt_roi[2]-gt_roi[0])*(gt_roi[3]-gt_roi[1])
    return ex_ar + gt_ar - ar_i

def intersection(ex_roi, gt_roi):
    x1 = max(ex_roi[0],gt_roi[0])
    y1 = max(ex_roi[1],gt_roi[1])
    x2 = min(ex_roi[2],gt_roi[2])
    y2 = min(ex_roi[3],gt_roi[3])
    if x1>x2 or y1>y2:
        return 0
    return (x2-x1)*(y2-y1)

def iou(ex_roi, gt_roi):
    if ex_roi[0]>ex_roi[2] or ex_roi[1]>ex_roi[3] or gt_roi[0]>gt_roi[2] or gt_roi[1]>gt_roi[3]:
        return 0
    ar_i = intersection(ex_roi,gt_roi)
    ar_u = union(ex_roi,gt_roi,ar_i)
    return ar_i/(1e-6 + ar_u) #to avoid divisibility by 0

def generate_anchor_boxes(imdb, width, height):
    anchor_scales = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    
    num_anchors = len(anchor_scales)*len(anchor_ratios)
    downscale = C.rpn_stride
    num_bboxes = len(imdb)

    out_width, out_height = width//downscale, height//downscale

    gt_bboxes = np.zeros((num_bboxes,4))
    gt_bboxes[:,0] = imdb['x1']
    gt_bboxes[:,1] = imdb['y1']
    gt_bboxes[:,2] = imdb['x2']
    gt_bboxes[:,3] = imdb['y2']
    
    y_is_box_valid = np.zeros((out_height, out_width, num_anchors)).astype(np.float32)
    y_rpn_overlap = np.zeros((out_height, out_width, num_anchors)).astype(np.float32)
    y_rpn_regr = np.zeros((out_height, out_width, 4*num_anchors)).astype(np.float32)

    #to mark the best iou for every gt bbox
    best_iou_for_bbox = np.ones((num_bboxes))
    best_iou_for_bbox *= -1 #all -1 if not visited
    best_anchor_for_bbox = np.zeros((num_bboxes,4))
    best_dx_for_bbox = np.zeros((num_bboxes,4)) #to store tx,ty,tw,th
    
    for anchor_scale_idx in range(len(anchor_scales)):
        for anchor_ratio_idx in range(len(anchor_ratios)):
            anchor_x = anchor_scales[anchor_scale_idx]*anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_scales[anchor_scale_idx]*anchor_ratios[anchor_ratio_idx][1]

            anchor_idx = len(anchor_ratios)*anchor_scale_idx + anchor_ratio_idx
            for x in range(out_width):
                x1 = x*downscale - anchor_x/2
                x2 = x*downscale + anchor_x/2
                if x1<0 or x2>=width: #x coordinate out of image
                    continue
                for y in range(out_height):
                    y1 = y*downscale - anchor_y/2
                    y2 = y*downscale + anchor_y/2
                    if y1<0 or y2>=height: #y coordinate out of image
                        continue
                    
                    #centre of current anchor box
                    cx = (x1+x2)/2    
                    cy = (y1+y2)/2

                    best_iou_for_anchor = 0 #best iou for current anchor
                    rpn_regr = [0,0,0,0] #best regr for current anchor

                    bbox_type = 'neg'
                    for bbox_num in range(num_bboxes):
                        curr_iou = iou([x1,y1,x2,y2],gt_bboxes[bbox_num])
                        
                        if imdb['class'][bbox_num]=='bg': #if background class, ignore
                            continue

                        #setting up bbox_type
                        if curr_iou >= C.rpn_max_overlap: #positive sample
                            bbox_type = 'pos'
                        
                        if C.rpn_min_overlap<curr_iou<C.rpn_max_overlap and bbox_type!='pos': #neutral sample
                            bbox_type = 'neutral'

                        #centre of current gt_bbox
                        gt_cx = (gt_bboxes[bbox_num,0] + gt_bboxes[bbox_num,2])/2
                        gt_cy = (gt_bboxes[bbox_num,1] + gt_bboxes[bbox_num,3])/2

                        #calculating dx (as per original paper)
                        tx = (cx - gt_cx)/(x2 - x1)
                        ty = (cy - gt_cy)/(y2 - y1)
                        tw = np.log((gt_bboxes[bbox_num,2]-gt_bboxes[bbox_num,0])/(x2-x1))
                        th = np.log((gt_bboxes[bbox_num,3]-gt_bboxes[bbox_num,1])/(y2-y1))

                        if curr_iou > best_iou_for_bbox[bbox_num]:
                            best_iou_for_bbox[bbox_num] = curr_iou
                            best_anchor_for_bbox[bbox_num] = [y,x,anchor_scale_idx,anchor_ratio_idx]
                            best_dx_for_bbox[bbox_num] = [tx,ty,tw,th]
                        
                        if curr_iou > best_iou_for_anchor:
                            curr_iou = best_iou_for_anchor
                            rpn_regr = [tx,ty,tw,th]
                        
                    if bbox_type=='neg': #bounding box type is negative
                        y_is_box_valid[y,x,anchor_idx]=1
                        y_rpn_overlap[y,x,anchor_idx]=0
                    
                    elif bbox_type=='neutral': #bounding box type is neutral
                        y_is_box_valid[y,x,anchor_idx]=0
                        y_rpn_overlap[y,x,anchor_idx]=0

                    else: #bounding box type is positive
                        y_is_box_valid[y,x,anchor_idx]=1
                        y_rpn_overlap[y,x,anchor_idx]=1
                        y_rpn_regr[y,x,4*anchor_idx:4*anchor_idx + 4] = rpn_regr

    # treating all best_boxes as positive samples
    for bbox_num in range(num_bboxes):
        if best_iou_for_bbox[bbox_num]==-1: #never visited
            continue
        y,x,anchor_scale_idx, anchor_ratio_idx = best_anchor_for_bbox[bbox_num].astype(int)
        anchor_idx = len(anchor_ratios)*anchor_scale_idx + anchor_ratio_idx

        y_is_box_valid[y,x,anchor_idx]=1
        y_rpn_overlap[y,x,anchor_idx]=1
        y_rpn_regr[y,x,4*anchor_idx:4*anchor_idx + 4] = best_dx_for_bbox[bbox_num]
    
    positive_anchors = np.where((y_is_box_valid==1)&(y_rpn_overlap==1)) #stores the indices of positive anchors
    negative_anchors = np.where((y_is_box_valid==1)&(y_rpn_overlap==0)) #stores the indices of negative anchors

    num_pos, num_neg = len(positive_anchors[0]), len(negative_anchors[0])

    if num_pos>C.num_samples/2: #turning extra positive anchors off
        non_selected = random.sample(range(num_pos), num_pos-C.num_samples//2)
        y_is_box_valid[positive_anchors[0][non_selected],positive_anchors[1][non_selected], positive_anchors[2][non_selected]]=0 
        num_pos = C.num_samples//2
    
    if num_neg>C.num_samples - num_pos: #turning extra negative anchors off
        non_selected = random.sample(range(num_neg), num_neg + num_pos - C.num_samples)
        y_is_box_valid[negative_anchors[0][non_selected], negative_anchors[1][non_selected], negative_anchors[2][non_selected]]=0 
        num_neg = C.num_samples - num_pos
    
    #converting the numpy arrays from 3D to 4D
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0) #dims: (1,out_width, out_height, num_anchors)
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0) #dims: (1,out_width, out_height, num_anchors)
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0) #dims: (1,out_width, out_height, 4*num_anchors)

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=3) #dims: (1,out_width, out_height, 2*num_anchors)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap,4,axis=3),y_rpn_regr],axis=3) #dims: (1,out_width, out_height, 4*num_anchors)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr), num_pos 

                            