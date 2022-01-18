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

def generate_anchor_boxes():
    pass