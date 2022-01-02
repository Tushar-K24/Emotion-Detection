

def getRoIfeatures(s_ratio, gt_roi):
    (x,y,w,h) = gt_roi
    
    x_new = x/s_ratio
    y_new = y/s_ratio
    w_new = w/s_ratio
    h_new = h/s_ratio

    return (x_new, y_new, w_new, h_new)

def RoIPooling(fv, H, W):
    pass

