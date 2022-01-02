Ideas:
Assume that we have the region proposals for now. (We'll work around that part later)


Components:

1) CNN to extract a feature map
2) A pooling layer to pool the RoI projection (in order to get a fixed size output)
3) Bounding box regressor (for region refinement)
4) Classifier (Expressions + background class)


Questions:

1) How do we take the projection of RoI on the feature map?
Ans: https://stackoverflow.com/questions/40925052/fast-rcnn-roi-projection

2) How do one do Spatial Pyramid Pooling? (if the feature map is not exactly dividable, how does on pad the feature map)
Ans: https://medium.datadriveninvestor.com/review-on-fast-rcnn-202c9eadd23b

3) How to train the network?
Ans: For now, let's just train for 1 image at a time. (it will have multiple RoIs)


Implementation Remarks:

RoIPooling.py :

Make sure to check the axis of the pooled_rois (ln: 64)