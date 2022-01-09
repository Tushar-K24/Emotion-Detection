Components:

1) CNN to extract a feature map
2) A pooling layer to pool the RoI projection (in order to get a fixed size output)
3) Bounding box regressor (for region refinement)
4) Classifier (Expressions + background class)
5) RPN (region proposal network) to generate region proposals