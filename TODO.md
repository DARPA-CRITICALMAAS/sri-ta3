#X priority of tasks
0. Positive / Negative Sample Balancing
    a. Over sampling positives #0 (done)
    b. SMOTE? #0
    c. Down sampling negatives - randomly sample XX% of negatives within each "geo-cell" #0 (done)
    d. Gain on positives? (done)
    IDEAL RESULT - a consistent strategy to balanced datasets
    CONSIDERATIONS - What "vector space" is used to generate psuedo samples (i.e. geo- or input- space)
1. Fix the train / valid / test splitting once and for all
    a. Random splits on points (ignore negative/positive label) #0 (done)
    b. Random splits on points BUT proportianate (consider positive/negative label) #0 (done)
    c. Random splits on "geo-cells" (modification of existing spatial cross val) #1
    g. Clustering based splits where # clusters goes from # samples down to 3. #2
    IDEAL RESULT - optimizing a metric in training split results in improved metric in test/valid split,
    AND IMPORTANTLY, a better qualitative evaluation of the map
    CONSIDERATIONS - What "vector space" is used to generate splits (i.e. geo- or input- space)
2. Speed and Quality of Mapping
    i. Speed
        a. More GPUs #1
        b. Lower resolution GeoTiffs #0 (done)
        c. Fill no data on geotiffs along edges #0 (done)
        d. Skip pixels #0
        e. Confirm precision (lower if feasible) #0
        f. Timing - Training & Mapping
    ii. Quality
        a. Upsampling after mapping output #1 (done)
        b. Smoothing after upsampling / map output #1 (done)
        c. Coloring accroding to USGS conventions
        d. Thresholding (e.g. expected % of coverage or by 1,2,3 sigmas, etc)
    IDEAL RESULT - near real-time map output
=================