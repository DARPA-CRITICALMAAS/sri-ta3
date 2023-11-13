1. Fix the splitting / labels used within DNN code
    * split on deposits only
    * train / test / val on deposits & occurences only
2. Build MAE pipline
    * run existing hyperspectral MAE repo  - confirm it works
    * trasnfer APPROPRIATE PORTION of existing hyperspectral MAE repo
    * self-supervised training
    * supervised training
    * initial MAE - monolothic, all inputs
    * modality specific MAEs - subsets of inputs
    * multi-modal MAE (CLS token reconstruction)
3. Implement / test hyperbolic geometry classifier / decoder
    * run existing semantic segmentation code for reference - confirm it works
    * Likelihood
    * Uncertainty
    * Hierarchy
X. Need to address the issues with spatial cross validation
