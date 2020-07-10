# UAVImgSegmentation
UAV Aerial image segmentation using PyTorch/Catalyst (with useful tools and utils)
________________________________________________________________________________________
Research based on DroneDeploy dataset, but you may use something else.
Main idea:
Image has multiclass label. Each class has its own color marker and mask and the mask is just an RGB image.
## **Procedure**
Use Jupyter Notebooks:
+ DataPrepLarge for make base data preparation:
  - crop to tiles
  - create DataFrame with RLE encoded pixels by each class for all images
  - drop useless mask (with ~100% empty labeled pixels)
+ FPN for segmentation images:
  - visualize images with masks
  - setting up data for training
  - define model parametrs
  - training models
  - save and visualize the result
