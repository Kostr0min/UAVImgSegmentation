# UAVImgSegmentation
UAV aerial image segmentation using PyTorch/Catalyst (with useful tools and utils)
________________________________________________________________________________________
Research based on DroneDeploy dataset, but you may use something else.
Main idea:
Image has multiclass label. Each class has its own color marker and mask and the mask is just an RGB image.
Train the model and select the best prediction object to use in  the comparision algorithm based on EM-algorithm and CPD
## **Progress**
:white_check_mark: Data set preparation    
:white_check_mark: Sematic segmentation model    
:white_check_mark: Post-processing the result    
:white_check_mark: Object selection algorithm    
:white_check_mark: Comparison using CPD-based algorithm    
:black_square_button: Create an instance segmentation model to improve algorithm efficiency    
:black_square_button: Modify object selection algorithm    
:black_square_button: Create comparison algorithm able to find coordinate of selecting object in all masks

## **Procedure**
Use Jupyter Notebooks:
+ __DataPrepLarge__ for make base data preparation:
  - crop to tiles
  - create DataFrame with RLE encoded pixels by each class for all images
  - drop useless mask (with ~100% empty labeled pixels)
+ __FPN__ for segmentation images:
  - visualize images with masks
  - setting up data for training
  - define model parametrs
  - training models
  - save and visualize the result 
+ __Inference__ for model inference and save predictions for further research
+ __Object_comparison__:
  - find the best object to compare using multi-criteria optimization
  - make object transformation
  - compare objects using CPD algorithm
  - to be continued :clock2::clock230::clock3:
__________________________________________  
  ## **Utils**
  + get_img -> return image from path
  + rle_decode -> make mask as 2darray from rle-code
  + mask2rle -> create rle code from mask
  + make_mask -> return ndarray (concatenate masks for all classes to one array)
  + post_process -> in 2 words: apply thresholding to predicted masks
  + visualize / visualize_with_raw -> plot images and masks
  + plot_with_augmentation -> plot images with augmentation
  + CityDataset - PyTorch Dataset object
  + Create_houses_df
    - get_edge -> return contour points (only key points, without lines between them)
