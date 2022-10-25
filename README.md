# ASR-Net
This module calls Minkowski Engine, which is implemented based on sparse convolution operation. The experimental dataset adopts the Semantic KITTI dataset.  
The following instructions will help you install and run the project on your local machine for development and testing. Please refer to the following instructions for specific operations.  

## Development environment requirements  
***●***  Ubuntu >= 14.04  
***●***  CUDA >= 10.1.243 and **the same CUDA version used for pytorch** (e.g. if you use conda cudatoolkit=11.1, use CUDA=11.1 for MinkowskiEngine     compilation)  
***●***  pytorch >= 1.7 To specify CUDA version, please use conda for installation. You must match the CUDA version pytorch uses and CUDA version used for Minkowski Engine installation. `conda install -y -c nvidia -c pytorch pytorch=1.8.1 cudatoolkit=10.2)`  
***●***  python >= 3.6   
***●***  ninja (for installation)    
***●***  GCC >= 7.4.0    

## Detailed description   
Unet1 is the basic structure of the baseline model. Unet2 combines the spatial attention module and channel attention module with the baseline structure.    
Unet3 combines the spatial attention module with baseline, and Unet4 combines the channel attention module with baseline.    
`attention` directory is the directory of attention module files.    
`draw_kitti.py`  Renderer, which can visually draw based on the existing weight file and the original coordinates of the point cloud.  
`frame_extration.py`  Frame sampling module, taking one frame for every n frames of point cloud frame.    
`lablename.py`  Stores the mapping relationship between labels and categories in the Semantic KITTI dataset, as well as the color of drawings in visualization.    
`mIoU.py`  Calculation module, calculate the confusion matrix, and obtain the statistical values of acc, mIoU and other indicators.    
`split_dataset.py`  Automatic partitioning of the entire dataset, using `torch.utils.data.random_split` method.    
`train_data.txt`  The txt file of the directory corresponding to the training set file is used to read the training file.    
`validate_data.txt`  The txt file of the directory corresponding to the cross validation set file for cross validation file reading.     
`test_data.txt`  The txt file of the directory corresponding to the test set file, which is used as the script for reading the test set file.    
`test_kitti.py`  Test script for model effect test.    
`write_config.py`  The full dataset scrambling and partitioning script is used to scramble data and partition data sets. Training set: cross validation set: test set 6:2:2   

## Super parameter setting    
In this experiment, the super parameters in training are set as follows：  
  
`GRID_SIZE_R` Size of semantic segmentation task execution range: [20, 20, 20]. Points beyond this range are ignored.  
`quantization_size` ：0.1  
`max_epoch`  Maximum training rounds：500  
`batch_size` ：16  
`learning_rate`  ：1e-4  
`weight_decay` ：1e-4  
#### Data Argumentation  
`tx` Translation of point cloud around x-axis：np.random.uniform(-2, 2)  
`ty` Translation of point cloud around y-axis：np.random.uniform(-2, 2)  
`tz` Translation of point cloud around z-axis：np.random.uniform(-2, 2)  
`rz` Rotation amount of point cloud around z-axis：np.random.uniform(-np.pi,np.pi)  
`factor` Point Cloud Scale Factor： np.random.uniform(0.95,1.05)  

##  Experimental environment  
GPU：3090  
CPU：i7-10700K CPU@3.8GHz  
Memory：32GB  
Disk：500GB+1TB  
python version：3.8  
linux：18.04  
Dataset：Semantic KITTI  
tensorboard  
CUDA：11.1  
##  Use process  
After downloading the dataset, first configure the development environment.  
Then use `frame_extraction.py`，`write_ config.py`. The script partitions the dataset.  
Adjust the network structure, super parameters, file directory, and execute `python train.py` for training.  
After training, use `draw_Kitti.py` file for visualization or `test_Kitti.py` test.  
## Experiment effect picture    
![Image text](https://raw.githubusercontent.com/wuzhaoo/ASR-Net/main/figures/1.jpg)  

## Effect Diagram of Attention Module    
![Image text](https://raw.githubusercontent.com/wuzhaoo/ASR-Net/main/figures/2.png)  

## Forecast time/power statistic chart    
![Image text](https://raw.githubusercontent.com/wuzhaoo/ASR-Net/main/figures/3.png)  

## Rendering of mobile robot deployment     
![Image text](https://raw.githubusercontent.com/wuzhaoo/ASR-Net/main/figures/4.png)    
