# ASR-Net
本模块调用Minkowski Engine，基于稀疏卷积运算实现,实验数据集采用Semantic KITTI数据集。  
以下说明将帮助你在本地机器上安装和运行该项目，进行开发和测试。具体操作，请参考如下说明。  

## 开发环境要求
***⭐***  Ubuntu >= 14.04  
***⭐***CUDA >= 10.1.243 and **the same CUDA version used for pytorch** (e.g. if you use conda cudatoolkit=11.1, use CUDA=11.1 for MinkowskiEngine     compilation)  
***⭐***pytorch >= 1.7 To specify CUDA version, please use conda for installation. You must match the CUDA version pytorch uses and CUDA version used for Minkowski Engine installation. `conda install -y -c nvidia -c pytorch pytorch=1.8.1 cudatoolkit=10.2)`  
***⭐***python >= 3.6   
***⭐***ninja (for installation)    
***⭐***GCC >= 7.4.0    

## 详细说明
Unet1为Unet基本结构，Unet3将空间注意力模块和通道注意力模块与基础Unet结构进行结合。  
`attention` 目录为注意力模块文件目录
`draw_kitti.py`  画图程序，可基于现有的权重文件和点云原始坐标进行可视化绘图  
`frame_extration.py`  帧采样模块，对点云帧每n帧取一帧  
`lablename.py`  存储Semantic KITTI数据集中标签和类别的映射关系，以及可视化中绘图的颜色  
`mIoU.py`  进行混淆矩阵的计算，得到acc，mIoU等指标的统计值  
`split_dataset.py`  对全数据集进行自动划分，使用`torch.utils.data.random_split`方法  
`train_data.txt`  训练集文件对应目录的txt文件，用于训练文件读取  
`validate_data.txt`  交叉验证集文件对应目录的txt文件，用于交叉验证文件读取  
`test_data.txt`  测试集文件对应目录的txt文件，用于测试脚本文件读取   
`test_kitti.py`  测试脚本，用于模型最终效果测试  
`write_config.py`  全数据集打乱划分脚本，训练集：交叉验证集：测试集 6：2：2  

## 超参数设置  
本实验中，训练中超参数设置如下：  
  
`GRID_SIZE_R` 语义分割任务执行范围大小：[20,20,20]，超出该范围的点进行忽略  
`quantization_size` 栅格大小：0.1
`batch_size` 批次大小：16  
`max_epoch`  最大训练轮数：500
`learning_rate`  学习率：1e-4  
`weight_decay` 权值衰减系数：1e-4  
#### Data Argumentation  
`tx` 点云绕x轴平移量：np.random.uniform(-2, 2)  
`ty` 点云绕y轴平移量：np.random.uniform(-2, 2)  
`tz` 点云绕z轴平移量：np.random.uniform(-2, 2)  
`rz` 点云绕z轴旋转量：np.random.uniform(-np.pi,np.pi)  
`factor` 点云缩放因数： np.random.uniform(0.95,1.05)  

##实验环境
GPU：3090  
CPU：i7-10700K CPU@3.8GHz  
内存：32GB  
硬盘：500GB+1TB  
python版本：3.8  
linux：18.04  
数据集：Semantic KITTI  
tensorboard  
CUDA：11.1  
##使用流程
下载数据集后，首先进行开发环境配置。  
然后使用`frame_extraction.py`，`write_config.py`，脚本进行数据集划分。  
调整网络结构，调整超参数，调整文件目录，执行`python train.py`进行训练。  
训练完成后，使用`draw_kitti.py`文件进行可视化或`test_kitti.py`进行测试。  
