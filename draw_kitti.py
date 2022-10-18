import numpy as np
import os
import open3d
import torch
import time
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
import argparse
import numpy as np
# from Unet1 import MyMinkUNet
# from Unet2 import MyMinkUNet
from Unet3 import MyMinkUNet
# from Unet4 import MyMinkUNet

colorpred = []
xyzpred=[]
# VALID_CLASS_IDS=[
#     0, 71, 81, 50, 80, 70, 52, 10, 99, 255, 48, 44, 40, 72, 51, 1, 60, 254,
#     20, 252, 32, 253, 11, 49, 15, 30, 31, 18, 257, 259]
VALID_CLASS_LEARNING_MAP= {
  0 : 0,     # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5 ,   # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5 ,   # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}
learning_map_inv={
  0: 0 ,     # "unlabeled", and others ignored
  1: 10  ,   # "car"
  2: 11  ,   # "bicycle"
  3: 15  ,   # "motorcycle"
  4: 18  ,   # "truck"
  5: 20   ,  # "other-vehicle"
  6: 30   ,  # "person"
  7: 31  ,   # "bicyclist"
  8: 32  ,   # "motorcyclist"
  9: 40 ,    # "road"
  10: 44  ,  # "parking"
  11: 48  ,  # "sidewalk"
  12: 49  ,  # "other-ground"
  13: 50  ,  # "building"
  14: 51 ,   # "fence"
  15: 70  ,  # "vegetation"
  16: 71  ,  # "trunk"
  17: 72  ,  # "terrain"
  18: 80  ,  # "pole"
  19: 81    # "traffic-sign"
}
CLASS_LABELS = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}
COLOR_MAP={
    0 : [0, 0, 0],
  1 : [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0]
}

#total:4540
a=[0, 71, 81, 50, 80, 70, 52, 10, 99, 255, 48, 44, 40, 72, 51, 1, 60, 254, 20, 252, 32, 253, 11, 49, 15, 30, 31, 18, 257, 259]
dataname="dataset/sequences/02/velodyne/000150.bin"
lablename="dataset/sequences/02/labels/000150.label"

"""
dataset/sequences/00/velodyne/003101.bin"       [0, np.pi/2, 0]
"dataset/sequences/05/velodyne/000150.bin"
"""

class TestDataset(Dataset):
    def __init__(self, quantization_size=0.1):  # quantization_size=0.02
        self.quantization_size = quantization_size

        self.all_data_name = [dataname]

    def __len__(self):
        self.len = len(self.all_data_name)
        return self.len

    def __getitem__(self, index):
        cur_filename = self.all_data_name[index]

        data_filename = cur_filename

        data = np.fromfile(data_filename, dtype=np.float32).reshape(-1, 4)

        xyz = torch.from_numpy(data[:, :3])

        input = xyz
        feats = torch.ones(input.shape[0], 1)

        discrete_coords, unique_feats = ME.utils.sparse_quantize(
            coordinates=input,
            features=feats,
            quantization_size=self.quantization_size,
            ignore_label=0
        )

        return discrete_coords, unique_feats
def test(test_dataloader, model, device):
    with torch.no_grad():
        weight_dict = torch.load('predictpth/epoch 790.pt')
        # weight_dict = torch.load('checkpoints_unet1_lr1e-4/epoch 226.pt')

        model.load_state_dict(weight_dict)
        for batch, data in enumerate(test_dataloader):
            coords, feats = data
            # print("coords",coords)
            xyzpred.append(coords[:,1:])
            input = ME.SparseTensor(feats.float().to(device), coords.to(device))
            start= time.time()
            output = model(input)
            timecost = time.time()-start
            print("耗时 = %.2f"%timecost ,"s")
            _, pred = torch.max(output.F, axis=1)
            # print(f'Batch: {int((batch))}')
            # print(pred)
            # print(len(pred))
            for i in range(len(pred)):
                # print(COLOR_MAP[learning_map_inv[int(i)]])
                colorpred.append(COLOR_MAP[learning_map_inv[int(pred[i])]])
            # with open("predictresult11.label", "w") as f:
            #     f.write(str(pred))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    config = parser.parse_args()
    test_dataset = TestDataset()
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 collate_fn=ME.utils.batch_sparse_collate,
                                 num_workers=8)
    device = torch.device('cuda')
    model = MyMinkUNet(
        in_channels=1,
        out_channels=20,
        D=3
    ).to(device)
    print("*"*55)
    print("当前点云帧:",dataname)


    test(test_dataloader=test_dataloader, model=model, device=device)
main()

data = np.fromfile(dataname, dtype=np.float32).reshape(-1, 4)
label = np.fromfile(lablename, dtype=np.uint32).reshape(-1,1)
# the lower 16 bits correspond to the label. The upper 16 bits encode the instance id
sem_label = label & 0xFFFF # semantic label in lower half
inst_label = label >> 16    # instance id in upper half
xyz=data[:,:3]

print("帧内点数量:",inst_label.shape[0])
print("*"*55)

# np.set_printoptions(threshold=np.inf)
# print(sem_label.shape,max(sem_label))
# print(inst_label.shape,max(inst_label))
colors=[]*len(label)
# sem_label=sem_label.astype(np.int)
sem_label=list(map(int,sem_label))
for i in range(len(label)):
    colors.append(COLOR_MAP[sem_label[i]])

#groundtruth
test_pcd=open3d.geometry.PointCloud()
test_pcd.points=open3d.utility.Vector3dVector(xyz)
colors=np.array(colors)
test_pcd.voxel_down_sample(voxel_size=0.01)
test_pcd.colors=open3d.utility.Vector3dVector(colors/255)

#predict
# print("xyzpred=",xyzpred,type(xyzpred[0]))
xyzpred=xyzpred[0].numpy()
pred_pcd=open3d.geometry.PointCloud()
pred_pcd.points=open3d.utility.Vector3dVector(xyzpred)
colorpred=np.array(colorpred)
pred_pcd.voxel_down_sample(voxel_size=0.01)
pred_pcd.colors=open3d.utility.Vector3dVector(colorpred/255)

pred_pcd.points = open3d.utility.Vector3dVector(
        np.array(pred_pcd.points) + np.array([1000, 1200, 0]))
test_pcd.scale(10.0,(0,0,0))

######
R = pred_pcd.get_rotation_matrix_from_axis_angle(np.array([0, np.pi/2, 0]).T)#向量方向为旋转轴，模长等于旋转角度，绕y轴旋转-90°
pred_pcd.rotate(R)
######
open3d.visualization.draw_geometries([pred_pcd],window_name="test")
#open3d.visualization.draw_geometries([test_pcd,pred_pcd],window_name="test")

