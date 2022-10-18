import argparse
import numpy as np
import torch.nn as nn
import torch
import os
import  time
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
import torch.nn.functional as F
from examples.unet import UNet
from examples.minkunet import MinkUNet34C
import open3d

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
LEARNING_MAP_INV={
  0: 0,
  1: 10,
  2: 11,
  3: 15,
  4: 18,
  5: 20,
  6: 30,
  7: 31,
  8: 32,
  9: 40,
  10: 44,
  11: 48,
  12: 49,
  13: 50,
  14: 51,
  15: 70,
  16: 71,
  17: 72,
  18: 80,
  19: 81
}
class MyDataset(Dataset):
    # Warning: read using mutable obects for default input arguments in python.
    def __init__(self,quantization_size=0.02,DATA_PATH="./00/velodyne/"):
        self.DATA_PATH=DATA_PATH
        self.all_data_name=os.listdir(self.DATA_PATH)
        self.quantization_size=quantization_size
        # axis_pcd = open3d.geometry.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # test_pcd=open3d.geometry.PointCloud()
        # test_pcd.points=open3d.utility.Vector3dVector(self.xyz)
        # test_pcd.colors=open3d.utility.Vector3dVector(self.xyz)
        # open3d.visualization.draw_geometries([test_pcd],window_name="test")
    def __len__(self):
        return len(self.all_data_name)

    def __getitem__(self, index):


        data = np.fromfile(os.path.join(self.DATA_PATH,'%06d.bin'%index), dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile("./00/labels/%06d.label"%index, dtype=np.uint32).reshape(-1, 1)
        # the lower 16 bits correspond to the label. The upper 16 bits encode the instance id
        sem_label = labels & 0xFFFF  # semantic label in lower half
        sem_label=sem_label/1.0
        inst_label = labels >> 16  # instance id in upper half
        xyz = data[:, :3]
        print()

        # print(str(os.path.join(self.DATA_PATH,'%06d.bin'%index)))
        for i in range(len(labels)):
            sem_label[i]=VALID_CLASS_LEARNING_MAP[int(sem_label[i])]

        input=xyz
        feats=input
        # feats=data
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=input,
            features=feats,
            labels=sem_label,
            quantization_size=self.quantization_size,
            ignore_label=0)

        return discrete_coords, unique_feats, unique_labels


def collation_fn(data_labels):
    coords, feats, labels = list(zip(*data_labels))
    coords_batch, feats_batch, labels_batch = [], [], []

    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0))

    return coords_batch, feats_batch, labels_batch


class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2)
        self.ReLU = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.MaxPool2d(2)
        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(4608, 2048)
        self.f7 = nn.Linear(2048, 2048)
        self.f8 = nn.Linear(2048, 1000)
        self.f9 = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s3(x)
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = F.dropout(x, p=0.5)
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)

        x = self.f9(x)
        x=self.ReLU(x)

        return x
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main(config):
    time_start = time.time()
    # Binary classification
    # mynet=Mynet()
    # x=torch.rand([1,3,224,224])
    # print("x=",x.shape)
    # y=mynet(x)
    # model = TheModelClass()
    # optimizer2 = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device('cuda' if (
            torch.cuda.is_available() and not config.use_cpu) else 'cpu')
    net = UNet(
        3,  # in_nchannel
        20,  # out_nchannel
        D=3)
    # net=MinkUNet34C(4,20)
    net.to(device)
    print(f"Using {device}")

    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    # print(net)

    params=list(net.parameters())
    print(len(params))
    # print("params",(params[12].size()))

    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # Dataset, data loader
    train_dataset = MyDataset()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        # 1) collate_fn=collation_fn,
        # 2) collate_fn=ME.utils.batch_sparse_collate,
        # 3) collate_fn=ME.utils.SparseCollation(),
        collate_fn=ME.utils.batch_sparse_collate,
        num_workers=1)
    accum_loss, accum_iter, tot_iter = 0, 0, 0

    for epoch in range(config.max_epochs):

        train_iter = iter(train_dataloader)
        # Training
        net.train()
        for i, data in enumerate(train_iter):
            coords, feats, labels = data
            # print("coords.shape",coords.shape)
            # print("feats.shape",feats.shape)
            # print(labels.shape)
            # out = net(ME.SparseTensor(feats.float(), coords))
            out = net(ME.SparseTensor(feats.float().to(device), coords.to(device)))


            # out = net()

            # print("out..F.shape",out.F.shape)
            # print(out.shape)
            # print(out.F.squeeze().shape)
            # print(labels.long().squeeze().shape)
            optimizer.zero_grad()
            # loss = criterion(out.F, labels.long().squeeze())
            loss = criterion(out.F, labels.long().squeeze().to(device))
            loss.backward()

            optimizer.step()
            accum_loss += loss.item()
            accum_iter += 1
            tot_iter += 1

            print(
                f'Epoch: {epoch} iter: {tot_iter}, Loss: {accum_loss / accum_iter}'
            )
            with open('test.txt', 'a+') as f:
                if epoch==0 and tot_iter==1:
                    f.write("Epoch     iter     Loss\n")
                f.write('%d         %d      %10f\n' % (epoch, tot_iter, (accum_loss / accum_iter)))


            accum_loss, accum_iter=0, 0


        torch.save(net.state_dict(), "./checkpoints/%d.pth"%epoch)
    time_end=time.time()
    print("time_cost",time_end-time_start)
    print("time_cost",time_end-time_start,'s')


def load_file(file_name):
    data = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    xyz = data[:, :3]
    colors=xyz
    return xyz, colors
def normalize_color(color: torch.Tensor, is_color_in_range_0_255: bool = False) -> torch.Tensor:
    r"""
    Convert color in range [0, 1] to [-0.5, 0.5]. If the color is in range [0,
    255], use the argument `is_color_in_range_0_255=True`.

    `color` (torch.Tensor): Nx3 color feature matrix
    `is_color_in_range_0_255` (bool): If the color is in range [0, 255] not [0, 1], normalize the color to [0, 1].
    """
    if is_color_in_range_0_255:
        color /= 255
    color -= 0.5
    return color.float()
def testpth():
    device = torch.device('cuda' if (
            torch.cuda.is_available() and not config.use_cpu) else 'cpu')
    model = UNet(
        3,  # in_nchannel
        20,  # out_nchannel
        D=3)
    model=model.to(device)
    model_dict = torch.load(config.weights)
    print(model_dict,"model_dict")
    model.load_state_dict(model_dict)
    model.eval()
    coords, colors= load_file(config.file_name)
    # Measure time
    with torch.no_grad():
        voxel_size = 0.02
        # Feed-forward pass and get the prediction
        in_field = ME.TensorField(
            features=normalize_color(torch.from_numpy(colors)),
            coordinates=ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=device,
        )
        # Convert to a sparse tensor
        sinput = in_field.sparse()
        # Output sparse tensor
        soutput = model(sinput)
        # get the prediction on the input tensor field
        out_field = soutput.slice(in_field)
        logits = out_field.F
        _, pred = logits.max(1)
        pred = pred.cpu().numpy()
        print("pred=",pred.shape,pred)
        #draw source
        lablename = "./00/labels/003250.label"
        label = np.fromfile(lablename, dtype=np.uint32).reshape(-1, 1)
        colors = [] * len(label)
        sem_label = label & 0xFFFF  # semantic label in lower half
        sem_label = sem_label.astype(np.int)
        sem_label = list(map(int, sem_label))
        for i in sem_label:
            colors.append(COLOR_MAP[i])

        test_pcd = open3d.geometry.PointCloud()
        test_pcd.points = open3d.utility.Vector3dVector(coords)
        colors = np.array(colors)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(coords)
        pcd.colors = open3d.utility.Vector3dVector(colors / 255)
        pcd.estimate_normals()
        #draw test
        correct=0
        #accuracy
        for i in range(len(label)):
            if VALID_CLASS_LEARNING_MAP[int(sem_label[i])]==pred[i]:
                correct+=1
        accuracy=correct/len(pred)
        print("accuracy=",accuracy)

        colors=[]

        for i in pred:
            colors.append(COLOR_MAP[LEARNING_MAP_INV[i]])

        colors=np.array(colors)
        print("colors.shape",colors.shape)
        pred_pcd = open3d.geometry.PointCloud()
        pred_pcd.points = open3d.utility.Vector3dVector(coords)
        pred_pcd.colors = open3d.utility.Vector3dVector(colors / 255)
        pred_pcd.estimate_normals()
        pcd.points = open3d.utility.Vector3dVector(
                np.array(pcd.points) + np.array([0, 100, 0]))

        open3d.visualization.draw_geometries([pcd]+[pred_pcd])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #train parser
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    #

    #test dataset parser
    parser.add_argument('--file_name', type=str, default='00/velodyne/003250.bin')
    parser.add_argument('--weights', type=str, default='checkpoints/10.pth')
    parser.add_argument('--use_cpu', action='store_true')
    #

    config = parser.parse_args()
    # main(config)
    testpth()



