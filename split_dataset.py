import argparse
import numpy as np
from torch import nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms
import MinkowskiEngine as ME
import os
import labelname
import numpy as np
import time
from tqdm import *
import torch.nn.functional as F
import open3d
from tensorboardX import SummaryWriter
from MyUnet4D_source import MinkUNet4D
class FullDataset(Dataset):
    def __init__(self, quantization_size=0.1, DATA_PATH="./00/velodyne/"):  # quantization_size=0.02
        self.DATA_PATH = DATA_PATH
        self.all_data_name = os.listdir(self.DATA_PATH)
        self.quantization_size = quantization_size

    def __len__(self):
        self.len = len(self.all_data_name) - 2
        return self.len

    def __getitem__(self, index):
        print('__getitem__')
        index += 1
        data0 = np.fromfile(os.path.join(self.DATA_PATH, '%06d.bin' % (index - 1)), dtype=np.float32).reshape(-1, 4)
        labels0 = np.fromfile("./00/labels/%06d.label" % (index - 1), dtype=np.uint32).reshape(-1, 1)
        data1 = np.fromfile(os.path.join(self.DATA_PATH, '%06d.bin' % index), dtype=np.float32).reshape(-1, 4)
        labels1 = np.fromfile("./00/labels/%06d.label" % index, dtype=np.uint32).reshape(-1, 1)
        data2 = np.fromfile(os.path.join(self.DATA_PATH, '%06d.bin' % (index + 1)), dtype=np.float32).reshape(-1, 4)
        labels2 = np.fromfile("./00/labels/%06d.label" % (index + 1), dtype=np.uint32).reshape(-1, 1)

        # the lower 16 bits correspond to the label. The upper 16 bits encode the instance id
        sem_label0 = torch.from_numpy((labels0 & 0xFFFF) / 1.0)  # semantic label in lower half
        sem_label1 = torch.from_numpy((labels1 & 0xFFFF) / 1.0)  # semantic label in lower half
        sem_label2 = torch.from_numpy((labels2 & 0xFFFF) / 1.0)  # semantic label in lower half

        xyz0 = torch.from_numpy(data0[:, :3])
        xyz1 = torch.from_numpy(data1[:, :3])
        xyz2 = torch.from_numpy(data2[:, :3])

        # print(index)
        print('\n', "./00/labels/%06d.bin" % (index - 1), '\n'"./00/labels/%06d.bin" % (index), '\n',
              "./00/labels/%06d.bin" % (index + 1))

        for i in range(len(labels0)):
            sem_label0[i] = labelname.VALID_CLASS_LEARNING_MAP[int(sem_label0[i])]
        for i in range(len(labels1)):
            sem_label1[i] = labelname.VALID_CLASS_LEARNING_MAP[int(sem_label1[i])]
        for i in range(len(labels2)):
            sem_label2[i] = labelname.VALID_CLASS_LEARNING_MAP[int(sem_label2[i])]

        t0 = torch.full((labels0.shape[0], 1), 0)
        t1 = torch.full((labels1.shape[0], 1), 1)
        t2 = torch.full((labels2.shape[0], 1), 2)
        txyz0 = torch.cat((t0, xyz0), dim=1)
        txyz1 = torch.cat((t1, xyz1), dim=1)
        txyz2 = torch.cat((t2, xyz2), dim=1)

        print(txyz0.shape, txyz1.shape, txyz2.shape)
        input = torch.cat((txyz0, txyz1, txyz2), dim=0)
        feats = torch.ones(input.shape[0], 3)
        # (n0+n1+n2,4)
        print('连接后的维度:', input.shape)
        labels = torch.cat((sem_label0, sem_label1, sem_label2), dim=0)
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=input,
            features=feats,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=0,
        )
        coord,unique_map, inverse_map = ME.utils.sparse_quantize(discrete_coords, return_index=True, return_inverse=True)

        print(ME.utils.sparse_quantize(input,feats,labels,quantization_size=self.quantization_size,
return_index=True, return_inverse=False))
        # print(unique_map)
        # unique_coords = input[unique_map]
        # print(unique_coords)
        # print(unique_coords[inverse_map] == input)  # True
        print('体素化后的shape:', end='')
        print(discrete_coords.shape, unique_feats.shape, unique_labels.shape)
        return discrete_coords, unique_feats, unique_labels

class Num_Dataset(Dataset):
    def __init__(self):
        self.x=list(range(20))
    def __len__(self):
        l=len(self.x)-2
        return l
    def __getitem__(self, index):
        index+=1
        return self.x[index]

full_dataset = Num_Dataset()

train_size = int(0.8*full_dataset.__len__())
test_size= full_dataset.__len__()-train_size


train_dataset,test_dataset = torch.utils.data.random_split(full_dataset,
                                                        [train_size,test_size],
                                                        generator=torch.Generator().manual_seed(0))
full_dataloader=DataLoader(full_dataset,batch_size=3,
                                  num_workers=0)

train_dataloader=DataLoader(train_dataset,batch_size=1,
                                  num_workers=0)

for batch,data in enumerate(full_dataloader):
    print(batch,data)