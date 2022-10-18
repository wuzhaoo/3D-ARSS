# from Unet1 import MyMinkUNet
# from Unet2 import MyMinkUNet
# from Unet3 import MyMinkUNet
# from Unet4 import MyMinkUNet
from Unet5 import MyMinkUNet
from testUnet5 import MyMinkUNet
# from testUnet5_2 import MyMinkUNet
# from oneconvnet import MyMinkUNet
"""
net
pth
save_dir
epoch
"""
import random
import torch
from mIoU import SegmentationMetric
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms
import MinkowskiEngine as ME
import labelname
import argparse
import numpy as np
import torch.nn as nn
import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from tensorboardX import SummaryWriter

GRID_SIZE_R=[20, 20, 20]

class TestDataset(Dataset):
    def __init__(self, quantization_size=0.1, DATA_PATH="./config_test.txt"):  # quantization_size=0.02
        self.DATA_PATH = DATA_PATH
        self.quantization_size = quantization_size
        with open(DATA_PATH, "r") as f:
            self.all_data_name = f.read()
        self.all_data_name = self.all_data_name.split('\n')

    def __len__(self):
        self.len = len(self.all_data_name)
        return self.len

    def __getitem__(self, index):
        cur_filename = self.all_data_name[index]
        sequence = cur_filename.split('.')[1].split('/')[-3]
        count = cur_filename.split('.')[1].split('/')[-1]

        data1_filename = cur_filename
        labels1_filename = './dataset/sequences/' + sequence + '/labels/' + '%06d.label' % (int(count))

        data1 = np.fromfile(data1_filename, dtype=np.float32).reshape(-1, 4)
        labels1 = np.fromfile(labels1_filename, dtype=np.uint32).reshape(-1, 1)

        # the lower 16 bits correspond to the label. The upper 16 bits encode the instance id
        sem_label1 = torch.from_numpy((labels1 & 0xFFFF) / 1.0)  # semantic label in lower half

        xyz1 = torch.from_numpy(data1[:, :3])

        # print('\n',data0_filename,'\n', data1_filename,
        #       '\n',data2_filename)

        for i in range(len(labels1)):
            sem_label1[i] = labelname.VALID_CLASS_LEARNING_MAP[int(sem_label1[i])]
        bound_x = np.logical_and(
            xyz1[:, 0] >= -GRID_SIZE_R[0], xyz1[:, 0] < GRID_SIZE_R[0]
        )
        bound_y = np.logical_and(
            xyz1[:, 1] >= -GRID_SIZE_R[1], xyz1[:, 1] < GRID_SIZE_R[1]
        )
        bound_z = np.logical_and(
            xyz1[:, 2] >= -GRID_SIZE_R[2], xyz1[:, 2] < GRID_SIZE_R[2]
        )
        bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z).bool()
        xyz1 = xyz1[bound_box]
        input = xyz1
        feats = torch.ones(input.shape[0], 1)
        labels = sem_label1
        labels = labels[bound_box]


        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=input,
            features=feats,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=0,
        )

        return discrete_coords, unique_feats, unique_labels


def test(config,test_dataloader,model,loss_fn,device):
    Testloss,Testcurrent,TestmIoU,n = 0.0,0.0,0.0,0
    metric = SegmentationMetric(20)
    total_acc,total_mIoU=0.0,0.0
    acc=[]
    totalmIoU=[]
    for  i in range(790,791):

        with torch.no_grad():

            weight_dict = torch.load('predictpth/epoch %d.pt'%i)
            print('checkpoints/epoch %d.pt'%i)
            # model.load_state_dict(weight_dict)
            for batch,data in enumerate(test_dataloader):
                n+=1
                print(f'Batch: {int((batch))}')

                coords, feats, labels = data

                input = ME.SparseTensor(feats.float().to(device), coords.to(device))

                labels=labels.to(device)
                ###
                output = model(input)

                cur_loss=loss_fn(output.F, labels.long().squeeze().to(device))

                _,pred=torch.max(output.F,axis=1)
                correct_count = torch.sum(labels.long().squeeze()==pred)
                all_count=output.F.shape[0]
                cur_acc=correct_count/all_count

                metric.reset()
                metric.addBatch(pred.cpu(),labels.cpu())
                mIoU = metric.mIoU()

                print("cur_loss=%6f\t"%(cur_loss.item()),'\t',end='')
                print("cur_acc=%6f\t"%cur_acc.item(),end='')
                print("cur_mIoU=%6f"%mIoU)
                total_acc+=cur_acc.item()
                total_mIoU+=float(mIoU)
                torch.cuda.empty_cache()
            acc.append(total_acc/n)
            totalmIoU.append(total_mIoU/n)
            total_acc,total_mIoU,n=0.0,0.0,0
    print('acc')
    print(acc)
    print('max_acc')
    print(max(acc))

    print('mIoU')
    print(totalmIoU)
    print('maxmIoU')
    print(max(totalmIoU))

    with open("testunet2.txt", "w") as f:
        f.write('acc\n')
        for i in acc:
            f.write(str(i) + '\n')
        f.write('mIoU\n')
        for i in totalmIoU:
            f.write(str(i) + '\n')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=int)

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


    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    time_start=time.time()
    test(config, test_dataloader=test_dataloader,  model=model,loss_fn=loss_fn, device=device)
    time_end = time.time()
    print("time_cost", time_end - time_start, 's')

if __name__ == '__main__':
    main()