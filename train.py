# from Unet1 import MyMinkUNet
from Unet2 import MyMinkUNet
# from Unet3 import MyMinkUNet
# from Unet4 import MyMinkUNet
# PATH_pth="./checkpoints_unet2/epoch %d.pt"
PATH_pth="./tmp/epoch %d.pt"

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

"""
net
pth_save_dir
max_epoch
lr_scheduler
weight_dict
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
from tensorboardX import SummaryWriter
GRID_SIZE_R=[20, 20, 20]

def transforms(points):
    #input points(N,3)
    tx = np.random.uniform(-2, 2)
    ty = np.random.uniform(-2, 2)
    tz = np.random.uniform(-2, 2)
    if random.random()>0.5:
        # anglex = np.random.uniform(-np.pi/18,np.pi/18)
        # angley = np.random.uniform(-np.pi/18,np.pi/18)
        anglez = np.random.uniform(-np.pi,np.pi)

        points = point_transform(points,tx,ty,tz,rz=anglez)
    else:
        factor = np.random.uniform(0.95,1.05)
        points = points*factor
    return points

def point_transform(points,tx,ty,tz,rx=0,ry=0,rz=0):
    #input points(N,3)

    N=points.shape[0]
    points= np.hstack([points,np.ones((N,1))])
    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)

    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)

    if rz!=0:
        mat = np.zeros((4,4))
        mat[2,2]=1
        mat[3,3]=1
        mat[0,0]=np.cos(rz)
        mat[0,1]=-np.sin(rz)
        mat[1,0]=np.sin(rz)
        mat[1,1]=np.cos(rz)
        points= np.matmul(points,mat)
        mat1=np.eye(4)
        mat1[3,0:3] = tx,ty,tz
        points = np.matmul(points,mat1)
    return points[:,0:3]

class TrainDataset(Dataset):
    def __init__(self,quantization_size=0.1,DATA_PATH="./config_train.txt"):#quantization_size=0.02
        self.DATA_PATH=DATA_PATH
        self.quantization_size=quantization_size
        with open(DATA_PATH, "r") as f:
            self.all_data_name = f.read()
        self.all_data_name = self.all_data_name.split('\n')

    def __len__(self):
        self.len=len(self.all_data_name)
        return self.len

    def __getitem__(self, index):
        cur_filename=self.all_data_name[index]
        sequence = cur_filename.split('.')[1].split('/')[-3]
        count = cur_filename.split('.')[1].split('/')[-1]

        data1_filename = cur_filename
        labels1_filename = './dataset/sequences/' + sequence + '/labels/' + '%06d.label' % (int(count))

        data1 = np.fromfile(data1_filename, dtype=np.float32).reshape(-1, 4)
        labels1 = np.fromfile(labels1_filename, dtype=np.uint32).reshape(-1, 1)

        # the lower 16 bits correspond to the label. The upper 16 bits encode the instance id
        sem_label1 = torch.from_numpy((labels1 & 0xFFFF) / 1.0) # semantic label in lower half

        # xyz1 = torch.from_numpy(data1[:, :3])#(N,3)
        xyz1 = (data1[:, :3])

        # print('\n',data0_filename,'\n', data1_filename,
        #       '\n',data2_filename)

        for i in range(len(labels1)):
            sem_label1[i]=labelname.VALID_CLASS_LEARNING_MAP[int(sem_label1[i])]
####
        bound_x = np.logical_and(
            xyz1[:,0] >= -GRID_SIZE_R[0], xyz1[:, 0] < GRID_SIZE_R[0]
        )
        bound_y = np.logical_and(
            xyz1[:, 1] >= -GRID_SIZE_R[1], xyz1[:, 1] < GRID_SIZE_R[1]
        )
        bound_z = np.logical_and(
            xyz1[:, 2] >= -GRID_SIZE_R[2], xyz1[:, 2] < GRID_SIZE_R[2]
        )
        bound_box = np.logical_and(np.logical_and(bound_x,bound_y),bound_z)

        xyz1 =transforms(xyz1)
        input = xyz1[bound_box]



        feats=torch.ones(input.shape[0],1)
        labels=sem_label1[bound_box]
####
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=input,
            features=feats,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=0,
            )

        return discrete_coords, unique_feats, unique_labels

class ValidateDataset(Dataset):
    def __init__(self, quantization_size=0.1, DATA_PATH="./config_validate.txt"):  # quantization_size=0.02
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
        labels = sem_label1[bound_box]


        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=input,
            features=feats,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=0,
        )

        return discrete_coords, unique_feats, unique_labels

def train(config,train_dataloader,validate_dataloader,model,loss_fn,optimizer,scheduler,device):
    writer = SummaryWriter(log_dir='logs',flush_secs=60)
    loss,current,n = 0.0,0.0,0
    Testloss,Testcurrent,t = 0.0,0.0,0
    TestmIoU=0
    time_start=time.time()
    metric = SegmentationMetric(20)
    for epoch in range(config.max_epochs):
        print(f"\nEpoch: {epoch}\n-------------------------------")
        for batch,data in enumerate(train_dataloader):
            print(f'Batch: {int((batch))}')

            model.train()

            coords, feats, labels = data

            input = ME.SparseTensor(feats.float().to(device), coords.to(device))

            labels=labels.to(device)
            ###
            optimizer.zero_grad()
            output = model(input)

            # del input
            # torch.cuda.empty_cache()

            cur_loss=loss_fn(output.F, labels.long().squeeze().to(device))

            cur_loss.backward()
            optimizer.step()
            ###
            # output = model(input)
            # cur_loss=loss_fn(output.F, labels.long().squeeze().to(device))/config.accumulate_step
            # cur_loss.backward()
            # if (batch+1) %config.accumulate_step==0:
            #     optimizer.step()
            #     optimizer.zero_grad()

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
            writer.add_scalar('Train_loss',cur_loss.item(),n)
            writer.add_scalar('Train_Accuracy',cur_acc,n)
            writer.add_scalar('Train_mIoU',mIoU,n)
            # writer.add_scalars('data/scalar_group',{'Train_mIoU':mIoU,
            #                                        'Train_Accuracy':cur_acc,
            #                                        'Train_loss':cur_loss},n)


            n +=1
            # loss += cur_loss.item()
            # current += cur_acc.item()
            ####################
            torch.cuda.empty_cache()
        torch.save(model.state_dict(), PATH_pth % (epoch))

        #测试循环
        with torch.no_grad():
            for batch,data in enumerate(validate_dataloader):
                print(f'\n--------------Epoch: {epoch}--------------')
                print(f'Batch: {batch}')
                model.eval()
                coords, feats, labels = data

                input = ME.SparseTensor(feats.float().to(device), coords.to(device))

                output = model(input)
                _,pred=torch.max(output.F,axis=1)
                labels = labels.to(device)

                cur_loss = loss_fn(output.F, labels.long().squeeze().to(device))
                cur_acc = torch.sum(labels.long().squeeze() == pred) / output.F.shape[0]
                t+=1

                metric.reset()
                metric.addBatch(pred.cpu(), labels.cpu())
                mIoU = metric.mIoU()


                Testloss += cur_loss.item()
                Testcurrent += cur_acc.item()
                TestmIoU+=mIoU
                print("Test_cur_loss",cur_loss.item())
                print('Test_cur_acc',cur_acc)
                print('Test_mIoU',mIoU)
                torch.cuda.empty_cache()
        # scheduler.step()
        writer.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        writer.add_scalar('Test_loss', Testloss/(batch+1), epoch)
        writer.add_scalar('Test_Accuracy', Testcurrent/(batch+1), epoch)
        writer.add_scalar('Test_mIoU', TestmIoU/(batch+1), epoch)
        Testloss, Testcurrent, TestmIoU = 0.0, 0.0, 0.0
        # writer.add_scalars('data/scalar_group_test', {'Test_mIoU': TestmIoU/t,
        #                                          'Test_Accuracy': Testcurrent/t,
        #                                          'Test_loss': Testloss/t}, epoch)

        ######################
    time_end = time.time()
    print("time_cost", time_end - time_start, 's')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)


    config = parser.parse_args()

    train_dataset = TrainDataset()
    validate_dataset =ValidateDataset()

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=ME.utils.batch_sparse_collate,
                                  num_workers=4)
    validate_dataloader = DataLoader(validate_dataset,
                                     batch_size=config.batch_size,
                                     collate_fn=ME.utils.batch_sparse_collate,
                                     num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MyMinkUNet(
        in_channels=1,
        out_channels=20,
        D=3
    ).to(device)

    weight_dict = torch.load("./tmp/epoch 86.pt")
    model.load_state_dict(weight_dict)


    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    optimizer =  torch.optim.Adam(model.parameters(),
                                  lr=config.lr,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=config.weight_decay)

    milestones=[100,150]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    train(config, train_dataloader=train_dataloader, validate_dataloader=validate_dataloader, model=model, loss_fn=loss_fn,
          optimizer=optimizer,scheduler=scheduler,device=device)

if __name__ == '__main__':
    main()