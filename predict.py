# from Unet1 import MyMinkUNet
# from Unet2 import MyMinkUNet
from Unet3 import MyMinkUNet
# from Unet4 import MyMinkUNet
import torch
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
'''
DATA_PATH
model
pth
'''
class TestDataset(Dataset):
    def __init__(self, quantization_size=0.1, DATA_PATH="./predictfilename/predict21.txt"):  # quantization_size=0.02
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

        data_filename = cur_filename

        data = np.fromfile(data_filename, dtype=np.float32).reshape(-1, 4)

        xyz = torch.from_numpy(data[:, :3])


        input = xyz
        feats = torch.ones(input.shape[0], 1)

        discrete_coords,unique_feats,index,inverse= ME.utils.sparse_quantize(
            coordinates=input,
            features=feats,
            quantization_size=self.quantization_size,
            ignore_label=0,
            return_index=True,
            return_inverse=True
        )
        # print(index)
        # print(inverse)
        # print(len(index))
        # print(len(inverse))

        return discrete_coords, unique_feats,inverse


def test(test_dataloader,model,device):
    with torch.no_grad():
        weight_dict = torch.load('predictpth/epoch 791.pt')
        # weight_dict = torch.load('checkpoints_unet1_lr1e-4/epoch 100.pt')
        model.load_state_dict(weight_dict)
        for batch,data in enumerate(test_dataloader):
            print(batch)
            coords, feats ,inverse= data
            pred_source=[0]*len(inverse)
            input = ME.SparseTensor(features=feats.to(device), coordinates=coords.to(device))

            output = model(input)
            _,pred=torch.max(output.F,axis=1)
            # print(f'Batch: {int((batch))}')
            pred=pred.cpu().numpy()
            ###
            for p in range(len(inverse)):
                pred_source[p] = pred[inverse[p]]
            # print(pred_source)
            print("len(pred_source)",len(pred_source))
            # print(pred)
            print("len(pred)",len(pred))

            ##
            pred=np.array(pred_source).astype('uint32')
            ##
            for i in range(len(pred)):
                pred[i]=labelname.LEARNING_MAP_INV[pred[i]]
            print(pred.dtype)

            pred.tofile("method_predictions/sequences/21/predictions/%06d.label" % batch)
            print(pred)
            print(len(pred))
            # with open("method_predictions/sequences/11/predictions/%06d.label" % batch, "w") as f:
            #     for i in range(len(pred)):
            #         # pred[i]=labelname.LEARNING_MAP_INV[pred[i]]
            #         f.write((pred[i].astype('int32')))
            #         # if i!=len(pred)-1:
            #         #     f.write('\n')

            # break




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)

    config = parser.parse_args()

    test_dataset = TestDataset()
    # test_dataset = TestDataset()

    test_dataloader = DataLoader(test_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=ME.utils.batch_sparse_collate,
                                  num_workers=1)

    device = torch.device('cuda')

    model = MyMinkUNet(
        in_channels=1,
        out_channels=20,
        D=3
    ).to(device)



    time_start=time.time()
    test( test_dataloader=test_dataloader,  model=model, device=device)
    time_end = time.time()

    print("time_cost", time_end - time_start, 's')

if __name__ == '__main__':
    main()