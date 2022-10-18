import torch
import MinkowskiEngine as ME
import numpy as np
import torch.nn as nn
import math
class ChannelAttention(ME.MinkowskiNetwork):
    def __init__(self, in_planes,D,ratio=16,gamma=2,b=1):
        super(ChannelAttention, self).__init__(D)
        self.ratio = ratio
        t = int(abs((math.log2(in_planes) + b) / gamma))
        self.k = t if t % 2 else t + 1
        self.conv1 =ME.MinkowskiConvolution(in_planes,in_planes//2,self.k,dimension=3)
        self.conv2 =ME.MinkowskiConvolution(in_planes,in_planes//2,self.k,dimension=3)
        self.sigmoid = ME.MinkowskiSigmoid()

    def forward(self, x):
        max_pool_out = torch.max(x.F, dim=0)[0].unsqueeze(dim=0)
        avg_pool_out = torch.mean(x.F, dim=0).unsqueeze(dim=0)
        max_pool_out = ME.SparseTensor(
            features=max_pool_out,
            coordinates=torch.Tensor([[0, 0, 0, 0]]).int().cuda()
        )
        avg_pool_out = ME.SparseTensor(
            features=avg_pool_out,
            coordinate_manager=max_pool_out.coordinate_manager,
            coordinate_map_key=max_pool_out.coordinate_map_key
        )
        avg_fc_out = self.conv1(avg_pool_out)
        max_fc_out = self.conv2(max_pool_out)
        out=ME.cat([avg_fc_out, max_fc_out])
        out = self.sigmoid(out)
        # out =x*out
        out_feat = out.F * x.F
        out = ME.SparseTensor(
            features=out_feat,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key
        )
        return out

class SpatialAttention(ME.MinkowskiNetwork):
    def __init__(self, D,kernel_size=7,ratio=16):
        super(SpatialAttention, self).__init__(D)
        self.sigmoid = ME.MinkowskiSigmoid()
        self.conv1 = ME.MinkowskiConvolution(1, ratio, kernel_size=1, dimension=3)
        self.conv2 = ME.MinkowskiConvolution(1, ratio, kernel_size=1, dimension=3)
        self.conv = ME.MinkowskiConvolution(ratio*2,1,kernel_size=kernel_size,dimension=3)
        self.relu =ME.MinkowskiReLU()

    def forward(self, x):
        out1_features = torch.max(x.F, dim=1)[0].unsqueeze(dim=-1)
        out2_features = torch.mean(x.F, dim=1).unsqueeze(dim=-1)
        out1=ME.SparseTensor(
            features=out1_features,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key
        )
        out2 = ME.SparseTensor(
            features=out2_features,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key
        )
        out1=self.relu(self.conv1(out1))
        out2=self.relu(self.conv2(out2))
        out = ME.cat([out1,out2])
        out=self.conv(out)
        out=self.sigmoid(out)
        out=x*out
        return out


###
# datafilename = '../dataset/sequences/00/velodyne/000000.bin'
# labels1_filename= '../dataset/sequences/00/labels/000000.label'
# data0 = np.fromfile(datafilename, dtype=np.float32).reshape(-1, 4)
# labels1 = np.fromfile(labels1_filename, dtype=np.uint32).reshape(-1, 1)
# sem_label1 = torch.from_numpy((labels1 & 0xFFFF) / 1.0)  # semantic label in lower half
#
# coords =torch.from_numpy(data0[:,:3]).int()
# feats = torch.ones(coords.shape[0],1)
# discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
#             coordinates=coords,
#             features=feats,
#             labels=sem_label1,
#             quantization_size=0.1,
#             ignore_label=0,
#             )
# print(coords.shape)
if __name__ == '__main__':

    device='cuda'
    torch.manual_seed(3)
    # #
    N = 1
    coords = torch.randint(1, 100, size=(N, 4)).int()
    feats = torch.rand([N, 32])
    input = ME.SparseTensor(feats.float().to(device), coords.to(device))
    # #
    print('input.C.shape',input.C.shape)
    print('input.F.shape',input.F.shape)
    output = ME.MinkowskiConvolution(32,32,kernel_size=1,dimension=3,dilation=1).to(device)(input)
    # print('output.C.shape',output.C.shape)
    # print('output.F.shape',output.F.shape)

    # global_max_pool = ME.MinkowskiGlobalMaxPooling()
    # global_avg_pool = ME.MinkowskiGlobalAvgPooling()
    # max_pool = ME.MinkowskiMaxPooling(1,2,dimension=3)
    # avg_pool = ME.MinkowskiAvgPooling(1,2,dimension=3)
    # avg = ME.MinkowskiAvgPooling(kernel_size=[16,16,16],stride=[2,2,2],dimension=3)
    # t=ME.MinkowskiToFeature()
    # a=ChannelAttention(32,3).to(device)
    # out=a(input)
    # print(out.F.shape)
    # print(ME.mean(input,out).F.shape)
    # b=SpatialAttention(32,D=3).to(device)
    # print(b(input).F.shape)

    # print(max_pool(input).F==avg_pool(input).F)
    # print('!!!!')
    # print('global_max_pool(input).F.shape', global_max_pool(input).F.shape)
    # print('global_avg_pool(input).F.shape', global_avg_pool(input).F.shape)
    # print('_max_pool(input).F.shape', max_pool(input).F.shape)
    # print('_avg_pool(input).F.shape', avg_pool(input).F.shape)
