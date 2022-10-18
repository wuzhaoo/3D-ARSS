import torch
import MinkowskiEngine as ME
import numpy as np
import torch.nn as nn
import math
class SpatialAttention(ME.MinkowskiNetwork):
    def __init__(self, D):
        super(SpatialAttention, self).__init__(D)
        self.sigmoid = ME.MinkowskiSigmoid()
        self.conv = ME.MinkowskiConvolution(2,1,kernel_size=7,dimension=D)

    def forward(self, x):
        max_pool_out = torch.max(x.F, dim=1)[0].unsqueeze(dim=-1)
        avg_pool_out = torch.mean(x.F, dim=1).unsqueeze(dim=-1)
        max_pool_out=ME.SparseTensor(
            features=max_pool_out,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key
        )
        avg_pool_out = ME.SparseTensor(
            features=avg_pool_out,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key
        )

        out = ME.cat([max_pool_out,avg_pool_out])
        out=self.conv(out)
        out=self.sigmoid(out)
        out=x*out
        return out

class ChannelAttention(ME.MinkowskiNetwork):
    def __init__(self, in_planes,D,gamma=2,b=1):
        super(ChannelAttention, self).__init__(D)
        t = int(abs((math.log2(in_planes) + b) / gamma))
        self.k = t if t % 2 else t + 1
        self.conv1 =ME.MinkowskiConvolution(in_planes,in_planes,self.k,dimension=D)
        self.conv2 =ME.MinkowskiConvolution(in_planes,in_planes,self.k,dimension=D)
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
        avg_conv_out = self.conv1(avg_pool_out)
        max_conv_out = self.conv2(max_pool_out)
        out=avg_conv_out+max_conv_out
        out = self.sigmoid(out)
        out_feat = out.F * x.F
        out = ME.SparseTensor(
            features=out_feat,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key
        )
        return out