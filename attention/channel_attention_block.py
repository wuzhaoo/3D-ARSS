from examples.minkunet import MinkUNet34C
import argparse

import torch
import MinkowskiEngine as ME
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size() # b为batch
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        #return x * y.expand_as(x)
        return torch.mul(x,y)
net =SELayer(64)

xyz=torch.randn(10000,3)
feats=torch.ones(xyz.shape[0],3)
labels=torch.randn(10000)
quantization_size=0.1
discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=xyz,
            features=feats,
            labels=labels,
            quantization_size=quantization_size,
            ignore_label=0)
device='cuda'
input = ME.SparseTensor(feats.float().to(device), xyz.to(device))
print(input.shape,type(input))
output =net(input)
print(output.shape)
