import torch
import MinkowskiEngine as ME
import torch.nn as nn

class conv_block(ME.MinkowskiNetwork):
    def __init__(self, ch_in, ch_out,D):
        super(conv_block, self).__init__(D)
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(ch_in, ch_out, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiBatchNorm(ch_out),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(ch_out, ch_out, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiBatchNorm(ch_out),
            ME.MinkowskiReLU()
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class SqueezeAttentionBlock(ME.MinkowskiNetwork):
    def __init__(self, ch_in, ch_out,D):
        super(SqueezeAttentionBlock, self).__init__(D)
        self.avg_pool = ME.MinkowskiAvgPooling(kernel_size=2,stride=2,dimension=D)
        self.conv = conv_block(ch_in,ch_out,D)
        self.conv_atten =  conv_block(ch_in,ch_out,D)
        self.upsample = ME.MinkowskiConvolutionTranspose(ch_out,ch_out,kernel_size=1,dimension=D)

    def forward(self, x):
        # print(x.shape)
        x_res = self.conv(x)
        # print(x_res.shape)
        y = self.avg_pool(x)
        # print(y.shape)
        y = self.conv_atten(y)
        # print(y.shape)
        y = self.upsample(y)
        # print(y.shape, x_res.shape)
        return (y * x_res) + y
coords = torch.randint(1,200,size=(100,4)).int()
print(coords)
feats=torch.ones(100,256)
device='cuda'
input = ME.SparseTensor(feats.float().to(device), coords.to(device))

print('input.C.shape',input.C.shape)

print('input.F.shape',input.F.shape)
model1  = SqueezeAttentionBlock(256,20,D=3).to(device)

output = model1(input)
print(output.shape)