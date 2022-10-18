import torch
import MinkowskiEngine as ME
import torch.nn as nn
class SENet(ME.MinkowskiNetwork):
    def __init__(self, channel, D,reduction=4):
        super(SENet, self).__init__(D)
        self.avg_pool = ME.MinkowskiAvgPooling(kernel_size=1,dimension=3)
        self.channel_sxcitation = nn.Sequential(ME.MinkowskiLinear(channel,int(channel//reduction)),
                                                ME.MinkowskiReLU(),
                                                ME.MinkowskiLinear(int(channel//reduction),channel),
                                                ME.MinkowskiSigmoid()
                                                )
    def forward(self, x):
        x = self.avg_pool(x)
        out = self.channel_sxcitation(x)
        return x*out

coords = torch.randint(1,200,size=(120,4)).int()
print(coords)
feats=torch.ones(120,256)
device='cuda'
input = ME.SparseTensor(feats.float().to(device), coords.to(device))

print('input.C.shape',input.C.shape)

print('input.F.shape',input.F.shape)
model  = SENet(256,3).to(device)

output = model(input)
print('output.F.shape',output.F.shape)