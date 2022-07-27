import torch
import torch.nn as nn
import torch.nn.functional as F

"""Implementation of some channel attention blocks such as:
SEBlock from squeeze and excitation in https://arxiv.org/pdf/1709.01507v4.pdf
CBAMBlock from Convolution Block Attention Module in https://arxiv.org/pdf/1807.06521.pdf
ECABlock from Efficient Channel Attention for Deep Convolutional Neural Networks in https://arxiv.org/pdf/1910.03151.pdf

Obs: The last one have some changes from the original.
"""  

## channel attention from squeeze and excitation paper https://arxiv.org/pdf/1709.01507v4.pdf   
class SEBlock(nn.Module):
    def __init__(self, inplanes, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(inplanes, inplanes//reduction_ratio, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(inplanes//reduction_ratio, inplanes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x*y
              
## channel attention from cbam paper https://arxiv.org/pdf/1807.06521.pdf
class CBAMBlock(nn.Module):
    def __init__(self, inplanes, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Flatten(),
                   nn.Linear(inplanes, inplanes//reduction_ratio),
                   nn.ReLU(inplace=False),
                   nn.Linear(inplanes//reduction_ratio, inplanes))
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        avg_pool = self.avg_pool(x) # same in squeeze op
        max_pool = self.max_pool(x)

        channel_att1 = self.mlp(avg_pool)
        channel_att2 = self.mlp(max_pool)

        channel_att_sum = channel_att1 + channel_att2

        scale = self.activation(channel_att_sum).unsqueeze(2).unsqueeze(3)

        return x*scale
 
## modified channel attention from ECA-NET paper https://arxiv.org/pdf/1910.03151.pdf  
class ECABlock(nn.Module):
    """Constructs a ECA module.
    Args:
    k_size: Size of convolutional filter use (3,3) as default
    
    op : Operation type for feature concatenation, use "sum" 
    or "mult" for sum or multiplication.
    """
    def __init__(self, k_size=3, op=None):
        super(ECABlock, self).__init__()
        self.op = op
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)

        # Two different branches of ECA module
        out1 = self.conv(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out2 = self.conv(y2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)   
        
        # Multi-scale information fusion
        if self.op == "sum":
            out = out1+out2
        elif self.op == "mult":
            out = out1*out2
        else:
            out = out1

        # Normalization 
        out = self.activation(out)

        return x*out
        
## spatial blocks
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        
class BasicConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
        groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(planes,eps=1e-5, momentum=0.01, affine=True) 
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialBlock(nn.Module):
    def __init__(self):
        super(SpatialBlock, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.activation(x_out) # broadcasting
        return x*scale
        
## Attention Block for Residue Module        
class AttentionBlock(nn.Module):
    def __init__(self, inplanes, channelAtt = "CBAM", spatialAtt=True, op=None):
        super(AttentionBlock, self).__init__()
        self.op = op
        self.spatialAtt = spatialAtt
        self.spatial = SpatialBlock()

        # choosing channel attention
        if channelAtt == "SE":
            self.channel = SEBlock(inplanes)
        elif channelAtt == "CBAM":
            self.channel = CBAMBlock(inplanes)
        elif channelAtt == "ECA":
            self.channel = ECABlock(op=self.op)
        else:
            raise NotImplemented
        
    def forward(self, x):
        out = (self.spatial(x) if self.spatialAtt else x)
        out = self.channel(out)
        
        return out

## Residue Module        
class ResidualAttention(nn.Module):
    def __init__(self, inplanes, planes, attention=True, channelAtt="CBAM", op=None):
        super(ResidualAttention, self).__init__()
        self.residue = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                       nn.BatchNorm2d(planes),)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.01, affine=True)
        # add attention
        if attention:
            self.att  = AttentionBlock(planes, channelAtt=channelAtt, op=op)
        else:
            self.att = None

    def forward(self,x):
        residue = self.residue(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.att(out)
        out = torch.cat([out,residue], dim=1)
        out = self.relu(out)

        return out