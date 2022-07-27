import torch
import torch.nn as nn
from Blocks import AttentionBlock, ResidualAttention
from utils import mel_layer, mfcc_layer
from timm.models import vit_base_patch16_224_in21k

# DeiT transformer
def DeiT(num_classes=1000, pretrained=False, model="small", freeze=False):
    if model == "small":
        deit = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=pretrained)
    elif model == "base":
        deit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=pretrained)
    else:
        raise NotImplemented
    
    if freeze:
        for param in deit.parameters():
            param.requires_grad = False
         
    n_inputs = deit.head.in_features 
    deit.head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes))
    return deit

# Vision Transformer
def ViT(num_classes=1000, pretrained=False, freeze=False):
    vit = vit_base_patch16_224_in21k(pretrained=pretrained)

    if freeze:
        for param in vit.parameters():
            param.requires_grad = False

    n_inputs = vit.head.in_features 
    vit.head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes))
    return vit


# Attention Model
class AttentionModel(nn.Module):
    def __init__(self, attention=True, channelAtt="CBAM", spatialAtt=False, op=None, num_classes=4):
        super(AttentionModel, self).__init__()
        self.mel = mel_layer()
        self.mfcc = mfcc_layer()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)       
        self.ResAtt = ResidualAttention(16,16, attention=attention, channelAtt=channelAtt, op=op)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.ResAtt2 = ResidualAttention(64,64, attention=attention, channelAtt=channelAtt, op=op)

        self.attBlock = AttentionBlock(128, channelAtt=channelAtt, spatialAtt=spatialAtt, op=op)
        #self.conv3 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn3 = nn.BatchNorm2d(128)
        #self.ResAtt2 = ResidualAttention(32,64, op=self.op)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 32)    
        self.fc2 = nn.Linear(32, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x1 = self.mfcc(x)
        x2 = self.mel(x)

        x = self.bn1(self.conv1(x1))
        x = self.maxpool(self.relu(x))
        x2 = self.maxpool(x2)
        x = self.ResAtt(x, x2)
        
        x = self.bn2(self.conv2(x))
        x = self.maxpool(self.relu(x))
        x2 = self.maxpool(x2)
        x = self.ResAtt2(x, x2)

        x = self.attBlock(x)
        #x = self.bn3(self.conv3(x))
        x = self.avgpool(self.relu(x))    
        x = torch.flatten(x, 1)     
        x = self.fc(x)

        x = self.fc2(x)

        return x  

# ViT Cnn Model
class ViTCnn(nn.Module):
    def __init__(self, num_classes, pretrained=True, attention=True, channelAtt="CBAM", spatialAtt=False, freeze=False):
        super(ViTCnn, self).__init__()
        self.att = attention
        self.cnn = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2, bias=False)                                 
        self.attention = AttentionBlock(16, channelAtt=channelAtt, spatialAtt=spatialAtt)
        self.cnn2 = nn.Sequential(nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2, bias=False),
                                  nn.BatchNorm2d(3)   
                                 )
        self.vit = ViT(num_classes=num_classes, pretrained=pretrained, freeze=freeze)

    def forward(self, x):
        out = self.cnn(x)
        att = (self.attention(out) if self.att else out)
        out = self.cnn2(att)
        out = self.vit(out)

        return out

# DeiT Cnn with braches
class DeiTCnn(nn.Module):
    def __init__(self, num_classes, pretrained=True, model="small", freeze=False, attention=True, channelAtt="CBAM", spatialAtt=True):
        super(DeiTCnn, self).__init__()
        self.att = attention
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False) 
        
        self.attBlock = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                      AttentionBlock(16, channelAtt=channelAtt, spatialAtt=spatialAtt),
                                      nn.Conv2d(16,3, kernel_size=3, stride=1, padding=1, bias=False))
    
        self.deit = DeiT(num_classes=num_classes, pretrained=pretrained, model=model, freeze=freeze)

    def forward(self, x):
        x1 = self.conv1(x[:,0,:,:].unsqueeze(dim=1))
        x2 = self.conv1(x[:,1,:,:].unsqueeze(dim=1))
        x3 = self.conv1(x[:,2,:,:].unsqueeze(dim=1))
        out = torch.cat([x1,x2,x3], dim=1)
        att = (self.attBlock(out) if self.att else out)
        out = self.deit(att)

        return out