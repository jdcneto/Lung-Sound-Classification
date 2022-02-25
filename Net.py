import torch
import torch.nn as nn
import torch.functional as F
from models import*
from Block import ConvBlock, ProjectorBlock, LinearAttentionBlock

sample_rate=32000 
window_size=1024 
hop_size=320
mel_bins=64 
fmin=50 
fmax=14000 
classes_num=527

model = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)

checkpoint = torch.load(r'Cnn14_mAP=0.431.pth')
model.load_state_dict(checkpoint['model'])

for param in model.parameters(): param.requires_grad = False

# for param in model.conv_block1.parameters(): param.requires_grad = True
# for param in model.conv_block2.parameters(): param.requires_grad = True    
# for param in model.conv_block3.parameters(): param.requires_grad = True
# for param in model.conv_block4.parameters(): param.requires_grad = True
for param in model.conv_block5.parameters(): param.requires_grad = True
for param in model.conv_block6.parameters(): param.requires_grad = True

# Without Attention Network
class Cnn(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn, self).__init__()
        
        # Conv Layers
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block1.conv1.weight = model.conv_block1.conv1.weight
        self.conv_block1.bn1.weight   = model.conv_block1.bn1.weight
        self.conv_block1.conv2.weight = model.conv_block1.conv2.weight
        self.conv_block1.bn2.weight   = model.conv_block1.bn2.weight
        
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block2.conv1.weight = model.conv_block2.conv1.weight
        self.conv_block2.bn1.weight   = model.conv_block2.bn1.weight
        self.conv_block2.conv2.weight = model.conv_block2.conv2.weight
        self.conv_block2.bn2.weight   = model.conv_block2.bn2.weight
        
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block3.conv1.weight = model.conv_block3.conv1.weight
        self.conv_block3.bn1.weight   = model.conv_block3.bn2.weight
        self.conv_block3.conv2.weight = model.conv_block3.conv2.weight
        self.conv_block3.bn2.weight   = model.conv_block3.bn2.weight
        
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block4.conv1.weight = model.conv_block4.conv1.weight
        self.conv_block4.bn1.weight   = model.conv_block4.bn2.weight
        self.conv_block4.conv2.weight = model.conv_block4.conv2.weight
        self.conv_block4.bn2.weight   = model.conv_block4.bn2.weight
        
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block5.conv1.weight = model.conv_block5.conv1.weight
        self.conv_block5.bn1.weight   = model.conv_block5.bn2.weight
        self.conv_block5.conv2.weight = model.conv_block5.conv2.weight
        self.conv_block5.bn2.weight   = model.conv_block5.bn2.weight
        
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.conv_block6.conv1.weight = model.conv_block6.conv1.weight
        self.conv_block6.bn1.weight   = model.conv_block6.bn2.weight
        self.conv_block6.conv2.weight = model.conv_block6.conv2.weight
        self.conv_block6.bn2.weight   = model.conv_block6.bn2.weight
        
#         self.fc1 = nn.Linear(2048, 512, bias=True)
#         self.fc2 = nn.Linear(512, 128, bias=True)
#         self.fc_audioset = nn.Linear(128, classes_num, bias=True)
        self.fc1 = nn.Linear(2048, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.fc3 = nn.Linear(512, 128, bias=True)
        self.fc4 = nn.Linear(128, classes_num, bias=True)
                
    def forward(self, x):
        """
        Input: (batch_size, 1, time_steps, mel_bins)
        
        """

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        x = F.dropout(x, p=0.5)
        x = self.fc4(x)
        
        return x
        #return self.softmax(x)

# Attention Network 

class Cnn_Att(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn_Att, self).__init__()

        # Conv Layers
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block1.conv1.weight = model.conv_block1.conv1.weight
        self.conv_block1.bn1.weight   = model.conv_block1.bn1.weight
        self.conv_block1.conv2.weight = model.conv_block1.conv2.weight
        self.conv_block1.bn2.weight   = model.conv_block1.bn2.weight
        
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block2.conv1.weight = model.conv_block2.conv1.weight
        self.conv_block2.bn1.weight   = model.conv_block2.bn1.weight
        self.conv_block2.conv2.weight = model.conv_block2.conv2.weight
        self.conv_block2.bn2.weight   = model.conv_block2.bn2.weight
        
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block3.conv1.weight = model.conv_block3.conv1.weight
        self.conv_block3.bn1.weight   = model.conv_block3.bn2.weight
        self.conv_block3.conv2.weight = model.conv_block3.conv2.weight
        self.conv_block3.bn2.weight   = model.conv_block3.bn2.weight
        
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block4.conv1.weight = model.conv_block4.conv1.weight
        self.conv_block4.bn1.weight   = model.conv_block4.bn2.weight
        self.conv_block4.conv2.weight = model.conv_block4.conv2.weight
        self.conv_block4.bn2.weight   = model.conv_block4.bn2.weight
        
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block5.conv1.weight = model.conv_block5.conv1.weight
        self.conv_block5.bn1.weight   = model.conv_block5.bn2.weight
        self.conv_block5.conv2.weight = model.conv_block5.conv2.weight
        self.conv_block5.bn2.weight   = model.conv_block5.bn2.weight
        
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.conv_block6.conv1.weight = model.conv_block6.conv1.weight
        self.conv_block6.bn1.weight   = model.conv_block6.bn2.weight
        self.conv_block6.conv2.weight = model.conv_block6.conv2.weight
        self.conv_block6.bn2.weight   = model.conv_block6.bn2.weight
        
        self.conv = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, padding=0, bias=True)
        
        # Projectors & Compatibility functions
        self.projector1 = ProjectorBlock(64, 512)
        self.projector2 = ProjectorBlock(256, 512)
        self.projector3 = ProjectorBlock(1024, 512)
         
        self.attn1 = LinearAttentionBlock(in_channels=512)
        self.attn2 = LinearAttentionBlock(in_channels=512)
        self.attn3 = LinearAttentionBlock(in_channels=512)
        
        # full connected layer for classification
                
        self.fc1 = nn.Linear(512*3, 512*2, bias=True)
        self.fc2 = nn.Linear(512*2, 256, bias=True)
        self.fc3 = nn.Linear(256, 128, bias=True)
        self.fc_audioset = nn.Linear(128, classes_num, bias=True)
        
        
    def forward(self, x):
        """
        Input: Input: (batch_size, 1, time_steps, mel_bins)
        
        """

        x1 = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x1, p=0.5)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x2 = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x2, p=0.5)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x3 = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x3, p=0.5)
        x = self.conv_block6 (x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        g = self.conv(x)
       
        g1 = self.attn1(self.projector1(x1), g)
        
        g2 = self.attn1(self.projector2(x2), g)
        
        g3 = self.attn3(self.projector3(x3), g)
        
        g = torch.cat((g1, g2, g3), 1) # batch_sizexC
        
        x = self.fc1(g)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        x = F.dropout(x, p=0.5)
        x = self.fc_audioset(x)
        
        return x
    
class ConvNet(nn.Module):
    def __init__(self, classes_num):
        
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False)
        self.b1  = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv1.weight,nonlinearity='relu')
        
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False)
        self.b2  = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv2.weight,nonlinearity='relu')
        
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False)
        self.b3  = nn.BatchNorm2d(128)
        nn.init.kaiming_normal_(self.conv3.weight,nonlinearity='relu')
        
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False)
        self.b4  = nn.BatchNorm2d(256)
        nn.init.kaiming_normal_(self.conv4.weight,nonlinearity='relu')
        
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False)
        self.b5  = nn.BatchNorm2d(512)
        nn.init.kaiming_normal_(self.conv5.weight,nonlinearity='relu')
        
        self.conv6 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False)
        self.b6  = nn.BatchNorm2d(512)
        nn.init.kaiming_normal_(self.conv6.weight,nonlinearity='relu')
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, classes_num)
        
    def forward(self, x):
        x = self.pool(self.b1(F.relu(self.conv1(x))))
        x = F.dropout(x, p=0.5)
        x = self.pool(self.b2(F.relu(self.conv2(x))))
        x = F.dropout(x, p=0.5)
        x = self.pool(self.b3(F.relu(self.conv3(x))))
        x = F.dropout(x, p=0.5)
        x = self.pool(self.b4(F.relu(self.conv4(x))))
        x = F.dropout(x, p=0.5)
        x = self.pool(self.b5(F.relu(self.conv5(x))))
        x = F.dropout(x, p=0.5)
        x = self.pool(self.b6(F.relu(self.conv6(x))))
        x = F.dropout(x, p=0.5)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        
        return x