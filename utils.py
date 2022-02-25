import torch
from torch import Tensor
from torchaudio import functional as F
from torch.utils.data import Dataset
import librosa

class Compose:
    '''
    Data augmentation module that transforms any given data 
    example with a chain of audio augmentations.
    '''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        x = self.transform(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "\t{0}".format(t)
        format_string += "\n)"
        return format_string

    def transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    
class DataFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
class SelectSamples(Dataset):
    def __init__(self, subset, samples=None, transform=None):
        '''
        samples must be a list of labels which some transformation
        will be applied
        '''
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if y in samples:
            x = self.transform(x)
        else:
            pass
        
        return x, y

class SpecPermute(torch.nn.Module):
    def __init__(self):
        super(SpecPermute, self).__init__()
        
    def forward(self, x):
        
        return torch.permute(x, (0, 2, 1))
    
class Permute(torch.nn.Module):
    def __init__(self):
        super(Permute, self).__init__()
        
    def forward(self, x):
        
        return torch.permute(x, (1, 0, 2))

class addChannel(torch.nn.Module):
    def __init__(self):
        super(addChannel, self).__init__()
       
    def forward(self,x):
        x = x.expand(3,-1,-1)
        return x

class dbScale(torch.nn.Module):
    def __init__(self):
        super(dbScale, self).__init__()
       
    def forward(self,x):
        return librosa.power_to_db(x)
