import torch
import torch.nn as nn
from torch import Tensor
from torchaudio import functional as F
from torch.utils.data import Dataset
import librosa
import torchaudio.transforms as T
import numpy as np

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
    
""" class SelectSamples(Dataset):
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
        
        return x, y """

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

class addChannel(object):
    def __init__(self):
        self.channels
       
    def __cal__(self,x):
        x = x.unsqueeze(dim=0)
        return x
    
    def __repr__(self):
        return print("Add two channel in a gray scaled imaged")
    
# extracting mel features     
class Mel(object):
    def __init__(self, sr = 22050, n_mels=96):
        self.sr = sr
        self.n_mels = n_mels
        super(Mel, self).__init__()
       
    def __cal__(self,x):
        return librosa.feature.melspectrogram(x, sr=self.sr, n_mels=self.n_mels, fmax=8000)    

    def __repr__(self):
        return print("Mel Features")
    
# decibel scale
class dbScale(object):
    def __init__(self):
        super(dbScale, self).__init__()
       
    def __cal__(self,x):
        return librosa.power_to_db(x,) 
    
    def __repr__(self):
        return print("Decibel Scale")

# extracting mfcc features
class MFCC(object):
    def __init__(self, sr=22050, n_mfcc=96):
        self.sr = sr
        self.n_mfcc = n_mfcc
        
    def __call__(self, x):
        x = librosa.feature.mfcc(x, sr=self.sr, n_mfcc=self.n_mfcc, hop_length=512, htk=True)
        x = torch.from_numpy(x)
        return x.unsqueeze(dim=0)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
        
# extracting mfcc features
class MakeImage(object):
    def __init__(self, sr=22050, n_bins=96):
        self.sr = sr
        self.n_bins = n_bins
        
    def __call__(self, x):
        mfcc = librosa.power_to_db(librosa.feature.mfcc(x, sr=self.sr, n_mfcc=self.n_bins, hop_length=512, htk=True),ref=np.max)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(x, sr=self.sr, n_mels=self.n_bins),ref=np.max)
        cqt_abs = np.abs(librosa.cqt(x, sr=self.sr, hop_length=512, n_bins=self.n_bins))
        cqt = librosa.amplitude_to_db(cqt_abs,ref=np.max)
        image = np.stack((mfcc, mel, cqt), axis=0)

        return torch.from_numpy(image)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


def mel_layer(n_mels=96, sample_rate=22050, f_min=0, f_max=8000, n_fft=2048, win_length=2048, hop_length=512):
    """extract Mel Spectrogram features"""
    return T.MelSpectrogram(sample_rate=sample_rate,n_fft=n_fft,win_length=win_length,hop_length=hop_length,
    center=True,pad_mode="reflect",power=2.0,norm="slaney",onesided=True,n_mels=n_mels,mel_scale="htk",)


def mfcc_layer(n_mfcc=96, n_mels=96, sample_rate=22050, f_min=0, f_max=8000, n_fft=2048, win_length=2048, hop_length=512):
    """Extract MFCC features"""
    return T.MFCC(sample_rate=sample_rate,n_mfcc=n_mfcc,melkwargs={"n_fft": n_fft,"n_mels": n_mels,"hop_length": hop_length,
        "mel_scale": "htk",},)