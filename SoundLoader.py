import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import os

"""
--------------------------------------------------
             Sound Dataset 
--------------------------------------------------    
"""
class LungSoundDS(Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.data_path = data_path
        self.transform = transform
        self.sr = 22050
        self.n_bins = 96
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
    def __len__(self):
        return len(self.df)    

  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
    def __getitem__(self, idx):
        # File path of the audio file 
        # full path
        audio_path = os.path.join(self.data_path,self.df.filename.iloc[idx])

        # Get the Class ID
        class_id = self.df.id.iloc[idx]
            
        # Load an audio file. Return the signal and the sample rate
        x, _ = librosa.load(audio_path)

        mfcc = librosa.power_to_db(librosa.feature.mfcc(x, sr=self.sr, n_mfcc=self.n_bins, hop_length=512, htk=True),ref=np.max)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(x, sr=self.sr, n_mels=self.n_bins),ref=np.max)
        cqt_abs = np.abs(librosa.cqt(x, sr=self.sr, hop_length=512, n_bins=self.n_bins))
        cqt = librosa.amplitude_to_db(cqt_abs,ref=np.max)
        image = np.stack((mfcc, mel, cqt), axis=0)

        image = torch.from_numpy(image)

        # Applying Transformations
        if self.transform:
            image = self.transform(image)
                
        return image, class_id