from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import numpy as np

def get_accuracy(predictions,labels):
    return np.sum(np.equal(labels, predictions)) / len(labels)


def get_project_root():
    """
    gets root directory of project
    :return: root (str): filepath of project's root directory
    """
    root = Path(__file__).parent
    return root


class GTZANDerivedDataDataset(Dataset):
    def __init__(self, genre_label_file_path, audio_data_file_path): # genre_label_file,  transform=None, target_transform=None):
        # classes
        self.classes = ["0 - blues", "1 - classical", "2 - country", "3 - disco", "4 - hiphop", "5 - jazz",
                        "6 - metal", "7 - pop", "8 - reggae", "9 - rock", ]
        # classes dict
        self.class_to_idx = {}
        for i in range(len(self.classes)):
            self.class_to_idx[self.classes[i]] = i

        self.data_file_path = audio_data_file_path
        self.data = torch.tensor(pd.read_csv(audio_data_file_path, header=None).to_numpy()).to(torch.float32)
        self.targets = torch.tensor(pd.read_csv(genre_label_file_path, header=None).to_numpy())
        self.targets = torch.reshape(self.targets,(-1,)).to(torch.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        row = self.data[idx,:]
        label = self.targets[idx]
        return row, label