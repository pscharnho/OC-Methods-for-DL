import numpy as np # linear algebra
import pandas as pd
import os
print(os.listdir("../input"))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms



class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.len = len(self.data['label'].values)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        self.features = torch.tensor(self.data.drop('label',axis=1).values.astype(np.float32), device=self.device).reshape(-1,1,28,28)/255
        self.labels = self.one_hot(self.data['label'].values,10)
    
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def one_hot(self, labels, num_classes):
        result = torch.zeros([len(labels), num_classes], dtype=torch.float32, device=self.device)
        for index, label in enumerate(labels):
            result[index][label]=1
        return result


def loadMNIST(root_train, root_test):
    train = MNISTDataset(root_train)
    train_loader = torch.utils.data.DataLoader(train)

    test = MNISTDataset(root_test)
    test_loader = torch.utils.data.DataLoader(test)
    return train_loader, test_loader