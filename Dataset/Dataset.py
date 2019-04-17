import numpy as np 
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.datasets import make_moons



class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.len = len(self.data['label'].values)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        self.features = torch.tensor(self.data.drop('label',axis=1).values.astype(np.float32), device=self.device).reshape(-1,1,28,28)/255   #.reshape(-1,784)/255 
        #self.labels = self.one_hot(self.data['label'].values,10)
        self.labels = self.data['label']
    
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
    train_loader = torch.utils.data.DataLoader(train, batch_size=100)

    test = MNISTDataset(root_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=100)
    return train_loader, test_loader


class MoonsDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples):
        self.n_samples = n_samples
        features, labels = make_moons(n_samples=self.n_samples, noise=0.1)

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels)#, dtype=torch.float32

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def makeMoonsDataset(dataset_size, batch_size):
    torch.manual_seed(0)
    #dataset_size = 1000
    dataset = MoonsDataset(dataset_size)
    #batch_size = 100
    test_split = .2
    shuffle_dataset = True
    random_seed= 42
    torch.manual_seed(42)

    # Creating data indices for training and validation splits:
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)
    return train_loader, test_loader
