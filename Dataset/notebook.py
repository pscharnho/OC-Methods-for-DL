
#%%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timeit

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


#%%
def loadMNIST(root_train, root_test):
    train = MNISTDataset(root_train)
    train_loader = torch.utils.data.DataLoader(train)

    test = MNISTDataset(root_test)
    test_loader = torch.utils.data.DataLoader(test)
    return train_loader, test_loader

print(os.getcwd())
train_loader, test_loader = loadMNIST('Dataset/input/mnist_train.csv', 'Dataset/input/mnist_test.csv')


#%%
class ResNet(nn.Module):
    '''ResNet class with variable size, specified in the constructor'''
    def __init__(self, num_conv, subs_points, num_kernels, num_fc=1, sizes_fc=[1000], res_skips=1):#, subs_style='conv'
        super(ResNet, self).__init__()
        self.num_conv = num_conv
        self.num_fc = num_fc
        self.subs_points = subs_points
        self.res_skips = res_skips
        
        self.conv_keys = ['Conv'+str(i) for i in range(num_conv)]
        self.fc_keys = ['FC'+str(i) for i in range(num_fc)]
        self.subs_keys = ['Conv1x1'+str(i) for i in range(len(subs_points))]
        
        self._init_conv_layers(num_kernels)
        num_pix = num_kernels[-1] * 784 / (4**len(subs_points))
        self._init_fc_layers(sizes_fc, num_pix)
        self._init_conv_subs_layers(num_kernels)
        #keys = ['Conv'+str(i) for i in range(self.num_conv)]
            
        
    def _init_conv_layers(self, num_kernels):
        #keys = ['Conv'+str(i) for i in range(self.num_conv)]
        num_kernel_counter = 0
        current_key = 0
        channels = 1
        for i in range(self.num_conv):
            if i in self.subs_points:
                num_kernel_counter += 1
                setattr(self, self.conv_keys[current_key], nn.Conv2d(channels, num_kernels[num_kernel_counter], 3, stride=2, padding=1))
                channels = num_kernels[num_kernel_counter]
                current_key += 1
            else:
                setattr(self, self.conv_keys[current_key], nn.Conv2d(channels, num_kernels[num_kernel_counter], 3, padding=1))
                channels = num_kernels[num_kernel_counter]
                current_key += 1
                
    def _init_fc_layers(self, sizes_fc, num_pix):
        #keys = ['FC'+str(i) for i in range(num_fc)]
        sizes = [num_pix]+sizes_fc
        for i in range(self.num_fc):
            setattr(self, self.fc_keys[i], nn.Linear(int(sizes[i]), int(sizes[i+1])))
            #print(sizes[i], sizes[i+1])
            
            
    def _init_conv_subs_layers(self, num_kernels):
        '''subsampling of identity via 1x1 convolution'''
        #channels = [1] + num_kernels
        for layer in range(len(self.subs_points)):
            setattr(self, self.subs_keys[layer], nn.Conv2d(num_kernels[layer], num_kernels[layer+1], 1, stride=2))
        
    def forward(self, x):
        '''single res_skip, x_neu = x+Relu(W*x)'''
        #print(x.size())
        for layer in range(self.num_conv):
            if layer in self.subs_points:
                x = self.subsample(x, self.subs_points.index(layer)) + F.relu(getattr(self, self.conv_keys[layer])(x))
                #print(x.size())
            else:
                x = x + F.relu(getattr(self, self.conv_keys[layer])(x))
                #print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        #print(x.size())
        for layer in range(self.num_fc-1):
            x = F.relu(getattr(self, self.fc_keys[layer])(x))
            #print(x.size())
        x = F.softmax(getattr(self, self.fc_keys[self.num_fc-1])(x), dim=1)      
        #print(x.size())
        return x
    
    def subsample(self, x, index):
        x = getattr(self, self.subs_keys[index])(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def train_backprop(self, num_epochs, dataloader, output_frequency=1000):
        tic=timeit.default_timer()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        loss_sum = 0
        for epoch in range(num_epochs):
            for index, (data, target) in enumerate(dataloader):
                output = net(data)
                loss = criterion(output, target)
                loss_sum = loss_sum + loss.data
                if index % (10*(output_frequency)) == 0:
                    print("#  Epoch  #  Batch  #  Avg-Loss ###############")
                if index % (output_frequency) == 0 and index > 0:
                    print("#  %d  #  %d  #  %f  #" % (epoch+1, index, loss_sum/output_frequency))
                    loss_sum = 0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        toc=timeit.default_timer()
        print('Time elapsed: ',toc-tic)
        
    def test(self, dataloader, test_set_size):
        with torch.no_grad():
            correct_pred = 0
            for index, (data, label) in enumerate(dataloader):
                prediction = self.forward(data)
                val, ind = torch.max(prediction,1)
                val_label, ind_label = torch.max(label, 1)
                if ind_label == ind:
                    correct_pred += 1
            print('Test set accuracy: ', correct_pred/test_set_size)


#%%
net = ResNet(3,[1],[3,6],num_fc=2,sizes_fc=[100,10])
print(net)


#%%
net.train_backprop(1,train_loader)


#%%
#net = ResNet(12,[1,5],[8,16,32],num_fc=3,sizes_fc=[1000,100,10])
#print(net)
#print(net.parameters)


#%%
for index, (data,target) in enumerate(train_loader):
    if index == 5:
        print(net(data))


#%%
import timeit
tic=timeit.default_timer()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 1
#print(train.labels.is_cuda)
for epoch in range(num_epochs):
    for index, (data, target) in enumerate(train_loader):
        #print(data)
        output = net(data)
        #print(target)
        #print(output)
        #print(target.is_cuda)
        loss = criterion(output, target)
        #print(loss.data)
        if index % 1000 == 0:
            print(epoch, index, loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
toc=timeit.default_timer()
print('Time elapsed: ',toc-tic)


#%%
with torch.no_grad():
    correct_pred = 0
    for index, (data, label) in enumerate(test_loader):
        prediction = net(data)
        val, ind = torch.max(prediction,1)
        val_label, ind_label = torch.max(label, 1)
        if ind_label == ind:
            correct_pred += 1
    print('Test set accuracy: ', correct_pred/10000)


#%%
class ResNet(nn.Module):
    '''Special test ResNet '''
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


