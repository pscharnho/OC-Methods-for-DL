
import numpy as np 
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timeit



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
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        loss_sum = 0
        for epoch in range(num_epochs):
            for index, (data, target) in enumerate(dataloader):
                output = self.forward(data)
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
            for _, (data, label) in enumerate(dataloader):
                prediction = self.forward(data)
                _, ind = torch.max(prediction,1)
                _, ind_label = torch.max(label, 1)
                if ind_label == ind:
                    correct_pred += 1
            print('Test set accuracy: ', correct_pred/test_set_size)