import numpy as np 
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSAConvLayer(nn.Module):
    '''Convolutional layer with additional parameters (states and co-states) and the Hamilton-Function'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(MSAConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.hamilton_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.accumulated_m = None
        self.lambda_shape = None
        

    def forward(self, x):
        self.states = nn.Parameter(x)
        x = self.conv(x)
        if self.lambda_shape == None:
            self.lambda_shape = x.shape
        return x

    def dHdx(self, lambda_t_new):
        '''needs reshaping!!! Consider num_channels'''
        hamilton = torch.mm(lambda_t_new.view(1,-1), self.conv(self.states).view(1,-1).transpose(0,1))
        hamilton.backward()
        lambda_t = self.states.grad
        self.states.grad.zero_()
        return lambda_t

    def hamilton_max(self, lambda_t_new):
        shape = list(self.hamilton_conv.parameters())[0].shape
        list(self.hamilton_conv.parameters())[0] = nn.Parameter(lambda_t_new.reshape(self.lambda_shape))
        if self.accumulated_m == None:
            self.accumulated_m = self.hamilton_conv(self.states)
        else:
            self.accumulated_m += self.hamilton_conv(self.states)

        
    def update_weights(self):
        pass


class MSALinearLayer(nn.Module):
    '''Linear layer with additional parameters (states and co-states) and the Hamilton-Function'''
    def __init__(self, in_features, out_features, bias=True):
        super(MSALinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.initialize_weights(in_features*out_features)

    def initialize_weights(self,num_weights):
        weights = torch.rand(1,num_weights)
        shape = self.linear.weight.shape
        weights.reshape_(shape)
        self.linear.weight = nn.Parameter(weights)
        

    def forward(self, x):
        x = self.linear(x)
        return x

    def hamilton(self, x, lambd):
        x_neu = self.linear(x)
        res = torch.dot(x_neu,lambd)
        return res

    def set_weights(self, layer_index, x_dict, lambda_dict, batch_size):
        m = torch.zeros(self.linear.out_features, self.linear.in_features)
        for i in range(batch_size):
            lambda_key = 'lambda_batch'+str(i)+'FC'+str(layer_index+1)
            x_key = 'x_batch'+str(i)+'FC'+str(layer_index)
            m += torch.matmul(lambda_dict.get(lambda_key).reshape(-1,1),x_dict.get(x_key).reshape(1,-1))
        new_weights = nn.Parameter(torch.sign(m))
        self.linear.weight = new_weights