import numpy as np 
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSAConvLayer(nn.Module):
    '''Convolutional layer with additional parameters (states and co-states) and the Hamilton-Function'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        

    def forward(self, x):
        self.states = nn.Parameter(x)
        x = self.conv(x)
        return x

    def dhamiltondx(self, lambda_t_new):
        '''needs reshaping!!!'''
        hamilton = torch.mm(lambda_t_new.view(1,-1), self.conv(self.states).view(1,-1).transpose(0,1))
        hamilton.backward()
        lambda_t = self.states.grad
        self.states.grad.zero_()
        return lambda_t

        



