import numpy as np 
import pandas as pd
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit



class ReluLayer(nn.Module):
    def __init__(self):
        super(ReluLayer, self).__init__()
        self.name = 'Relu'

    def forward(self, x):
        x = F.relu(x)
        return x

    def hamilton(self, x, lambd):
        x_neu = self.forward(x)
        res = torch.dot(x_neu,lambd)
        return res


class TanhLayer(nn.Module):
    def __init__(self):
        super(TanhLayer, self).__init__()
        self.name = 'Tanh'

    def forward(self, x):
        x = torch.tanh(x)
        return x

    def hamilton(self, x, lambd):
        x_neu = self.forward(x)
        res = torch.dot(x_neu,lambd)
        return res


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
    def __init__(self, in_features, out_features, bias=True, test=False):
        super(MSALinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.has_bias = bias
        self.initialize_weights(in_features,out_features, test)
        self.name = 'Linear'
        self.rho = 0.5
        self.ema_alpha = 0.99
        

    def initialize_weights(self,in_features,out_features, test):
        if test:
            weights = np.ones((out_features,in_features))
                
            for i in range(out_features):
                for j in range(in_features):
                    if (i+j)%2 == 0:
                        weights[i][j] = -1
            self.linear.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
            self.m_accumulated = 0.001 * self.linear.weight.clone().detach()

            if self.has_bias:
                biases = np.ones(out_features)
                for i in range(out_features):
                    if i%2==0:
                        biases[i] = -1
                self.linear.bias = nn.Parameter(torch.tensor(biases, dtype=torch.float32))
                self.m2_accumulated = 0.001 * self.linear.bias.clone().detach()

        else:
            weights = nn.Parameter(torch.sign(self.linear.weight))
            self.linear.weight = nn.Parameter(weights)
            self.m_accumulated = 0.001 * self.linear.weight.clone().detach()
            if self.has_bias:
                biases = nn.Parameter(torch.sign(self.linear.bias))
                self.linear.bias = nn.Parameter(biases)
                self.m2_accumulated = 0.001 * self.linear.bias.clone().detach()
            #print(self.linear.weight)
        

    def forward(self, x):
        x = self.linear(x)
        return x

    def hamilton(self, x, lambd):
        x_neu = self.forward(x)
        #print(x_neu)
        res = torch.dot(x_neu,lambd)
        return res

    def set_weights(self, layer_index, x_dict, lambda_dict, batch_size):
        m = torch.zeros(self.linear.out_features, self.linear.in_features, dtype=torch.float32)#
        for i in range(batch_size):
            lambda_key = 'lambda_batch'+str(i)+'FC'+str(layer_index+1)
            x_key = 'x_batch'+str(i)+'FC'+str(layer_index)
            #print(lambda_dict.get(lambda_key))
            m += torch.matmul(lambda_dict.get(lambda_key).reshape(-1,1),x_dict.get(x_key).reshape(1,-1))
        #m = (1 / batch_size) * m
        self.m_accumulated = self.ema_alpha * self.m_accumulated + ( 1 - self.ema_alpha ) * m
        old_weight_signs = self.linear.weight
        new_weight_signs = torch.sign(self.m_accumulated)
        compared_weight_signs = torch.tensor(torch.ne(new_weight_signs, old_weight_signs).detach(), dtype=torch.float32)
        reduced_m = torch.abs(compared_weight_signs * self.m_accumulated)
        
        max_weight_elem = torch.max(reduced_m)
        
        new_weights = new_weight_signs*torch.tensor(torch.ge(torch.abs(self.m_accumulated),self.rho*max_weight_elem*torch.ones_like(self.m_accumulated)).detach(),dtype=torch.float32)
        zero_weight_indices = (new_weights == 0).nonzero()

        for index in zero_weight_indices:
            new_weights[index[0]][index[1]] = old_weight_signs[index[0]][index[1]]

        new_matr = nn.Parameter(new_weights)
        self.linear.weight = new_matr

        if self.has_bias:
            m2 = torch.zeros(self.linear.out_features, dtype=torch.float32)
            for i in range(batch_size):
                lambda_key = 'lambda_batch'+str(i)+'FC'+str(layer_index+1)
                m2 += lambda_dict.get(lambda_key)
            #m2 = (1 / batch_size) * m2
            self.m2_accumulated = self.ema_alpha * self.m2_accumulated + ( 1 - self.ema_alpha ) * m2
            old_bias_signs = self.linear.bias
            new_bias_signs = torch.sign(self.m2_accumulated)
            compared_bias_signs = torch.tensor(torch.ne(new_bias_signs, old_bias_signs).detach(), dtype=torch.float32)
            reduced_m2 = torch.abs(compared_bias_signs * self.m2_accumulated)
            max_bias_elem = torch.max(reduced_m2)
            new_biases = new_bias_signs*torch.tensor(torch.ge(torch.abs(self.m2_accumulated),self.rho*max_bias_elem*torch.ones_like(self.m2_accumulated)).detach(),dtype=torch.float32)
            zero_bias_indices = (new_biases == 0).nonzero()
            for index in zero_bias_indices:
                new_biases[index[0]] = old_bias_signs[index[0]]
            new_vec = nn.Parameter(new_biases)
            self.linear.bias = new_vec
