import numpy as np 
import pandas as pd
import random
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit



class ReluLayer(nn.Module):
    """ ReLU activation layer for fc nets.
    
    Attributes:
            name (str): Name of the layer.

    Methods:
            forward(x): Forward propagation of x.
            hamilton(x, lambd): Computes the hamilton function at point x and lambd.
    """
    def __init__(self):
        super(ReluLayer, self).__init__()
        self.name = 'Relu'

    def forward(self, x):
        """ Forward propagation of x.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """
        x = F.relu(x)
        return x

    def hamilton(self, x, lambd):
        """ Computes the hamilton function at point x and lambd.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
                lambd (torch.FloatTensor): Tensor containing the co-states.
        """
        x_neu = self.forward(x)
        res = torch.sum(x_neu*lambd, dim=1)
        return res


class Relu2dLayer(nn.Module):
    """ ReLU activation layer for conv nets.
    
    Attributes:
            name (str): Name of the layer.

    Methods:
            forward(x): Forward propagation of x.
            hamilton(x, lambd): Computes the hamilton function at point x and lambd.
    """
    def __init__(self):
        super(Relu2dLayer, self).__init__()
        self.name = 'Relu'

    def forward(self, x):
        """ Forward propagation of x.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """
        x = F.relu(x)
        return x

    def hamilton(self, x, lambd):
        """ Computes the hamilton function at point x and lambd.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
                lambd (torch.FloatTensor): Tensor containing the co-states.
        """
        x_neu = self.forward(x)
        res = torch.sum(x_neu*lambd, dim=(1,2,3))
        return res


class TanhLayer(nn.Module):
    """ Tanh activation layer for fc nets.
    
    Attributes:
            name (str): Name of the layer.

    Methods:
            forward(x): Forward propagation of x.
            hamilton(x, lambd): Computes the hamilton function at point x and lambd.
    """
    def __init__(self):
        super(TanhLayer, self).__init__()
        self.name = 'Tanh'

    def forward(self, x):
        """ Forward propagation of x.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """
        x = torch.tanh(x)
        return x

    def hamilton(self, x, lambd):
        """ Computes the hamilton function at point x and lambd.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
                lambd (torch.FloatTensor): Tensor containing the co-states.
        """
        x_neu = self.forward(x)
        res = torch.sum(x_neu*lambd, dim=1)
        return res


class Tanh2dLayer(nn.Module):
    """ Tanh activation layer for conv nets.
    
    Attributes:
            name (str): Name of the layer.

    Methods:
            forward(x): Forward propagation of x.
            hamilton(x, lambd): Computes the hamilton function at point x and lambd.
    """
    def __init__(self):
        super(Tanh2dLayer, self).__init__()
        self.name = 'Tanh'

    def forward(self, x):
        """ Forward propagation of x.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """
        x = torch.tanh(x)
        return x

    def hamilton(self, x, lambd):
        """ Computes the hamilton function at point x and lambd.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
                lambd (torch.FloatTensor): Tensor containing the co-states.
        """
        x_neu = self.forward(x)
        res = torch.sum(x_neu*lambd, dim=(1,2,3))
        return res


class CustomBatchNorm1d(nn.BatchNorm1d):
    """ Batchnorm layer for fc nets.
    
    Attributes:
            name (str): Name of the layer.

    Methods:
            hamilton(x, lambd): Computes the hamilton function at point x and lambd.
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True):#True
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.name = 'BatchNorm'

    def hamilton(self, x, lambd):
        """ Computes the hamilton function at point x and lambd.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
                lambd (torch.FloatTensor): Tensor containing the co-states.
        """
        x_neu = self.forward(x)
        res = torch.sum(x_neu*lambd, dim=1)
        return res


class CustomBatchNorm2d(nn.BatchNorm2d):
    """ Batchnorm layer for conv nets.
    
    Attributes:
            name (str): Name of the layer.

    Methods:
            hamilton(x, lambd): Computes the hamilton function at point x and lambd.
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True):#True
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.name = 'BatchNorm'

    def hamilton(self, x, lambd):
        """ Computes the hamilton function at point x and lambd.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
                lambd (torch.FloatTensor): Tensor containing the co-states.
        """
        x_neu = self.forward(x)
        res = torch.sum(x_neu*lambd, dim=(1,2,3))
        return res


class MSAConvLayer(nn.Module):
    '''Convolutional layer for MSA training.
    
    Attributes:
            conv (layer): PyTorchs 2d conv layer.
            m_accumulated (nn.Parameter): Accumulative EMA m of the MSA.
            name (str): Name of the layer.
            rho (float): MSA rho.
            ema_alpha (float): Alpha for computing new m.
    
    Methods:
            forward(x): Forward propagation of x.
            hamilton(x, lambd): Computes the hamilton function at point x and lambd.
            set_weights(layer_index, x_dict, lambda_dict): Updates the layers weights.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """ Parameters:
                in_channels (int): Number of ingoing channels.
                out_channels (int): Number of outgoing channels.
                kernel_size (int): Spatial size of the filters (default is 3).
                stride (int): Stride of the filters (default is 1).
                padding (int): Zero padding of the filters (default is 1).
        """
        super(MSAConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.m_accumulated = None
        self.name = 'Conv'
        self._initialize_weights()
        self.rho = 0.5
        self.ema_alpha = 0.99
        
    def _initialize_weights(self):
        weights = nn.Parameter(torch.sign(self.conv.weight))
        self.conv.weight = nn.Parameter(weights)
        self.m_accumulated = 0.001 * self.conv.weight.clone().detach()

    def forward(self, x):
        """ Forward propagation of x.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """
        x = self.conv(x)
        return x

    def hamilton(self, x, lambd):
        """ Computes the hamilton function at point x and lambd.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
                lambd (torch.FloatTensor): Tensor containing the co-states.
        """
        x_neu = self.forward(x)
        res = torch.sum(x_neu*lambd, dim=(1,2,3)) # x.dim() => 4
        return res
        
    def set_weights(self, layer_index, x_dict, lambda_dict):
        """ Updates the layers weights.
        
        Parameters:
                layer_index (str): Number of the layer.
                x_dict (dict): Dict containing the intermediate variables.
                lambda_dict (dict): Dict containing the co-states.
        """
        self.conv.weight.grad.data.zero_()
        lambda_key = 'lambda_Conv'+str(layer_index+1)
        x_key = 'x_Conv'+str(layer_index)
        temp = torch.sum(self.hamilton(x_dict.get(x_key),lambda_dict.get(lambda_key)))
        temp.backward()
        m = self.conv.weight.grad 
        self.m_accumulated = self.ema_alpha * self.m_accumulated + ( 1 - self.ema_alpha ) * m
        old_weight_signs = self.conv.weight
        new_weight_signs = torch.sign(self.m_accumulated)
        compared_weight_signs = torch.ne(new_weight_signs, old_weight_signs).clone().detach()#.type(torch.FloatTensor)
        abs_acc = torch.abs(self.m_accumulated)
        reduced_m = torch.where(compared_weight_signs, abs_acc, torch.zeros_like(abs_acc))
        max_weight_elem = torch.max(reduced_m)      
        new_weights = torch.where(abs_acc>=(self.rho*max_weight_elem), new_weight_signs, torch.zeros_like(abs_acc))
        new_matr = nn.Parameter(torch.where(new_weights == 0, old_weight_signs, new_weights))
        self.conv.weight = new_matr


class MSALinearLayer(nn.Module):
    '''Linear layer for MSA training.
    
    Attributes:
            linear (layer): PyTorchs linear layer.
            has_bias (bool): Whether to use biases.
            name (str): Name of the layer.
            rho (float): MSA rho.
            ema_alpha (float): Alpha for computing new m.
    
    Methods:
            forward(x): Forward propagation of x.
            hamilton(x, lambd): Computes the hamilton function at point x and lambd.
            set_weights_and_biases(layer_index, x_dict, lambda_dict): Updates the layers weights and biases.
    '''

    def __init__(self, in_features, out_features, bias=True, test=False):
        """ Parameters:
                in_features (int): Number of ingoing features.
                out_features (int): Number of outgoing features.
                bias (bool): Whether to use biases (default is True).
                test (bool): Whether to use test initialization (default is False).
        """
        super(MSALinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.has_bias = bias
        self._initialize_weights(in_features, out_features, test)
        self.name = 'Linear'
        self.rho = 0.5
        self.ema_alpha = 0.99
        

    def _initialize_weights(self, in_features, out_features, test):
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
        

    def forward(self, x):
        """ Forward propagation of x.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """
        x = self.linear(x)
        return x

    def hamilton(self, x, lambd):
        """ Computes the hamilton function at point x and lambd.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
                lambd (torch.FloatTensor): Tensor containing the co-states.
        """
        x_neu = self.forward(x)
        res = torch.sum(x_neu*lambd, dim=1)
        return res

    def set_weights_and_biases(self, layer_index, x_dict, lambda_dict):
        """ Updates the layers weights and biases.
        
        Parameters:
                layer_index (str): Number of the layer.
                x_dict (dict): Dict containing the intermediate variables.
                lambda_dict (dict): Dict containing the co-states.
        """
        self.linear.weight.grad.data.zero_()   
        self._set_weights(layer_index, x_dict, lambda_dict)
        if self.has_bias:
            self._set_biases(layer_index, x_dict, lambda_dict)


    def _set_weights(self, layer_index, x_dict, lambda_dict):
        lambda_key = 'lambda_'+'FC'+str(layer_index+1)
        x_key = 'x_'+'FC'+str(layer_index)
        temp = torch.sum(self.hamilton(x_dict.get(x_key),lambda_dict.get(lambda_key)))
        temp.backward()
        m = self.linear.weight.grad
        self.m_accumulated = self.ema_alpha * self.m_accumulated + ( 1 - self.ema_alpha ) * m
        old_weight_signs = self.linear.weight
        new_weight_signs = torch.sign(self.m_accumulated)
        compared_weight_signs = torch.ne(new_weight_signs, old_weight_signs).clone().detach()
        abs_acc = torch.abs(self.m_accumulated)
        reduced_m = torch.where(compared_weight_signs, abs_acc, torch.zeros_like(abs_acc))
        max_weight_elem = torch.max(reduced_m)
        new_weights = torch.where(abs_acc>=(self.rho*max_weight_elem), new_weight_signs, torch.zeros_like(abs_acc))
        new_matr = nn.Parameter(torch.where(new_weights == 0, old_weight_signs, new_weights))
        self.linear.weight = new_matr

    def _set_biases(self, layer_index, x_dict, lambda_dict):
        m2 = torch.zeros(self.linear.out_features, dtype=torch.float32)
        lambda_key = 'lambda_'+'FC'+str(layer_index+1)
        x_old_key = 'x_FC'+str(layer_index)
        x_new_old_key = 'x_FC'+str(layer_index+1)
        x_new_new = self.forward(x_dict[x_old_key])
        x = x_dict[x_new_old_key] - x_new_new
        m2 = torch.sum(lambda_dict.get(lambda_key), dim=0) + torch.sum(x, dim=0)
        self.m2_accumulated = self.ema_alpha * self.m2_accumulated + ( 1 - self.ema_alpha ) * m2
        new_bias_signs = torch.sign(self.m2_accumulated)
        new_vec = nn.Parameter(new_bias_signs)
        self.linear.bias = new_vec


class AntiSymLayer(nn.Module):
    """ Layer with weight matrix constructed by antisymmetric matrix.
    
    Attributes:
            features (int): Number of ingoing and outgoing features.
            gamma (float): Gamma of weight matrix.
            has_bias (bool): Whether to use biases.
            weight (nn.Parameter): Weight used for constructing weight matrix.
            eye (torch.Tensor): Identity used for constructing weight matrix.

    Methods:
            forward(x): Forward propagation of x. 
    """
    
    def __init__(self, features, gamma, bias=False):
        """ Parameters:
                features (int): Number of ingoing and outgoing features.
                gamma (float): Gamma of weight matrix.
                bias (bool): Whether to use biases (default is False).
        """
        super().__init__()
        self.features = features
        self.gamma = gamma
        self.has_bias = bias
        self.weight = nn.Parameter(torch.zeros(features,features).uniform_(-1/math.sqrt(features),1/math.sqrt(features)))
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(features).uniform_(-1/math.sqrt(features),1/math.sqrt(features)))
        self.eye = torch.eye(features)

    def forward(self, x):
        """ Forward propagation of x.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """
        if self.has_bias:
            res = torch.addmm(self.bias, x, 1/2 * (self.weight - self.weight.t() - self.gamma * self.eye))
        else:
            res = x.matmul(1/2 * (self.weight - self.weight.t() - self.gamma * self.eye))
        return res