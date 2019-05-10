
import numpy as np 
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timeit
from Layers.Layers import MSALinearLayer, ReluLayer, TanhLayer, CustomBatchNorm1d, AntiSymLayer, Relu2dLayer, Tanh2dLayer, CustomBatchNorm2d, MSAConvLayer




#################################################################
##
## Basic FCNet with MSA training
##
#################################################################


class BasicVarFCNet(nn.Module):
    """Base class for FC network with msa training. For each hidden FC layer, 
    there is an activation layer (Relu or tanh).
    
    Attributes:
            num_fc (int): number of fc layers including batchnorm and 
                activation layers.
            fc_keys (list of str): names of the fc layers.
            layers_dict (dict): dict containing the layer objects.
            x_dict (dict): dict containing the intermediate values .
            track_test (bool): Whether to track the test results.

    Methods:
            forward(x): forward propagation of x. Needs to be implemented for subclasses.
            set_test_tracking(value): Set the option to track test results per epoch.
    """

    def __init__(self, num_fc, sizes_fc, bias, batchnorm=True, test=False):
        """Parameters:
                num_fc (int): number of layer blocks containing activation and batchnorm
                sizes_fc (list of int): contains the number of neurons per layer. Needs 
                        to be of size num_fc+1.
                bias (bool): Whether to include a bias for the layers
                batchnorm (bool): Whether to include batchnorm layers (default is True)
                test (bool): Whether to use a specific initialization of the layer 
                        weights (default is False)
        """

        super(BasicVarFCNet, self).__init__()
        if batchnorm:
            self.num_fc = num_fc * 3 - 1 # Additional counting for batchnorm and activation layers
        else:
            self.num_fc = num_fc * 2 - 1 # Additional counting for activation layers
        
        self.fc_keys = ['FC'+str(i) for i in range(self.num_fc+1)] # Keys for intermediate variables

        self.layers_dict = {}
        self.x_dict = {}
        self.lambda_dict = {}
        
        self._init_fc_layers(sizes_fc, bias, test, batchnorm)

        self.track_test = False 
                
    def _init_fc_layers(self, sizes_fc, bias, test, batchnorm):
        fc_layer_index = 0
        if batchnorm:
            for i in range(0,self.num_fc,3):
                self.layers_dict.update({self.fc_keys[i]:MSALinearLayer(int(sizes_fc[fc_layer_index]), int(sizes_fc[fc_layer_index+1]), bias=bias, test=test)})
                fc_layer_index += 1
            fc_layer_index = 1
            for i in range(1,self.num_fc,3):
                self.layers_dict.update({self.fc_keys[i]:CustomBatchNorm1d(int(sizes_fc[fc_layer_index]))})
                fc_layer_index += 1
            for i in range(2, self.num_fc,3):
                self.layers_dict.update({self.fc_keys[i]:ReluLayer()})#TanhLayer
        else:
            for i in range(self.num_fc):
                if i % 2 == 0:
                    self.layers_dict.update({self.fc_keys[i]:MSALinearLayer(int(sizes_fc[fc_layer_index]), int(sizes_fc[fc_layer_index+1]), bias=bias, test=test)})
                    fc_layer_index += 1
                else:
                    self.layers_dict.update({self.fc_keys[i]:ReluLayer()})#TanhLayer


    def set_test_tracking(self, value):
        """ Set the option to track test results per epoch.
        
        Parameters:
                value (bool): Whether to track the test results."""

        self.track_test = value

    def forward(self, x):
        """Forward propagation of x. Needs to be implemented for subclasses"""

        pass



#################################################################
##
## FCNet with MSA training
##
#################################################################


class FCMSANet(BasicVarFCNet):
    """ Class for a MSA trained FC-Net. Inherits from BasicVarFCNet.
    
    Methods:
            forward(x): Forward propagation of x and registering of intermediate values.
            train_msa(num_epochs, dataloader): Trains the net using a modified version of the MSA algorithm.
            train_epoch(epoch, dataloader, criterion): Train the net for a single epoch.
            set_ema_alpha(value): Set the value for EMA alpha in linear layers
            decay_ema_alpha(rate): Decay the EMA alpha in every layer with a certain rate
            set_rho(value): Set the rho value for minimizing the error term in the MSA.
            test(dataloader, test_set_size): Evaluate the accuracy of the net on the test set.
    """

    def __init__(self, num_fc, sizes_fc, bias, batchnorm=True, test=False):
        super(FCMSANet, self).__init__(num_fc, sizes_fc, bias, batchnorm, test)
        

        
    def forward(self, x):
        """Forward propagation of x and registering of intermediate values.
        
        Parameter:
                x(torch.FloatTensor): Batch of training samples
                
        Returns:
                torch.FloatTensor: Result of forward propagation
        """

        for layer in range(self.num_fc):
            x_key = 'x_'+self.fc_keys[layer]
            self.x_dict.update({x_key:x.clone().detach().requires_grad_(True)})
            x = self.layers_dict[self.fc_keys[layer]](x) 
        x_key = 'x_'+self.fc_keys[self.num_fc]
        self.x_dict.update({x_key:x.clone().detach().requires_grad_(True)})
        return x


    def train_msa(self, num_epochs, dataloader, testloader=None, print_output=True):
        """ Trains the net using a modified version of the MSA algorithm.

        Introduces new attributes:
                avg_losses (torch.tensor): Average loss per epoch.
                avg_correct_pred (torch.tensor): Average correct predictions per epoch.
                batch_size (int): Batchsize specified by the dataloader.

        Parameters:
                num_epochs (int): Number of epochs for training the net.
                dataloader (torch.Dataloader): Container for training samples.
                testloader (torch.Dataloader): Container for test samples if self.track_test is True (default is None).
                print_output (bool): Whether to print output (default is True).
        """

        tic = timeit.default_timer()
        criterion = nn.CrossEntropyLoss(reduction='sum')#
        train_size = len(dataloader.dataset)*0.8
        self.avg_losses = torch.zeros(num_epochs)
        self.avg_correct_pred = torch.zeros(num_epochs)
        self.batch_size = dataloader.batch_size
        if self.track_test:
            self.test_results = torch.zeros(num_epochs)
            test_size = len(testloader.dataset)*0.2
        for epoch in range(num_epochs):
            if print_output and epoch % 10 == 0:
                print("#  Epoch  #  Avg-Loss  #  Train-Acc  ###############")
            for layer in self.layers_dict:
                if self.layers_dict[layer].name == 'BatchNorm':
                    # set batchnorm layer into train mode
                    self.layers_dict[layer].train()
            self.train_epoch(epoch, dataloader, criterion)
            self.decay_ema_alpha(0.95)
            if print_output:
                print("#  %d  #  %f  #  %f  #" % (epoch+1, self.loss_sum/train_size, self.correct_pred/train_size))
            self.avg_correct_pred[epoch] = self.correct_pred/train_size
            self.avg_losses[epoch] =  self.loss_sum/train_size
            if self.track_test:
                self.test_results[epoch] = self.test(testloader, test_size, print_output)
            
        toc = timeit.default_timer()
        if print_output:
            print('Time elapsed: ',toc-tic)

                    
    def train_epoch(self, epoch, dataloader, criterion):
        """ Train the net for a single epoch.

        Parameters:
                epoch (int): Current epoch number. Used for output.
                dataloader (torch.Dataloader): Container for training samples.
                criterion (): The Loss function.
        """

        self.correct_pred = 0
        self.loss_sum = 0
        for _, (data, label) in enumerate(dataloader):
            # MSA step 1
            self._msa_step_1(data,label)
            # MSA step 2
            self._msa_step_2(epoch,criterion,label)
            # MSA step 3
            self._msa_step_3()


    def _msa_step_1(self, x, label):
        output = self.forward(x)
        _, ind = torch.max(output,1)
        self.correct_pred += torch.sum(ind==label).item()


    def _msa_step_2(self,epoch,criterion,label):
        self._compute_lambda_T(epoch,criterion, label)
        self._compute_lambdas()
        


    def _compute_lambda_T(self, epoch, criterion, label):
        x_T_key = 'x_'+self.fc_keys[self.num_fc]
        loss = criterion(self.x_dict[x_T_key],label)
        self.loss_sum += loss
        loss.backward()
        lambda_T_key = 'lambda_'+self.fc_keys[self.num_fc]
        self.lambda_dict.update({lambda_T_key:-1/self.batch_size*self.x_dict[x_T_key].grad})


    def _compute_lambdas(self):
        for layer in reversed(range(self.num_fc)):
            x_key = 'x_'+self.fc_keys[layer]
            lambda_key = 'lambda_'+self.fc_keys[layer+1]
            res = self.layers_dict[self.fc_keys[layer]].hamilton(self.x_dict.get(x_key), self.lambda_dict.get(lambda_key))
            res.backward(torch.FloatTensor(np.ones(len(res))))
            new_lambda_key = 'lambda_'+self.fc_keys[layer]
            self.lambda_dict.update({new_lambda_key:self.x_dict.get(x_key).grad})


    def _msa_step_3(self):
        for layer in range(self.num_fc):
            if self.layers_dict[self.fc_keys[layer]].name == 'Linear':
                self.layers_dict[self.fc_keys[layer]].set_weights_and_biases(layer, self.x_dict, self.lambda_dict)


    def set_ema_alpha(self,value):
        """ Set the value for EMA alpha in linear layers
        
        Parameters:
                value (float): The new alpha value for all linear layers
        """

        for layer in self.layers_dict:
            if self.layers_dict[layer].name == 'Linear':
                self.layers_dict[layer].ema_alpha = value


    def decay_ema_alpha(self, rate):
        """ Decay the EMA alpha in every layer with a certain rate
        
        Parameters:
                rate (float): Rate of decay.
        """

        for layer in self.layers_dict:
            if self.layers_dict[layer].name == 'Linear':
                self.layers_dict[layer].ema_alpha = 1 - (1 - self.layers_dict[layer].ema_alpha) * rate


    def set_rho(self,value):
        """ Set the rho value for minimizing the error term in the MSA.
        
        Parameters:
                value (float): New rho value.
        """

        for layer in self.layers_dict:
            if self.layers_dict[layer].name == 'Linear':
                self.layers_dict[layer].rho = value


    def test(self, dataloader, test_set_size, print_output=True):
        """ Evaluate the accuracy of the net on the test set.
        
        Parameters:
                dataloader (torch.Dataloader): Container for the test samples.
                test_set_size (int): Size of the test set.
                print_output (bool): Whether to print output (default is True).
        
        Returns:
                correct_pred (int): Number of correct predicted test samples.
        """

        with torch.no_grad():
            for layer in self.layers_dict:
                if self.layers_dict[layer].name == 'BatchNorm':
                    # set batchnorm layer into evaluation mode
                    self.layers_dict[layer].eval()
            correct_pred = 0
            for _, (data, label) in enumerate(dataloader):
                
                prediction = self.forward(data)
                _, ind = torch.max(prediction,1)
                correct_pred += torch.sum(ind==label).item()

            if print_output:
                print('Correct predictions: '+str(correct_pred/test_set_size))
        return correct_pred



#################################################################
##
## Backprop trained Nets
##
#################################################################


class BasicBackpropNet(nn.Module):
    """ Base class for a FC Net with backpropagation training.

    Attributes:
            track_test (bool): Whether to track the test results.
    
    Methods:
            forward(x): Forward propagation of x. Needs to be implemented for subclasses.
            train(dataloader, num_epochs): Trains the net using backpropagation.
            train_epoch(epoch, optimizer, dataloader, criterion, train_size): Train the net for a single epoch.
            test(dataloader, test_set_size):  Evaluate the accuracy of the net on the test set.
            set_test_tracking(value): Set the option to track test results per epoch.
    """

    def __init__(self):
        super().__init__()
        self.track_test = False

    def forward(self, x):
        """ Forward propagation of x. Needs to be implemented for subclasses.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """

        pass

    def train(self, num_epochs, dataloader, testloader=None, print_output=True):
        """ Trains the net using backpropagation.

        Introduces new attributes:
                avg_losses (torch.tensor): Average loss per epoch.
                avg_correct_pred (torch.tensor): Average correct predictions per epoch.

        Parameters:
                num_epochs (int): Number of epochs for training the net.
                dataloader (torch.Dataloader): Container for training samples.
                testloader (torch.Dataloader): Container for test samples if track_test is True (default is None).
                print_output (bool): Whether to print output (default is True).
        """

        tic=timeit.default_timer()
        criterion = nn.CrossEntropyLoss()#reduction='sum'
        optimizer = torch.optim.SGD(self.parameters(), lr=0.5)
        train_size = len(dataloader.dataset)*0.8
        self.avg_losses = torch.zeros(num_epochs)
        self.avg_correct_pred = torch.zeros(num_epochs)
        if self.track_test:
            self.test_results = torch.zeros(num_epochs)
            test_size = len(testloader.dataset)*0.2
        for epoch in range(num_epochs):

            self.train_epoch(epoch, optimizer, dataloader, criterion, train_size, print_output)
            if self.track_test:
                self.test_results[epoch] = self.test(testloader, test_size, print_output)

        toc=timeit.default_timer()
        if print_output:
            print('Time elapsed: ',toc-tic)


    def train_epoch(self, epoch, optimizer, dataloader, criterion, train_size, print_output=True):
        """ Train the net for a single epoch.

        Parameters:
                epoch (int): Current epoch number. Used for output.
                optimizer (): Torch optimizer for performing the weight updates.
                dataloader (torch.Dataloader): Container for training samples.
                criterion (): The Loss function.
                train_size (int): Size of the training set.
                print_output (bool): Whether to print output (default is True).
        """

        if print_output and epoch % 10 == 0:
            print("#  Epoch  #  Avg-Loss  #  Train-Acc  ###############")
        correct_pred = 0
        loss_sum = 0
        for _, (data, target) in enumerate(dataloader):
            output = self.forward(data)
            loss = criterion(output, target)
            _, ind = torch.max(output,1)
            correct_pred += torch.sum(ind==target).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum + loss.data

        if print_output:    
            print("#  %d  #  %f  #  %f  #" % (epoch+1, loss_sum/train_size, correct_pred/train_size))
        self.avg_correct_pred[epoch] = correct_pred/train_size
        self.avg_losses[epoch] =  loss_sum/train_size

    def set_test_tracking(self, value):
        """ Set the option to track test results per epoch.
        
        Parameters:
                value (bool): Whether to track the test results."""

        self.track_test = value

    def test(self, dataloader, test_set_size, print_output=True):
        """ Evaluate the accuracy of the net on the test set.
        
        Parameters:
                dataloader (torch.Dataloader): Container for the test samples.
                test_set_size (int): Size of the test set.
                print_output (bool): Whether to print output (default is True).

        Returns:
                correct_pred (int): Number of correct predicted test samples.
        """

        with torch.no_grad():
            correct_pred = 0
            for _, (data, label) in enumerate(dataloader):
                prediction = self.forward(data)
                _, ind = torch.max(prediction,1)
                correct_pred += torch.sum(ind==label).item()
            if print_output:
                print('Correct predictions: '+str(correct_pred/test_set_size))
        return correct_pred

    
#################################################################
##
## ResNet with antisymmetric weighted layers
##
#################################################################


class ResAntiSymNet(BasicBackpropNet):
    """ Class for ResNets with weight matrices constructed using antisymmetric matrices.
    
    Attributes:
            layer_keys (list of str): List of layer names.
            num_layers (int): Number of FC Layers
            gamma (float): Initialization parameter for AntiSymLayers
            h (float): Time step size of explicit Euler
            has_bias (bool): Whether to use biases 
            
    Methods:
            forward(x): Forward propagation of the sample x.
    """

    def __init__(self, features, classes, num_layers, gamma, h, bias=False, hidden_size=None):
        """ Parameters:
                features (int): Number of features
                classes (int): Number of classes
                num_layers (int): Number of FC Layers
                gamma (float): Initialization parameter for AntiSymLayers
                h (float): Time step size of explicit Euler
                bias (bool): Whether to use biases (default is False)
                hidden_size (int): Optional size of the hidden layers, if different from features or classes (default is None)
        """

        super().__init__()
        if hidden_size == None:
            if features != classes:
                if features > classes:
                    self.layer_keys = ['AntSym'+str(i) for i in range(num_layers-1)]
                    self.layer_keys.append('FC')
                else:
                    self.layer_keys = ['FC']
                    self.layer_keys.extend(['AntSym'+str(i) for i in range(num_layers-1)])
            else:
                self.layer_keys = ['AntSym'+str(i) for i in range(num_layers)]
        else:
            layer_fc_count = 0
            layer_ant_sym_count = 0
            if features == hidden_size:
                self.layer_keys = ['AntSym'+str(layer_ant_sym_count)]
                layer_ant_sym_count += 1
            else:
                self.layer_keys = ['FC'+str(layer_fc_count)]
                layer_fc_count += 1
            for _ in range(num_layers-1):
                self.layer_keys.append('AntSym'+str(layer_ant_sym_count))
                layer_ant_sym_count += 1
            if hidden_size == classes:
                self.layer_keys.append('AntSym'+str(layer_ant_sym_count))
            else:
                self.layer_keys.append('FC'+str(layer_fc_count))
            
        self.num_layers = num_layers
        self.gamma = gamma
        self.h = h
        self.has_bias = bias
        self._init_layers(features, classes, hidden_size)


    def _init_layers(self, features, classes, hidden_size):
        if hidden_size != None:
            start = 0
            if features != hidden_size:
                setattr(self,self.layer_keys[0],nn.Linear(features,hidden_size,bias=self.has_bias))
                start = 1
            for index in range(start,self.num_layers-1):
                setattr(self,self.layer_keys[index],AntiSymLayer(hidden_size,self.gamma,bias=self.has_bias))
            if hidden_size == classes:
                setattr(self,self.layer_keys[self.num_layers-1],AntiSymLayer(classes,self.gamma,bias=self.has_bias))
            else:
                setattr(self,self.layer_keys[self.num_layers-1],nn.Linear(hidden_size,classes,bias=self.has_bias))
        else:
            for key in self.layer_keys:
                if key == 'FC':
                    setattr(self,key,nn.Linear(features,classes,bias=self.has_bias))
                else:
                    setattr(self,key,AntiSymLayer(features,self.gamma,bias=self.has_bias))


    def forward(self, x):
        """ Forward propagation of the sample x.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """

        start = 0
        if self.layer_keys[0] == 'FC' or self.layer_keys[0] == 'FC0':
            x = F.relu(getattr(self, self.layer_keys[0])(x)) 
            start = 1
        for key in range(start, self.num_layers-1):
            x = x + self.h * F.relu(getattr(self, self.layer_keys[key])(x))
        x = getattr(self, self.layer_keys[self.num_layers-1])(x)
        return x



#################################################################
##
## Simple FCNet with backpropagation
##
#################################################################

class FCNet(BasicBackpropNet):
    """ Simple fully connected net with backpropagation training. Used as reference.
    
    Attributes:
            num_layers (int): Number of layers.
            has_bias (bool): Whether the layers have biases.
            layer_keys (list of str): List of layer names.
            
    Methods:
            forward(x): Forward propagation of x."""

    def __init__(self, num_layers, layers, bias=False):
        """Parameters:
                num_layers (int): Number of layers.
                layers (list of int): List of number of neurons per layer. 
                        len(layers) needs to be == num_layers+1
                bias (bool): Whether to use biases (default is False).
        """

        super().__init__()
        self.num_layers = num_layers
        self.has_bias = bias
        self.layer_keys = ['FC'+str(i) for i in range(num_layers)]
        self._init_layers(layers)

    def _init_layers(self, layers):
        for layer in range(self.num_layers):
            setattr(self,self.layer_keys[layer],nn.Linear(layers[layer],layers[layer+1],bias=self.has_bias))


    def forward(self, x):
        """ Forward propagation of x.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """

        for layer in range(self.num_layers-1):
            x = F.relu(getattr(self, self.layer_keys[layer])(x))
        x = getattr(self, self.layer_keys[self.num_layers-1])(x)
        return x



#################################################################
##
## Simple ResNet with backpropagation
##
#################################################################

class ResFCNet(FCNet):
    """ Simple ResNet with backpropagation training. Used as reference.
    
    Attributes:
            layers (list of int): List containing the numbers of neurons in hidden layers.
            
    Methods:
            forward(x): Forward propagation of x.
    """

    def __init__(self, num_layers, layers, bias=False):
        """Parameters:
                num_layers (int): Number of layers.
                layers (list of int): List of number of neurons per layer. 
                        len(layers) needs to be == num_layers+1
                bias (bool): Whether to use biases (default is False).
        """
        super().__init__(num_layers, layers, bias)
        self.layers = layers
        for i in range(num_layers-1):
            assert layers[i] <= layers[i+1], 'Decreasing hidden layer size!'

    def forward(self, x):
        """ Forward propagation of x.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """
        for layer in range(self.num_layers-1):
            if x.size()[1] != self.layers[layer+1]:
                additional_zeros = self.layers[layer+1] - x.size()[1]
                cat_tensor =  torch.zeros((x.size()[0], additional_zeros), dtype=torch.float32)
                #print(x)
                #print(cat_tensor)
                x = torch.cat((x,cat_tensor),1) + F.relu(getattr(self, self.layer_keys[layer])(x))
            else:
                x = x + F.relu(getattr(self, self.layer_keys[layer])(x))
        x = getattr(self, self.layer_keys[self.num_layers-1])(x)
        return x



#################################################################
##
## ConvNet with backprop
##
#################################################################


class ConvNet(BasicBackpropNet):
    """ Simple Convolutional neural net with backpropagation training. Used as reference.
    
    Attributes:"""
    def __init__(self, num_conv, num_channels, subsample_points, num_fc, sizes_fc):
        super().__init__()
        self.num_conv = num_conv
        self.conv_keys = ['Conv'+str(i) for i in range(num_conv)]
        self._init_conv_layers(num_channels, subsample_points)
        self.num_fc = num_fc
        self.fc_keys = ['FC'+str(i) for i in range(num_fc)]
        self._init_fc_layers(sizes_fc)

    
    def _init_conv_layers(self, num_channels, subsample_points):
        for i in range(self.num_conv):
            if i in subsample_points:
                setattr(self, self.conv_keys[i], nn.Conv2d(num_channels[i], num_channels[i+1], kernel_size=3, stride=2, padding=1, bias=False))
            else:
                setattr(self, self.conv_keys[i], nn.Conv2d(num_channels[i], num_channels[i+1], kernel_size=3, stride=1, padding=1, bias=False))


    def _init_fc_layers(self, sizes_fc):
        for i in range(self.num_fc):
            setattr(self, self.fc_keys[i], nn.Linear(sizes_fc[i], sizes_fc[i+1], bias=False))
            
    def forward(self, x):
        """ Forward propagation of x.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """
        for layer in range(self.num_conv):
            x = F.relu(getattr(self, self.conv_keys[layer])(x))
        x = x.view(x.size()[0],-1)
        for layer in range(self.num_fc-1):
            x = F.relu(getattr(self, self.fc_keys[layer])(x))
        x = getattr(self, self.fc_keys[self.num_fc-1])(x)
        return x


#################################################################
##
## ConvNet with MSA Training
##
#################################################################




class BasicVarConvNet(BasicVarFCNet):
    def __init__(self, num_conv, num_channels, subsample_points, num_fc, sizes_fc, batchnorm=True, test=False):
        super(BasicVarConvNet, self).__init__(num_fc, sizes_fc, False, batchnorm, test)
        if batchnorm:
            self.num_conv = num_conv * 3 - 1 # Additional counting for batchnorm and activation layers
        else:
            self.num_conv = num_conv * 2 - 1 # Additional counting for activation layers
        
        self.conv_keys = ['Conv'+str(i) for i in range(self.num_conv+1)]
        self.subsample_points = subsample_points

        self._init_conv_layers(num_channels, batchnorm)
            
        
    def _init_conv_layers(self, num_channels, batchnorm):
        conv_layer_index = 0
        if batchnorm:
            for i in range(0,self.num_conv,3):
                if conv_layer_index in self.subsample_points:
                    #print(conv_layer_index)
                    #print(self.conv_keys[i])
                    self.layers_dict.update({self.conv_keys[i]:MSAConvLayer(num_channels[conv_layer_index], num_channels[conv_layer_index+1], stride=2)})
                else:
                    self.layers_dict.update({self.conv_keys[i]:MSAConvLayer(num_channels[conv_layer_index], num_channels[conv_layer_index+1])})
                conv_layer_index += 1
            conv_layer_index = 1
            for i in range(1,self.num_conv,3):
                self.layers_dict.update({self.conv_keys[i]:CustomBatchNorm2d(num_channels[conv_layer_index])})
                conv_layer_index += 1
            for i in range(2, self.num_conv,3):
                self.layers_dict.update({self.conv_keys[i]:Relu2dLayer()})#Tanh2dLayer
        else:
            for i in range(self.num_conv):
                if i % 2 == 0:
                    if conv_layer_index in self.subsample_points:
                        self.layers_dict.update({self.conv_keys[i]:MSAConvLayer(num_channels[conv_layer_index], num_channels[conv_layer_index+1], stride=2)})
                    else:
                        self.layers_dict.update({self.conv_keys[i]:MSAConvLayer(num_channels[conv_layer_index], num_channels[conv_layer_index+1])})
                    conv_layer_index += 1
                else:
                    self.layers_dict.update({self.conv_keys[i]:Tanh2dLayer()})#ReluLayer
                
        
    def forward(self, x):
        """ Forward propagation of x. Needs to be implemented for subclasses.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """
        pass
    


class ConvMSANet(BasicVarConvNet):
    '''ConvNet class with variable size, specified in the constructor'''
    def __init__(self, num_conv, num_channels, subsample_points, num_fc, sizes_fc, batchnorm=True, test=False):
        super(ConvMSANet, self).__init__(num_conv, num_channels, subsample_points, num_fc, sizes_fc, batchnorm, test)

        
    def forward(self, x):
        """ Forward propagation of x and registering of the intermediate values.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """
        for layer in range(self.num_conv):
            x_key = 'x_'+self.conv_keys[layer]
            self.x_dict.update({x_key:x.clone().detach().requires_grad_(True)})
            x = self.layers_dict[self.conv_keys[layer]](x)

        x_key = 'x_'+self.conv_keys[self.num_conv]
        self.x_dict.update({x_key:x.clone().detach().requires_grad_(True)})

        x = x.view(-1, self._num_flat_features(x))

        for layer in range(self.num_fc-1):
            #print(x)
            x_key = 'x_'+self.fc_keys[layer]
            self.x_dict.update({x_key:x.clone().detach().requires_grad_(True)})
            x = self.layers_dict[self.fc_keys[layer]](x)
        x_key = 'x_'+self.fc_keys[self.num_fc-1]
        self.x_dict.update({x_key:x.clone().detach().requires_grad_(True)})
        x = self.layers_dict[self.fc_keys[self.num_fc-1]](x) #F.softmax(, dim=0)  
        x_key = 'x_'+self.fc_keys[self.num_fc]
        self.x_dict.update({x_key:x.clone().detach().requires_grad_(True)})
        return x
    
    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    


    def train_msa(self, num_epochs, dataloader, testloader=None):
        tic=timeit.default_timer()
        criterion = nn.CrossEntropyLoss()
        self.avg_losses = torch.zeros(num_epochs)
        self.avg_correct_pred = torch.zeros(num_epochs)
        self.batch_size = dataloader.batch_size
        train_size = len(dataloader.dataset)
        if self.track_test:
            self.test_results = torch.zeros(num_epochs)
            test_size = len(testloader.dataset)
        for epoch in range(num_epochs):
            if epoch % 10 == 0:
                print("#  Epoch  #  Avg-Loss  #  Train-Acc  ###############")
            for layer in self.layers_dict:
                if self.layers_dict[layer].name == 'BatchNorm':
                    # set batchnorm layer into train mode
                    self.layers_dict[layer].train()
            self.train_epoch(epoch, dataloader, criterion)
            self.decay_ema_alpha(0.95)
            print("#  %d  #  %f  #  %f  #" % (epoch+1, self.loss_sum/train_size, self.correct_pred/train_size))
            self.avg_correct_pred[epoch] = self.correct_pred/train_size
            self.avg_losses[epoch] =  self.loss_sum/train_size
            if self.track_test:
                self.test_results[epoch] = self.test(testloader, test_size)
        toc=timeit.default_timer()
        print('Time elapsed: ',toc-tic)
                    
    def train_epoch(self, epoch, dataloader, criterion):
        """ Train the net for a single epoch.

        Parameters:
                epoch (int): Current epoch number. Used for output.
                dataloader (torch.Dataloader): Container for training samples.
                criterion (): The Loss function.
        """

        self.correct_pred = 0
        self.loss_sum = 0
        for _, (data, label) in enumerate(dataloader):
            # MSA step 1
            #tic=timeit.default_timer()
            self._msa_step_1(data,label)
            #toc=timeit.default_timer()
            #print('Time MSA1 elapsed: ',toc-tic)
            # MSA step 2
            #tic=timeit.default_timer()
            self._msa_step_2(epoch,criterion,label)
            #toc=timeit.default_timer()
            #print('Time MSA2 elapsed: ',toc-tic)
            # MSA step 3
            #tic=timeit.default_timer()
            self._msa_step_3()
            #toc=timeit.default_timer()
            #print('Time MSA3 elapsed: ',toc-tic)

    def _msa_step_1(self, x, label):
        output = self.forward(x)
        _, ind = torch.max(output,1)
        self.correct_pred += torch.sum(ind == label).item()


    def _msa_step_2(self,epoch,criterion,label):
        self._compute_lambda_T(epoch,criterion, label)
        self._compute_lambdas()
        


    def _compute_lambda_T(self, epoch, criterion, label):
        x_T_key = 'x_'+self.fc_keys[self.num_fc]
        loss = criterion(self.x_dict[x_T_key],label)
        self.loss_sum += loss
        loss.backward()
        lambda_T_key = 'lambda_'+self.fc_keys[self.num_fc]
        self.lambda_dict.update({lambda_T_key:-1/self.batch_size*self.x_dict[x_T_key].grad})


    def _compute_lambdas(self):
        for layer in reversed(range(self.num_fc)):
            x_key = 'x_'+self.fc_keys[layer]
            lambda_key = 'lambda_'+self.fc_keys[layer+1]
            res = self.layers_dict[self.fc_keys[layer]].hamilton(self.x_dict.get(x_key), self.lambda_dict.get(lambda_key))
            res.backward(torch.FloatTensor(np.ones(len(res))))
            new_lambda_key = 'lambda_'+self.fc_keys[layer]
            self.lambda_dict.update({new_lambda_key:self.x_dict.get(x_key).grad})
        lambda_transform_key = 'lambda_'+self.fc_keys[0]
        x_transform_key = 'x_'+self.conv_keys[self.num_conv]
        new_lambda = self.lambda_dict[lambda_transform_key].view(self.x_dict[x_transform_key].size())
        new_lambda_key = 'lambda_'+self.conv_keys[self.num_conv]
        self.lambda_dict.update({new_lambda_key:new_lambda})
        for layer in reversed(range(self.num_conv)):
            x_key = 'x_'+self.conv_keys[layer]
            lambda_key = 'lambda_'+self.conv_keys[layer+1]
            res = self.layers_dict[self.conv_keys[layer]].hamilton(self.x_dict.get(x_key), self.lambda_dict.get(lambda_key))
            res.backward(torch.FloatTensor(np.ones(len(res))))
            new_lambda_key = 'lambda_'+self.conv_keys[layer]
            self.lambda_dict.update({new_lambda_key:self.x_dict.get(x_key).grad})


    def _msa_step_3(self):
        
        for layer in range(self.num_fc):
            if self.layers_dict[self.fc_keys[layer]].name == 'Linear':
                self.layers_dict[self.fc_keys[layer]].set_weights_and_biases(layer, self.x_dict, self.lambda_dict)
        for layer in range(self.num_conv):
            if self.layers_dict[self.conv_keys[layer]].name == 'Conv':
                self.layers_dict[self.conv_keys[layer]].set_weights(layer, self.x_dict, self.lambda_dict)

    def set_ema_alpha(self,value):
        """ Set the value for EMA alpha in linear and conv layers
        
        Parameters:
                value (float): The new alpha value for all linear and conv layers
        """

        for layer in self.layers_dict:
            if self.layers_dict[layer].name == 'Linear' or self.layers_dict[layer].name == 'Conv':
                self.layers_dict[layer].ema_alpha = value


    def decay_ema_alpha(self, rate):
        """ Decay the EMA alpha in every layer with a certain rate
        
        Parameters:
                rate (float): Rate of decay.
        """

        for layer in self.layers_dict:
            if self.layers_dict[layer].name == 'Linear' or self.layers_dict[layer].name == 'Conv':
                self.layers_dict[layer].ema_alpha = 1 - (1 - self.layers_dict[layer].ema_alpha) * rate


    def set_rho(self,value):
        """ Set the rho value for minimizing the error term in the MSA.
        
        Parameters:
                value (float): New rho value.
        """

        for layer in self.layers_dict:
            if self.layers_dict[layer].name == 'Linear' or self.layers_dict[layer].name == 'Conv':
                self.layers_dict[layer].rho = value


    def test(self, dataloader, test_set_size):
        with torch.no_grad():
            correct_pred = 0
            for _, (data, label) in enumerate(dataloader):
                prediction = self.forward(data)
                _, ind = torch.max(prediction,1)
                #print(prediction)
                #print(ind)
                #print(label)
                #_, ind_label = torch.max(label, 1)
                correct_pred += torch.sum(ind == label).item()
            print('Test set accuracy: ', correct_pred/test_set_size)
