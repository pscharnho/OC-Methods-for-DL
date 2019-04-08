
import numpy as np 
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timeit
from Layers.Layers import MSALinearLayer, ReluLayer, TanhLayer, CustomBatchNorm1d, AntiSymLayer



class BasicVarConvNet(nn.Module):
    def __init__(self, num_conv, subs_points, num_kernels, num_fc=1, sizes_fc=[1000], test=False):
        super(BasicVarConvNet, self).__init__()
        self.num_conv = num_conv
        self.num_fc = num_fc
        self.subs_points = subs_points
        
        self.conv_keys = ['Conv'+str(i) for i in range(num_conv)]
        self.fc_keys = ['FC'+str(i) for i in range(num_fc+1)]

        self.layers_dict = {}
        self.x_dict = {}
        self.lambda_dict = {}
        self.batch_element = None
        
        self._init_conv_layers(num_kernels)
        if num_conv != 0:
            num_pix = num_kernels[-1] * 784 / (4**len(subs_points))
        else:
            num_pix = None
        self._init_fc_layers(sizes_fc, num_pix, test)
        #self._init_conv_subs_layers(num_kernels)
        #keys = ['Conv'+str(i) for i in range(self.num_conv)]
            
        
    def _init_conv_layers(self, num_kernels):
        #keys = ['Conv'+str(i) for i in range(self.num_conv)]
        num_kernel_counter = 0
        current_key = 0
        channels = 1
        for i in range(self.num_conv):
            if i in self.subs_points:
                num_kernel_counter += 1
                self.layers_dict.update({self.conv_keys[current_key]:nn.Conv2d(channels, num_kernels[num_kernel_counter], 3, stride=2, padding=1)})
                #setattr(self, self.conv_keys[current_key], nn.Conv2d(channels, num_kernels[num_kernel_counter], 3, stride=2, padding=1))
                channels = num_kernels[num_kernel_counter]
                current_key += 1
            else:
                self.layers_dict.update({self.conv_keys[current_key]:nn.Conv2d(channels, num_kernels[num_kernel_counter], 3, padding=1)})
                #setattr(self, self.conv_keys[current_key], nn.Conv2d(channels, num_kernels[num_kernel_counter], 3, padding=1))
                channels = num_kernels[num_kernel_counter]
                current_key += 1
                
    def _init_fc_layers(self, sizes_fc, num_pix, test):
        #keys = ['FC'+str(i) for i in range(num_fc)]
        if num_pix != None:
            sizes = [num_pix]+sizes_fc
        else:
            sizes = sizes_fc
        for i in range(self.num_fc):
            self.layers_dict.update({self.fc_keys[i]:MSALinearLayer(int(sizes[i]), int(sizes[i+1]), test=test)})
            #setattr(self, self.fc_keys[i], nn.Linear(int(sizes[i]), int(sizes[i+1])))
            #print(sizes[i], sizes[i+1])
            
        
    def forward(self, x):
        pass
    





class ResConvNet(BasicVarConvNet):
    '''ResNet class with variable size, specified in the constructor'''
    def __init__(self, num_conv, subs_points, num_kernels, num_fc=1, sizes_fc=[1000], res_skips=1):#, subs_style='conv'
        super(ResConvNet, self).__init__(num_conv, subs_points, num_kernels, num_fc, sizes_fc)
        self.subs_keys = ['Conv1x1'+str(i) for i in range(len(subs_points))]
        self._init_conv_subs_layers(num_kernels)
            
    def _init_conv_subs_layers(self, num_kernels):
        '''subsampling of identity via 1x1 convolution'''
        #channels = [1] + num_kernels
        for layer in range(len(self.subs_points)):
            self.layers_dict.update({self.subs_keys[layer]:nn.Conv2d(num_kernels[layer], num_kernels[layer+1], 1, stride=2)})
            #setattr(self, self.subs_keys[layer], nn.Conv2d(num_kernels[layer], num_kernels[layer+1], 1, stride=2))
        
    def forward(self, x):
        '''single res_skip, x_new = x+Relu(W*x)'''
        for layer in range(self.num_conv):
            if layer in self.subs_points:
                #intermediate_x = getattr(self, self.conv_keys[layer])(x)
                #intermediate_x = self.layers_dict[self.conv_keys[layer]](x)
                x_key = 'x_batch'+str(self.batch_element)+self.conv_keys[layer]
                self.x_dict.update({x_key:torch.tensor(x, requires_grad=True)})
                x = self.subsample(x, self.subs_points.index(layer)) + F.relu(self.layers_dict[self.conv_keys[layer]](x))
            else:
                #intermediate_x = self.layers_dict[self.conv_keys[layer]](x)
                x_key = 'x_batch'+str(self.batch_element)+self.conv_keys[layer]
                self.x_dict.update({x_key:torch.tensor(x, requires_grad=True)})
                x = x + F.relu(self.layers_dict[self.conv_keys[layer]](x))
        x = x.view(-1, self.num_flat_features(x))
        for layer in range(self.num_fc-1):
            x_key = 'x_batch'+str(self.batch_element)+self.fc_keys[layer]
            self.x_dict.update({x_key:torch.tensor(x, requires_grad=True)})
            x = F.relu(self.layers_dict[self.fc_keys[layer]](x))
        x_key = 'x_batch'+str(self.batch_element)+self.fc_keys[self.num_fc-1]
        self.x_dict.update({x_key:torch.tensor(x, requires_grad=True)})
        x = F.softmax(self.layers_dict[self.fc_keys[self.num_fc-1]](x), dim=1)
        x_key = 'x_batch'+str(self.batch_element)+self.fc_keys[self.num_fc]
        self.x_dict.update({x_key:torch.tensor(x, requires_grad=True)})
        return x
    
    def subsample(self, x, index):
        x = self.layers_dict[self.subs_keys[index]](x)
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






class ConvNet(BasicVarConvNet):
    '''ConvNet class with variable size, specified in the constructor'''
    def __init__(self, num_conv, subs_points, num_kernels, num_fc=1, sizes_fc=[1000], test=False):#, subs_style='conv'
        super(ConvNet, self).__init__(num_conv, subs_points, num_kernels, num_fc, sizes_fc, test)

        
    def forward(self, x):
        '''x_new = Relu(W*x)'''
        for layer in range(self.num_conv):
            x_key = 'x_batch'+str(self.batch_element)+self.conv_keys[layer]
            self.x_dict.update({x_key:torch.tensor(x, requires_grad=True)})
            x = F.relu(self.layers_dict[self.conv_keys[layer]](x))

        #x = x.view(-1, self.num_flat_features(x))

        for layer in range(self.num_fc-1):
            #print(x)
            x_key = 'x_batch'+str(self.batch_element)+self.fc_keys[layer]
            self.x_dict.update({x_key:torch.tensor(x, requires_grad=True)})
            x = F.relu(self.layers_dict[self.fc_keys[layer]](x))
        x_key = 'x_batch'+str(self.batch_element)+self.fc_keys[self.num_fc-1]
        self.x_dict.update({x_key:torch.tensor(x, requires_grad=True)})
        x = self.layers_dict[self.fc_keys[self.num_fc-1]](x) #F.softmax(, dim=0)  
        x_key = 'x_batch'+str(self.batch_element)+self.fc_keys[self.num_fc]
        self.x_dict.update({x_key:torch.tensor(x, requires_grad=True)})
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def train_backprop(self, num_epochs, dataloader, output_frequency=1000):
        tic=timeit.default_timer()
        criterion = nn.CrossEntropyLoss()
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

    def train_msa(self, num_epochs, dataloader):
        tic=timeit.default_timer()
        criterion = nn.CrossEntropyLoss()
        
        self.batch_size = dataloader.batch_size
        for epoch in range(num_epochs):

            self.train_epoch(dataloader, criterion)
            
        toc=timeit.default_timer()
        print('Time elapsed: ',toc-tic)
                    
    def train_epoch(self, dataloader, criterion):
        for index, (data, label) in enumerate(dataloader):
            if index % 10 == 0:
                print(index*self.batch_size)
            for batch_element in range(self.batch_size):
                self.batch_element = batch_element
                # MSA step 1
                self.msa_step_1(data[batch_element])
                # MSA step 2
                self.msa_step_2(criterion,label)
            # MSA step 3
            self.msa_step_3(dataloader.batch_size)
            for x in self.x_dict:
                self.x_dict[x].grad.zero_()

    def msa_step_1(self,x):
        self.forward(x)


    def msa_step_2(self,criterion,label):
        self.compute_lambda_T(criterion, label)
        self.compute_lambdas()



    def compute_lambda_T(self, criterion, label):
        x_T_key = 'x_batch'+str(self.batch_element)+self.fc_keys[self.num_fc]
        loss = criterion(self.x_dict[x_T_key].unsqueeze(0),label[self.batch_element].unsqueeze(0))
        loss.backward()
        lambda_T_key = 'lambda_batch'+str(self.batch_element)+self.fc_keys[self.num_fc]
        self.lambda_dict.update({lambda_T_key:-self.x_dict[x_T_key].grad})
    
    def compute_lambdas(self):
        for layer in reversed(range(self.num_fc)):
            x_key = 'x_batch'+str(self.batch_element)+self.fc_keys[layer]
            lambda_key = 'lambda_batch'+str(self.batch_element)+self.fc_keys[layer+1]
            
            res = self.layers_dict[self.fc_keys[layer]].hamilton(self.x_dict.get(x_key), self.lambda_dict.get(lambda_key))
            res.backward()
            new_lambda_key = 'lambda_batch'+str(self.batch_element)+self.fc_keys[layer]
            self.lambda_dict.update({new_lambda_key:self.x_dict.get(x_key).grad})


    def msa_step_3(self, batchsize):
        for layer in range(self.num_fc):
            self.layers_dict[self.fc_keys[layer]].set_weights(layer, self.x_dict, self.lambda_dict, batchsize)



    def test(self, dataloader, test_set_size):
        with torch.no_grad():
            correct_pred = 0
            for _, (data, label) in enumerate(dataloader):
                prediction = self.forward(data)
                _, ind = torch.max(prediction,1)
                print(prediction)
                print(ind)
                print(label)
               # _, ind_label = torch.max(label, 1)
                #if label == ind: #ind_
                #    correct_pred += 1
            #print('Test set accuracy: ', correct_pred/test_set_size)


#################################################################
##
## FC Nets with MSA training
##
#################################################################


class BasicVarFCNet(nn.Module):
    """Base class for FC network with msa training. For each hidden FC layer, 
    there is an activation layer (Relu or tanh).
    
    Attributes:
            num_fc (int): number of fc layers including batchnorm and 
                activation layers
            fc_keys (list of str): names of the fc layers
            layers_dict (dict): dict containing the layer objects
            x_dict (dict): dict containing the intermediate values 

    Methods:
            forward(x): forward propagation of x. Needs to be implemented for subclasses
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
                
    def _init_fc_layers(self, sizes_fc, bias, test, batchnorm):
        fc_layer_index = 0
        if batchnorm:
            for i in range(0,self.num_fc,3):
                self.layers_dict.update({self.fc_keys[i]:MSALinearLayer(int(sizes_fc[fc_layer_index]), int(sizes_fc[fc_layer_index+1]), bias=bias, test=test)})
                fc_layer_index += 1
            fc_layer_index = 1
            for i in range(1,self.num_fc,3):
                self.layers_dict.update({self.fc_keys[i]:CustomBatchNorm1d(int(sizes_fc[fc_layer_index]))})#ReluLayer
                fc_layer_index += 1
            for i in range(2, self.num_fc,3):
                self.layers_dict.update({self.fc_keys[i]:ReluLayer()})#TanhLayer
        else:
            for i in range(self.num_fc):
                if i % 2 == 0:
                    self.layers_dict.update({self.fc_keys[i]:MSALinearLayer(int(sizes_fc[fc_layer_index]), int(sizes_fc[fc_layer_index+1]), bias=bias, test=test)})
                    fc_layer_index += 1
                else:
                    self.layers_dict.update({self.fc_keys[i]:TanhLayer()})#ReluLayer
            
    def forward(self, x):
        """Forward propagation of x. Needs to be implemented for subclasses"""

        pass



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


    def train_msa(self, num_epochs, dataloader):
        """ Trains the net using a modified version of the MSA algorithm.

        Introduces new attributes:
                avg_losses (torch.tensor): Average loss per epoch.
                avg_correct_pred (torch.tensor): Average correct predictions per epoch.
                batch_size (int): Batchsize specified by the dataloader.

        Parameters:
                num_epochs (int): Number of epochs for training the net.
                dataloader (torch.Dataloader): Container for training samples.
        """

        tic=timeit.default_timer()
        criterion = nn.CrossEntropyLoss(reduction='sum')#
        train_size = len(dataloader.dataset)*0.8
        self.avg_losses = torch.zeros(num_epochs)
        self.avg_correct_pred = torch.zeros(num_epochs)
        self.batch_size = dataloader.batch_size
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
            self._msa_step_1(data,label)
            # MSA step 2
            self._msa_step_2(epoch,criterion,label)
            # MSA step 3
            self._msa_step_3()


    def _msa_step_1(self, x, label):
        output = self.forward(x)
        _, ind = torch.max(output,1)
        for i in range(len(label)):
            if ind[i].data == label[i].data:
                self.correct_pred +=1


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
                self.layers_dict[self.fc_keys[layer]].set_weights_and_biases(layer, self.x_dict, self.lambda_dict, self.batch_size)


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


    def test(self, dataloader, test_set_size):
        """ Evaluate the accuracy of the net on the test set.
        
        Parameters:
                dataloader (torch.Dataloader): Container for the test samples.
                test_set_size (int): Size of the test set.
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
                for i in range(len(label)):
                    if ind[i].data == label[i].data:
                        correct_pred +=1
            print('Correct predictions: '+str(correct_pred/test_set_size))



#################################################################
##
## Backprop trained Nets
##
#################################################################


class BasicBackpropNet(nn.Module):
    """ Base class for a FC Net with backpropagation training.
    
    Methods:
            forward(x): Forward propagation of x. Needs to be implemented for subclasses.
            train(dataloader, num_epochs): Trains the net using backpropagation.
            train_epoch(epoch, optimizer, dataloader, criterion, train_size): Train the net for a single epoch.
            test(dataloader, test_set_size):  Evaluate the accuracy of the net on the test set.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ Forward propagation of x. Needs to be implemented for subclasses.
        
        Parameters:
                x (torch.FloatTensor): Tensor containing a batch of training samples.
        """

        pass

    def train(self, dataloader, num_epochs):
        """ Trains the net using backpropagation.

        Introduces new attributes:
                avg_losses (torch.tensor): Average loss per epoch.
                avg_correct_pred (torch.tensor): Average correct predictions per epoch.

        Parameters:
                dataloader (torch.Dataloader): Container for training samples.
                num_epochs (int): Number of epochs for training the net.
        """

        tic=timeit.default_timer()
        criterion = nn.CrossEntropyLoss()#reduction='sum'
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        train_size = len(dataloader.dataset)*0.8
        self.avg_losses = torch.zeros(num_epochs)
        self.avg_correct_pred = torch.zeros(num_epochs)
        for epoch in range(num_epochs):

            self.train_epoch(epoch, optimizer, dataloader, criterion, train_size)

        toc=timeit.default_timer()
        print('Time elapsed: ',toc-tic)


    def train_epoch(self, epoch, optimizer, dataloader, criterion, train_size):
        """ Train the net for a single epoch.

        Parameters:
                epoch (int): Current epoch number. Used for output.
                optimizer (): Torch optimizer for performing the weight updates.
                dataloader (torch.Dataloader): Container for training samples.
                criterion (): The Loss function.
                train_size (int): Size of the training set.
        """

        if epoch % 10 == 0:
            print("#  Epoch  #  Avg-Loss  #  Train-Acc  ###############")
        correct_pred = 0
        loss_sum = 0
        for _, (data, target) in enumerate(dataloader):
            output = self.forward(data)
            #print(output)
            loss = criterion(output, target)
            _, ind = torch.max(output,1)
            for i in range(len(target)):
                if ind[i].data == target[i].data:
                    correct_pred +=1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum + loss.data
            
        print("#  %d  #  %f  #  %f  #" % (epoch+1, loss_sum/train_size, correct_pred/train_size))
        self.avg_correct_pred[epoch] = correct_pred/train_size
        self.avg_losses[epoch] =  loss_sum/train_size


    def test(self, dataloader, test_set_size):
        """ Evaluate the accuracy of the net on the test set.
        
        Parameters:
                dataloader (torch.Dataloader): Container for the test samples.
                test_set_size (int): Size of the test set.
        """

        with torch.no_grad():
            correct_pred = 0
            for _, (data, label) in enumerate(dataloader):
                prediction = self.forward(data)
                _, ind = torch.max(prediction,1)
                for i in range(len(label)):
                    if ind[i].data == label[i].data:
                        correct_pred +=1
            print('Correct predictions: '+str(correct_pred/test_set_size))

    
#################################################################
##
## ResNet with antisymmetric weighted layers
##
#################################################################


class ResAntiSymNet(BasicBackpropNet):
    """ Class for ResNets with weight matrices constructed using antisymmetric matrices.
    
    Attributes:
            layer_keys (list of str): List of layer names.
            num_layers (int): """
    def __init__(self, features, classes, num_layers, gamma, h, bias=False, hidden_size=None):
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
        self.init_layers(features, classes, hidden_size)

    def init_layers(self, features, classes, hidden_size):
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
    def __init__(self, num_layers, layers, bias=False):
        super().__init__()
        self.num_layers = num_layers
        self.has_bias = bias
        self.layer_keys = ['FC'+str(i) for i in range(num_layers)]
        self.init_layers(layers)

    def init_layers(self, layers):
        for layer in range(self.num_layers):
            setattr(self,self.layer_keys[layer],nn.Linear(layers[layer],layers[layer+1],bias=self.has_bias))


    def forward(self, x):
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
    def __init__(self, num_layers, layers, bias=False):
        super().__init__(num_layers, layers, bias)
        self.layers = layers
        for i in range(num_layers-1):
            assert layers[i] <= layers[i+1], 'Decreasing hidden layer size!'

    def forward(self, x):
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
