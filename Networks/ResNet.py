
import numpy as np 
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timeit
from Layers.Layers import MSALinearLayer, ReluLayer, TanhLayer



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
    


class BasicVarFCNet(nn.Module):
    '''Base class for FC network with msa training. For each hidden FC layer, there is an activation layer (Relu)'''
    def __init__(self, num_fc, sizes_fc, bias, test=False):
        super(BasicVarFCNet, self).__init__()
        self.num_fc = num_fc * 2 - 1 # Additional counting for activation layers
        
        self.fc_keys = ['FC'+str(i) for i in range(self.num_fc+1)] # Keys for intermediate variables

        self.layers_dict = {}
        self.x_dict = {}
        self.lambda_dict = {}
        self.batch_element = None
        
        self._init_fc_layers(sizes_fc, bias, test)
                
    def _init_fc_layers(self, sizes_fc, bias, test):
        fc_layer_index = 0
        for i in range(self.num_fc):
            if i % 2 == 0:
                self.layers_dict.update({self.fc_keys[i]:MSALinearLayer(int(sizes_fc[fc_layer_index]), int(sizes_fc[fc_layer_index+1]), bias=bias, test=test)})
                fc_layer_index += 1
            else:
                self.layers_dict.update({self.fc_keys[i]:TanhLayer()})#ReluLayer
            
        
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





class FCNet(BasicVarFCNet):
    def __init__(self, num_fc, sizes_fc, bias, test=False):
        super(FCNet, self).__init__(num_fc, sizes_fc, bias, test)

        
    def forward(self, x):

        for layer in range(self.num_fc):
            x_key = 'x_batch'+str(self.batch_element)+self.fc_keys[layer]
            self.x_dict.update({x_key:torch.tensor(x, requires_grad=True)})
            x = self.layers_dict[self.fc_keys[layer]](x) 
        x_key = 'x_batch'+str(self.batch_element)+self.fc_keys[self.num_fc]
        self.x_dict.update({x_key:torch.tensor(x, requires_grad=True)})
        return x


    def train_msa(self, num_epochs, dataloader):
        self.best_avg = 0
        tic=timeit.default_timer()
        criterion = nn.CrossEntropyLoss()
        dataset_size = len(dataloader.dataset)
        self.avg_loss = torch.zeros(num_epochs, dtype=torch.float32)
        #print(dataloader.batch_size)
        self.batch_size = dataloader.batch_size
        for epoch in range(num_epochs):

            self.train_epoch(epoch, dataloader, criterion)
            self.avg_loss[epoch] = self.avg_loss[epoch] / dataset_size
            print('Epoch '+str(epoch+1)+' ##############')
            self.test(dataloader,160)
            self.decay_ema_alpha(0.95)
            
        toc=timeit.default_timer()
        print('Time elapsed: ',toc-tic)

                    
    def train_epoch(self, epoch, dataloader, criterion):
        for index, (data, label) in enumerate(dataloader):
            # print(str(index)+' ##########')
            # for layer in self.layers_dict:
            #     if self.layers_dict[layer].name == 'Linear':
            #         print(layer)
            #         print('Weight')
            #         print(self.layers_dict[layer].linear.weight)
            #         print('Bias')
            #         print(self.layers_dict[layer].linear.bias)
            #         print('Macc')
            #         print(self.layers_dict[layer].m_accumulated)
            #         print('Macc2')
            #         print(self.layers_dict[layer].m2_accumulated)
            for batch_element in range(self.batch_size):
                self.batch_element = batch_element
                # MSA step 1
                self.msa_step_1(data[batch_element])
                # MSA step 2
                self.msa_step_2(epoch,criterion,label)
            # MSA step 3
            self.msa_step_3(self.batch_size)
            #for x in self.x_dict:
                #self.x_dict[x].grad.zero_()


    def msa_step_1(self,x):
        self.forward(x)


    def msa_step_2(self,epoch,criterion,label):
        self.compute_lambda_T(epoch,criterion, label)
        self.compute_lambdas()


    def compute_lambda_T(self, epoch, criterion, label):
        x_T_key = 'x_batch'+str(self.batch_element)+self.fc_keys[self.num_fc]
        loss = criterion(self.x_dict[x_T_key].unsqueeze(0),label[self.batch_element].unsqueeze(0))
        self.avg_loss[epoch] += loss
        loss.backward()
        lambda_T_key = 'lambda_batch'+str(self.batch_element)+self.fc_keys[self.num_fc]
        self.lambda_dict.update({lambda_T_key:-1/self.batch_size*self.x_dict[x_T_key].grad})


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
            if self.layers_dict[self.fc_keys[layer]].name == 'Linear':
                self.layers_dict[self.fc_keys[layer]].set_weights(layer, self.x_dict, self.lambda_dict, batchsize)


    def decay_ema_alpha(self, rate):
        for layer in self.layers_dict:
            if self.layers_dict[layer].name == 'Linear':
                self.layers_dict[layer].ema_alpha = 1 - (1 - self.layers_dict[layer].ema_alpha) * rate

    def set_rho(self,value):
        for layer in self.layers_dict:
            if self.layers_dict[layer].name == 'Linear':
                self.layers_dict[layer].rho = value


    def test(self, dataloader, test_set_size):
        with torch.no_grad():
            correct_pred = 0
            for _, (data, label) in enumerate(dataloader):
                for i in range(len(data)):
                    prediction = self.forward(data[i])
                    #print(prediction)
                    _, ind = torch.max(prediction,0)
                    #print(prediction)
                    #print(ind)
                    #print(label)
                    if ind.data == label[i].data:
                        correct_pred +=1
            print('Correct predictions: '+str(correct_pred/test_set_size))
            if correct_pred/test_set_size > self.best_avg:
                self.best_avg = correct_pred/test_set_size