from Dataset.Dataset import loadMNIST
from Networks.ResNet import ConvNet

#net = ConvNet(3, [1], [3,6], num_fc=2, sizes_fc=[100,10])
net = ConvNet(0,[],[],num_fc=5,sizes_fc=[784,1000,500,200,100,10])
train_loader, test_loader = loadMNIST('Dataset/input/mnist_train.csv', 'Dataset/input/mnist_test.csv')
#net.train_backprop(1,train_loader)
#net.test(test_loader,10000)
print(train_loader.batch_size)

net.train_msa(1,train_loader)
net.test(test_loader,10000)
#for index, (data, target) in enumerate(train_loader):
#    if index == 1:
#        print(data.shape)
        