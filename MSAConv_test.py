from Networks.ResNet import ConvMSANet
from Dataset.Dataset import loadMNIST

def main():
    net = ConvMSANet(num_conv=3, num_channels=[1,3,6,12], subsample_points=[1,2], num_fc=2,sizes_fc=[588,1000,10], batchnorm=True, test=False)
    train_loader, test_loader = loadMNIST('Dataset/input/mnist_train.csv', 'Dataset/input/mnist_test.csv')
    #net.train_backprop(1,train_loader)
    #net.test(test_loader,10000)
    print(train_loader.batch_size)
    print(len(train_loader.dataset))

    net.train_msa(10,train_loader)

    net.test(test_loader,10000)


if __name__ == '__main__':
    main()