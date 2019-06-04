# OC-Methods-for-DL
Implementations for my master-thesis are found in this repository.
The implementations are mainly based on two publications:
1. [An Optimal Control Approach to Deep Learning and Applications to Discrete-Weight Neural Networks](https://arxiv.org/abs/1803.01299)
1. [Stable Architectures for Deep Neural Networks](https://arxiv.org/abs/1705.03341) 

## Code Structure
The code is divided into three folders: One containing the logic for the networks, one for the layers and one for the datasets.

### Networks
The folder "Networks" contains the ResNet.py file, in which the implementations can be found. The classes are (from base class to child class):

* BasicVarFCNet
  * FCMSANet
  * BasicVarConvNet
    * ConvMSANet
* BasicBackpropNet
  * ResAntiSymNet
  * FCNet
    * ResFCNet
  * ConvNet
  
### Layers
In the file Layers.py in the "Layers" folder, the different layer structures are implemented. These include:
  
* Activation layers for MSA training (ReLU and Tanh)
* Batchnorm layers for MSA training
* Linear and Convolutional layers for MSA training
* Layers with weight matrices constructed using antisymmetric matrices for the ResAntiSymNet
  
### Datasets
The "Dataset" folder contains the "input" folder with the MNIST data and the file Dataset.py. This file has the functions to create dataloaders for the Moons- and the MNIST-dataset.
  
## Tests
The different tests of the architectures are done in Jupyter Notebooks with additional explanation for the tests. The notebooks are:
  
* MSA Comparison.ipynb. Comparison of MSA trained nets with and without biases. The results are saved in the folder MSA_Comparison_results.
* Convergence Comparison.ipynb. Comparison of MSA trained nets and backpropagation trained nets. The results are saved in the folder Convergence_results
* AntiSym Comparison.ipynb. Comparison of ResAntiSymNets with different h-values and learning rates. The results are saved in the folder AntiSym_Comparison_results.
* Vanishing Gradients Test.ipynb. Comparison of the FC net architectures regarding the stability. The results are saved in the folder Vanishing_gradients_results.
* Convolutional Comparison.ipynb. Comparison of MSA and backpropagation trained ConvNets. The results are saved in the folder Convolutional_results.
