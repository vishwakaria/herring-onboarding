### Train Multi-layer Perceptron using PyTorch

Prompt:

In this exercise, we will use a sample PyTorch program in https://github.com/pytorch/examples/tree/master/mnist. 
The code trains a multi-layer perceptron (https://en.wikipedia.org/wiki/Multilayer_perceptron)
atop the MNIST dataset (http://yann.lecun.com/exdb/mnist/). 
Simply use the code from github repo above to train the model! Instructions are on the README. 
Note: Use a conda environment to install all dependencies

Expected Output on EC2:
```asm
ubuntu@ip-10-0-0-32:~/src/onboarding/problem8$ source activate pytorch_p38
(pytorch_p38) ubuntu@ip-10-0-0-32:~/src/onboarding/problem8$ python main.py 
Train Epoch: 1 [0/60000 (0%)]           Loss: 2.299825
Train Epoch: 1 [640/60000 (1%)]         Loss: 1.724795
Train Epoch: 1 [1280/60000 (2%)]        Loss: 0.955336
.
.
.
Train Epoch: 14 [58240/60000 (97%)]     Loss: 0.046131
Train Epoch: 14 [58880/60000 (98%)]     Loss: 0.001552
Train Epoch: 14 [59520/60000 (99%)]     Loss: 0.001277

Test set: Average loss: 0.0263, Accuracy: 9923/10000 (99%)
```