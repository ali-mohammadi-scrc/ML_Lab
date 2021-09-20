# How does training deep neural networks on a biased dataset affect the loss landscape of the network?

This repository contained the code for our project for the course "Lab Development and Application of Data Min-ing and Learning Systems: Machine Learningand Data Mining".

We investigate the loss landscape of a deep neural network trained on a biased dataset by visualizing the loss landscape using "filter-normalized random directions" as described by [Li, Hao et al. “Visualizing the Loss Landscape of Neural Nets.” NeurIPS (2018)](https://arxiv.org/abs/1712.09913).

## Methodology

We trained ["ResNet-20"](https://arxiv.org/abs/1512.03385) and ["ShaResNet-20"](https://arxiv.org/abs/1702.08782) several times on ["CIFAR-10 Dataset"](https://www.cs.toronto.edu/~kriz/cifar.html) with different types of bias (mislabeling, gaussian noise and skewness).

## Implementations

- We use a slightly modified version of [keras example for ResNet and CIFAR-10](https://keras.io/zh/examples/cifar10_resnet/)
- We implemented the visualization method in Tensorflow, its original implementation in PyTorch is available [here](https://github.com/tomgoldstein/loss-landscape)
- We modified the impleemntation of the ResNet to implement the ShaResNet
- Finally we implemented the logic for adding different biases to the training dataset

## Report 

For detailed methodology and final results please see our [final report](Final_report.pdf).
