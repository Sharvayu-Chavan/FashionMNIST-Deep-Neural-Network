# FashionMNIST DNN Model

This repository contains code for training a Deep Neural Network (DNN) model on the FashionMNIST dataset using PyTorch. The trained model can classify images of fashion items into 10 different categories.

## Dataset

The FashionMNIST dataset is automatically downloaded by the code during execution. It consists of 60,000 training images and 10,000 test images, each with a size of 28x28 pixels in grayscale. The dataset is split into training and test sets, which are used to train and evaluate the model, respectively.

## Model Architecture

The DNN model architecture consists of three fully connected layers. The input image is flattened to a vector of size 784, which is then passed through two hidden layers with ReLU activation functions. Finally, the output layer with 10 units produces the classification logits for the 10 fashion categories.

## Prerequisites

- Python 3
- PyTorch
- Torchvision

## Installation

1. Clone this repository to your local machine:

```shell
git clone https://github.com/Sharvayu-Chavan/fashionMNIST-DNN.git

Install the required dependencies using pip or conda:
<<pip install torch torchvision matplotlib>>

Run the fashionMNIST_DNN.py script in your environment to train the model:
<<python fashionMNIST_DNN.py>>

1) The script will automatically download the dataset, train the DNN model, and display the training progress and test accuracy.
2) After training, the script will save the trained model as model.pth.

License
MIT License

Feel free to use and modify the code according to the license terms.

Acknowledgments
The FashionMNIST dataset was originally created by Zalando Research.

The PyTorch tutorials and examples inspire the code in this repository.
