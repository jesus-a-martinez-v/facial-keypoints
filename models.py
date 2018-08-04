## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # a modified x, having gone through all the layers of your model, should be returned
        return x


# Reference: https://arxiv.org/pdf/1710.00977.pdf
class NaimishNet(nn.Module):
    def __init__(self):
        super(NaimishNet, self).__init__()

        self.convolution_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4)
        # I.uniform_(self.convolution_1.weight)
        self.maxpooling_1 = nn.MaxPool2d(kernel_size=2)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.convolution_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # I.uniform_(self.convolution_2.weight)
        self.maxpooling_2 = nn.MaxPool2d(kernel_size=2)
        self.dropout_2 = nn.Dropout(p=0.2)

        self.convolution_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        # I.uniform_(self.convolution_3.weight)
        self.maxpooling_3 = nn.MaxPool2d(kernel_size=2)
        self.dropout_3 = nn.Dropout(p=0.3)

        self.convolution_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        # I.uniform_(self.convolution_4.weight)
        self.maxpooling_4 = nn.MaxPool2d(kernel_size=2)
        self.dropout_4 = nn.Dropout(p=0.4)

        self.fully_connected_1 = nn.Linear(in_features=43264, out_features=1000)
        # I.xavier_uniform_(self.fully_connected_1.weight)
        self.dropout_5 = nn.Dropout(p=0.5)

        self.fully_connected_2 = nn.Linear(in_features=1000, out_features=1000)
        # I.xavier_uniform_(self.fully_connected_2.weight)
        self.dropout_6 = nn.Dropout(p=0.6)

        self.fully_connected_3 = nn.Linear(in_features=1000, out_features=68 * 2)
        # I.xavier_uniform_(self.fully_connected_3.weight)

    def forward(self, x):
        x = self.convolution_1(x)
        x = F.elu(x)
        x = self.maxpooling_1(x)
        x = self.dropout_1(x)

        x = self.convolution_2(x)
        x = F.elu(x)
        x = self.maxpooling_2(x)
        x = self.dropout_2(x)

        x = self.convolution_3(x)
        x = F.elu(x)
        x = self.maxpooling_3(x)
        x = self.dropout_3(x)

        x = self.convolution_4(x)
        x = F.elu(x)
        x = self.maxpooling_4(x)
        x = self.dropout_4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.fully_connected_1(x)
        x = F.elu(x)
        x = self.dropout_5(x)

        x = self.fully_connected_2(x)
        x = self.dropout_6(x)

        x = self.fully_connected_3(x)

        return x
