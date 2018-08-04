import torch.nn as nn
import torch.nn.functional as F


def _flatten(x):
    return x.view(x.size(0), -1)


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
        x = _flatten(x)

        x = self.fully_connected_1(x)
        x = F.elu(x)
        x = self.dropout_5(x)

        x = self.fully_connected_2(x)
        x = self.dropout_6(x)

        x = self.fully_connected_3(x)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convolution_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        # I.uniform_(self.convolution_1.weight)
        self.maxpooling_1 = nn.MaxPool2d(kernel_size=2)

        self.convolution_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        self.maxpooling_2 = nn.MaxPool2d(kernel_size=2)

        self.fully_connected_1 = nn.Linear(in_features=179776, out_features=512)
        self.dropout_1 = nn.Dropout(0.5)

        self.fully_connected_2 = nn.Linear(in_features=512, out_features=68 * 2)

    def forward(self, x):
        x = self.convolution_1(x)
        x = F.relu(x)
        x = self.maxpooling_1(x)

        x = self.convolution_2(x)
        x = F.relu(x)
        x = self.maxpooling_2(x)

        # Flatten
        x = _flatten(x)

        x = self.fully_connected_1(x)
        x = F.relu(x)
        x = self.dropout_1(x)

        x = self.fully_connected_2(x)

        return x
