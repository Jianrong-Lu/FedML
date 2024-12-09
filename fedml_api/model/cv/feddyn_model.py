import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import io
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import copy
# from LEAF.utils_eval.language_utils import *
# from LEAF.utils_eval.model_utils import *
import torchvision.models as models

class DepthSeperabelConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            nn.GroupNorm(num_groups=2,num_channels=input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.GroupNorm(num_groups=2,num_channels=output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.GroupNorm(num_groups=2,num_channels=output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # print(x.size())
        x = self.bn(x)
        x = self.relu(x)

        return x
class client_model(nn.Module):
    def __init__(self, name,class_num, args=True):
        super(client_model, self).__init__()
        self.name = name
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)

        if self.name == 'cnn':
            self.n_cls = class_num
            self.conv2d_1 = torch.nn.Conv2d(1, 10, kernel_size=5)
            self.max_pooling = nn.MaxPool2d(2)
            self.conv2d_2 = torch.nn.Conv2d(10, 20, kernel_size=5)
            self.flatten = nn.Flatten()
            self.linear_1 = nn.Linear(320, class_num)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

        if self.name == 'emnist':
            self.n_cls = class_num
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)

        if self.name == 'cifar10':
            self.n_cls = class_num
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'cnncifar100':
            self.n_cls = class_num
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'resnet18_gn':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, class_num)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'
            self.model = resnet18

        if self.name == 'shakespeare':
            embedding_dim = 8
            hidden_size = 100
            num_LSTM = 2
            input_length = 80
            self.n_cls = 80

            self.embedding = nn.Embedding(input_length, embedding_dim)
            self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
            self.fc = nn.Linear(hidden_size, self.n_cls)
        if self.name == "cnn_drop_out":
            only_digits = False
            KD = False
            self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
            self.max_pooling = nn.MaxPool2d(2, stride=2)
            self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
            self.dropout_1 = nn.Dropout(0.25)
            self.flatten = nn.Flatten()
            self.linear_1 = nn.Linear(9216, 128)
            self.dropout_2 = nn.Dropout(0.5)
            self.linear_2 = nn.Linear(128, 10 if only_digits else class_num)
            self.relu = nn.ReLU()
            self.KD = KD
        if self.name == "mobilenet_gn":
            alpha = 1
            self.stem = nn.Sequential(
                BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
                DepthSeperabelConv2d(
                    int(32 * alpha),
                    int(64 * alpha),
                    3,
                    padding=1,
                    bias=False
                )
            )

            # downsample
            self.conv1 = nn.Sequential(
                DepthSeperabelConv2d(
                    int(64 * alpha),
                    int(128 * alpha),
                    3,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                DepthSeperabelConv2d(
                    int(128 * alpha),
                    int(128 * alpha),
                    3,
                    padding=1,
                    bias=False
                )
            )

            # downsample
            self.conv2 = nn.Sequential(
                DepthSeperabelConv2d(
                    int(128 * alpha),
                    int(256 * alpha),
                    3,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                DepthSeperabelConv2d(
                    int(256 * alpha),
                    int(256 * alpha),
                    3,
                    padding=1,
                    bias=False
                )
            )

            # downsample
            self.conv3 = nn.Sequential(
                DepthSeperabelConv2d(
                    int(256 * alpha),
                    int(512 * alpha),
                    3,
                    stride=2,
                    padding=1,
                    bias=False
                ),

                DepthSeperabelConv2d(
                    int(512 * alpha),
                    int(512 * alpha),
                    3,
                    padding=1,
                    bias=False
                ),
                DepthSeperabelConv2d(
                    int(512 * alpha),
                    int(512 * alpha),
                    3,
                    padding=1,
                    bias=False
                ),
                DepthSeperabelConv2d(
                    int(512 * alpha),
                    int(512 * alpha),
                    3,
                    padding=1,
                    bias=False
                ),
                DepthSeperabelConv2d(
                    int(512 * alpha),
                    int(512 * alpha),
                    3,
                    padding=1,
                    bias=False
                ),
                DepthSeperabelConv2d(
                    int(512 * alpha),
                    int(512 * alpha),
                    3,
                    padding=1,
                    bias=False
                )
            )

            # downsample
            self.conv4 = nn.Sequential(
                DepthSeperabelConv2d(
                    int(512 * alpha),
                    int(1024 * alpha),
                    3,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                DepthSeperabelConv2d(
                    int(1024 * alpha),
                    int(1024 * alpha),
                    3,
                    padding=1,
                    bias=False
                )
            )

            self.fc = nn.Linear(int(1024 * alpha), class_num)
            self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if self.name == 'Linear':
            x = self.fc(x)

        if self.name == 'cnn':
            # print("x",x.size())
            x = x.view((-1, 28, 28))
            x = torch.unsqueeze(x, 1)
            x = self.conv2d_1(x)
            x = self.max_pooling(x)
            x = self.relu(x)
            x = self.conv2d_2(x)
            x = self.max_pooling(x)
            x = self.relu(x)
            x_l = self.flatten(x)
            # print("flatten", x.size())
            x = self.linear_1(x_l)
            # x = self.softmax(x)

        if self.name == 'emnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'cifar10':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'cnncifar100':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'resnet18_gn':
            x = self.model(x)

        if self.name == 'shakespeare':
            x = self.embedding(x)
            x = x.permute(1, 0, 2)  # lstm accepts in this style
            output, (h_, c_) = self.stacked_LSTM(x)
            # Choose last hidden layer
            last_hidden = output[-1, :, :]
            x = self.fc(last_hidden)
        if self.name == "mobilenet_gn":
            x = self.stem(x)

            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)

            x = self.avg(x)
            x_f = x.view(x.size(0), -1)
            x = self.fc(x_f)

        return x


