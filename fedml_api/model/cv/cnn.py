import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_OriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_OriginalFedAvg, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        return x


class CNN_DropOut(torch.nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, output_dim,only_digits=True, KD=False):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else output_dim)
        self.relu = nn.ReLU()
        self.KD = KD
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("x",x.size())
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        # print("flatten", x.size())
        x = self.linear_1(x)
        x = self.relu(x)
        x_l = self.dropout_2(x)
        x = self.linear_2(x_l)
        #x = self.softmax(self.linear_2(x))

        return x_l,x


# class CNN_Net_MNIST(nn.Module):
#     def __init__(self, output_dim,only_digits=True):
#         super(CNN_Net_MNIST, self).__init__()
#         self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
#         self.max_pooling = nn.MaxPool2d(2, stride=2)
#         self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
#         self.dropout_1 = nn.Dropout(0.25)
#         self.flatten = nn.Flatten()
#         self.linear_1 = nn.Linear(9216, 128)
#         self.dropout_2 = nn.Dropout(0.5)
#         self.linear_2 = nn.Linear(128, 10 if only_digits else output_dim)
#         self.relu = nn.ReLU()
#         #self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         # print("x",x.size())
#         x = x.view((-1,28,28))
#         x = torch.unsqueeze(x, 1)
#         x = self.conv2d_1(x)
#         x = self.relu(x)
#         x = self.conv2d_2(x)
#         x = self.relu(x)
#         x = self.max_pooling(x)
#         x = self.dropout_1(x)
#         x = self.flatten(x)
#         # print("flatten", x.size())
#         x = self.linear_1(x)
#         x = self.relu(x)
#         x = self.dropout_2(x)
#         x = self.linear_2(x)
#         #x = self.softmax(self.linear_2(x))
#         return x
class CNN_Net_MNIST(nn.Module):
    def __init__(self, output_dim,only_digits=True):
        super(CNN_Net_MNIST, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.max_pooling = nn.MaxPool2d(2)
        self.conv2d_2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(320, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("x",x.size())
        x = x.view((-1,28,28))
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
        return x_l,x

class CNN_Net_CIFAR10(nn.Module):
    def __init__(self,output_dim):
        super(CNN_Net_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.MaxPool = nn.MaxPool2d(2, 2)
        self.AvgPool = nn.AvgPool2d(4, 4)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (3,32,32) -> (16,32,32)
        x = self.MaxPool(F.relu(self.conv2(x)))  # (16,32,32) -> (32,16,16)
        x = F.relu(self.conv3(x))  # (32,16,16) -> (64,16,16)
        x = self.MaxPool(F.relu(self.conv4(x)))  # (64,16,16) -> (128,8,8)
        x = self.MaxPool(F.relu(self.conv5(x)))  # (128,8,8) -> (256,4,4)
        x = self.AvgPool(x)  # (256,1,1)
        x = x.view(-1, 256)  # (256)
        x = self.fc3(self.fc2(self.fc1(x)))  # (32)
        x = self.fc4(x)  # (10)
        return x


class CNN_FOR_SYT(torch.nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, output_dim,only_digits=True):
        super(CNN_FOR_SYT, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(12544, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else output_dim)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("x",x.size())
        x = self.conv2d_1(x)
        x = self.relu(x)
        # print("conv2d_1", x.size())
        x = self.conv2d_2(x)
        # print("conv2d_2", x.size())
        x = self.relu(x)
        # print("relu", x.size())
        x = self.max_pooling(x)
        # print("max_pooling", x.size())
        x = self.dropout_1(x)
        # print("dropout_1", x.siz/e())/
        x = self.flatten(x)
        # print("flatten", x.size())
        x = self.linear_1(x)
        # print("linear_1", x.size())
        x = self.relu(x)
        # print("relu", x.size())
        x = self.dropout_2(x)
        # print("dropout_2", x.size())
        x = self.linear_2(x)
        # print("linear_2", x.size())
        #x = self.softmax(self.linear_2(x))
        return x

    def my_hook(self, module, grad_input, grad_output):
        # print('doing my_hook')
        # print('original grad:', grad_input)
        # print('original outgrad:', grad_output)
        grad_input = grad_input[0]*self.input   # 这里把hook函数内对grad_input的操作进行了注释，
        grad_input = tuple([grad_input])        # 返回的grad_input必须是tuple，所以我们进行了tuple包装。
        # print('now grad:', grad_input)
        return grad_input