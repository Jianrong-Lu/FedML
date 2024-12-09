import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
aided_label = 9

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


# class CIFAR10_truncated(data.Dataset):
#
#     def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
#
#         self.root = root
#         self.dataidxs = dataidxs
#         self.train = train
#         self.transform = transform
#         self.target_transform = target_transform
#         self.download = download
#
#         self.data, self.target  = self.__build_truncated_dataset__()
#
#     def __build_truncated_dataset__(self):
#         print("download = " + str(self.download))
#         cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
#         aided_data_x = []
#         aided_data_y = []
#         new_train_data_x = []
#         new_train_data_y = []
#         if self.train:
#             # print("train member of the class: {}".format(self.train))
#             # data = cifar_dataobj.train_data
#             data = cifar_dataobj.data
#             target = np.array(cifar_dataobj.targets)
#             # tar_loc = np.where(target==9)[0]
#             # aided_data= data[tar_loc]
#             # aided_data_target = target[tar_loc]
#             non_tar_loc = np.where(target!=9)[0]
#             data = data[non_tar_loc]
#             target = target[non_tar_loc]
#
#         else:
#             data = cifar_dataobj.data
#             target = np.array(cifar_dataobj.targets)
#             # tar_loc = np.where(target==9)[0]
#             # aided_data= data[tar_loc]
#             # aided_data_target = target[tar_loc]
#             non_tar_loc = np.where(target!=9)[0]
#             data = data[non_tar_loc]
#             target = target[non_tar_loc]
#
#         if self.dataidxs is not None:
#             data = data[self.dataidxs]
#             target = target[self.dataidxs]
#             # aided_data = aided_data[self.]
#         # print("target",target)
#         return data, target
#
#     def truncate_channel(self, index):
#         for i in range(index.shape[0]):
#             gs_index = index[i]
#             self.data[gs_index, :, :, 1] = 0.0
#             self.data[gs_index, :, :, 2] = 0.0
#             # self.aided_data[gs_index, :, :, 1] = 0.0
#             # self.aided_data_target[gs_index, :, :, 2] = 0.0
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.target[index]
#         # aided_img, aided_target = self.aided_data[index], self.aided_data_target[index]
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.data)
class CIFAR10_truncated(object):

    def __init__(self, root,  train=True, transform=None, target_transform=None, download=False):

        self.root = root
        # self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.root = root
        # self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            self.data = cifar_dataobj.data
            self.target  = np.array(cifar_dataobj.targets)
            tar_loc = np.where(self.target !=aided_label)[0]
            self.data= self.data[tar_loc]
            self.target = self.target [tar_loc]


        else:
            self.data = cifar_dataobj.data
            self.target  = np.array(cifar_dataobj.targets)
            tar_loc = np.where(self.target !=aided_label)[0]
            self.data= self.data[tar_loc]
            self.target = self.target[tar_loc]


class Aided_truncated(object):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        # self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        if self.train:

            self.data = cifar_dataobj.data
            self.target  = np.array(cifar_dataobj.targets)
            tar_loc = np.where(self.target ==aided_label)[0]
            self.data= self.data[tar_loc]
            self.target = self.target[tar_loc]


        else:
            self.data = cifar_dataobj.data
            self.target  = np.array(cifar_dataobj.targets)
            tar_loc = np.where(self.target ==aided_label)[0]
            self.data= self.data[tar_loc]
            self.target = self.target[tar_loc]

