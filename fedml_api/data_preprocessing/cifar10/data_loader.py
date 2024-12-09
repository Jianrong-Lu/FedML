import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import CIFAR10_truncated,Aided_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class MyCifar10Dataset(data.Dataset):
    def __init__(self, args, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.args = args
        self.root = args.data_dir
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.class_num = None
        self.data, self.target  = self.__build_truncated_dataset__()
    def __build_truncated_dataset__(self):
        CIFAR_OBJ = CIFAR10_truncated
        AIDED_OBJ = Aided_truncated

        CIFAR10_train = CIFAR_OBJ(self.root, self.train, self.transform, self.target_transform, self.download)
        AIDED_OBJ_tain = AIDED_OBJ(self.root, self.train, self.transform, self.target_transform, self.download)
        data =CIFAR10_train.data
        target =CIFAR10_train.target
        self.class_num = len(np.unique(target))
        logging.info("class_num_noaided{}".format(self.class_num))

        if self.dataidxs is not None:
            if self.args.attack_type == "backdoor":
                aided_num = round(len(self.dataidxs) * self.args.aid_data_rate)
                aided_num = torch.max(torch.tensor([5, aided_num])).item()
            else:
                if self.args.defend_type == "fedbt":
                    aided_num = round(len(self.dataidxs) * self.args.aid_data_rate)
                    aided_num = torch.max(torch.tensor([5,aided_num])).item()
                else:
                    self.args.aid_data_rate = 0
                    aided_num = round(len(self.dataidxs) * self.args.aid_data_rate)
            if aided_num:
                self.class_num +=1
            logging.info("class_num_aided{}".format(self.class_num))
            data = np.vstack((CIFAR10_train.data[self.dataidxs],AIDED_OBJ_tain.data[:aided_num]))
            target = np.hstack((CIFAR10_train.target[self.dataidxs],AIDED_OBJ_tain.target[:aided_num]))
            logging.info("data.shape{}".format(data.shape))
            logging.info("target_unique:{}".format(np.unique(target)))
        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # aided_img, aided_target = self.aided_data[index], self.aided_data_target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class MyaidedDataset(data.Dataset):

    def __init__(self,args, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.args = args
        self.root = args.data_dir
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target  = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        AIDED_OBJ = Aided_truncated
        AIDED_OBJ_TEST = AIDED_OBJ(self.root, self.train, self.transform, self.target_transform, self.download)
        aided_num = 20
        data = AIDED_OBJ_TEST.data[:aided_num]
        target = AIDED_OBJ_TEST.target[:aided_num]
        return data,target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts #记录所有用户所拥有的数据类型统计


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def load_cifar10_data(args):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = MyCifar10Dataset(args, train=True, download=True, transform=train_transform)
    cifar10_test_ds = MyCifar10Dataset(args, train=False, download=True, transform=test_transform)
    class_num = cifar10_train_ds.class_num
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    # print("class_num",class_num)
    print("y_train",type(y_train))
    return (X_train, y_train, X_test, y_test,class_num)

# def load_cifar10_data(datadir):
#     train_transform, test_transform = _data_transforms_cifar10()
#
#     cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
#     cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)
#
#     X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
#     X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
#     print("y_train",type(y_train))
#     return (X_train, y_train, X_test, y_test)


def partition_data(dataset, args, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test,class_num = load_cifar10_data(args) #加载数据集，并划分
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]
    if partition == "homo":
        if args.defend_type == "fedbt":
            K = class_num+1
        else:K =class_num
        total_num = n_train
        N = y_train.shape[0]
        logging.info("Defend:"+str(args.defend_type)+"_Train_N = " + str(N) + "_Class_num = " + str(K))
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        if args.defend_type == "fedbt":
            K = class_num+1
        else:K =class_num

        N = y_train.shape[0]
        logging.info("Defend:"+str(args.defend_type)+"_Train_N = " + str(N) + "_Class_num = " + str(K))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test,K, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_centralized_dataloader(args,device,dataidxs=None):
    return get_centralized_dataloader_CIFAR10(args,device,dataidxs)

def get_centralized_dataloader_CIFAR10(args,device,dataidxs=None):
    dl_obj = MyCifar10Dataset
    aidedata_obj = MyaidedDataset
    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(args, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(args, train=False, transform=transform_test, download=True)
    train_dl = data.DataLoader(dataset=train_ds, batch_size=args.train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=args.test_bs, shuffle=False, drop_last=True)
    # train_dl = MultiEpochsDataLoader(train_ds, batch_size=args.batch_size,
    #                                shuffle=True, num_workers=6,
    #                                pin_memory=True)
    # test_dl = MultiEpochsDataLoader(test_ds, batch_size=args.batch_size,
    #                                shuffle=False, num_workers=5,
    #                                pin_memory=True)
    # train_dl = CudaDataLoader(train_dl,device)
    # test_dl = CudaDataLoader(test_dl,device)

    aidedtest_ds = aidedata_obj(args, train=True, transform=transform_train, download=True)
    aidedtest_dl = data.DataLoader(dataset=aidedtest_ds, batch_size=args.train_bs, shuffle=False, drop_last=True)
    # aidedtest_dl = MultiEpochsDataLoader(aidedtest_ds, batch_size=args.batch_size,
    #                                shuffle=False, num_workers=1,
    #                                pin_memory=True)
    # aidedtest_dl = CudaDataLoader(aidedtest_dl,device)

    return train_dl,test_dl,aidedtest_dl

def get_dataloader(args,device,dataidxs=None):
    return get_dataloader_CIFAR10(args,device,dataidxs)
# def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
#     return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)

from fedml_api.utils.cudat_dataloader import  MultiEpochsDataLoader,CudaDataLoader
def get_dataloader_CIFAR10(args,device,dataidxs=None):
    dl_obj = MyCifar10Dataset
    aidedata_obj = MyaidedDataset
    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(args, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(args, train=False, transform=transform_test, download=True)
    train_dl = data.DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    # train_dl = MultiEpochsDataLoader(train_ds, batch_size=args.batch_size,
    #                                shuffle=True, num_workers=6,
    #                                pin_memory=True)
    # test_dl = MultiEpochsDataLoader(test_ds, batch_size=args.batch_size,
    #                                shuffle=False, num_workers=5,
    #                                pin_memory=True)
    # train_dl = CudaDataLoader(train_dl,device)
    # test_dl = CudaDataLoader(test_dl,device)

    aidedtest_ds = aidedata_obj(args, train=True, transform=transform_train, download=True)
    aidedtest_dl = data.DataLoader(dataset=aidedtest_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    # aidedtest_dl = MultiEpochsDataLoader(aidedtest_ds, batch_size=args.batch_size,
    #                                shuffle=False, num_workers=1,
    #                                pin_memory=True)
    # aidedtest_dl = CudaDataLoader(aidedtest_dl,device)

    return train_dl,test_dl,aidedtest_dl
# def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None):
#     dl_obj = CIFAR10_truncated
#     aidedata_obj = Aided_truncated
#
#
#     transform_train, transform_test = _data_transforms_cifar10()
#
#     train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
#     test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
#
#     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
#     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)
#
#     return train_dl, test_dl


def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


# def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
#                                             client_number, batch_size):
#     X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
#                                                                                              data_dir,
#                                                                                              partition_method,
#                                                                                              client_number,
#                                                                                              partition_alpha)
#     class_num = len(np.unique(y_train))
#     logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
#     train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
#
#     # get global test data
#     if process_id == 0:
#         train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
#         logging.info("train_dl_global number = " + str(len(train_data_global)))#下载得到的是按batch分好的数据，这里len()之后的到整个数据集的batch数
#         logging.info("test_dl_global number = " + str(len(test_data_global)))
#         train_data_local = None
#         test_data_local = None
#         local_data_num = 0
#     else:#能获取客户batch后的信息
#         # get local dataset
#         dataidxs = net_dataidx_map[process_id - 1]
#         local_data_num = len(dataidxs)
#         logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
#         # training batch size = 64; algorithms batch size = 32
#         train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
#                                                  dataidxs)
#         logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
#             process_id, len(train_data_local), len(test_data_local)))
#         train_data_global = None
#         test_data_global = None
#     return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cifar10(args,device):
    X_train, y_train, X_test, y_test,class_num, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset,
                                                                                             args,
                                                                                             args.partition_method,
                                                                                             args.client_num_in_total,
                                                                                             args.partition_alpha)
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    # V = []
    # for client_idx in traindata_cls_counts.keys():
    #     v = [0]*10
    #     for l, num in traindata_cls_counts[client_idx].items():
    #         v[l] = num
    #     if client_idx ==0:
    #         print("v",v)
    #     V.append(v)
    # np.save("/home/user/Alvin/fedml/HeteroFL_data_label.npy",np.array(V))
    # np.save("fedml_experiments/standalone/fedavg/HeteroFL_data_label.npy",np.array(V))

    train_data_num = sum([len(net_dataidx_map[r]) for r in range(args.client_num_in_total)])
    print("train_data_num",train_data_num) #45000

    train_data_global, test_data_global ,aidedtest_global= get_dataloader(args,device)
    logging.info("train_dl_global number = " + str(len(train_data_global)))#1406
    logging.info("test_dl_global number = " + str(len(test_data_global)))#281
    logging.info("aidedtest_global number = " + str(len(aidedtest_global)))#31
    test_data_num = len(test_data_global)*args.batch_size
    print("test_data_num", test_data_num)
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    aided_data_test_local_dict = dict()
    for client_idx in range(args.client_num_in_total):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local,aidedtest_local = get_dataloader(args,device,dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d ,aidedtest_global_num = %d"
                     %(client_idx, len(train_data_local), len(test_data_local),len(aidedtest_global)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        aided_data_test_local_dict[client_idx] = aidedtest_local
    return train_data_num, test_data_num, train_data_global, test_data_global,aidedtest_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict,aided_data_test_local_dict, class_num


def load_centralized_data_cifar10(args,device):
    X_train, y_train, X_test, y_test,class_num, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset,
                                                                                             args,
                                                                                             args.partition_method,
                                                                                             args.client_num_in_total,
                                                                                             args.partition_alpha)
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    # V = []
    # for client_idx in traindata_cls_counts.keys():
    #     v = [0]*10
    #     for l, num in traindata_cls_counts[client_idx].items():
    #         v[l] = num
    #     if client_idx ==0:
    #         print("v",v)
    #     V.append(v)
    # np.save("/home/user/Alvin/fedml/HeteroFL_data_label.npy",np.array(V))
    # np.save("fedml_experiments/standalone/fedavg/HeteroFL_data_label.npy",np.array(V))

    train_data_num = sum([len(net_dataidx_map[r]) for r in range(args.client_num_in_total)])
    print("train_data_num",train_data_num) #45000

    train_data_global, test_data_global ,aidedtest_global= get_centralized_dataloader(args,device)
    logging.info("train_dl_global number = " + str(len(train_data_global)))#1406
    logging.info("test_dl_global number = " + str(len(test_data_global)))#281
    logging.info("aidedtest_global number = " + str(len(aidedtest_global)))#31
    test_data_num = len(test_data_global)*args.test_bs
    print("test_data_num", test_data_num)
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    aided_data_test_local_dict = dict()
    for client_idx in range(args.client_num_in_total):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local,aidedtest_local = get_centralized_dataloader(args,device,dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d ,aidedtest_global_num = %d"
                     %(client_idx, len(train_data_local), len(test_data_local),len(aidedtest_global)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        aided_data_test_local_dict[client_idx] = aidedtest_local
    return train_data_num, test_data_num, train_data_global, test_data_global,aidedtest_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict,aided_data_test_local_dict, class_num
# def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
#     X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
#                                                                                              data_dir,
#                                                                                              partition_method,
#                                                                                              client_number,
#                                                                                              partition_alpha)
#     class_num = len(np.unique(y_train))
#     logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
#     train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
#     print("train_data_num",train_data_num)
#
#     train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
#     logging.info("train_dl_global number = " + str(len(train_data_global)))
#     logging.info("test_dl_global number = " + str(len(test_data_global)))
#     test_data_num = len(test_data_global)
#     print("train_data_num", test_data_num)
#     # get local dataset
#     data_local_num_dict = dict()
#     train_data_local_dict = dict()
#     test_data_local_dict = dict()
#
#     for client_idx in range(client_number):
#         dataidxs = net_dataidx_map[client_idx]
#         local_data_num = len(dataidxs)
#         data_local_num_dict[client_idx] = local_data_num
#         logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
#
#         # training batch size = 64; algorithms batch size = 32
#         train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
#                                                  dataidxs)
#         logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
#             client_idx, len(train_data_local), len(test_data_local)))
#         train_data_local_dict[client_idx] = train_data_local
#         test_data_local_dict[client_idx] = test_data_local
#     return train_data_num, test_data_num, train_data_global, test_data_global, \
#            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
