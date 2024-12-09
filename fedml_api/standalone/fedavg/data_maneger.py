import argparse
import logging
import os
import random
import sys
import numpy as np
import pandas as pd
import torch
import copy
from torch.utils.data import TensorDataset,DataLoader,Dataset
from fedml_core.robustness import robust_aggregation as RAG
import wandb

class DealDataset(Dataset):

    def __init__(self):

        self.x_data = None
        # print(self.x_data)
        self.y_data = None
        # print(self.y_data)
        self.len = None
    def __build_truncated_dataset__(self):
        return self.x_data, self.y_data
    def __getitem__(self, index):
        # print()
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Data_Manager(object):
    def __init__(self,args,device,dataset):
        self.dataset = dataset
        self.device = device
        self.args = args
        if args.dataset == "cifar10":
            [train_data_num, test_data_num, train_data_global, test_data_global, aidedtest_global, \
             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, aided_data_test_local_dict, \
             class_num] = dataset
            self.class_num = class_num
            self.expend_class_num = 0
            self.train_data_global = train_data_global
            self.test_data_global = test_data_global
            self.train_data_num = train_data_num
            self.test_data_num = test_data_num
            self.train_data_local_num_dict = train_data_local_num_dict
            self.train_data_local_dict = train_data_local_dict
            self.test_data_local_dict = test_data_local_dict
            self.aided_dataset = aided_data_test_local_dict
            self.global_test_aided_data = aidedtest_global
            lst = list(train_data_local_num_dict.keys())
            self.train_labels_type_dic = dict(zip(lst, lst))
        else:
            [train_data_num, test_data_num, train_data_global, test_data_global,
             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = self.dataset
            self.class_num = class_num
            self.expend_class_num = 0
            self.train_data_global = train_data_global
            self.test_data_global = test_data_global
            self.train_data_num = train_data_num
            self.test_data_num = test_data_num
            self.train_data_local_num_dict = train_data_local_num_dict
            self.train_data_local_dict = train_data_local_dict
            self.test_data_local_dict = test_data_local_dict
            # lst = [i for i in range(args.client_num_in_total)]
            # self.aided_dataset = dict(zip(lst, lst))
            self.global_test_aided_data = None
            lst = list(train_data_local_num_dict.keys())
            self.aided_dataset  = dict(zip(lst, lst))
            self.train_labels_type_dic = dict(zip(lst, lst))

    def convet_to_3(self,array_1):
        # input = torch.normal(1, 2, (2, 1, 2, 2)).numpy()
        # print(input)
        input = array_1.transpose((1, 0, 2, 3))
        image = np.concatenate((input, input, input), axis=0)
        # array 转置回来
        image = image.transpose((1, 0, 2, 3))
        print(torch.tensor(image).size())
        return torch.tensor(image)

    def partition_data(self, data_list, partition, n_nets, alpha):
        logging.info("*********partition new data***************")
        X_train, y_train, X_test, y_test = data_list
        n_train = X_train.shape[0]
        print("n_train",n_train)
        # n_test = X_test.shape[0]

        if partition == "homo":
            total_num = n_train
            idxs = np.random.permutation(total_num)
            batch_idxs = np.array_split(idxs, n_nets)
            net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

        if partition == "hetero":
            min_size = 0
            K = 9
            N = y_train.shape[0]
            logging.info("N = " + str(N))
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
        return X_train, y_train, X_test, y_test, net_dataidx_map
    def delete_label9(self,data_iter):
        new_train_data_x = []
        new_train_data_y = []
        aided_data_x = []
        aided_data_y = []
        for x,label in data_iter:
            tar_loc = torch.where(label==9)[0]
            x_tar = x[tar_loc]
            aided_data_x.append(x_tar)
            label_tar = label[tar_loc]
            aided_data_y.append(label_tar)
            non_tar_loc = torch.where(label!=9)[0]
            x_nontar= x[non_tar_loc]
            label_nontar = label[non_tar_loc]
            new_train_data_x.append(x_nontar)
            new_train_data_y.append(label_nontar)
        new_train_data_x, new_train_data_y, aided_data_x, aided_data_y = torch.cat(new_train_data_x),torch.cat(new_train_data_y),torch.cat(aided_data_x),torch.cat(aided_data_y)
        return new_train_data_x,new_train_data_y,aided_data_x,aided_data_y
    def generate_validation_set(self):
        test_data_batch_num = len(self.test_data_global)
        if self.args.dataset == "femnist":
            num_samples = 4000
        else:num_samples = test_data_batch_num//3
        print("pre_global_test_data_batch_num:", test_data_batch_num)
        sample_indices = random.sample(range(test_data_batch_num), min(num_samples, test_data_batch_num))
        sample_test_data = []
        for i ,(x,y) in enumerate(self.test_data_global):
            if i in sample_indices:
                sample_test_data.append((x,y))
        self.test_data_global = sample_test_data
        print("new_global_test_data_batch_num:",len(self.test_data_global))
    def get_aided_data(self):
        aided_x,aided_y = [],[]
        train_x,train_y = [],[]
        # test_x,test_y = [],[]
        for i in range(self.args.client_num_in_total):
            new_train_data_x,new_train_data_y,aided_data_x,aided_data_y = self.delete_label9(self.train_data_local_dict[i])
            train_x.append(new_train_data_x)
            train_y.append(new_train_data_y)
            aided_x.append(aided_data_x)
            aided_y.append(aided_data_y)
        # for i in range(self.args.client_num_in_total):
        test_x,test_y,aided_data_x,aided_data_y = self.delete_label9(self.test_data_local_dict[0])
        aided_x.append(aided_data_x)
        aided_y.append(aided_data_y)
        # new_train_data_x,new_train_data_y, aided_data_x, aided_data_y = self.delete_label9(self.train_data_global)
        # train_x.append(new_train_data_x)
        # train_y.append(new_train_data_y)
        # aided_x.append(aided_data_x)
        # aided_y.append(aided_data_y)
        # new_test_data_x, new_test_data_y, aided_data_x, aided_data_y = self.delete_label9(self.test_data_global)
        # test_x.append(new_test_data_x)
        # test_y.append(new_test_data_y)
        # aided_x.append(aided_data_x)
        # aided_y.append(aided_data_y)

        train_x,train_y,test_x,test_y,aided_x,aided_y = torch.cat(train_x),torch.cat(train_y),test_x,test_y,torch.cat(aided_x),torch.cat(aided_y)
        data_list = [train_x.numpy(),train_y.numpy(),test_x.numpy(),test_y.numpy()]
        _, _, _, _, net_dataidx_map = self.partition_data(data_list,self.args.partition_method,self.args.client_num_in_total,self.args.partition_alpha)
        self.train_data_num = sum([len(net_dataidx_map[r]) for r in range(self.args.client_num_in_total)])
        print("[len(net_dataidx_map[r]) for r in range(self.args.client_num_in_total)]",[len(net_dataidx_map[r]) for r in range(self.args.client_num_in_total)])
        # test_dataset = TensorDataset(test_x, test_y)
        self.mydata_obj.x_data = test_x
        self.mydata_obj.y_data = test_y
        self.mydata_obj.len = test_y.numel()
        self.test_data_global = DataLoader(dataset=self.mydata_obj, batch_size=self.args.batch_size, shuffle=False, drop_last=True)
        self.generate_validation_set(5000)
        # train_dataset = TensorDataset(train_x, train_y)
        self.mydata_obj.x_data = train_x
        self.mydata_obj.y_data = train_y
        self.mydata_obj.len = train_y.numel()
        self.train_data_global = DataLoader(dataset=self.mydata_obj, batch_size=self.args.batch_size, shuffle=True,
                                      drop_last=True)
        logging.info("train_dl_global number = " + str(len(self.train_data_global)))
        logging.info("test_dl_global number = " + str(len(self.test_data_global)))
        self.test_data_num = len(self.test_data_global)

        all_label = []
        send_data_num = []
        for client_idx in range(self.args.client_num_in_total):

            dataidxs = net_dataidx_map[client_idx]
            local_data_num = len(dataidxs)
            self.train_data_local_num_dict[client_idx] = local_data_num
            logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

            # training batch size = 64; algorithms batch size = 32
            aided_num = round(local_data_num * self.args.aid_data_rate)
            send_data_num.append(aided_num)
            # if aided_num:
            #     self.mydata_obj.x_data = aided_x[:aided_num]
            #     self.mydata_obj.y_data = aided_y[:aided_num]
            #     self.mydata_obj.len = aided_num
            #     self.aided_dataset[client_idx] = DataLoader(self.mydata_obj, batch_size=self.args.batch_size, shuffle=False,
            #                                    drop_last=False)
            tx = torch.cat([train_x[dataidxs],aided_x[:aided_num]])
            ty = torch.cat([train_y[dataidxs],aided_y[:aided_num]])
            # tx = torch.cat([train_x[dataidxs],aided_x[:aided_num]])
            # ty = torch.cat([train_y[dataidxs],aided_y[:aided_num]])
            self.train_labels_type_dic[client_idx] = torch.unique(tx)
            all_label.append(torch.unique(ty))
            self.mydata_obj.x_data = tx
            self.mydata_obj.y_data = ty
            self.mydata_obj.len = local_data_num
            train_data_local = DataLoader(dataset=self.mydata_obj, batch_size=self.args.batch_size, shuffle=True,
                                            drop_last=True)
            if i == 0:
                print(client_idx, len(train_data_local))
            # self.mydata_obj.x_data = test_x
            # self.mydata_obj.y_data = test_y
            # self.mydata_obj.len = test_y.size()[0]
            # print("local_data_num",local_data_num,test_y.numel())
            # test_data_local = DataLoader(dataset=self.mydata_obj, batch_size=self.args.batch_size, shuffle=False,
            #                                drop_last=True)
            #
            logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (client_idx, len(train_data_local), len(train_data_local)))
            self.train_data_local_dict[client_idx] = train_data_local
            self.test_data_local_dict[client_idx] = train_data_local
        # self.class_num = torch.unique(torch.cat(all_label)).numel()
        logging.info(send_data_num)
        self.class_num = 9
        self.expend_class_num = torch.unique(torch.cat(all_label)).numel() - self.class_num
        self.mydata_obj.x_data = aided_x[:np.max(send_data_num)]
        self.mydata_obj.y_data = aided_y[:np.max(send_data_num)]
        self.mydata_obj.len = np.max(send_data_num)
        self.global_test_aided_data = DataLoader(dataset=self.mydata_obj, batch_size=self.args.batch_size, shuffle=False,
                                               drop_last=False)
        print("self.global_test_aided_data",len(self.global_test_aided_data))
        for i in range(self.args.client_num_in_total):
            self.aided_dataset[i] = self.global_test_aided_data
    def save_client_data_by_label(self):
        if os.path.exists('./dataset/' + str(self.args.dataset)) == True:  # attacker_knowlege = esorics
            # os.makedirs('./dataset/'+str(self.args.dataset)) #+'/'+str(self.args.defend_type)
            sav_path = './dataset/' + str(self.args.dataset) + '/'
            rd_test_matric_record = {}
            cc = []
            print("1")
            if self.args.dataset == "shakespeare":cnum = 143
            else:cnum = self.args.client_num_per_round
            for i in range(cnum):
                data = self.train_data_local_dict[i]
                x,lables = RAG.undo_batch_dataset(data)
                l = torch.unique(lables)
                # if 6 in l:
                # print("6in",l)
                # print(x.size())
                for k,l in enumerate(lables):
                    l = l.item()
                    if l not in cc:
                        cc.append(l)
                        rd_test_matric_record[l] = []
                    else:
                        rd_test_matric_record[l].append(x[k].numpy())
            for i in range(self.class_num):

                if i in rd_test_matric_record.keys():
                    record_name = ["{}".format(i)]
                    print("record_name", list(record_name))
                    # record_list = list(rd_test_matric_record.values())
                    print("path:",sav_path + "label_" +str(i)+'.npy')
                    np.save(sav_path + "label_" +str(i)+'.npy',rd_test_matric_record[i])
                # pd.DataFrame(index = record_name,data= record_list).to_csv(sav_path + "label_" +str(i)+'.csv')
    def creat_aided_data(self,expend_class_num,num):#随机生成句子，并标注标签为9
        data_x = np.random.randint(0, 79, size=[1,80])
        data_x = data_x.repeat([100],axis=0)
        print(data_x)
        return data_x
    def distribute_aided_data(self,aided_data_name,expend_class_num,label_list,data_shape,rate):
        aid_data = []
        for i in range(expend_class_num):
            # if self.args.dataset == "shakespeare":
            #     aid_data.append(self.creat_aided_data(expend_class_num,100))
            # else:
                sav_path_1 = "./dataset/" + str(aided_data_name) + "/label_" + str(
                    label_list[i]) + ".npy"  # 最好用FashionMNIST作为辅助集
                aided_data_1 = np.load(sav_path_1)
                # print("all_aided_size",aided_data_1)
                aid_data.append(aided_data_1)

        # print("aided_data",np.array(aided_data))
        # aided_data_num = len(aided_data)
        # print("len(aided_data)", len(aided_data))
        data_num = list(self.train_data_local_num_dict.values())
        distruibute_num = []
        for n in data_num:
            distruibute_num.append(np.max(np.array([round(n*rate),2])))
        print("client_data_num", data_num)
        print("send_data_num", distruibute_num[:5])
        global_test_num = np.max(np.array(distruibute_num))
        global_test_data_x = []
        global_test_data_y = []
        print("global_test_num", global_test_num)
        for j in range(expend_class_num):
            if self.args.dataset == "cifar10":
                glabol_x = aid_data[j][:500]
                glabol_y = torch.tensor([self.class_num + j] * len(glabol_x))
                global_test_data_x.append(glabol_x)
                global_test_data_y.append(glabol_y)
            if self.args.dataset == "shakespeare":
                glabol_x = torch.tensor(aid_data[j][:500]).view(data_shape)
                glabol_y = torch.tensor([label_list[0]] * len(glabol_x))
                global_test_data_x.append(glabol_x)
                global_test_data_y.append(glabol_y)
            else:
                glabol_x = torch.tensor(aid_data[j][:500]).view(data_shape)
                glabol_y = torch.tensor([self.class_num + j] * len(glabol_x))
                global_test_data_x.append(glabol_x)
                global_test_data_y.append(glabol_y)

        self.global_test_aided_data = RAG.generate_dataset(self.args, self.args.fisher_batch_size,
                                                           torch.cat(global_test_data_x),
                                                           torch.cat(global_test_data_y))
        for i in range(self.args.client_num_in_total):
            aid_x_list = []
            aid_y_list = []
            m = distruibute_num[i]
            for j in range(expend_class_num):
                if self.args.dataset == "cifar10":
                    aided_data_1_x = aid_data[j][:m]
                    aided_data_1_lables = torch.tensor([self.class_num + j] * len(aided_data_1_x))
                    aid_x_list.append(aided_data_1_x)
                    aid_y_list.append(aided_data_1_lables)
                if self.args.dataset == "shakespeare":
                    aided_data_1_x = torch.tensor(aid_data[j][:m]).view(data_shape)
                    aided_data_1_lables = torch.tensor([label_list[0]] * len(aided_data_1_x))
                    aid_x_list.append(aided_data_1_x)
                    aid_y_list.append(aided_data_1_lables)
                else:
                    aided_data_1_x = torch.tensor(aid_data[j][:m]).view(data_shape)
                    aided_data_1_lables = torch.tensor([self.class_num + j] * len(aided_data_1_x))
                    aid_x_list.append(aided_data_1_x)
                    aid_y_list.append(aided_data_1_lables)
            aided_data_x = torch.cat(aid_x_list)
            aided_data_lables = torch.cat(aid_y_list)
            self.aided_dataset[i] = RAG.generate_dataset(self.args, self.args.fisher_batch_size, aided_data_x,
                                                    aided_data_lables)

            if i <5:
                print("aided", aided_data_lables.size(), aided_data_x.size())
            self.train_data_local_num_dict[i] = self.train_data_local_num_dict[i] + expend_class_num * m
            data = self.train_data_local_dict[i]
            x, lables = RAG.undo_batch_dataset(data)
            # print(lables.size())
            # print(lables)
            idx = torch.where(torch.bincount(lables) > 0)[0]
            # print("idx",idx)
            self.train_labels_type_dic[i] = idx
            x = torch.cat([x, aided_data_x])
            lables = torch.cat([lables, aided_data_lables])
            self.train_data_local_dict[i] = RAG.generate_dataset(self.args, self.args.batch_size, x, lables)  # 合成后的数据
        # self.train_data_num = sum(list(self.train_data_local_num_dict.values()))

            # print(lables.numel(), x.size())
    def divide_aided_data(self):

        # self.save_client_data_by_label()
            ##///读取数据部分
            ##///读取数据集
        method = ["fedbt"]
        if self.args.defend_type in method:
            if self.args.dataset == "mnist":  # [28*28]
                self.distribute_aided_data(aided_data_name="femnist",expend_class_num=1,
                                           label_list=[24],data_shape=(-1, 28 *28),rate=self.args.aid_data_rate)
                self.expend_class_num = 2
            if self.args.dataset == "femnist":  # [28*28]
                self.distribute_aided_data(aided_data_name="synthetic",expend_class_num=1,
                                           label_list=[0],data_shape=(-1, 28, 28),rate=self.args.aid_data_rate)
                self.expend_class_num = 1
                # self.generate_validation_set()
            # if self.args.dataset == "cifar10":  # 3, 32, 32
            #     self.generate_validation_set()

            if self.args.dataset == "shakespeare":  # 3, 32, 32
                self.distribute_aided_data(aided_data_name="shakespeare",expend_class_num=1,
                                           label_list=[81],data_shape=(-1, 80),rate=self.args.aid_data_rate)
                self.generate_validation_set()
            if self.args.dataset == "cifar100":  # 3, 32, 32
                self.distribute_aided_data(aided_data_name="cifar100",expend_class_num=2,
                                           label_list=[40,60],data_shape=(-1, 3, 32, 32),rate=self.args.aid_data_rate)
                self.expend_class_num = 2
        # else:self.generate_validation_set()
        # dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
        #            train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
        #            class_num]
        dataset = [self.train_data_num, self.test_data_num, self.train_data_global, self.test_data_global,
                   self.train_data_local_num_dict, self.train_data_local_dict, self.train_labels_type_dic, self.test_data_local_dict,
                   self.aided_dataset,self.global_test_aided_data, self.class_num + self.expend_class_num]
        return dataset
