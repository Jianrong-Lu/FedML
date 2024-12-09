import argparse
import logging
import os
import random
import sys
import numpy as np
import pandas as pd
import torch
import wandb
import copy

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,'
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_core.robustness import robust_aggregation as RAG
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
# from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
# from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
# from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
# from fedml_api.data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
# from fedml_api.data_preprocessing.UCI import data_loader_for_susy_and_ro as uci
# from fedml_api.data_preprocessing.lending_club_loan.lending_club_dataset import loan_load_two_party_data
# from fedml_api.data_preprocessing.synthetic_1_1.data_loader import load_partition_data_federated_synthetic_1_1
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.mobilenet_v3 import MobileNetV3
from fedml_api.model.cv.feddyn_model import client_model
from fedml_api.model.cv.resnet import resnet20_gn
from fedml_api.model.cv.cnn import CNN_DropOut ,CNN_FOR_SYT,CNN_Net_CIFAR10,CNN_Net_MNIST
from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow
from fedml_api.data_preprocessing.edge_case_examples.data_loader import load_poisoned_dataset
from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from fedml_api.model.linear.lr import LogisticRegression,moon_LogisticRegression
from fedml_api.model.cv.resnet_gn import resnet18
from fedml_api.model.cv.vgg import vgg11
from fedml_api.model.cv.Alexnet import AlexNet
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
# from fedml_api.standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
# from fedml_api.standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
from fedml_api.standalone.fedavg.data_maneger import Data_Manager,DealDataset
from fedml_api.utils import cudat_dataloader
# from nfnetsh import replace_conv
def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='lr', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='mnist', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/mnist',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.2
                        , metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--sample_method',type=str, default='cross-silo',
                        help='cross_silo:sellect all,cross_device:sellect some')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)
    parser.add_argument('--gmf', help='global momentum factor;', type=float, default=0)

    parser.add_argument('--mu', help='mu;', type=float, default=0)

    parser.add_argument('--momentum', help='momentum factor;', type=float, default=0)

    parser.add_argument('--dampening', help='dampening factor;', type=float, default=0)

    parser.add_argument('--nesterov', help='use nesterov;', type=float, default=False)

    parser.add_argument('--epochs', type=int, default=2, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=20, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=30, metavar='NN',
                        help='number of workers')
    parser.add_argument('--attacker_num',type=int, default=0, metavar='NN',
                        help='number of attackers')

    parser.add_argument('--comm_round', type=int, default=1,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='作为临时使用的参数，ci=1则表示实现功能，如选用固定用户',)
    parser.add_argument('--defend_type',type=str,default='fedBT',
                        help='choose the defend method'
                             'fedavg;norm_diff_clipping;add_week_noise;Krum;MultiKrum'
                             'Faba,Zeno,DiverseFL,Trimmed,RFA,Resample,Fisher,...,'
                        )
    parser.add_argument('--defend_module',type=str,default='ABCD',
                        help='choose the defend MODULE for HeteroFL'

                        )
    parser.add_argument('--norm_bound',type=int,default=5,
                        help='for norm diff clipping and weak DP defenses')
    parser.add_argument('--stddev',type=float,default=0.025,
                        help='for weak DP defenses')
    parser.add_argument("--b", help="b, number of trimmed values in Zeno", type=int,default=5)
    parser.add_argument("--p", help="p, Regular term coefficient for Zeno", type=float,default=0.0005)
    parser.add_argument("--k1",help="k1, Hyperparameters in DiverseFL,e.g. k1<C2<k2",type=float,default=0.5)
    parser.add_argument("--k2",help="k2, Hyperparameters in DiverseFL,e.g. k1<C2<k2",type=float,default=2)
    parser.add_argument("--s",help="s, sample_rate in DiverseFL:0.01~0.03,", type=float,default=0.03)
    parser.add_argument("--i", help="I, Regular term coefficient in DiverseFL,", type=float,default=1)
    parser.add_argument("--fisher_batch_size",type = int,default=5)
    parser.add_argument("--aid_data_rate",type = float,default=0.03)
    parser.add_argument("--init_r",type = float,default=0.01) #用来给feddyn设置超参数alpha
    parser.add_argument('--attack_type',type=str,default="label_flipping")
    parser.add_argument("--attacker_knowlege",type=str,default="esorics",help = "arg_and_update,update_only")
    parser.add_argument('--poison_type', type=str, default='southwest',
                        help='specify source of data poisoning: |ardis|(for EMNIST), |southwest|howto|(for CIFAR-10)')

    return parser

def load_data(args, dataset_name,device):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False
    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size)
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir,args.batch_size,device)
        args.client_num_in_total = client_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "ILSVRC2012":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_ImageNet(dataset=dataset_name, data_dir=args.data_dir,
                                                 partition_method=None, partition_alpha=None,
                                                 client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_dir, 'mini_gld_train_split.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'mini_gld_test.csv')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_dir, 'federated_train.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'test.csv')

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
                                                  fed_train_map_file=fed_train_map_file,
                                                  fed_test_map_file=fed_test_map_file,
                                                  partition_method=None, partition_alpha=None,
                                                  client_number=args.client_num_in_total, batch_size=args.batch_size)
    elif dataset_name == "lending_club_loan":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = loan_load_two_party_data(data_dir=args.data_dir)
    elif dataset_name == "usuy":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_list = [i for i in range(100)]
        sample_num_in_total = 5000000
        beta = 0.5
        usi_obj = uci.DataLoader(dataset_name, args.data_dir, client_list, sample_num_in_total, beta)
        train_data_local_dict = usi_obj.StreamingDataDict()
        [train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num] = None
    elif dataset_name == "synthetic":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = load_partition_data_federated_synthetic_1_1(data_dir=args.data_dir,batch_size=args.batch_size)
    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10

        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10
        if dataset_name == "cifar10":
            train_data_num, test_data_num, train_data_global, test_data_global, aidedtest_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, aided_data_test_local_dict,\
            class_num = data_loader(args,device)
        else:
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                    args.partition_alpha, args.client_num_in_total, args.batch_size,device)

    if args.ci == 1:
        if args.client_num_per_round == 20:
            selected_index = [507, 818, 452, 368, 242, 929, 262, 810, 318, 49, 446, 142, 968, 345, 971, 133, 104, 6,
                              600, 496]
        else:
            selected_index = [i for i in range(args.client_num_per_round)]
        print("selected_index", selected_index)
        if centralized:#centralized 时，初始总用户要设置为1
            user_train_data_num = 0
            for index in selected_index:
                user_train_data_num += train_data_local_num_dict[index]
            train_data_local_num_dict = {0: user_train_data_num}
            train_data_num = user_train_data_num
            train_data_local_dict = {
                0: [batch for cid in sorted(selected_index) for batch in train_data_local_dict[cid]]}
            test_data_local_dict = {
                0: [batch for cid in sorted(selected_index) for batch in test_data_local_dict[cid]]}
            centralized = False
            args.client_num_in_total = 1
        else:
            print("selected_index1", selected_index)
            user_train_data_num = 0
            for index in selected_index:
                user_train_data_num += train_data_local_num_dict[index]
            dict1 = {}
            dict2 = {}
            dict3 = {}
            for i,index in enumerate(selected_index):
                dict1[i] = train_data_local_num_dict[index]
                dict2[i] = train_data_local_dict[index]
                dict3[i] = test_data_local_dict[index]
            train_data_local_num_dict= dict1
            train_data_num = user_train_data_num
            train_data_local_dict= dict2
            test_data_local_dict= dict3
            args.client_num_in_total = len(selected_index)
            print(train_data_local_dict.keys())
    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size
    if  args.dataset == "cifar10":
        # for i in range(args.client_num_in_total):
        print("cifar10")
        dataset = [train_data_num, test_data_num,train_data_global,test_data_global, aidedtest_global,
                    train_data_local_num_dict, train_data_local_dict,test_data_local_dict , aided_data_test_local_dict,
                     class_num]
    else:
        dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict, train_data_local_dict , test_data_local_dict , class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None

    if model_name == "lr" and args.dataset == "mnist":
        logging.info("lr + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    if model_name == "cnn" and args.dataset == "mnist":                                   # lr mnist
        logging.info("CNN + MNIST")
        model = CNN_Net_MNIST(output_dim,False)
    elif model_name == "cnn" and args.dataset == "femnist":                              # cnn,femnist
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(output_dim,False)
    elif model_name == "cnn" and args.dataset == "synthetic":                              # cnn,femnist
        logging.info("CNN + synthetic")
        model = CNN_DropOut(output_dim,False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet34(output_dim)
    elif model_name == "rnn" and args.dataset == "shakespeare":                           #rnn
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg(output_dim)
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":                                                         # resnet56,cifar10
        model = resnet56(class_num=output_dim)
    elif model_name == "resnet20":                                                         # resnet56,cifar10
        model = resnet20(class_num=output_dim)
    elif model_name == "mobilenet": #
        logging.info("mobilenet")
        model = mobilenet(class_num=output_dim)
    elif model_name == "mobilenet_v3": #
        logging.info("mobilenet_v3")
        model = MobileNetV3(class_num=output_dim)
    elif model_name == "resnet20_gn":                                                      # cnn,合成数据集
        logging.info("resnet20_gn")
        model = resnet20_gn(output_dim)
    elif model_name == "vgg11":
        logging.info("vgg11")
        model = vgg11(output_dim)
    elif model_name == "cnn":
        logging.info("cnn")
        model = CNN_DropOut(output_dim,False)
    elif model_name == "cnncf10":
        logging.info("cnncf10")
        model = CNN_Net_CIFAR10(output_dim)
    elif model_name == "AlexNet":
        model = AlexNet(num_classes=output_dim, init_weights=True)
    return model


def custom_model_trainer(args, model):
    if args.dataset == "stackoverflow_lr":
        return MyModelTrainerTAG(model)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        return MyModelTrainerNWP(model)
    else: # default model trainer is for classification problem
        return MyModelTrainerCLS(model)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args() #可以调用参数
    # if args.client_num_in_total  == 60:args.partition_alpha = 0.05

    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("Cuda:",torch.cuda.is_available())
    logger.info(device)

    #
    # wandb = None
    wandb.init(
        project='Accuracy_improvement',
        # project='test',
        name= "alpha=" + str(args.init_r) + "-" + "niid_degree" + str(args.partition_alpha) + "-" + str(args.defend_type)+"-"+ str(args.dataset) + "-ep" + str(args.epochs) + "-lr" + str(args.lr)
        +"-bs" + str(args.batch_size) + "-{}/{}".format(args.client_num_per_round,args.client_num_in_total) + "-" + str(args.partition_method) +
         "-"+ str(args.model),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)

    # aid_data_rate = args.aid_data_rate

    if  args.dataset == "cifar10":

        # for i in range(args.client_num_in_total):
        # [train_data_num, test_data_num, device, train_data_global, test_data_global, aidedtest_global,
        #  train_data_local_num_dict, train_data_local_dict, test_data_local_dict, aided_data_test_local_dict,
        #  class_num] = load_data(args, args.dataset,device)
        [train_data_num, test_data_num, train_data_global, test_data_global, aidedtest_global,train_data_local_num_dict,
         train_data_local_dict,test_data_local_dict , aided_data_test_local_dict,
         class_num] = load_data(args, args.dataset,device)
        dataset=[train_data_num, test_data_num, train_data_global, test_data_global,aidedtest_global,
        train_data_local_num_dict,train_data_local_dict, test_data_local_dict,
        aided_data_test_local_dict, class_num]
    else:

       [train_data_num, test_data_num, train_data_global, test_data_global,
                   train_data_local_num_dict,train_data_local_dict, test_data_local_dict, class_num] = load_data(args, args.dataset,device)
       dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                  train_data_local_num_dict, train_data_local_dict,  test_data_local_dict,
                  class_num]

    method = ["fedbt"]
    if args.defend_type in method:
        # train_data_num += sum(distruibute_num)
        # dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
        #  train_data_local_num_dict, train_data_local_dict,train_labels_type_dic, test_data_local_dict, aided_dataset,class_num + add_class_num]
        data_maneger = Data_Manager(args,device,dataset)
        dataset = data_maneger.divide_aided_data()
        model = create_model(args, model_name=args.model, output_dim=dataset[-1])
        print("defend:",args.defend_type)
        print("class_num:",dataset[-1])
    else: #这里看model = create_model(args, model_name=args.model, output_dim=dataset[-1])就行
        if args.dataset == "shakespeare":
            args.aid_data_rate = 0.01
        lst = [i for i in range(args.client_num_in_total)]
        train_labels_type_dic = dict(zip(lst,lst))
        data_maneger = Data_Manager(args,device,dataset)
        dataset = data_maneger.divide_aided_data()
        seleected_index = data_maneger.train_data_local_dict.keys()
        data_maneger.train_labels_type_dic = train_labels_type_dic
        dataset = data_maneger.divide_aided_data()
        model = create_model(args, model_name=args.model, output_dim=dataset[-1])
        print("defend:",args.defend_type)
        print("class_num:",dataset[-1])
        import gc
        del data_maneger
        gc.collect()
        # for i in range(500):
    #     print("user{}:"+str(i) + str(test_data_local_dict[i]))
    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    # print("aid_data_rate",aid_data_rate,args.aid_data_rate)
    # args.aid_data_rate = aid_data_rate
    model_func = lambda: client_model(args.model,dataset[-1])
    model = model_func()
    model_trainer = custom_model_trainer(args, model)

    # else:model_func = None
    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer, model_func)

    fedavgAPI.train()


