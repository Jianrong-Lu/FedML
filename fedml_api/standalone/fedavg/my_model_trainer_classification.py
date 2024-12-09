import torch
import collections
import numpy as np
import copy
# from torch import nn
import torch.nn.functional as F
import torch.nn as nn
from fedml_api.standalone.fedavg import client
# from fedml_api.standalone.fednova import fednova
try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer
# grad = []
def hook(module, grad_input, grad_output):
    print('grad_input: ',grad_input[2].size() ,grad_input)
    print('grad_output: ', grad_output)
class MyModelTrainer(ModelTrainer):

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def vectorize_weight(self,state_dict,device):
        weight_list = []
        for (k, v) in state_dict.items():
            if self.is_weight_param(k):  #
                weight_list.extend(v.view(1, -1).to(device))
        return torch.cat(weight_list)  # torch.cat的用法必须注意
    def vectorize_model(self,model):
        weight_list = []
        for k, v in model.named_parameters():
            if self.is_weight_param(k):  #
                weight_list.extend(v.view(1, -1))
        return torch.cat(weight_list)  # torch.cat的用法必须注意
    def is_weight_param(self,k):
        return ("running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k)
    def is_w_r_var_mean(self,k):
        return "num_batches_tracked" not in k


    def get_local_grad(self,args, device,cur_params, init_params):
        update_dict = collections.OrderedDict()
        for k in cur_params.keys():
            # if self.is_weight_param(k):
            update_dict[k] = init_params[k].to(device) - cur_params[k].to(device)
            # else:update_dict[k] = cur_params[k].to(device)
        return update_dict
    def get_local_trainable_grad(self,args, device,cur_params, init_params):
        update_dict = collections.OrderedDict()
        for k in cur_params.keys():
            if self.is_weight_param(k):
                update_dict[k] = (init_params[k].to(device) - cur_params[k].to(device))/args.lr

        return update_dict
    def get_local_all_grad(self,args, device,cur_params, init_params):
        update_dict = collections.OrderedDict()
        for k in cur_params.keys():
            update_dict[k] = (init_params[k].to(device) - cur_params[k].to(device))/args.lr
        return update_dict

    def feddyn_train(self, round_idx,local_training_data, device, args,cld_model, alpha_coef,avg_mdl_param,local_grad_vector):
        model = cld_model # 这个model不是字典，需要get_model_state_dict
        model.to(device)
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        print("alpha_coef",alpha_coef)
        if args.client_optimizer == "sgd":
            # optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=alpha_coef )
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        for epoch in range(args.epochs):  # 本地训练的轮次。
            for bidx ,(x,labels) in enumerate(local_training_data):
                x, labels = x.to(device), labels.to(device).long()
                model.zero_grad()
                log_probs = model(x)
                ## Get f_i estimate

                loss_f_i = loss_fn(log_probs, labels.reshape(-1).long())
                # loss_f_i = loss_f_i / list(labels.size())[0]

                # Get linear penalty on the current parameter estimates
                local_par_list = None
                for param in model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
                loss = loss_f_i + loss_algo
                print("loss",loss,loss_f_i,loss_algo)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)  # Clip gradients
                optimizer.step()
        # Freeze model
        for params in model.parameters():
            params.requires_grad = False
        model.eval()
        return model

    def train(self,round_idx, train_data, device, args,last_model_dict):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        # train and updates
        criterion = loss.to(device)
        init_params = copy.deepcopy(last_model_dict)
        method = ["fedavg","heterofl","Zeno","median","resample","faba","fltrust","fedbt","dnc","fedba","MKrum","expsmoo","MAB_FL","moon",
                  "fedavgM","fedIR","fedmy","fedBT"]
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(),lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        epoch_loss = []
        train_data_list = [(x, labels) for i, (x, labels) in enumerate(train_data)]
        batch_list = []
        client_bt_num = len(train_data)
        print("client_bt_num", client_bt_num)
        for k in range(40):
            sample_idx = np.random.choice(range(client_bt_num), 1)[0]
            batch_list.append(sample_idx)
        for epoch in range(args.epochs):#本地训练的轮次。
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
            # for batch_idx in batch_list:
            #     (x, labels) = train_data_list[batch_idx]
                # print("tt",(batch_idx,labels))
                if epoch == 0 and batch_idx==0:
                    print("x.size",x.size())
                    # print("x:", x)
                    # print("x:", labels)
                x, labels = x.to(device), labels.to(device).long()
                model.zero_grad()
                # if args.attack_type == "ESP_attack":#攻击
                #     if args.model == "resnet18":
                #         loss= 0
                #     if args.model == "cnn":loss= 0
                # else:#正常计算分类损失
                if args.defend_module== "ABCD":
                    log_probs = model(x)
                else:
                    _,log_probs = model(x)

                loss = criterion(log_probs, labels)
                if args.defend_type == 'fedprox':
                    # global_model = copy.deepcopy(self.model)
                    # global_model.load_state_dict(last_model_dict)
                    # global_weight_collector = list(global_model.to(device).parameters())
                    fed_prox_reg = 0.0
                    # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                    for param_index, param in model.named_parameters():
                        # print(param)
                        fed_prox_reg += ((args.init_r / 2) * torch.norm((param - last_model_dict[param_index].to(device))) ** 2)
                    loss =loss + fed_prox_reg
                    loss.backward()
                if args.defend_type == 'DiverseFL':
                    loss = loss + self.L2Losss(model,args.i) #L2正则化
                    loss.backward()
                if args.defend_type in method:
                    # print(args.defend_type)
                    # bia = copy.deepcopy(model.linear.bias.grad)
                    # print(bia)
                    loss.backward()
                optimizer.step() #若出现cuda bug,可能是batchsize 太大导致
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # updata = collections.OrderedDict()
        # for p,v in model.named_parameters():
        #     updata[p] = copy.deepcopy(v.grad)*args.lr

        updata = self.get_local_grad(args,device, copy.deepcopy(self.get_model_params()), init_params)

        return updata

    def RTysever_train(self, round_idx, train_data, device, args,w_global,previous_w_global,mean_loss,sellected,last_grad,last_loss):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        init_params = copy.deepcopy(w_global)
        # train and updates
        criterion = loss.to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        bth_mean_grad_dict = collections.OrderedDict()
        if round_idx == 0:
            for key in previous_w_global.keys():
                bth_mean_grad_dict[key] = torch.zeros_like(previous_w_global[key]).to(device)
        else:
            if sellected:
                mean_loss = mean_loss*args.client_num_per_round - last_loss
                for key in previous_w_global.keys():
                    bth_mean_grad_dict[key] = (args.client_num_per_round*((previous_w_global[key].to(device) - w_global[key].to(device))/args.lr) - last_grad[key]/args.lr)
            else:
                mean_loss = mean_loss*args.client_num_per_round
                for key in previous_w_global.keys():
                    bth_mean_grad_dict[key] =args.client_num_per_round*((previous_w_global[key].to(device) - w_global[key].to(device))/args.lr)
        for epoch in range(args.epochs):  # 本地训练的轮次。
            batch_loss = []
            for bidx ,(x,labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device).long()
                model.zero_grad()
                if args.defend_module== "ABCD":
                    log_probs = model(x)
                else:
                    _,log_probs = model(x)
                loss1 = criterion(log_probs, labels)
                loss2 = torch.tensor(0.,requires_grad=True)
                for p,v in model.named_parameters():
                    # print("v",v)
                    # print("ini",init_params[p])
                    # if self.is_weight_param(p):
                    f1 = v - w_global[p].to(device)
                    f2 = bth_mean_grad_dict[p].to(device)
                    lss = torch.sum(f1.mul(f2))
                    loss2 = loss2 + lss
                    # else:
                loss2 = loss2 + mean_loss
                # loss2 = np.max(np.sign(loss2.item()),0)*loss2
                # loss2 = np.max(np.sign(loss2.item()),0)*loss2
                loss2 = torch.abs(loss2)
                print("loss1_2",round_idx,epoch,bidx,loss1,loss2,mean_loss)
                mul = args.init_r
                # mul = (args.client_num_per_round*args.client_num_per_round*0.2-args.client_num_per_round)/(args.client_num_per_round-1)
                loss = loss1 + mul* loss2 #cifar10,0.01,femnist0.1
                loss.backward()
                # optimizer.step()
                for p, v in model.named_parameters():
                    # if v.grad is None:
                    #     continue
                    v.data.add_(-args.lr, v.grad.data)
                batch_loss.append(loss1.item())

        grad_dict = self.get_local_grad(args, device, copy.deepcopy(model.state_dict()), init_params)

        return grad_dict,np.mean(np.array(batch_loss))

    def RTye_train(self, round_idx, train_data, device, args,w_global,previous_w_global,global_grad,global_meta_grad,mean_loss,selected_batch_totoal_num):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        mul = args.init_r
        init_params = copy.deepcopy(w_global)
        # train and updates
        criterion = loss.to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        bth_mean_grad_dict = collections.OrderedDict()
        # # print(global_grad.keys())
        # # for key in global_grad.keys():
        # #     bth_mean_grad_dict[key] = global_grad[key] -  prepre_meta_gradient[key]
        # for key in global_grad.keys():
        #     bth_mean_grad_dict[key] =prepre_meta_gradient[key]
        selected_batch_totoal_num = 40
        if args.defend_type == "fedTest_NoMetaGrad":
            if round_idx > 0:
                mean_loss = mean_loss * args.client_num_per_round
                for key in previous_w_global.keys():
                    global_meta_grad[key] = global_grad[key] / args.lr
        if args.defend_type == "fedTest_momemtum":
            if round_idx > 0:
                mean_loss = mean_loss * args.client_num_per_round
                for key in previous_w_global.keys():
                    # global_meta_grad[key] = global_grad[key] / args.lr - mul*selected_batch_totoal_num * global_meta_grad[key]
                    global_meta_grad[key] = (1-mul)*global_grad[key] + mul* global_meta_grad[key]
        else:
            if round_idx > 0:
                mean_loss = mean_loss * args.client_num_per_round
                for key in previous_w_global.keys():
                    # global_meta_grad[key] = args.client_num_per_round * ((previous_w_global[key].to(device) - w_global[key].to(device)) / args.lr - mul*args.epochs*selected_batch_totoal_num * global_meta_grad[key])
                    global_meta_grad[key] = global_grad[key] / args.lr - mul*selected_batch_totoal_num * global_meta_grad[key]
        # meta_gradient = collections.OrderedDict()
        client_bt_num = len(train_data)
        print("client_bt_num", client_bt_num)
        train_data_list = [(x, labels) for i, (x, labels) in enumerate(train_data)]
        batch_list = []
        for k in range(selected_batch_totoal_num):
            sample_idx = np.random.choice(range(client_bt_num), 1)[0]
            batch_list.append(sample_idx)
            # print(sample_idx)

        for epoch in range(args.epochs):  # 本地训练的轮次。
            batch_loss = []
            for bidx ,(x,labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device).long()
                model.zero_grad()
                if args.defend_module== "ABCD":
                    log_probs = model(x)
                else:
                    _,log_probs = model(x)
                loss1 = criterion(log_probs, labels)
                loss2 = torch.tensor(0., requires_grad=True)
                for p, v in model.named_parameters():
                    f1 = v - w_global[p].to(device)
                    f2 = args.client_num_per_round * global_meta_grad[p]
                    lss = torch.sum(f1.mul(f2))
                    loss2 = loss2 + lss
                if args.defend_type == "fedTest_momemtum":
                    loss2 = torch.tensor(0., requires_grad=True)
                if args.defend_type == "fedTest_Nomeanloss":
                    loss2 = loss2
                if args.defend_type == "fedTest":
                    loss2 = loss2 + mean_loss
                if args.defend_type == "fedTest_NoMetaGrad":
                    loss2 = loss2 + mean_loss
                # loss2 = np.max(np.sign(loss2.item()),0)*loss2
                # loss2 = np.max(np.sign(loss2.item()),0)*loss2
                print("round{}_epoch{}_batch{}_suploss{}_matchloss{}_meanloss{}".format(round_idx,epoch,bidx,loss1,loss2,mean_loss))
                loss2 = torch.abs(loss2)

                loss = loss1 + mul* loss2 #cifar10,0.01,femnist0.1
                loss.backward()
                # optimizer.step()
                for p, v in model.named_parameters():
                    # if v.grad is None:
                    #     continue
                    if args.defend_type == "fedTest_momemtum":
                        d_p = v.grad.data
                        # d_p.mul_(1-mul).add_(global_meta_grad[p].mul(mul))
                        d_p.add_(global_meta_grad[p].mul(mul))
                        v.data.add_(-args.lr, d_p)

                    else:
                        v.data.add_(-args.lr, v.grad.data)
                batch_loss.append(loss1.item())

        grad_dict = self.get_local_grad(args, device, copy.deepcopy(model.state_dict()), init_params)
        # for key in grad_dict.keys():
        #     meta_gradient[key] = grad_dict[key]/args.lr - mul*args.epochs*len(train_data)* bth_mean_grad_dict[key]
            # grad_dict[key] = grad_dict[key]

        return grad_dict,global_meta_grad,np.mean(np.array(batch_loss))

    def Tye_train(self, round_idx, train_data, device, args,last_model_dict,bth_mean_grad_dict,mean_loss):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        init_params = copy.deepcopy(last_model_dict)
        # train and updates
        criterion = loss.to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        # for epoch in range(args.epochs):  # 本地训练的轮次。
        grad_dict = collections.OrderedDict()
        The_first_batch_gradient = collections.OrderedDict()
        client_bt_num = len(train_data)
        first_bt_num = 0
        # print("client_bt_num",client_bt_num)
        for bidx ,(x,labels) in enumerate(train_data):

            x, labels = x.to(device), labels.to(device).long()
            model.zero_grad()
            _,log_probs = model(x)
            loss1 = criterion(log_probs, labels)

            if round_idx!=0:
                loss2 = torch.tensor(0.,requires_grad=True)
                for p,v in model.named_parameters():
                    if self.is_weight_param(p):
                        f1 = torch.flatten(v - init_params[p].to(device))
                        f2 = torch.flatten(bth_mean_grad_dict[p]).to(device)
                        lss = torch.sum(f1 * f2)
                        # print("lss",lss)
                        loss2 = loss2 + lss
                    # else:
                loss2 = loss2 + mean_loss
                loss2 = np.max(np.sign(loss2.item()),0)*loss2
                print("loss1_2",loss1,loss2,mean_loss)
                mul = args.init_r
                # if loss2.item() <= 0.8*mean_loss:mul=0
                loss = loss1 + mul*loss2#cifar10,0.01,femnist0.1
            else:
                loss = loss1
            loss.backward()
            optimizer.step()
            if bidx == 0:
                first_bt_num = len(labels)
                lo1 = loss1.item()
                # The_first_batch_gradient = self.get_local_trainable_grad(args, device, copy.deepcopy(model.state_dict()), init_params)
            if bidx == client_bt_num-1:
                grad_dict = self.get_local_grad(args, device, copy.deepcopy(self.get_model_params()), init_params)
                The_first_batch_gradient = self.get_local_trainable_grad(args, device, copy.deepcopy(model.state_dict()), init_params)

        return grad_dict,The_first_batch_gradient,lo1,first_bt_num
    def mime_train(self, round_idx, train_data, device, args,last_model_dict,s,c):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        model.to(device)
        model.train()
        initial_model = copy.deepcopy(model)
        initial_model.train()
        initial_model1 = copy.deepcopy(model)
        initial_model1.train()
        loss = nn.CrossEntropyLoss()
        init_params = copy.deepcopy(last_model_dict)
        # train and updates
        criterion = loss.to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(initial_model1.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        # grad_dict = collections.OrderedDict()
        client_bt_num = len(train_data)
        print("client_bt_num",client_bt_num)
        train_data_list = [(x, labels) for i, (x, labels) in enumerate(train_data)]
        full_x = torch.from_numpy(np.asarray([])).float().to(device)
        full_y = torch.from_numpy(np.asarray([])).long().to(device)
        for (batched_x, batched_y) in train_data:
            # print("batched_y",batched_y)
            full_x = torch.cat((full_x, batched_x.to(device)), 0)
            full_y = torch.cat((full_y, copy.deepcopy(batched_y.to(device))), 0)
        for k in range(args.epochs):
            # model_grad_dict = {}
            grad_dict = {}
            sample_idx = np.random.choice(range(client_bt_num), 1)[0]
            # print(sample_idx)
            (x, labels) = train_data_list[sample_idx]
            # print("x",x.size())
            x, labels = x.to(device), labels.to(device).long()
            model.zero_grad()
            initial_model.zero_grad()

            if args.defend_module == "ABCD":
                log_probs2 = initial_model(x)
            else:
                _, log_probs2 = initial_model(x)
            loss2 = criterion(log_probs2, labels)
            loss2.backward()
            # optimizer.step()
            for p,v in initial_model.named_parameters():
                grad_dict[p] = v.grad.data
            if args.defend_module == "ABCD":
                log_probs1 = model(x)
            else:
                _, log_probs1 = model(x)
            loss1 = criterion(log_probs1, labels)
            loss1.backward()
            for p,v in model.named_parameters():
                # print(c[p].size())
                d_p = v.grad.data - grad_dict[p] + c[p]
                d_p.mul_(1 - args.init_r).add_(s[p].mul(args.init_r))
                v.data.add_(-args.lr, d_p)
                s[p] = copy.deepcopy(d_p.detach())

        if args.defend_module == "ABCD":
            log_probs3 = initial_model1(full_x)
        else:
            _, log_probs3 = initial_model1(full_x)
        # print("full_x", full_x.size(),full_y.size())
        # print("log_probs3", log_probs3.size())
        loss3 = criterion(log_probs3, full_y)
        loss3.backward()
        optimizer.step()
        initial_model1.zero_grad()
        update = self.get_local_grad(args, device, model.state_dict(), init_params)
        full_grad_dict = self.get_local_all_grad(args, device, initial_model1.state_dict(), init_params)
        return full_grad_dict,update

    def mimelite_train(self, round_idx, train_data, device, args,last_model_dict,s):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        model.to(device)
        model.train()
        initial_model = copy.deepcopy(model)
        initial_model.train()
        loss = nn.CrossEntropyLoss()
        init_params = copy.deepcopy(last_model_dict)
        # train and updates
        criterion = loss.to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(initial_model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        # grad_dict = collections.OrderedDict()
        client_bt_num = len(train_data)
        print("client_bt_num",client_bt_num)
        train_data_list = [(x, labels) for i, (x, labels) in enumerate(train_data)]
        full_x = torch.from_numpy(np.asarray([])).float().to(device)
        full_y = torch.from_numpy(np.asarray([])).long().to(device)
        for (batched_x, batched_y) in train_data:
            # print("batched_y",batched_y)
            full_x = torch.cat((full_x, batched_x.to(device)), 0)
            full_y = torch.cat((full_y, copy.deepcopy(batched_y.to(device))), 0)
        #####################
        #####################
        for k in range(args.epochs):
            # model_grad_dict = {}
            sample_idx = np.random.choice(range(client_bt_num), 1)[0]
            # print(sample_idx)
            (x, labels) = train_data_list[sample_idx]
            # print("x",x.size())
            x, labels = x.to(device), labels.to(device).long()
            model.zero_grad()
            if args.defend_module == "ABCD":
                log_probs1 = model(x)
            else:
                _, log_probs1 = model(x)
            loss1 = criterion(log_probs1, labels)
            loss1.backward()
            # optimizer.step()
            for p,v in model.named_parameters():
                d_p = v.grad.data
                # if round_idx > 0:
                d_p.mul_(1-args.init_r).add_(s[p].mul(args.init_r))
                v.data.add_(-args.lr, d_p)
                s[p] = copy.deepcopy(d_p.detach())
        if args.defend_module == "ABCD":
            log_probs3 = initial_model(full_x)
        else:
            _, log_probs3 = initial_model(full_x)
        # print("full_x", full_x.size(),full_y)
        # print("log_probs3", log_probs3.size())
        loss3 = criterion(log_probs3, full_y)
        loss3.backward()
        optimizer.step()
        initial_model.zero_grad()
        update = self.get_local_grad(args, device, self.get_model_params(), init_params)
        full_grad_dict = self.get_local_all_grad(args, device, initial_model.state_dict(), init_params)
        return full_grad_dict,update

    def scaffold_train(self, round_idx, selected,train_data, device, args,last_model_dict,global_c,local_c):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        init_params = copy.deepcopy(last_model_dict)
        # train and updates
        criterion = loss.to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        # mu = args.init_r
        if selected == False:
            for key, va in init_params.items():
                local_c[key] = torch.zeros_like(va).to(device)
        for epoch in range(args.epochs):  # 本地训练的轮次。
            batch_loss = []
            for batch_idx, (x, target) in enumerate(train_data):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()
                _, out1= model(x)
                loss = criterion(out1, target)
                loss.backward()
                batch_loss.append(loss.item())
                optimizer.step()
                cur_model = model.state_dict()
                for p, v in cur_model.items():
                    # if self.is_weight_param(p):
                    init_params[p] = cur_model[p].to(device) + args.lr*(local_c[p] - global_c[p])
                # self.model_trainer.set_model_params(copy.deepcopy(init_params))  # 设置客户的模型,相对于广播
                model.load_state_dict(init_params)  # 这个model不是字典，需要get_model_state_dict
                model.to(device)
                model.train()
                loss = nn.CrossEntropyLoss()
                # train and updates
                criterion = loss.to(device)
                if args.client_optimizer == "sgd":
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
                else:
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                                 weight_decay=args.wd, amsgrad=True)

        updata = self.get_local_grad(args,device, copy.deepcopy(self.get_model_params()), last_model_dict)
        for k, u in updata.items():
            local_c[k] = local_c[k] - global_c[k] + (1/(len(train_data)*args.epochs))*u
        return updata,local_c
    def my_train(self, round_idx, train_data, device, args,last_model_dict,pre_local_model_dict):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        pre_model = copy.deepcopy(self.model)
        cur_global_model = copy.deepcopy(self.model)
        cur_global_model.load_state_dict(copy.deepcopy(last_model_dict))
        cur_global_model.to(device)
        cur_global_model.eval()
        for param in cur_global_model.parameters():
            param.requires_grad = False
        pre_model.load_state_dict(pre_local_model_dict)
        pre_model.to(device)
        pre_model.eval()
        for param in pre_model.parameters():
            param.requires_grad = False
        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        init_params = copy.deepcopy(last_model_dict)
        # train and updates
        criterion = loss.to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        cos = torch.nn.CosineSimilarity(dim=-1)
        temperature = 0.5
        # model_name == "resnet20":
        # if args.dataset == "cifar10"
        mu = args.init_r
        for epoch in range(args.epochs):  # 本地训练的轮次。
            batch_loss = []
            for batch_idx, (x, target) in enumerate(train_data):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                # print(batch_idx)
                # print(x)
                # model.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()
                # print(model(x))
                pro1, out1= model(x)
                loss1 = criterion(out1, target)
                # # print("pro1,out",pro1, out)
                # pro2, out2 = cur_global_model(x)
                # # loss2 = criterion(out2, target)
                # posi = cos(pro1, pro2)
                # logits = posi.reshape(-1, 1)
                #
                # pro3,out3 = pre_model(x)
                # nega = cos(pro1, pro3)
                # logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
                #
                # logits /= temperature
                # labels = torch.zeros(x.size(0)).to(device).long()
                #
                # loss2 = mu * criterion(logits, labels)
                # print("loss22",loss2)

                # loss3 += fed_prox_reg
                loss = loss1
                # print("fed_prox_reg",fed_prox_reg)
                # loss = loss1 + loss2
                # loss = loss1 + loss2 + fed_prox_reg
                # loss = loss1 + fed_prox_reg
                loss.backward()
                batch_loss.append(loss.item())
                optimizer.step()
            # pre_model.load_state_dict(copy.deepcopy(model.state_dict()))
            # pre_model.to(device)
            # pre_model.eval()
            # for param in pre_model.parameters():
            #     param.requires_grad = False
        updata = self.get_local_grad(args,device, copy.deepcopy(self.get_model_params()), init_params)
        return updata
    def pess_train(self, round_idx, train_data, device, args,last_model_dict,pre_local_model_dict):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        pre_model = copy.deepcopy(self.model)
        cur_global_model = copy.deepcopy(self.model)
        cur_global_model.load_state_dict(copy.deepcopy(last_model_dict))
        cur_global_model.to(device)
        cur_global_model.eval()
        for param in cur_global_model.parameters():
            param.requires_grad = False
        pre_model.load_state_dict(pre_local_model_dict)
        pre_model.to(device)
        pre_model.eval()
        for param in pre_model.parameters():
            param.requires_grad = False
        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        init_params = copy.deepcopy(last_model_dict)
        # train and updates
        criterion = loss.to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        cos = torch.nn.CosineSimilarity(dim=-1)
        temperature = 0.5
        # model_name == "resnet20":
        # if args.dataset == "cifar10"
        mu = args.init_r
        for epoch in range(args.epochs):  # 本地训练的轮次。
            batch_loss = []
            for batch_idx, (x, target) in enumerate(train_data):
                x, target = x.to(device), target.to(device)
                # optimizer.zero_grad()
                model.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()
                # print(model(x))
                pro1, out1= model(x)
                pro2, out2 = cur_global_model(x)
                pro3, out3 = pre_model(x)
                loss1 = criterion(out1, target)
                # print("pro1,out",pro1, out)
                posi = cos(pro1, pro3)
                logits = posi.reshape(-1, 1)

                nega = cos(pro1, pro2)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                logits /= temperature
                labels = torch.zeros(x.size(0)).to(device).long()

                loss2 = mu * criterion(logits, labels)
                print("loss22",loss2)

                loss = loss2
                # loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
        updata = self.get_local_grad(args,device, copy.deepcopy(self.get_model_params()), init_params)
        return updata
    def contrastive_train(self, round_idx, train_data, device, args,last_model_dict,pre_local_model_dict):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        pre_model = copy.deepcopy(self.model)
        cur_global_model = copy.deepcopy(self.model)
        cur_global_model.load_state_dict(copy.deepcopy(last_model_dict))
        cur_global_model.to(device)
        cur_global_model.eval()
        for param in cur_global_model.parameters():
            param.requires_grad = False
        pre_model.load_state_dict(pre_local_model_dict)
        pre_model.to(device)
        pre_model.eval()
        for param in pre_model.parameters():
            param.requires_grad = False
        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        init_params = copy.deepcopy(last_model_dict)
        # train and updates
        criterion = loss.to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        cos = torch.nn.CosineSimilarity(dim=-1)
        temperature = 0.5
        # model_name == "resnet20":
        # if args.dataset == "cifar10"
        mu = args.init_r
        for epoch in range(args.epochs):  # 本地训练的轮次。
            batch_loss = []
            for batch_idx, (x, target) in enumerate(train_data):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()
                # print(model(x))
                pro1, out1= model(x)
                loss1 = criterion(out1, target)
                # print("pro1,out",pro1, out)
                pro2, out2 = cur_global_model(x)
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)
                pro3,out3 = pre_model(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                logits /= temperature
                labels = torch.zeros(x.size(0)).to(device).long()

                loss2 = mu * criterion(logits, labels)
                print("loss22",loss2)
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
        updata = self.get_local_grad(args,device, copy.deepcopy(self.get_model_params()), init_params)
        return updata
    def fedIR_train(self, round_idx, train_data, device, args,last_model_dict,labels_number_list):
        model = self.model  # 这个model不是字典，需要get_model_state_dict
        model.to(device)
        model.train()
        loss = nn.CrossEntropyLoss()
        init_params = copy.deepcopy(last_model_dict)
        # train and updates
        criterion = loss.to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        global_labels_dict, local_label_list= labels_number_list
        importance_weight_dict = {}
        for (l,number) in local_label_list:
            importance_weight_dict[l] = global_labels_dict[l]/number
        print("importance_weight_dict",importance_weight_dict)

        for epoch in range(args.epochs):  # 本地训练的轮次。
            batch_loss = []
            for batch_idx, (x, target) in enumerate(train_data):
                w_sum = 0
                w = []
                # print("ot",target)
                for label in target:
                    # print("bt_label",label)
                    label = label.item()
                    w_sum += importance_weight_dict[label]
                for label in target:
                    label = label.item()
                    w.append(importance_weight_dict[label]/w_sum)
                w_tensor = torch.tensor(w).to(device)
                print("w_tensor",w_tensor)
                x, target = x.to(device), target.to(device)
                # print("target",target)
                # weight = batch_label[label]
                optimizer.zero_grad()
                # model.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()
                _, out1= model(x)
                # print("target",target)
                # print("out1",out1.size()[1])
                one_hot = F.one_hot(target,num_classes =out1.size()[1]).float()  # 对标签进行one_hot编码
                softmax = torch.exp(out1) / torch.sum(torch.exp(out1), dim=1).reshape(-1, 1)
                logsoftmax = torch.log(softmax)
                # print("one_hot",one_hot)
                # print("logsoftmax",logsoftmax)
                # print("one_hot * logsoftmax",one_hot * logsoftmax)
                lss = one_hot * logsoftmax
                loss = -torch.sum(lss*w_tensor.reshape((target.shape[0],1)))
                # print(loss)
                # loss = criterion(out1, target)
                # print("nllloss,loss",nllloss,loss)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
        updata = self.get_local_grad(args,device, copy.deepcopy(self.get_model_params()), init_params)
        return updata

    def feddyn_test(self,test_data, device, args,model):
        # model = self.model
        model.to(device)
        model.eval()
        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                if args.attack_type == "backdoor":
                    # print(x.size())
                    x = torch.squeeze(x)
                    # print("x.size", x.size())
                    if x.size()[0] == 28:break
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                # if batch_idx == 0:
                    # print("pred",pred)
                    # print("target",target)
                    # print("predicted",predicted)
                    # print("predicted.eq(target).sum()",predicted.eq(target).sum())
                correct = predicted.eq(target).sum()
                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics
    def global_test(self, test_data, device, args):
        model = self.model
        model.to(device)
        model.eval()
        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                if args.attack_type == "backdoor":
                    # print(x.size())
                    x = torch.squeeze(x)
                    # print("x.size", x.size())
                    if x.size()[0] == 28:break
                x = x.to(device)
                target = target.to(device)
                if args.defend_module== "ABCD":
                    pred = model(x)
                else:
                    _,pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                # if batch_idx == 0:
                    # print("pred",pred)
                    # print("target",target)
                    # print("predicted",predicted)
                    # print("predicted.eq(target).sum()",predicted.eq(target).sum())
                correct = predicted.eq(target).sum()
                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test(self, test_data, device, args,guding_model,client_model):
        model = self.model
        model.to(device)
        model.eval()
        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }
        criterion = nn.CrossEntropyLoss().to(device)
        # label_count = []
        # rd_bt_idx = np.random.choice(range(len(list(enumerate(test_data)))),5,False)
        # print("tdx",rd_bt_idx)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                if args.attack_type == "backdoor":
                    # print(x.size())
                    x = torch.squeeze(x)
                    # print("x.size", x.size())
                    if x.size()[0] == 28:break
                x = x.to(device)
                target = target.to(device)
                _,pred = model(x)
                loss = criterion(pred, target)
                if args.defend_type == 'DiverseFL':
                    loss = loss + self.L2Losss(model,args.i) #L2正则化
                    # loss.backward()
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
