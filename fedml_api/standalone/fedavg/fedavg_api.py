import copy
import logging
import random
import argparse
import numpy as np
import torch
import wandb
import collections
import pandas as pd
import os
from fedml_api.standalone.fedavg.client import Client
from fedml_core.robustness.robust_aggregation import RobustAggregator
from fedml_core.robustness import robust_aggregation as RAG
from fedml_api.standalone.fedavg.model_poisioning_attack import Model_Attacker
from fedml_api.model.cv.mobilenet import mobilenet
# from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.cv.resnet import resnet20, moon_resnet20,resnet56
from fedml_api.model.cv.cnn import CNN_DropOut ,CNN_FOR_SYT,CNN_Net_CIFAR10,CNN_Net_MNIST


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def set_client_from_params(mdl, params,device):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length
    mdl.load_state_dict(dict_param)
    return mdl
class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer,model_func):
        self.device = device
        self.args = args
        self.model_func = model_func
        [train_data_num, test_data_num, train_data_global, test_data_global,
        train_data_local_num_dict,train_data_local_dict, train_labels_type_dic, test_data_local_dict,
        aided_dataset,global_test_aided_data, class_num] = dataset
        # dataset = [self.train_data_num, self.test_data_num, self.train_data_global, self.test_data_global,
        #            self.train_data_local_num_dict, self.train_data_local_dict, self.train_labels_type_dic, self.test_data_local_dict,
        #            self.aided_dataset,self.global_test_aided_data, self.class_num + self.expend_class_num]
        self.atk_knowlege = ["min_max_updates_only","min_max_agnostic","arg_and_update","arg_only","min_sum_updates_only","min_sum_agnostic"]
        self.train_global = train_data_global
        self.test_global = test_data_global
        # self.targetted_task_test_loader = targetted_task_test_loader
        # self.poisoned_train_loader = poisoned_train_loader
        # self.num_dps_poisoned_dataset = num_dps_poisoned_dataset
        self.global_test_aided_data = global_test_aided_data
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.train_labels_type_dic = train_labels_type_dic
        self.test_data_local_dict = test_data_local_dict
        self.auxiliary_data_dict = aided_dataset
        self.model_trainer = model_trainer
        self.global_loss = 0
        self.class_num = class_num
        self.model_dim = 0
        self.client_selected_record_dict = {i:0 for i in range(self.args.client_num_in_total)}
        self.client_succ = {i:1 for i in range(self.args.client_num_in_total)}
        self.client_fail = {i:1 for i in range(self.args.client_num_in_total)}
        self.client_momun = {i:None for i in range(self.args.client_num_in_total)}
        self.v = None
        self.momentum = args.init_r
        self.label_account_dict = {i:0 for i in range(class_num)}
        self.client_momemtum_dict = {}
        self.privious_global_grad = None
        self.privious_global_model = None
        self.preprivious_global_grad = None
        self.global_grad = None
        self.mean_loss = 0
        self.selected_diff_num = 0
        self.history_grad = []
        self.round_data_num = 0
        self.mine_c = None
        self.mine_s = None
        self.pre_full_grad_norm = 0
        self.fedDyn_h = None
        self.selected_batch_totoal_num = 0

        # for i in range(self.args.client_num_in_total):
        #     labels = train_labels_type_dic[i]
        #     for (l,num) in labels:
        #         self.label_account_dict[l] += num
        # print(self.label_account_dict)

        self._setup_clients(train_data_local_num_dict, train_data_local_dict,train_labels_type_dic, test_data_local_dict, model_trainer,aided_dataset)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict,train_labels_type_dic, test_data_local_dict, model_trainer,aided_data):
        logging.info("############setup_clients (START)#############")
        print("train_data_local_dict[client_idx]",train_data_local_dict.keys())
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx],train_labels_type_dic[client_idx], self.args, self.device, model_trainer,aided_data[client_idx])
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def get_local_grad(self, args, device, cur_params, init_params):
        update_dict = collections.OrderedDict()
        for k in cur_params.keys():
            # if self.is_weight_param(k):
            update_dict[k] = init_params[k].to(device) - cur_params[k].to(device)
            # else:update_dict[k] = cur_params[k].to(device)
        return update_dict
    def danweihua(self,grad, device):
        vec = RAG.vectorize_state_dict(grad,device)
        one_vec = vec / torch.norm(vec, 2)
        return one_vec
    def test_model(self,round_idx,client,w_global):
        model = collections.OrderedDict()
        for key, param in w_global.items():
            # if RAG.is_weight_param(key):
            model[key] = w_global[key].to(self.device) - client.grad[key].to(self.device)
            # else:model[key] = client.grad[key].to(self.device)
        self.model_trainer.set_model_params(model)
        # client.model_state_dict = model
        # self.model_trainer.set_model_params(client.model_state_dict)
        stats = self._aided_data_test(round_idx,client)
        client.aid_loss = self.global_loss - stats["aided_test_loss"]

    def train(self):

        n_clnt = self.args.client_num_in_total
        weight_list = np.asarray([self.train_data_local_num_dict[i] for i in range(n_clnt)])
        weight_list = weight_list / np.sum(weight_list)
        # print(weight_list)
        weight_list_dict = {}
        batch_num_list_dict = {}
        for i in range(n_clnt):
            weight_list_dict[i] = weight_list[i]
            batch_num_list_dict[i] = len(self.train_data_local_dict[i])

        if self.args.defend_type == "feddyn":
            init_model = self.model_func().to(self.device)
            model_func = self.model_func
            device = self.device
            alpha_coef = self.args.init_r
            n_par = len(get_mdl_params([model_func()])[0])
            # weight_list = weight_list * n_clnt
            local_param_list = np.zeros((n_clnt, n_par)).astype('float32')
            init_par_list = get_mdl_params([init_model], n_par)[0]
            clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                        -1)  # n_clnt X n_par
            clnt_models = list(range(n_clnt))

            # avg_model = model_func().to(device)
            # avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            #
            # all_model = model_func().to(device)
            # all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            cld_model = model_func().to(device)
            cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cld_model], n_par)[0]
        else:
            # 获取全局模型
            w_global = self.model_trainer.get_model_params()
            # print(self.model_dim)
            self.privious_global_model = copy.deepcopy(w_global)
            self.privious_global_grad = collections.OrderedDict()
            self.preprivious_global_grad = collections.OrderedDict()
            self.global_grad = collections.OrderedDict()
            for key, value in w_global.items():
                w_global[key] = value.to(self.device)
                self.privious_global_grad[key] = torch.zeros_like(value).to(self.device)
                if self.args.defend_type in ["mime", "mimelite"]:
                    self.mine_s = copy.deepcopy(self.privious_global_grad)
                    self.mine_c = copy.deepcopy(self.privious_global_grad)
                else:
                    self.preprivious_global_grad = copy.deepcopy(self.privious_global_grad)
                    global_grad = copy.deepcopy(self.preprivious_global_grad)
                    global_meta_grad = copy.deepcopy(self.preprivious_global_grad)
                self.model_dim += value.numel()

        if self.args.defend_type == "scaffold":
            global_c = collections.OrderedDict()
            for k,v in w_global.items():
                global_c[k] = torch.zeros_like(v).to(self.device)
        for round_idx in range(self.args.comm_round):
            self.selected_batch_totoal_num = 0

            logging.info("\n#######################################################Communication round : {}".format(round_idx))
            clients = []
            print("self.args.ci:", self.args.ci)

            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                               self.args.client_num_per_round)#random_sample

            logging.info("sample_client_indexes = " + str(client_indexes))
            if self.args.defend_type == "feddyn":

                # # Fix randomness in client selection
                selected_clnts = client_indexes
                # unselected_clnts = []
                # for se in range(self.args.client_num_in_total):
                #     if se not in selected_clnts:
                #         unselected_clnts.append(se)
                # unselected_clnts = np.array(selected_clnts)
                # print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
                cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=self.device)

            for idx, client in enumerate(self.client_list):#client_list 为所有客户的instance
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, False, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx],
                                            self.train_labels_type_dic[client_idx],
                                            self.auxiliary_data_dict[client_idx])
                if self.args.defend_type == "feddyn":
                    client_idx = selected_clnts[idx]
                    clnt = client_idx
                    # Train locally
                    print('---- Training client %d' % clnt)

                    clnt_models[clnt] = model_func().to(device)
                    model = clnt_models[clnt]
                    # Warm start from current avg model
                    model.load_state_dict(copy.deepcopy(dict(cld_model.named_parameters())))
                    for params in model.parameters():
                        params.requires_grad = True

                    # Scale down
                    alpha_coef_adpt = alpha_coef / (n_clnt*weight_list_dict[clnt])  # adaptive alpha coef
                    print(alpha_coef_adpt)
                    local_param_list_curr = torch.tensor(local_param_list[clnt], dtype=torch.float32, device=self.device)
                    clnt_models[clnt] = client.feddyn_train(round_idx, model,alpha_coef_adpt,cld_mdl_param_tensor,local_param_list_curr)
                    curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]

                    # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                    local_param_list[clnt] += curr_model_par - cld_mdl_param
                    clnt_params_list[clnt] = curr_model_par

                elif self.args.defend_type == "moon":
                    model, updata = client.moon_train(round_idx, copy.deepcopy(w_global))
                elif self.args.defend_type == "fedal":
                    model, updata = client.my_train(round_idx, copy.deepcopy(w_global))
                elif self.args.defend_type == "scaffold":
                    updata = client.scaffold_train(round_idx, copy.deepcopy(w_global),copy.deepcopy(global_c))
                    model = None
                elif self.args.defend_type == "fedIR":
                    model,updata = client.fedIR_train(round_idx,copy.deepcopy(w_global),self.label_account_dict)
                elif self.args.defend_type == "fedTye":
                    client.update_client_setting(client.last_client_grad, client.last_client_loss, client.selected,
                                                 client.local_c, self.privious_global_model)  #
                    model,updata = client.RTyserver_train(round_idx,copy.deepcopy(w_global),copy.deepcopy(self.privious_global_model),
                                                  self.mean_loss)
                elif self.args.defend_type == "fedTyetest":
                    client.update_client_setting(client.last_client_grad, client.last_client_loss, client.selected,
                                                 client.local_c, self.privious_global_model)  #
                    model,updata = client.RTyserver_train(round_idx,copy.deepcopy(w_global),copy.deepcopy(self.privious_global_model),
                                                  self.mean_loss)
                elif self.args.defend_type == "fedTest":
                    client.update_client_setting(client.last_client_grad, client.last_client_loss, client.selected,
                                                 client.local_c, self.privious_global_model)  #
                    model, updata,NEW_global_meta_grad = client.RTye_train(round_idx, copy.deepcopy(w_global),copy.deepcopy(global_grad),copy.deepcopy(self.privious_global_model),
                                                      copy.deepcopy(self.preprivious_global_grad), self.mean_loss,copy.deepcopy(global_meta_grad),self.selected_batch_totoal_num)
                elif self.args.defend_type == "fedTest_NoMetaGrad":#                if args.defend_type == "fedTest_NoMetaGrad":
                    client.update_client_setting(client.last_client_grad, client.last_client_loss, client.selected,
                                                 client.local_c, self.privious_global_model)  #
                    model, updata,NEW_global_meta_grad = client.RTye_train(round_idx, copy.deepcopy(w_global),copy.deepcopy(global_grad),copy.deepcopy(self.privious_global_model),
                                                      copy.deepcopy(self.preprivious_global_grad), self.mean_loss,copy.deepcopy(global_meta_grad),self.selected_batch_totoal_num)
                elif self.args.defend_type == "fedTest_Nomeanloss":
                    client.update_client_setting(client.last_client_grad, client.last_client_loss, client.selected,
                                                 client.local_c, self.privious_global_model)  #
                    model, updata,NEW_global_meta_grad = client.RTye_train(round_idx, copy.deepcopy(w_global),copy.deepcopy(global_grad),copy.deepcopy(self.privious_global_model),
                                                      copy.deepcopy(self.preprivious_global_grad), self.mean_loss,copy.deepcopy(global_meta_grad),self.selected_batch_totoal_num)
                elif self.args.defend_type == "fedTest_momemtum":
                    client.update_client_setting(client.last_client_grad, client.last_client_loss, client.selected,
                                                 client.local_c, self.privious_global_model)  #
                    model, updata,NEW_global_meta_grad = client.RTye_train(round_idx, copy.deepcopy(w_global),copy.deepcopy(global_grad),copy.deepcopy(self.privious_global_model),
                                                      copy.deepcopy(self.preprivious_global_grad), self.mean_loss,copy.deepcopy(global_meta_grad),self.selected_batch_totoal_num)

                elif self.args.defend_type == "mime":

                    model, updata = client.mime_train(round_idx, copy.deepcopy(w_global),
                                                      self.mine_s,
                                                      self.mine_c)
                elif self.args.defend_type == "mimelite":
                    # client.pi = self.train_data_local_num_dict[client.client_idx] / self.train_data_num_in_total
                    model, updata = client.mimelite_train(round_idx, copy.deepcopy(w_global),
                                                     self.mine_s,
                                                      )
                else:

                    model, updata = client.train(round_idx, copy.deepcopy(w_global))
                if self.args.defend_type != "feddyn":
                    client.class_num = self.class_num - 1
                    client.grad = updata
                    client.weight = weight_list_dict[client_idx]
                    # if self.args.defend_type == "fedTest":
                    #     client.client_meta_grad = None
                    # client.cur_local_model = model
                    # client.local_model = model
                    clients.append(client)


                # # #///本地测试
                # if round_idx % self.args.frequency_of_the_test == 0:#设置记录准确率的频率
                #     if self.args.dataset.startswith("stackoverflow"):
                #         self._local_test_on_validation_set(round_idx)
                #     else:
                #         self._local_test_on_client(round_idx,client)



            Aggregator_method_obj = RobustAggregator(self.args)  # 设置鲁邦聚合类的资源


            if self.args.defend_type == 'fedavg':
                for c in clients:
                    c.weight = 1/self.args.client_num_per_round
                global_grad = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                                                       self.args, self.device)# 按模型加权聚合
                w_global = self._grad_desc(global_grad,copy.deepcopy(w_global))
                # w_global = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                #                                        self.args, self.device)# 按模型加权聚合
            if self.args.defend_type == 'fedTye':
                self.privious_global_model = copy.deepcopy(w_global)
                self.mean_loss = 0
                for c in clients:
                    c.weight = 1/self.args.client_num_per_round
                    self.mean_loss += c.last_client_loss
                self.mean_loss = self.mean_loss / self.args.client_num_per_round
                global_grad = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                                                       self.args, self.device)# 按模型加权聚合
                w_global = self._grad_desc(global_grad,copy.deepcopy(w_global))

            if self.args.defend_type == 'fedTest':
                self.privious_global_model = copy.deepcopy(w_global)
                # self.privious_global_grad = copy.deepcopy(global_grad)
                # self.preprivious_global_grad = copy.deepcopy(prepre_meta_global_grad)
                self.selected_batch_totoal_num = 0
                self.mean_loss = 0
                for c in clients:
                    c.weight = 1/self.args.client_num_per_round
                    self.mean_loss += c.last_client_loss
                    self.selected_batch_totoal_num += batch_num_list_dict[c.client_idx]*self.args.epochs
                self.mean_loss = self.mean_loss / self.args.client_num_per_round
                self.selected_batch_totoal_num = self.selected_batch_totoal_num / self.args.client_num_per_round
                global_grad = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                                                       self.args, self.device)# 按模型加权聚合
                global_meta_grad = NEW_global_meta_grad
                # for c in clients:
                #     c.grad = c.client_meta_grad
                # global_meta_grad = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                #                                        self.args, self.device)# 按模型加权聚合
                w_global = self._grad_desc(global_grad,copy.deepcopy(w_global))
            if self.args.defend_type == 'fedTest_Nomeanloss':
                self.privious_global_model = copy.deepcopy(w_global)
                # self.privious_global_grad = copy.deepcopy(global_grad)
                # self.preprivious_global_grad = copy.deepcopy(prepre_meta_global_grad)
                self.selected_batch_totoal_num = 0
                self.mean_loss = 0
                for c in clients:
                    c.weight = 1/self.args.client_num_per_round
                    self.mean_loss += c.last_client_loss
                    self.selected_batch_totoal_num += batch_num_list_dict[c.client_idx]*self.args.epochs
                self.mean_loss = self.mean_loss / self.args.client_num_per_round
                self.selected_batch_totoal_num = self.selected_batch_totoal_num / self.args.client_num_per_round
                global_grad = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                                                       self.args, self.device)# 按模型加权聚合
                global_meta_grad = NEW_global_meta_grad
                # for c in clients:
                #     c.grad = c.client_meta_grad
                # global_meta_grad = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                #                                        self.args, self.device)# 按模型加权聚合
                w_global = self._grad_desc(global_grad,copy.deepcopy(w_global))
            if self.args.defend_type == 'fedTest_NoMetaGrad':
                self.privious_global_model = copy.deepcopy(w_global)
                # self.privious_global_grad = copy.deepcopy(global_grad)
                # self.preprivious_global_grad = copy.deepcopy(prepre_meta_global_grad)
                self.selected_batch_totoal_num = 0
                self.mean_loss = 0
                for c in clients:
                    c.weight = 1/self.args.client_num_per_round
                    self.mean_loss += c.last_client_loss
                    self.selected_batch_totoal_num += batch_num_list_dict[c.client_idx]*self.args.epochs
                self.mean_loss = self.mean_loss / self.args.client_num_per_round
                self.selected_batch_totoal_num = self.selected_batch_totoal_num / self.args.client_num_per_round
                global_grad = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                                                       self.args, self.device)# 按模型加权聚合
                global_meta_grad = NEW_global_meta_grad
                # for c in clients:
                #     c.grad = c.client_meta_grad
                # global_meta_grad = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                #                                        self.args, self.device)# 按模型加权聚合
                w_global = self._grad_desc(global_grad,copy.deepcopy(w_global))
            if self.args.defend_type == 'fedTest_momemtum':
                self.privious_global_model = copy.deepcopy(w_global)
                # self.privious_global_grad = copy.deepcopy(global_grad)
                # self.preprivious_global_grad = copy.deepcopy(prepre_meta_global_grad)
                self.selected_batch_totoal_num = 0
                self.mean_loss = 0
                for c in clients:
                    c.weight = 1/self.args.client_num_per_round
                    self.mean_loss += c.last_client_loss
                    self.selected_batch_totoal_num += batch_num_list_dict[c.client_idx]*self.args.epochs
                self.mean_loss = self.mean_loss / self.args.client_num_per_round
                self.selected_batch_totoal_num = self.selected_batch_totoal_num / self.args.client_num_per_round
                global_grad = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                                                       self.args, self.device)# 按模型加权聚合
                global_meta_grad = NEW_global_meta_grad
                # for c in clients:
                #     c.grad = c.client_meta_grad
                # global_meta_grad = Aggregator_method_obj.fedavg(round_idx, clients, copy.deepcopy(w_global),
                #                                        self.args, self.device)# 按模型加权聚合
                w_global = self._grad_desc(global_grad,copy.deepcopy(w_global))

            if self.args.defend_type in ['mimelite', "mime"] :

                self.mine_c,global_grad,_,global_norm = Aggregator_method_obj.mime(round_idx, clients, copy.deepcopy(w_global),
                                                       self.args, self.device)# 按模型加权聚合


                # if round_idx >0:
                #     wandb.log({"s_tK": self.pre_full_grad_norm.item()/global_norm.item(), "round": round_idx})
                self.pre_full_grad_norm = global_norm
                for key in self.mine_s.keys():#服务器状态更新
                    self.mine_s[key] = (1-self.args.init_r)*self.mine_c[key]  + self.args.init_r*self.mine_s[key]
                w_global = self._grad_desc(global_grad, copy.deepcopy(w_global))

            if self.args.defend_type == "feddyn":
                avg_mdl_param = np.mean(clnt_params_list[selected_clnts], axis=0)
                cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)

                # avg_model = set_client_from_params(model_func(), avg_mdl_param)
                # all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis=0))
                cld_model = set_client_from_params(model_func().to(device), cld_mdl_param,device)


                logging.info("\n")
                logging.info("global_test_on_validation_set_at_round : {}".format(round_idx))

                test_metrics = {
                    'num_samples': [],
                    'num_correct': [],
                    'losses': []
                }

                sever = self.client_list[0]  # 代用客户1来测试全局模型准确率
                sever.update_local_dataset(0, sever.byzantine, None, self.test_global, None, None, None)  # 设置全局验证数据
                # test data
                # print("set_gl_model",sever.model_trainer.get_model_params())
                test_global_metrics = sever.feddyn_test(cld_model)  # 使用全局模型在验证数据集上测试
                test_metrics['num_samples'].append(copy.deepcopy(test_global_metrics['test_total']))
                test_metrics['num_correct'].append(copy.deepcopy(test_global_metrics['test_correct']))
                test_metrics['losses'].append(copy.deepcopy(test_global_metrics['test_loss']))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                # test on test dataset
                test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
                test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
                stats = {'global_Acc': test_acc, 'global_Loss': test_loss}
                wandb.log({"global_Acc": test_acc, "round": round_idx})
                wandb.log({"global_Loss": test_loss, "round": round_idx})
                logging.info(stats)
            else:
                self.model_trainer.set_model_params(w_global)
                _ = self._global_test(round_idx)


    def _client_sampling(self,round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)# 每轮选不同用户
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        # logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
    def _Thomson_sampling(self,round_idx,clients_index):
        client_indexes = []
        if round_idx < 10:
            return clients_index
        else:
            pro = {}
            pro[round_idx] = []
            for i in clients_index:
                p = np.random.beta(self.client_succ[i], self.client_fail[i])
                pro[round_idx].append(p)
                if p >= 0.9:
                    client_indexes.append(i)
                if p > 0.2 and p <= 0.9 and np.random.random() < p:
                    client_indexes.append(i)
            if len(client_indexes) == 0:  # 如果本轮采样结果为空，则所有用户均被选中
                client_indexes = clients_index
            logging.info("each_round_pro:{}".format(pro[round_idx]))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num  = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset
    def _grad_desc(self, global_grad,current_model):#加权聚合
        for key, param in current_model.items():

            # if RAG.is_weight_param(key):
                current_model[key] = param.to(self.device) - global_grad[key]

            # else:
            #     current_model[key] = global_grad[key]
        return current_model
    def _global_c_desc(self, global_c,current_c):#加权聚合
        for key, param in current_c.items():
            current_c[key] = param.to(self.device) + (self.args.client_num_per_round/self.args.client_num_in_total) * global_c[key]
        return current_c

    def _aggregate(self, w_locals):#加权聚合
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                    # print("_aggregate1")
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _local_test_on_client(self, round_idx,client):
        logging.info("\n")
        logging.info("local_test_on_client_{}_at_round : {}".format(client.client_idx,round_idx))

        """
        Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
        the training client number is larger than the testing client number
        """
        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        ### train data
        train_local_metrics = client.local_test(1) #False使用训练集
        train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
        train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
        train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

        ### test data
        test_local_metrics = client.local_test(2)#Ture使用验证集
        test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))



        """
        Note: CI environment is CPU-based computing. 
        The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
        """

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        ## test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])



        # stats = {'training_acc': train_acc, 'training_loss': train_loss,'client_idx': client.client_idx}
        logging.info({"round": round_idx, "client_idx":client.client_idx, "client_identity":client.byzantine , "Train/Acc": train_acc, "Train/Loss": train_loss})
        stats = {'test_acc': test_acc, 'test_loss': test_loss,'client_idx': client.client_idx}
        logging.info({"round": round_idx, "client_idx":client.client_idx, "client_identity":client.byzantine , "Test/Acc": test_acc, "Test/Loss": test_loss})
        # logging.info({"round": round_idx, "client_idx":client.client_idx, "client_identity":client.byzantine , "Train/Acc": train_acc, "Train/Loss": train_loss})

        # return pubulic_stats

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients_at_round : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]
        client.model_trainer.set_model_params(client.local_model)
        # print("client.local_model",client.local_model)
        # print("client.model_trainer.set_model_params(client.local_model)",self.model_trainer.get_model_params())
        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0,client.byzantine, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx],self.train_labels_type_dic[client_idx])
            # train data
            train_local_metrics = client.local_test(1)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(2)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
            if client_idx == self.args.client_num_per_round :
                break
        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)



    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, client.byzantine,None, self.val_global, None,None)
        # test data
        test_metrics = client.local_test(2)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!"%self.args.dataset)

        logging.info(stats)
    def _backdoor_task_test(self,round_idx):
        logging.info("\n")
        logging.info("global_test_on_validation_set_at_round : {}".format(round_idx))

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        sever = self.client_list[0]  # 代用客户1来测试全局模型准确率
        sever.update_local_dataset(0, sever.byzantine, None, self.targetted_task_test_loader, None, None, None)  # 设置全局验证数据
        # test data
        # print("set_gl_model",sever.model_trainer.get_model_params())
        test_global_metrics = sever.global_test()  # 使用全局模型在验证数据集上测试
        test_metrics['num_samples'].append(copy.deepcopy(test_global_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_global_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_global_metrics['test_loss']))

        """
        Note: CI environment is CPU-based computing. 
        The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
        """
        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        stats = {'backdoor_Acc': test_acc, 'backdoor_Loss': test_loss}
        wandb.log({"backdoor_Acc": test_acc, "round": round_idx})
        wandb.log({"backdoor_Loss": test_loss, "round": round_idx})
        logging.info(stats)
        return stats
    def _global_test(self,round_idx):
        logging.info("\n")
        logging.info("global_test_on_validation_set_at_round : {}".format(round_idx))

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        sever = self.client_list[0] #代用客户1来测试全局模型准确率
        sever.update_local_dataset(0,sever.byzantine,None,self.test_global, None,None,None)#设置全局验证数据
        # test data
        # print("set_gl_model",sever.model_trainer.get_model_params())
        test_global_metrics = sever.global_test()#使用全局模型在验证数据集上测试
        test_metrics['num_samples'].append(copy.deepcopy(test_global_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_global_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_global_metrics['test_loss']))

        """
        Note: CI environment is CPU-based computing. 
        The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
        """
        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        stats = {'global_Acc': test_acc, 'global_Loss': test_loss}
        wandb.log({"global_Acc": test_acc, "round": round_idx})
        wandb.log({"global_Loss": test_loss, "round": round_idx})
        logging.info(stats)
        return stats
    def _aided_data_test(self,round_idx,client):

        if self.args.attack_type != "model_attack":
            logging.info("\n")
            logging.info("****Aided_data_test_on_cliet_{}_at_round : {}".format(client.client_idx,round_idx))

        """
        Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
        the training client number is larger than the testing client number
        """

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        # test data
        test_local_metrics = client.local_test(3)#使用辅助验证集
        test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        # logging.info({"round": round_idx, "client_idx":client.client_idx, "client_identity":client.byzantine , "Train/Acc": train_acc, "Train/Loss": train_loss})
        stats = {'aided_test_acc': test_acc, 'aided_test_loss': test_loss,'client_idx': client.client_idx}
        if self.args.attack_type != "model_attack":
            logging.info({"round": round_idx, "client_idx":client.client_idx, "client_identity":client.byzantine , "Aided_Test/Acc": test_acc, "Aided_Test/Loss": test_loss})
            logging.info(stats)
        # else:
        #     print("searching...")
        return stats