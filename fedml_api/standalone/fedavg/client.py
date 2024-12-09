import logging
import torch
import collections
import copy
from torch import nn

def is_weight_param(k):
    return ("running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k)

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number,train_labels_type, args, device,
                 model_trainer,aided_data):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.aided_test_data = aided_data
        self.labels = train_labels_type  #[(label,label number),(,)]
        # logging.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.cur_local_model = None
        self.pre_local_model = None
        self.guding_model = None
        self.acc_grad = None
        self.grad = None
        self.grad_norm = 0
        self.grad_norm_wo_mean_var = 0
        self.vec_grad_one = None
        self.model = collections.OrderedDict()
        self.model_state_dict = None
        self.model_norm = None
        self.model_dim =None
        self.vec_model_one = None
        self.vec_grad_no_meanvar =None
        self.loss = nn.CrossEntropyLoss()
        self.byzantine = False
        self.weight = 1/self.args.client_num_per_round
        self.min_norm = None
        self.aid_loss = 0
        self.class_num =0
        self.test = 0
        self.momentum = None
        self.succ = 1
        self.fail = 1
        self.seleted_epoch = 0
        self.miu = self.args.init_r
        self.test_num = 0
        self.label_dict ={}
        self.batch_num = 0
        self.delta_loss = 0
        self.data = None
        self.bth_mean_grad_dict = collections.OrderedDict()
        self.tye_pro = 0
        self.last_client_grad = collections.OrderedDict()
        self.prepre_meta_global_grad = None
        self.pre_global_grad = None
        self.last_client_loss = 0
        self.selected = False
        self.local_c = collections.OrderedDict()
        self.delta_c = collections.OrderedDict()
        self.w = 1
        self.aggr_w = 1/self.args.client_num_per_round
        self.full_grad = None
        self.pi = None
        self.client_feddyn_w = None
        self.client_feddyn_nabla = None
        self.client_meta_grad =None
        # self.set_guding_model = False
    def update_client_MAB_FL_setting(self,round_idx,client_selected_record,client_succ,client_fail):

        self.succ = client_succ
        self.fail = client_fail
        self.seleted_epoch = client_selected_record
    def update_local_dataset(self, client_idx, client_idx_state,local_training_data, local_test_data, local_sample_number,lables,aided_test_data):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.labels = lables
        self.byzantine = client_idx_state
        # self.acc_grad
        self.aided_test_data = aided_test_data
    def update_client_setting(self,last_client_grad,last_client_loss,selected,local_c,pre_local_model):
        self.last_client_grad = last_client_grad
        self.last_client_loss = last_client_loss
        self.selected = selected
        self.local_c = local_c
        self.pre_local_model = pre_local_model


    def accmulate_grad(self,curr_grad):
        if self.acc_grad == None:
            self.acc_grad = curr_grad
        else:
            for key,param in curr_grad.items():
                self.acc_grad[key] += param

    def get_sample_number(self):
        return self.local_sample_number

    def train(self,round_idx, w_global):#w_global为字典
        logging.info("client{}_byt{}_training...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播
        grads = self.model_trainer.train(round_idx,self.local_training_data, self.device, self.args,w_global)
        weights = self.model_trainer.get_model_params()

        return weights,grads

    # round_idx, copy.deepcopy(cld_model), alpha_coef_adpt, cld_mdl_param_tensor, local_param_list_curr
    def feddyn_train(self,round_idx, cld_model, alpha_coef,avg_mdl_param,local_grad_vector):#w_global为字典
        logging.info("client{}_feddyn_train...".format(self.client_idx))
        model = self.model_trainer.feddyn_train(round_idx,self.local_training_data, self.device, self.args,cld_model, alpha_coef,avg_mdl_param,local_grad_vector)
        return model
    def moon_train(self,round_idx, w_global):#w_global为字典
        logging.info("client{}_byt{}_training...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播
        grads = self.model_trainer.contrastive_train(round_idx,self.local_training_data, self.device, self.args,w_global,self.pre_local_model)
        weights = self.model_trainer.get_model_params()
            # weights = 1
        return weights,grads


    def RTyserver_train(self,round_idx, w_global,previous_w_global,mean_loss):#w_global为字典
        logging.info("client{}_byt{}_Tye_train...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播

        grads, train_loss = self.model_trainer.RTysever_train(round_idx,self.local_training_data,
                                                                                   self.device, self.args, w_global,
                                                                                   previous_w_global,mean_loss,self.selected,self.last_client_grad,self.last_client_loss)
        weights = self.model_trainer.get_model_params()

        self.last_client_grad = grads
        self.last_client_loss = train_loss
        self.selected = True
        return weights,grads
    def RTyserver_train_mul(self,round_idx, w_global,previous_w_global,mean_loss):#w_global为字典
        logging.info("client{}_byt{}_Tye_train...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播
        if self.byzantine == False:
            grads, train_loss = self.model_trainer.RTysever_train_pi(round_idx,self.local_training_data,
                                                                                       self.device, self.args, w_global,
                                                                                       previous_w_global,mean_loss,self.selected,self.last_client_grad,self.last_client_loss,self.pi)
            weights = self.model_trainer.get_model_params()
            # weights = 1
        else:#攻击者执行攻击
            # print("1")
            self.attack_based_on_data() #None表示无基于参数的攻击
            grads, train_loss = self.model_trainer.RTysever_train_pi(round_idx, self.local_training_data,
                                                                                       self.device, self.args, w_global,
                                                                                       previous_w_global,mean_loss,self.selected,self.last_client_grad,self.last_client_loss,self.pi)

            weights = self.model_trainer.get_model_params()
            grads, weights, = self.attack_based_on_param(grads,weights)

        self.last_client_grad = grads
        self.last_client_loss = train_loss
        self.selected = True
        return weights,grads
    def mime_train(self,round_idx, w_global,s,c):#w_global为字典
        logging.info("client{}_byt{}_mime_training...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播
        full_batch_grad,grads= self.model_trainer.mime_train(round_idx,self.local_training_data, self.device, self.args,w_global,s,c)
        self.full_grad = full_batch_grad
        weights = None
        return weights,full_batch_grad
    def mimelite_train(self,round_idx, w_global,s):#w_global为字典
        logging.info("client{}_byt{}_mimelite_training...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播
        full_batch_grad,grads= self.model_trainer.mimelite_train(round_idx,self.local_training_data, self.device, self.args,w_global,s)
        self.full_grad = full_batch_grad
        # weights = self.model_trainer.get_model_params()
        weights = None
        return weights,grads
    def RTye_train(self,round_idx, w_global,global_grad,previous_w_global,previous_global_gradient,mean_loss,global_meta_grad,selected_batch_totoal_num):#w_global为字典
        logging.info("client{}_byt{}_RTye_train...".format(self.client_idx,self.byzantine))
        # w_global = collections.OrderedDict()
        # for key, param in previous_global_model.items():
        #     # if RAG.is_weight_param(key):
        #         w_global[key] = param.to(self.device) - global_grad[key]

        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播

        grads, global_meta_grad, train_loss = self.model_trainer.RTye_train(round_idx,self.local_training_data,
                                                                                   self.device, self.args, w_global,previous_w_global,global_grad,
                                                                                   global_meta_grad,mean_loss,selected_batch_totoal_num)
        self.last_client_grad = None
        self.last_client_loss = train_loss
        self.selected = True
        self.batch_num = len(self.local_training_data)
        weights = self.model_trainer.get_model_params()

        return weights,grads,global_meta_grad
    def Tye_train(self,round_idx, w_global,bth_mean_grad_dict,mean_loss):#w_global为字典
        logging.info("client{}_byt{}_Tye_train...".format(self.client_idx,self.byzantine))
        # logging.info("client{}_byt{}_training...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播
        if self.byzantine == False:
            grads, batch_grad, batch_loss, first_bt_num = self.model_trainer.Tye_train(round_idx,
                                                                                       self.local_training_data,
                                                                                       self.device, self.args, w_global,
                                                                                       bth_mean_grad_dict, mean_loss)
            weights = self.model_trainer.get_model_params()
            # weights = 1
        else:#攻击者执行攻击
            # print("1")
            self.attack_based_on_data() #None表示无基于参数的攻击
            grads, batch_grad, batch_loss, first_bt_num = self.model_trainer.Tye_train(round_idx,
                                                                                       self.local_training_data,
                                                                                       self.device, self.args, w_global,
                                                                                       bth_mean_grad_dict, mean_loss)

            weights = self.model_trainer.get_model_params()
            grads,weights, = self.attack_based_on_param(grads,weights)

        # self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播
        # grads,batch_grad ,batch_loss,first_bt_num = self.model_trainer.Tye_train(round_idx,self.local_training_data, self.device, self.args,w_global,bth_mean_grad_dict,mean_loss)
        # weights = self.model_trainer.get_model_params()
        self.batch_grad = batch_grad
        self.batch_loss = batch_loss
        self.batch_num = first_bt_num
        return weights,grads
    def BT_train(self,round_idx, w_global):#w_global为字典
        logging.info("client{}_byt{}_training...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播
        grads,bt_num ,delta_loss,data = self.model_trainer.BT_train(round_idx,self.local_training_data, self.device, self.args,w_global)
        weights = self.model_trainer.get_model_params()
        self.batch_num =bt_num
        self.delta_loss = delta_loss
        self.data = data
        # self.
            # weights = 1
        return weights,grads
    def my_train(self,round_idx, w_global):#w_global为字典
        logging.info("client{}_byt{}_training...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播
        grads = self.model_trainer.my_train(round_idx,self.local_training_data, self.device, self.args,w_global,self.pre_local_model)
        weights = self.model_trainer.get_model_params()
            # weights = 1
        return weights,grads
    def scaffold_train(self,round_idx, w_global,global_c):#w_global为字典
        logging.info("client{}_byt{}_scaffold_train...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播

        grads,local_c_add = self.model_trainer.scaffold_train(round_idx,self.selected,self.local_training_data, self.device, self.args,w_global,global_c,copy.deepcopy(self.local_c))
        # weights = self.model_trainer.get_model_params()

        for k,v in local_c_add.items():
            local_c_add[k] = local_c_add[k] - self.local_c[k]
        self.local_c = copy.deepcopy(local_c_add)
        self.delta_c = local_c_add
        self.selected = True
            # weights = 1
        return grads
    def fedIR_train(self,round_idx, w_global,label_account_dict):#w_global为字典
        logging.info("client{}_byt{}_training...".format(self.client_idx,self.byzantine))
        self.model_trainer.set_model_params(w_global) #设置客户的模型,相对于广播
        grads = self.model_trainer.fedIR_train(round_idx,self.local_training_data, self.device, self.args,w_global,[label_account_dict,self.labels])
        weights = self.model_trainer.get_model_params()
            # weights = 1
        return weights,grads

    def local_test(self, b_use_test_dataset):

        if b_use_test_dataset == 1:
            test_data = self.local_training_data
            # print("client_model_ture",self.model_trainer.get_model_params())
        if b_use_test_dataset == 2:
            test_data = self.local_test_data
            # print("client_model_False",self.model_trainer.get_model_params())
        if b_use_test_dataset == 3:
            test_data = self.aided_test_data
            # print("client_model_False",self.model_trainer.get_model_params().items())
        metrics = self.model_trainer.test(test_data, self.device, self.args,self.guding_model,self.local_model)
        return metrics

    def global_test(self):
        test_data = self.local_test_data
        # print("global_model_T", self.model_trainer.get_model_params())
        metrics = self.model_trainer.global_test(test_data, self.device, self.args)
        return metrics
    def feddyn_test(self,model):
        test_data = self.local_test_data
        # print("global_model_T", self.model_trainer.get_model_params())
        metrics = self.model_trainer.feddyn_test(test_data, self.device, self.args,model)
        return metrics
    def attack_based_on_data(self):
        # model = []
        # print("2.1",self.args.attack_type)
        if self.args.attack_type == "label_flipping":

            flipped_data = []
            model = ["mobilenet","lr"]
            if self.args.model in model:
                # print("2.2")
                logging.info("label_flipping...")

                for (x,label) in self.local_training_data:
                    lbelow = label[torch.where(label < 10)[0]]
                    l = torch.ones(size=lbelow.size(), dtype=torch.int) * 9
                    label[torch.where(label < 10)[0]] = l - lbelow
                    flipped_data.append((x, label))
                self.local_training_data = flipped_data
            if self.args.model == "cnn" and self.args.dataset == "mnist":
                logging.info("label_flipping...")
                for (x,label) in self.local_training_data:
                    lbelow = label[torch.where(label < 10)[0]]
                    l = torch.ones(size=lbelow.size(), dtype=torch.int) * 9
                    label[torch.where(label < 10)[0]] = l - lbelow
                    flipped_data.append((x, label))
                self.local_training_data = flipped_data
            if self.args.dataset == "cifar10":
                logging.info("label_flipping...")
                for (x,label) in self.local_training_data:
                    lbelow = label[torch.where(label < 9)[0]]
                    l = torch.ones(size=lbelow.size(), dtype=torch.int) * 8
                    label[torch.where(label < 9)[0]] = l - lbelow
                    flipped_data.append((x, label))
                self.local_training_data = flipped_data
            if  self.args.model == "cnn" and self.args.dataset == "femnist":
                logging.info("label_flipping...")
                for (x,label) in self.local_training_data:
                    # print("x",x.size())
                    lbelow = label[torch.where(label<62)[0]]
                    l = torch.ones(size=lbelow.size(), dtype=torch.int) * 61
                    label[torch.where(label<62)[0]] = l - lbelow
                    flipped_data.append((x, label))
                self.local_training_data = flipped_data
            if  self.args.model == "rnn" and self.args.dataset == "shakespeare":
                logging.info("label_flipping...")
                for (x,label) in self.local_training_data:
                    # print("pre_label:",label[:20])
                    loc_below = torch.where(label<81)[0]
                    loc_above = torch.where(label>81)[0]
                    lbelow = label[loc_below]
                    l_1 = torch.ones(size=lbelow.size(), dtype=torch.int) * 90
                    labove = label[loc_above]
                    l_2 = torch.ones(size=labove.size(), dtype=torch.int) * 90
                    label[loc_below] = l_1 - lbelow
                    label[loc_above] = l_2 - labove
                    print("new_label:", label[:20])
                    flipped_data.append((x, label))
                self.local_training_data = flipped_data
        # if self.args.attack_type == "bitflip_attack":
        #     for (x,label) in self.local_training_data:
        #         x[:] = -x[0]

    def attack_based_on_param(self, grads, weight):

        if self.args.attack_type == "gaussian_attack":
            print("self.args.attack_type", self.args.attack_type)
            for (name,v) in weight.items():
                # if is_weight_param(name):
                print("name",name)
                gaussian_noise = torch.normal(mean=0,std=0.5,size=v.size()).to(self.device)
                # grads[name] = grads[name].to(self.device) + gaussian_noise
                weight[name] = weight[name].to(self.device) + gaussian_noise

        if self.args.attack_type == "sign_flipping":
            print("self.args.attack_type", self.args.attack_type)
            for (name,_) in weight.items():
                grads[name] = -20*grads[name] #2.5
                weight[name] = -20*weight[name]
        # if self.args.attack_type == "random_grad":
        #     print("self.args.attack_type", self.args.attacker_knowlege)
        #     for (name, parm) in grads.items():
        #         grads[name] = torch.normal(mean=0, std=2, size=parm.size()).div(1e20)
        #         weight[name] = torch.normal(mean=0, std=2, size=parm.size()).div(1e20)


        # if self.args.attack_type == "param_flipping":
        #     print("self.args.attack_type", self.args.attack_type)
        #     grads = 0
        #     weight = 0
        return grads,weight
    # def attack_based_parameter(self,weights):
    #     return weights





