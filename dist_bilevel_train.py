#!/usr/bin/env python3

import json

import argparse

try: import setGPU
except ImportError: pass
import numpy as np




import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Function, Variable, grad
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter


import os
import sys
import math
import copy
import shutil

import setproctitle

import densenet
import models
import yaml
# import make_graph

# from IPython.core import ultratb

from update import LocalUpdate, test_inference
from dist_train import FedDistManager

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return image.clone().detach().requires_grad_(True), label.clone().detach().requires_grad_(True)
        return torch.as_tensor(image), torch.as_tensor(label)
        # return Variable(image), Variable(label)

class DistManager(object):
    def __init__(self, save_path, seed=42) -> None:
        self.save_path = save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        self.logger = SummaryWriter('logs', flush_secs=1)
        self.slurm_id = os.getenv("SLURM_JOB_ID")
        if self.slurm_id:
            self.logger.add_text("slurm_id", self.slurm_id)
    
        
    def run_exp(self, config_dict) -> None:
        # record the experiment result to the dsirectory
        # make the global_model to the clients and complete the initial method
        # split dataset to different clients.
        # for number of epochs
            # every client complete local update and test the model            
            # every client upload information to the global             
            # global complete calculation and test the global model            
            # every client receive information from global and update model            
            # record the test information
        self.config_dict = config_dict
        
    
 
class BilevelFedDistManager(FedDistManager):
    '''
    This class manage the bilevel optimization training procedure for federated learning with central server
    The code comes from FedNest
    '''
    def __init__(self, save_path) -> None:
        super().__init__(save_path)
        
    def run_exp(self, config_dict) -> None:
        # super().run_exp(config_dict)
        self.config_dict = config_dict
        
        # record the training information
        self.trainF = open(os.path.join(self.save_path, 'train.csv'), 'w')
        self.trainF.write('glo_ep,loc_ep,user_id,loss,err\n')
        self.trainF.flush()
        
        self.valF = open(os.path.join(self.save_path, 'val.csv'), 'w')
        self.valF.write('glo_ep,user_id,loss,err\n')
        self.valF.flush()
        
        self.testF = open(os.path.join(self.save_path, 'test.csv'), 'w')
        self.testF.write('glo_ep,loss,err\n')
        self.testF.flush()
        

        # get the dataset from the origin dataset
        self.train_dataset, self.test_dataset, self.user_groups = self.get_dataset()
        
        # initialize the global model and save the weight
        self.global_net = self.get_net().to(self.device)
        
        
        net_weight = self.global_net.state_dict()
        # find the hyper param
        self.hyper_param = [k for n,k in self.global_net.named_parameters() if not "header" in n]
        self.hyper_param_init = [k for n,k in self.global_net.named_parameters() if not "header" in n]
        self.hyper_optimizer = torch.optim.SGD(self.hyper_param, lr=self.config_dict["hlr"])
        self.val_loss = self.cross_entropy
        self.loss_func = self.cross_entropy_reg
        self.beta = 1
        
        # print the net information for debug
        number_param = sum([p.data.nelement() for p in self.global_net.parameters()])
        print('  + Number of params: {}'.format(number_param))
        
        # start training and 
        train_loss, train_accuracy = [], []
        for epoch in range(1, config_dict["nEpoch"] + 1):
            local_weights, local_losses, list_acc = [], [], []
            print(f'\n | Global Training Round : {epoch} |\n')
            # change the network to train mode
            self.global_net.train()
            
            # find the number of users for next epoch
            m = max(int(config_dict['frac'] * config_dict['num_users']), 1)
            
            # Repeat FedIn for config_dict['inEpoch'] times
            for _ in range(1, config_dict["inEpoch"] + 1):
                idxs_users = np.random.choice(range(config_dict['num_users']), m, replace=False)
                for idx in idxs_users:
                    w, acc, loss = self.local_train(user_id=idx, epoch=epoch)
                    list_acc.append(acc)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
                loss_avg = sum(local_losses) / len(local_losses)
                # aggregate the local models and calculate the local training loss
                global_weights = self.aggregate(local_weights)
                self.global_net.load_state_dict(global_weights)
                
                train_loss.append(loss_avg)
                train_accuracy.append(sum(list_acc)/len(list_acc))
            
            # complete FedOut
            idxs_users = np.random.choice(range(config_dict['num_users']), m, replace=False)
            
            self.fed_out_train(idxs_users, epoch)
            
            if (epoch+1) % 5 == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
                
            test_acc, test_loss = self.global_test(self.global_net)
            if self.slurm_id:
                # run at Slurm job cluster
                self.logger.add_scalars(f"{self.slurm_id}/accuracy", {"train_acc": train_accuracy[-1], "test_acc": test_acc}, epoch)
                self.logger.add_scalars(f"{self.slurm_id}/loss",  {"train_loss": train_accuracy[-1], "test_loss": test_acc}, epoch)
            else:
                self.logger.add_scalars("accuracy", {"train_acc": train_accuracy[-1], "test_acc": test_acc}, epoch)
                self.logger.add_scalars("loss",  {"train_loss": train_accuracy[-1], "test_loss": test_acc}, epoch)
            self.testF.write('{},{},{}\n'.format(epoch, test_loss, 1 - test_acc))
            self.testF.flush()
            
            try:
                torch.save(self.global_net, os.path.join(config_dict['save'], 'latest.pth'))
            except:
                pass
        # os.system('./plot.py "{}" &'.format(config_dict['save']))

        self.trainF.close()
        self.valF.close()
        self.testF.close()
        
    def get_net(self):
        dataset = self.config_dict['dataset']
            
        if self.config_dict['model_name'] == 'cnn' and dataset == 'cifar-10':
            net = models.CNNCifar(args=self.config_dict).to(self.device)
        elif self.config_dict['model_name'] == 'cnn' and dataset == 'mnist':
            net = models.CNNMnist(args=self.config_dict).to(self.device)
        elif self.config_dict['model_name'] == 'mlp':
            len_in = 1
            for x in self.config_dict['img_size']:
                len_in *= x
            net = models.MLP(dim_in=len_in, dim_hidden=200,
                            dim_out=self.config_dict['num_classes']).to(self.device)
        elif self.config_dict['model_name'] == 'linear':
            net = models.Linear(args=self.config_dict).to(self.device)
        elif self.config_dict['model_name'] == 'fmnist_cnn':
            net = models.MM_CNN(args=self.config_dict).to(self.device)
        else:
            assert(False)

        return net
    
    def local_train(self, user_id, epoch):
        # split indexes for train, validation, and test (80, 10, 10)
        dataset = self.train_dataset
        idxs = list(self.user_groups[user_id])
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        idxs_train = idxs[:int(0.9*len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        self.trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.config_dict['batchSz'], shuffle=True, **kwargs)
        # self.validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=config_dict['batchSz'], shuffle=False, **kwargs)
        self.testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.config_dict['batchSz'], shuffle=False, **kwargs)
        train_model = copy.deepcopy(self.global_net)
        for name, w in train_model.named_parameters():
            if not "header" in name:
                w.requires_grad= False
        optimizer = self.get_optimizer(train_model.parameters())
        self.adjust_opt(optimizer, epoch)
        train_model.train()
        epoch_loss = []
        for iter_cnt in range(self.config_dict['local_ep']):
            batch_loss = []
            incorrect, total = 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                log_probs = F.log_softmax(train_model(images))
                loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
                pred = log_probs.data.max(1)[1]
                incorrect += pred.ne(labels.data).cpu().sum()
                total += len(images)
                err = 100.* incorrect / total
                if self.config_dict['verbose'] and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc:{:.2f}'.format(
                        epoch, iter_cnt, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item(), 100 - err))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            self.trainF.write('{},{},{},{:.6f},{}\n'.format(epoch, iter_cnt, user_id, loss.data, err))
            self.trainF.flush()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        correct = 0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = F.log_softmax(train_model(images))
            batch_loss = F.nll_loss(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct/total
        self.valF.write('{},{},{},{}\n'.format(epoch, user_id, loss, 1 - accuracy))
        self.valF.flush()
        
        return train_model.state_dict(), accuracy, sum(epoch_loss) / len(epoch_loss)
    
    def fed_out_train(self, idxs_users, epoch):
        # lfed_out procedure
        self.client_locals = []
        d_out_d_y_locals = []
        self.hyper_iter_locals = []
        self.counter = []
        for user_id in idxs_users:
            # client= Client(self.args, idx, copy.deepcopy(self.net_glob),self.dataset, self.dict_users, self.hyper_param)
            dataset = self.train_dataset
            idxs = list(self.user_groups[user_id])
            kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
            idxs_train = idxs[:int(0.9*len(idxs))]
            # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
            idxs_test = idxs[int(0.9*len(idxs)):]

            self.trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                    batch_size=self.config_dict['batchSz'], shuffle=True, **kwargs)
            # self.validloader = DataLoader(DatasetSplit(dataset, idxs_val),
            #                          batch_size=config_dict['batchSz'], shuffle=False, **kwargs)
            self.testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                    batch_size=self.config_dict['batchSz'], shuffle=False, **kwargs)
            temp_net = copy.deepcopy(self.global_net)
            
            self.client_locals.append(temp_net)
            self.hyper_iter_locals.append(0)
            d_out_d_y, _ = self.grad_d_out_d_y(temp_net, self.trainloader)
            d_out_d_y_locals.append(d_out_d_y)
        p = self.aggregateP(d_out_d_y_locals)
        
        p_locals=[]
        
        if self.config_dict['hvp_method'] == 'global_batch':
            for i in range(self.config_dict['neumann']):
                for cnt, client in enumerate(self.client_locals):
                    p_client = self.hvp_iter(p, self.config_dict['hpr_lr'], cnt)
                    p_locals.append(p_client)
                    self.hyper_iter_locals[cnt] += 1
                p=self.aggregateP(p_locals)
        elif self.config_dict['hvp_method'] == 'local_batch':
            for cnt, client in enumerate(self.client_locals):
                p_client=p.clone()
                for i in range(self.config_dict['neumann']):
                    p_client = self.hvp_iter(p_client, self.config_dict['hpr_lr'], cnt)
                    self.hyper_iter_locals[cnt] += 1
                p_locals.append(p_client)
            p=self.aggregateP(p_locals)
        # elif self.args.hvp_method == 'seperate':
        #     for client in client_locals:
        #         d_out_d_y,_=client.grad_d_out_d_y()
        #         p_client=d_out_d_y.clone()
        #         for _ in range(self.args.neumann):
        #             p_client = client.hvp_iter(p_client, self.args.hlr)
        #         p_locals.append(p_client)
        #     p=FedAvgP(p_locals, self.args)

        else:
            raise NotImplementedError
        
        r = 1+ self.config_dict['neumann']
        
        hg_locals =[]
        for client in self.client_locals:
            hg = self.hyper_grad(client, p.clone())
            hg_locals.append(hg)
        hg_glob=self.aggregateP(hg_locals)
        r += 1
        
        hg_locals =[]
        for client in self.client_locals:
            for _ in range(self.config_dict['outer_tau']):
                h = self.hyper_svrg_update(client, hg_glob)
            hg_locals.append(h)
        hg_glob=self.aggregateP(hg_locals)
        r += 1
        self.assign_hyper_gradient(self.hyper_param, hg_glob)
        self.hyper_optimizer.step()
        
    def aggregateP(self, w):
        # print(f"type of w[0]: {type(w[0])}")
        print(f"length of w[0]: {len(w[0])}")
        w_avg = torch.zeros(w[0].shape[0], dtype=w[0].dtype, device=self.device)
        for k in w:
            w_avg+=k
        w_avg = torch.div(w_avg, len(w)).detach()
        return w_avg
        # # simple gossip
        # new_weight = []
        # for user_idx in range(self.config_dict['num_users']):
        #     w_avg = copy.deepcopy(w[0]) * self.mix_mat[user_idx][0]
        #     for key in w_avg.keys():
        #         for i in range(1, len(w)):
        #             w_avg[key] += w[i][key] * self.mix_mat[user_idx][i]
        #     new_weight.append(w_avg)
        # return new_weight    
    
    def cross_entropy(self, logits, targets):
        return F.cross_entropy(logits, targets)
        
    def cross_entropy_reg(self, logits, targets, param):
        reg = self.beta*sum([torch.norm(k) for k in param])
        return F.cross_entropy(logits, targets)+0.5*reg
    
    # def fedIHGP(self,client_locals):
    #     d_out_d_y_locals=[]
    #     for client in client_locals:
    #         d_out_d_y,_=client.grad_d_out_d_y()
    #         d_out_d_y_locals.append(d_out_d_y)
    #     p=self.aggregateP(d_out_d_y_locals,self.args)
        
    #     p_locals=[]
    #     self.counter = []
    #     if self.config_dict['hvp_method'] == 'global_batch':
    #         for i in range(self.config_dict['neumann']):
    #             for client in client_locals:
    #                 p_client = client.hvp_iter(p, self.config_dict['hlr'])
    #                 p_locals.append(p_client)
    #             p=self.aggregateP(p_locals, self.args)
    #     elif self.config_dict['hvp_method'] == 'local_batch':
    #         for client in client_locals:
    #             p_client=p.clone()
    #             for _ in range(self.config_dict['neumann']):
    #                 p_client = client.hvp_iter(p_client, self.config_dict['hlr'])
    #             p_locals.append(p_client)
    #         p=self.aggregateP(p_locals, self.args)
    #     # elif self.args.hvp_method == 'seperate':
    #     #     for client in client_locals:
    #     #         d_out_d_y,_=client.grad_d_out_d_y()
    #     #         p_client=d_out_d_y.clone()
    #     #         for _ in range(self.args.neumann):
    #     #             p_client = client.hvp_iter(p_client, self.args.hlr)
    #     #         p_locals.append(p_client)
    #     #     p=FedAvgP(p_locals, self.args)

    #     else:
    #         raise NotImplementedError
    #     return p
    
    def gather_flat_grad(self, loss_grad):
    # convert the gradient output from list of tensors to to flat vector 
        return torch.cat([p.contiguous().view(-1) for p in loss_grad if not p is None])
    
    def hvp_iter(self, p, lr, loc_net_cnt):
        if self.hyper_iter_locals[loc_net_cnt] == 0:
            self.d_in_d_y, params, _ = self.grad_d_in_d_y(self.client_locals[loc_net_cnt], self.trainloader)
            self.counter.append(p.clone())
        else:
            params = [k for n,k in self.client_locals[loc_net_cnt].named_parameters() if "header" in n]
        old_counter = self.counter[loc_net_cnt]
        hessian_term = self.gather_flat_grad(
            torch.autograd.grad(self.d_in_d_y, params,
                 grad_outputs=self.counter[loc_net_cnt].view(-1), retain_graph=True)
        )
        self.counter[loc_net_cnt] = old_counter - lr * hessian_term
        p = p+self.counter[loc_net_cnt]
        return p
    
    def grad_d_in_d_y(self, local_net, dataset_loader):
        self.net0 = copy.deepcopy(local_net)
        self.net0.train()
        hyper_param = [k for n,k in self.net0.named_parameters() if not "header" in n]
        params = [k for n,k in self.net0.named_parameters() if "header" in n]
        num_weights = sum(p.numel() for p in params)
        d_in_d_y = torch.zeros(num_weights, device=self.device)
        for batch_idx, (images, labels) in enumerate(dataset_loader):
            images, labels = images.to(
                self.device), labels.to(self.device)
            
            self.net0.zero_grad()
            log_probs = F.softmax(self.net0(images))
            loss = self.loss_func(log_probs, labels, params)
            d_in_d_y += self.gather_flat_grad(grad(loss,
                                         params, create_graph=True))
        d_in_d_y /= (batch_idx+1.)
        return d_in_d_y, params, hyper_param
    
    def grad_d_out_d_y(self, local_net, dataset_loader):
        self.net0 = copy.deepcopy(local_net)
        self.net0.train()
        hyper_param = [k for n,k in self.net0.named_parameters() if not "header" in n]
        params = [k for n,k in self.net0.named_parameters() if "header" in n]
        num_weights = sum(p.numel() for p in params)
        d_out_d_y = torch.zeros(num_weights, device=self.device)
        for batch_idx, (images, labels) in enumerate(dataset_loader):
            images, labels = images.to(
                self.device), labels.to(self.device)
            self.net0.zero_grad()
            log_probs = F.softmax(self.net0(images))
            loss = self.val_loss(log_probs, labels)
            d_out_d_y += self.gather_flat_grad(grad(loss,
                                         params, create_graph=True))
        d_out_d_y /= (batch_idx+1.)
        return d_out_d_y, hyper_param
    
    def grad_d_out_d_x(self, local_net, dataset_loader, hyper_param = None):
        
        self.net0 = copy.deepcopy(local_net)
        if hyper_param == None:
            hyper_param = [k for n,k in self.net0.named_parameters() if not "header" in n]
        self.net0.train()
        num_weights = sum(p.numel() for p in hyper_param)
        d_out_d_x = torch.zeros(num_weights, device=self.device)
        for batch_idx, (images, labels) in enumerate(dataset_loader):
            images, labels = images.to(
                self.device), labels.to(self.device)
            self.net0.zero_grad()
            log_probs = F.softmax(self.net0(images))
            loss = self.val_loss(log_probs, labels)
            d_out_d_x += self.gather_flat_grad(grad(loss,
                                         self.get_trainable_hyper_params(hyper_param), create_graph=True))
        d_out_d_x /= (batch_idx+1.)
        return d_out_d_x
    
    def hyper_grad(self, client, p):
        d_in_d_y, _, hyper_param=self.grad_d_in_d_y(client, self.trainloader)
        indirect_grad= self.gather_flat_grad(
            grad(d_in_d_y,
                self.get_trainable_hyper_params(hyper_param),
                grad_outputs= p.view(-1),
                allow_unused= True)
        )
        # try:
        direct_grad= self.grad_d_out_d_x(client, self.testloader)
        hyper_grad=direct_grad-self.config_dict['hlr']*indirect_grad
        # except:
        #     print(" No direct grad, use only indirect gradient.")
        #     hyper_grad= - indirect_grad
        return hyper_grad
    
    def get_trainable_hyper_params(self, params):
        if isinstance(params,dict):
            return[params[k] for k in params if params[k].requires_grad]
        else:
            return params

    def hyper_svrg_update(self, client, hg):
        try:
            direct_grad = self.grad_d_out_d_x(client, self.testloader)
            direct_grad_0 = self.grad_d_out_d_x(client, self.testloader, hyper_param=self.hyper_param_init)
            h = direct_grad - direct_grad_0 + hg
            print(f"success hypergradient")
        except:
            print(f"hypergradient exception")
            h = hg
        self.assign_hyper_gradient(self.hyper_param, h)
        self.hyper_optimizer.step()
        return -self.gather_flat_hyper_params(self.hyper_param)+self.gather_flat_hyper_params(self.hyper_param_init)
        
    def assign_hyper_gradient(self, params, gradient):
        i = 0
        max_len=gradient.shape[0]
        if isinstance(params, dict):
            for k in params:
                para=params[k]
                if para.requires_grad:
                    num = para.nelement()
                    grad = gradient[i:min(i+num,max_len)].clone()
                    torch.reshape(grad, para.shape)
                    para.grad = grad.view(para.shape)
                    i += num
        else:
            for para in params:
                if para.requires_grad:     
                    num = para.nelement()
                    grad = gradient[i:min(i+num,max_len)].clone()
                    para.grad = grad.view(para.shape)
                    i += num
                    
    def gather_flat_hyper_params(self, params):
        if isinstance(params,dict):
            return torch.cat([params[k].view(-1) for k in params if params[k].requires_grad])
        else:
            return torch.cat([k.view(-1) for k in params if k.requires_grad])

       
