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
        
    
class FedDistManager(DistManager):
    '''
    This class manage the training and testing procedure for federated learning with central server
    '''
    def __init__(self, save_path) -> None:
        super().__init__(save_path)
        
    def run_exp(self, config_dict) -> None:
        super().run_exp(config_dict)
        
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
                log_probs = train_model(images)
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
                        100. * batch_idx / len(self.trainloader), loss.item(), 1 - err))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            self.trainF.write('{},{},{},{:.6f},{}\n'.format(epoch, iter_cnt, user_id, loss.data, err))
            self.trainF.flush()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = train_model(images)
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
    
    def aggregate(self, w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg
    
    def global_test(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

        criterion = nn.NLLLoss().to(self.device)
        testloader = DataLoader(self.test_dataset, batch_size=self.config_dict['batchSz'],
                                shuffle=False, **kwargs)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss / total
    
    def get_dataset(self):
        from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
        from sampling import cifar_iid, cifar_noniid
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        
        # different datasets
        if self.config_dict['dataset'] == 'mnist':
            # generate mnist dataset for train and test
            train_dataset = dset.MNIST('/storage/data/panbk/dataset/', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
            test_dataset= dset.MNIST('/storage/data/panbk/dataset/', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
            # TODO: add iid to the code
            if self.config_dict['iid']:
                user_groups = mnist_iid(train_dataset, self.config_dict['num_users'])
            else:
                if self.config_dict['unequal']:
                    user_groups = mnist_noniid_unequal(train_dataset, self.config_dict['num_users'])
                else:
                    user_groups = mnist_noniid(train_dataset, self.config_dict['num_users'])
        elif self.config_dict['dataset'] == 'cifar-10':
            normMean = [0.49139968, 0.48215827, 0.44653124]
            normStd = [0.24703233, 0.24348505, 0.26158768]
            normTransform = transforms.Normalize(normMean, normStd)

            trainTransform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normTransform
            ])
            testTransform = transforms.Compose([
                transforms.ToTensor(),
                normTransform
            ])

            train_dataset = dset.CIFAR10(root='/storage/data/panbk/dataset/', train=True, download=True,
                            transform=trainTransform)
            test_dataset = dset.CIFAR10(root='/storage/data/panbk/dataset/', train=False, download=True,
                            transform=testTransform)
            if self.config_dict['iid']:
                user_groups = cifar_iid(train_dataset, self.config_dict['num_users'])
            else:
                if self.config_dict['unequal']:
                    raise NotImplementedError()
                else:
                    user_groups = cifar_noniid(train_dataset, self.config_dict['num_users'])
        else:
            assert(False)

        return train_dataset, test_dataset, user_groups

    def get_net(self):
        dataset = self.config_dict['dataset']
        if self.config_dict['model_name'] == 'densenet':
            net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                                    bottleneck=True, nClasses=10, dataset=dataset)
        elif self.config_dict['model_name'] == 'lenet':
            net = models.Lenet(self.config_dict['nHidden'], 10, self.config_dict['proj'], dataset=dataset)
        elif self.config_dict['model_name'] == 'lenet-optnet':
            net = models.LenetOptNet(self.config_dict['nHidden'], self.config_dict['nineq'], dataset=dataset)
        elif self.config_dict['model_name'] == 'fc':
            net = models.FC(self.config_dict['nHidden'], self.config_dict['bn'], dataset=dataset)
        elif self.config_dict['model_name'] == 'optnet':
            net = models.OptNet(self.config_dict['nHidden'], self.self.config_dict['bn'], self.config_dict['nineq'], 
                                dataset=dataset, new_init=self.config_dict['new_init'])
        elif self.config_dict['model_name'] == 'optnet-eq':
            net = models.OptNetEq(self.config_dict['nHidden'], self.config_dict['neq'], dataset=dataset)
        elif self.config_dict['model_name'] == 'resoptnet1':
            net = models.ResOptNet1(self.config_dict['nHidden'], self.config_dict['bn'], self.config_dict['nineq'], 
                                    dataset=dataset)
        elif self.config_dict['model_name'] == 'resoptnet2':
            net = models.ResOptNet2(self.config_dict['nHidden'], self.config_dict['bn'], self.config_dict['nineq'], 
                                    dataset=dataset)
        else:
            assert(False)

        return net
    
    def get_optimizer(self, params):
        # model_hparam_dict = config_dict['model_hparam']
        if self.config_dict['dataset'] == 'mnist':
            if self.config_dict['model_name'] == 'optnet-eq':
                params = list(params)
                A_param = params.pop(0)
                assert(A_param.size() == (self.config_dict['neq'], self.config_dict['nHidden']))
                optimizer = optim.Adam([
                    {'params': params, 'lr': 1e-3},
                    {'params': [A_param], 'lr': 1e-1}
                ])
            else:
                optimizer = optim.Adam(params)
        elif self.config_dict['dataset'] in ('cifar-10', 'cifar-100'):
            if self.config_dict['opt'] == 'sgd':
                optimizer = optim.SGD(params, lr=1e-4, momentum=0.9, weight_decay=self.config_dict['weightDecay'])
            elif self.config_dict['opt'] == 'adam':
                optimizer = optim.Adam(params, weight_decay=self.config_dict['weightDecay'])
        else:
            assert(False)
        print(f"optimizer: {optimizer}")
        return optimizer
    
    def adjust_opt(self, optimizer, epoch):
        if self.config_dict['model_name'] == 'densenet':
            if self.config_dict['opt'] == 'sgd':
                if epoch == 150: self.update_lr(optimizer, 1e-2)
                elif epoch == 225: self.update_lr(optimizer, 1e-3)
                else: return
                
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class DecenDistManager(DistManager):
    '''
    This class manage the training and testing procedure for decentralized federated learning
    '''
    def __init__(self, save_path) -> None:
        super().__init__(save_path)
        
    def run_exp(self, config_dict) -> None:
        super().run_exp(config_dict)
        
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
        
        # print the net information for debug
        number_param = sum([p.data.nelement() for p in self.global_net.parameters()])
        print('  + Number of params: {}'.format(number_param))
        
        train_loss, train_accuracy = [], []
        self.local_weights = []
        # initialize all the local weight
        for idx in range(config_dict['num_users']):
            self.local_weights.append(copy.deepcopy(self.global_net))
        for epoch in range(1, config_dict["nEpoch"] + 1):
            self.local_weights, local_losses, list_acc = [], [], []
            print(f'\n | Global Training Round : {epoch} |\n')
            # change the network to train mode
            self.global_net.train()
            # find the number of users for next epoch
            # m = max(int(config_dict['frac'] * config_dict['num_users']), 1)
            # idxs_users = np.random.choice(range(config_dict['num_users']), m, replace=False)
            for idx in range(config_dict['num_users']):
                w, acc, loss = self.local_train(idxs=self.user_groups[idx], user_id=idx, epoch=epoch)
                list_acc.append(acc)
                self.local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            train_accuracy.append(sum(list_acc)/len(list_acc))
            # aggregate the local models and calculate the local training loss
            self.graph_update()
            new_weights = self.aggregate(self.local_weights)
            
            test_acc, test_loss = 0, 0
            for idx in range(self.config_dict['num_users']):
                self.global_net.load_state_dict(new_weights[idx])
                local_test_acc, local_test_loss = self.global_test(self.global_net)
                test_acc += local_test_acc / self.config_dict['num_users']
                test_loss += local_test_loss / self.config_dict['num_users']
            
            
            if (epoch+1) % 5 == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
                
            
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
        
    def local_train(self, idxs, user_id, epoch):
        # split indexes for train, validation, and test (80, 10, 10)
        dataset = self.train_dataset
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
                log_probs = train_model(images)
                loss = F.nll_loss(log_probs, labels)
                loss.backward()
                print(f"loss.item(): {loss.item()}")
                optimizer.step()
                pred = log_probs.data.max(1)[1]
                incorrect += pred.ne(labels.data).cpu().sum()
                total += len(images)
                err = 100.* incorrect / total
                if self.config_dict['verbose'] and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc:{:.2f}'.format(
                        epoch, iter_cnt, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item(), 1 - err))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            self.trainF.write('{},{},{},{:.6f},{}\n'.format(epoch, iter_cnt, user_id, loss.data, err))
            self.trainF.flush()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = train_model(images)
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
    
    def graph_update(self):
        num_users = self.config_dict['num_users']
        self.mix_mat = np.ones((num_users, num_users)) / num_users
    
    def aggregate(self, w):
        # simple gossip
        new_weight = []
        for user_idx in range(self.config_dict['num_users']):
            w_avg = copy.deepcopy(w[0]) * self.mix_mat[user_idx][0]
            for key in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key] * self.mix_mat[user_idx][i]
            new_weight.append(w_avg)
        return new_weight
    
    def global_test(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

        criterion = nn.NLLLoss().to(self.device)
        testloader = DataLoader(self.test_dataset, batch_size=self.config_dict['batchSz'],
                                shuffle=False, **kwargs)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss
    
    def get_dataset(self):
        from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
        from sampling import cifar_iid, cifar_noniid
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        
        # different datasets
        if self.config_dict['dataset'] == 'mnist':
            # generate mnist dataset for train and test
            train_dataset = dset.MNIST('/storage/data/panbk/dataset/', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
            test_dataset= dset.MNIST('/storage/data/panbk/dataset/', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
            # TODO: add iid to the code
            if self.config_dict['iid']:
                user_groups = mnist_iid(train_dataset, self.config_dict['num_users'])
            else:
                if self.config_dict['unequal']:
                    user_groups = mnist_noniid_unequal(train_dataset, self.config_dict['num_users'])
                else:
                    user_groups = mnist_noniid(train_dataset, self.config_dict['num_users'])
        elif self.config_dict['dataset'] == 'cifar-10':
            normMean = [0.49139968, 0.48215827, 0.44653124]
            normStd = [0.24703233, 0.24348505, 0.26158768]
            normTransform = transforms.Normalize(normMean, normStd)

            trainTransform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normTransform
            ])
            testTransform = transforms.Compose([
                transforms.ToTensor(),
                normTransform
            ])

            train_dataset = dset.CIFAR10(root='/storage/data/panbk/dataset/', train=True, download=True,
                            transform=trainTransform)
            test_dataset = dset.CIFAR10(root='/storage/data/panbk/dataset/', train=False, download=True,
                            transform=testTransform)
            if self.config_dict['iid']:
                user_groups = cifar_iid(train_dataset, self.config_dict['num_users'])
            else:
                if self.config_dict['unequal']:
                    raise NotImplementedError()
                else:
                    user_groups = cifar_noniid(train_dataset, self.config_dict['num_users'])
        else:
            assert(False)

        return train_dataset, test_dataset, user_groups

    def get_net(self):
        dataset = self.config_dict['dataset']
        if self.config_dict['model_name'] == 'densenet':
            net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                                    bottleneck=True, nClasses=10, dataset=dataset)
        elif self.config_dict['model_name'] == 'lenet':
            net = models.Lenet(self.config_dict['nHidden'], 10, self.config_dict['proj'], dataset=dataset)
        elif self.config_dict['model_name'] == 'lenet-optnet':
            net = models.LenetOptNet(self.config_dict['nHidden'], self.config_dict['nineq'], dataset=dataset)
        elif self.config_dict['model_name'] == 'fc':
            net = models.FC(self.config_dict['nHidden'], self.config_dict['bn'], dataset=dataset)
        elif self.config_dict['model_name'] == 'optnet':
            net = models.OptNet(self.config_dict['nHidden'], self.self.config_dict['bn'], self.config_dict['nineq'], 
                                dataset=dataset, new_init=self.config_dict['new_init'])
        elif self.config_dict['model_name'] == 'optnet-eq':
            net = models.OptNetEq(self.config_dict['nHidden'], self.config_dict['neq'], dataset=dataset)
        elif self.config_dict['model_name'] == 'resoptnet1':
            net = models.ResOptNet1(self.config_dict['nHidden'], self.config_dict['bn'], self.config_dict['nineq'], 
                                    dataset=dataset)
        elif self.config_dict['model_name'] == 'resoptnet2':
            net = models.ResOptNet2(self.config_dict['nHidden'], self.config_dict['bn'], self.config_dict['nineq'], 
                                    dataset=dataset)
        else:
            assert(False)

        return net
    
    def get_optimizer(self, params):
        # model_hparam_dict = config_dict['model_hparam']
        if self.config_dict['dataset'] == 'mnist':
            if self.config_dict['model_name'] == 'optnet-eq':
                params = list(params)
                A_param = params.pop(0)
                assert(A_param.size() == (self.config_dict['neq'], self.config_dict['nHidden']))
                optimizer = optim.Adam([
                    {'params': params, 'lr': 1e-3},
                    {'params': [A_param], 'lr': 1e-1}
                ])
            else:
                print("test optimizer")
                optimizer = optim.Adam(params, lr= 1e-4)
        elif self.config_dict['dataset'] in ('cifar-10', 'cifar-100'):
            if self.config_dict['opt'] == 'sgd':
                optimizer = optim.SGD(params, lr=1e-2, momentum=0.9, weight_decay=self.config_dict['weightDecay'])
            elif self.config_dict['opt'] == 'adam':
                optimizer = optim.Adam(params, weight_decay=self.config_dict['weightDecay'])
        else:
            assert(False)

        return optimizer
    
    def adjust_opt(self, optimizer, epoch):
        if self.config_dict['model_name'] == 'densenet':
            if self.config_dict['opt'] == 'sgd':
                if epoch == 150: self.update_lr(optimizer, 1e-2)
                elif epoch == 225: self.update_lr(optimizer, 1e-3)
                else: return
                
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    
