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
from torch.autograd import Function, Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
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

import sys
from IPython.core import ultratb

from update import LocalUpdate, test_inference


sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def get_loaders(config_dict):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    if config_dict['dataset'] == 'mnist':
        trainLoader = torch.utils.data.DataLoader(
            dset.MNIST('/storage/data/panbk/dataset/', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=config_dict['batchSz'], shuffle=True, **kwargs)
        testLoader = torch.utils.data.DataLoader(
            dset.MNIST('/storage/data/panbk/dataset/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=config_dict['batchSz'], shuffle=False, **kwargs)
   
    elif config_dict['dataset'] == 'cifar-10':
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

        trainLoader = DataLoader(
            dset.CIFAR10(root='/storage/data/panbk/dataset/', train=True, download=True,
                        transform=trainTransform),
            batch_size=config_dict['batchSz'], shuffle=True, **kwargs)
        testLoader = DataLoader(
            dset.CIFAR10(root='/storage/data/panbk/dataset/', train=False, download=True,
                        transform=testTransform),
            batch_size=config_dict['batchSz'], shuffle=False, **kwargs)
    else:
        assert(False)

    return trainLoader, testLoader

def get_dataset(config_dict):
    from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
    from sampling import cifar_iid, cifar_noniid
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    # different datasets
    if config_dict['dataset'] == 'mnist':
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
        if config_dict['iid']:
            user_groups = mnist_iid(train_dataset, config_dict['num_users'])
        else:
            if config_dict['unequal']:
                user_groups = mnist_noniid_unequal(train_dataset, config_dict['num_users'])
            else:
                user_groups = mnist_noniid(train_dataset, config_dict['num_users'])
    elif config_dict['dataset'] == 'cifar-10':
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
        if config_dict['iid']:
            user_groups = cifar_iid(train_dataset, config_dict['num_users'])
        else:
            if config_dict['unequal']:
                raise NotImplementedError()
            else:
                user_groups = cifar_noniid(train_dataset, config_dict['num_users'])
    else:
        assert(False)

    return train_dataset, test_dataset, user_groups
    

def get_net(config_dict, dataset):
    if config_dict['model_name'] == 'densenet':
        net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                                bottleneck=True, nClasses=10, dataset=dataset)
    elif config_dict['model_name'] == 'lenet':
        net = models.Lenet(config_dict['nHidden'], 10, config_dict['proj'], dataset=dataset)
    elif config_dict['model_name'] == 'lenet-optnet':
        net = models.LenetOptNet(config_dict['nHidden'], config_dict['nineq'], dataset=dataset)
    elif config_dict['model_name'] == 'fc':
        net = models.FC(config_dict['nHidden'], config_dict['bn'], dataset=dataset)
    elif config_dict['model_name'] == 'optnet':
        net = models.OptNet(config_dict['nHidden'], config_dict['bn'], config_dict['nineq'], dataset=dataset,
                            new_init=config_dict['new_init'])
    elif config_dict['model_name'] == 'optnet-eq':
        net = models.OptNetEq(config_dict['nHidden'], config_dict['neq'], dataset=dataset)
    elif config_dict['model_name'] == 'resoptnet1':
        net = models.ResOptNet1(config_dict['nHidden'], config_dict['bn'], config_dict['nineq'], dataset=dataset)
    elif config_dict['model_name'] == 'resoptnet2':
        net = models.ResOptNet2(config_dict['nHidden'], config_dict['bn'], config_dict['nineq'], dataset=dataset)
    else:
        assert(False)

    return net


def get_optimizer(config_dict, params):
    # model_hparam_dict = config_dict['model_hparam']
    if config_dict['dataset'] == 'mnist':
        if config_dict['model_name'] == 'optnet-eq':
            params = list(params)
            A_param = params.pop(0)
            assert(A_param.size() == (config_dict['neq'], config_dict['nHidden']))
            optimizer = optim.Adam([
                {'params': params, 'lr': 1e-3},
                {'params': [A_param], 'lr': 1e-1}
            ])
        else:
            optimizer = optim.Adam(params)
    elif config_dict['dataset'] in ('cifar-10', 'cifar-100'):
        if config_dict['opt'] == 'sgd':
            optimizer = optim.SGD(params, lr=1e-2, momentum=0.9, weight_decay=config_dict['weightDecay'])
        elif config_dict['opt'] == 'adam':
            optimizer = optim.Adam(params, weight_decay=config_dict['weightDecay'])
    else:
        assert(False)

    return optimizer


    

# TODO: change the code
def train(epoch, net, trainLoader, optimizer, trainF, device):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data, err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data, err))
        trainF.flush()



def test(epoch, net, testLoader, optimizer, testF, device):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

def adjust_opt(config_dict, optimizer, epoch):
    if config_dict['model_name'] == 'densenet':
        if config_dict['opt'] == 'sgd':
            if epoch == 150: update_lr(optimizer, 1e-2)
            elif epoch == 225: update_lr(optimizer, 1e-3)
            else: return

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def main(config_dict):
    # add log files
    logger = SummaryWriter('../logs', flush_secs=1)
    slurm_id = os.getenv("SLURM_JOB_ID")
    if slurm_id:
        logger.add_text("slurm_id", slurm_id)
    # model_hparam_dict = config_dict['model_hparam']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config_dict['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config_dict['seed'])

    
    train_dataset, test_dataset, user_groups = get_dataset(config_dict)
    # trainLoader, testLoader = get_loaders(config_dict)
    global_net = get_net(config_dict, config_dict['dataset'])
    
    
    
    # save the initial global weight of the net 
    net_weight = global_net.state_dict()

    number_param = sum([p.data.nelement() for p in global_net.parameters()])

    print('  + Number of params: {}'.format(number_param))
    
    # move net to device
    global_net = global_net.to(device)

    trainF = open(os.path.join(config_dict['save'], 'train.csv'), 'w')
    valF = open(os.path.join(config_dict['save'], 'val.csv'), 'w')
    testF = open(os.path.join(config_dict['save'], 'test.csv'), 'w')
    
    train_loss, train_accuracy = [], []
    # begin train
    for epoch in range(1, config_dict["nEpoch"] + 1):
        
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch} |\n')
        # change the network to train mode
        global_net.train()
        
        # find the number of users for next epoch
        m = max(int(config_dict['frac'] * config_dict['num_users']), 1)
        idxs_users = np.random.choice(range(config_dict['num_users']), m, replace=False)
        for idx in idxs_users:
            local_model = LocalUpdate(config_dict=config_dict, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, device=device)
            local_net = copy.deepcopy(global_net)
            optimizer = get_optimizer(config_dict, local_net.parameters())
            adjust_opt(config_dict, optimizer, epoch)
            w, loss = local_model.update_weights(
                model=local_net, epoch=epoch, optimizer=optimizer, trainF=trainF)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        
        # TODO: sth to add
        global_weights = average_weights(local_weights)
        global_net.load_state_dict(global_weights)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        
        global_net.eval()
        for c in range(config_dict['num_users']):
            local_model = LocalUpdate(config_dict=config_dict, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, device=device)
            acc, loss = local_model.inference(model=global_net, epoch=epoch, user_id=c, valF=valF)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        
        if (epoch+1) % 5 == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            
        test_acc, test_loss = test_inference(config_dict, global_net, test_dataset, device)
        if slurm_id:
            # run at Slurm job cluster
            logger.add_scalars(f"{slurm_id}/accuracy", {"train_acc": train_accuracy[-1], "test_acc": test_acc}, epoch)
            logger.add_scalars(f"{slurm_id}/loss",  {"train_loss": train_accuracy[-1], "test_loss": test_acc}, epoch)
        else:
            logger.add_scalars("accuracy", {"train_acc": train_accuracy[-1], "test_acc": test_acc}, epoch)
            logger.add_scalars("loss",  {"train_loss": train_accuracy[-1], "test_loss": test_acc}, epoch)
        testF.write('{},{},{}\n'.format(epoch, test_loss, 1 - test_acc))
        testF.flush()
        
        try:
            torch.save(global_net, os.path.join(config_dict['save'], 'latest.pth'))
        except:
            pass
        # os.system('./plot.py "{}" &'.format(config_dict['save']))

    trainF.close()
    valF.close()
    testF.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='lenet-mnist')
    parser.add_argument("--save_dir", type=str, default='lenet_basic_test')
    args = parser.parse_args()
    
    config_path = os.path.join('config', args.config + '.yaml')
    
    # load all yaml config dictionary settings
    with open(config_path, 'r') as f:
        config_dict = yaml.full_load(f)
    slurm_id = os.getenv("SLURM_JOB_ID")
    # if this job works in slurm cluster, then take slurm id as filefold name
    if slurm_id:
        t = slurm_id
    # else take default name
    elif config_dict['save'] is None:
        t = '{}.{}'.format(config_dict['dataset'], config_dict['model_name'])
        if config_dict['model_name'] == 'lenet':
            t += '.nHidden:{}.proj:{}'.format(config_dict['nHidden'], 
                                              config_dict['proj'])
        elif config_dict['model_name'] == 'fc':
            t += '.nHidden:{}'.format(config_dict['nHidden'])
            if config_dict['bn']:
                t += '.bn'
        elif config_dict['model_name'] == 'optnet':
            t += '.nHidden:{}.nineq:{}.eps:{}'.format(config_dict['nHidden'],
                                                      config_dict['nineq'], 
                                                      config_dict['eps'])
            if config_dict['bn']:
                t += '.bn'
        elif config_dict['model_name'] == 'optnet-eq':
            t += '.nHidden:{}.neq:{}'.format(config_dict['nHidden'], 
                                             config_dict['neq'])
        elif config_dict['model_name'] == 'lenet-optnet':
            t += '.nHidden:{}.nineq:{}.eps:{}'.format(config_dict['nHidden'], 
                                                      config_dict['nineq'], 
                                                      config_dict['eps'])
    setproctitle.setproctitle('bamos.'+t)
    config_dict['save'] = os.path.join("experiment", args.save_dir, t)
    if not os.path.exists(os.path.join("experiment", args.save_dir)):
        os.makedirs(os.path.join("experiment", args.save_dir), exist_ok=True)
    if os.path.exists(config_dict['save']):
        shutil.rmtree(config_dict['save'])
    os.makedirs(config_dict['save'], exist_ok=True)
    try:
        shutil.copy(config_path, config_dict['save'])
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
        print("Unexpected error:", sys.exc_info())
    new_config_dict = {'config_dict': config_dict}
    main(config_dict)
    


