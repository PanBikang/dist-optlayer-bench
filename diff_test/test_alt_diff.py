import json

import argparse
import time
try: import setGPU
except ImportError: pass

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import copy
import math
import numpy as np
import shutil
import random
import setproctitle

# import models

import torch_geometric
from qpth.qp import QPFunction
from newlayer import diff

class nondiff(nn.Module):
    def __init__(self):
        super().__init__()

        # self.nFeatures = nFeatures
        # self.nHidden = nHidden
        # self.bn = bn
        # self.nCls = nCls
        # self.nineq = nineq
        # self.neq = neq
        # self.eps = eps

        # if bn:
        #     self.bn1 = nn.BatchNorm1d(nHidden)
        #     self.bn2 = nn.BatchNorm1d(nCls)

        # self.fc1 = nn.Linear(nFeatures, nHidden)
        # self.fc2 = nn.Linear(nHidden, nCls)

        # self.M = Variable(torch.tril(torch.ones(nCls, nCls)).cuda())
        # self.L = Parameter(torch.tril(torch.rand(nCls, nCls).cuda()))
        # self.G = Parameter(torch.Tensor(nineq,nCls).uniform_(-1,1).cuda())
        # self.z0 = Parameter(torch.zeros(nCls).cuda())
        # self.s0 = Parameter(torch.ones(nineq).cuda())

    def forward(self, x):
        # nBatch = x.size(0)

        # # FC-ReLU-(BN)-FC-ReLU-(BN)-QP-Softmax
        # x = x.view(nBatch, -1)
        # x = F.relu(self.fc1(x))
        # if self.bn:
        #     x = self.bn1(x)
        # x = F.relu(self.fc2(x))
        # if self.bn:
        #     x = self.bn2(x)

        # L = self.M*self.L
        # Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nCls)).cuda()
        # h = self.G.mv(self.z0)+self.s0
        x_2 = -1 * torch.square(x)
        p = torch.cat((x_2, torch.tensor([0])), 0).unsqueeze(0)
        E = Variable(torch.Tensor([[0], [1]])).unsqueeze(0)
        e = Variable(torch.Tensor([1])).unsqueeze(0)
        Q = torch.Tensor([[1, 0], [0, 0]]).unsqueeze(0)
        G = torch.Tensor([[-1], [0]]).unsqueeze(0)
        h = x.unsqueeze(0)
        
        x = diff(verbose=False)(Q, p, G, h, E, e)

        return x

if __name__ == "__main__":
    model = nondiff().to('cpu')
    x = torch.tensor([-1.0001], requires_grad=True).to('cpu')
    y = model(x)
    y.backward()
    print(f"x.grad at [-1.0001]: {x.grad}")
    
    x = torch.tensor([-1.0], requires_grad=True)
    y = model(x)
    y.backward()
    print(f"x.grad at [-1.0]: {x.grad}")
    
    x = torch.tensor([-0.9999], requires_grad=True)
    y = model(x)
    y.backward()
    print(f"x.grad at [-0.9999]: {x.grad}")
    
    x = torch.tensor([-0.5], requires_grad=True)
    y = model(x)
    y.backward()
    print(f"x.grad at [-0.5]: {x.grad}")
    
    x = torch.tensor([0.0], requires_grad=True)
    y = model(x)
    y.backward()
    print(f"x.grad at [0.0]: {x.grad}")
    
    x = torch.tensor([0.0001], requires_grad=True)
    y = model(x)
    y.backward()
    print(f"x.grad at [0.0001]: {x.grad}")
    
    x = torch.tensor([-0.0001], requires_grad=True)
    y = model(x)
    y.backward()
    print(f"x.grad at [-0.0001]: {x.grad}")