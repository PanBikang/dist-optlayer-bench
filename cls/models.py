import torch

import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from easyfl.models.model import BaseModel

from qpth.qp import QPFunction, QPSolvers
import copy
import time

class Lenet(nn.Module):
    def __init__(self, nHidden, nCls=10, proj='softmax', dataset='mnist'):
        super(Lenet, self).__init__()
        if dataset == "mnist":
            input_channel = 1
            output_width = 4
        # cifar-10
        else:
            input_channel = 3
            output_width = 5
        
        self.conv1 = nn.Conv2d(input_channel, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(50*output_width*output_width, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)

        self.proj = proj
        self.nCls = nCls

        if proj == 'simproj':
            self.Q = Variable(0.5*torch.eye(nCls).double().cuda())
            self.G = Variable(-torch.eye(nCls).double().cuda())
            self.h = Variable(-1e-5*torch.ones(nCls).double().cuda())
            self.A = Variable((torch.ones(1, nCls)).double().cuda())
            self.b = Variable(torch.Tensor([1.]).double().cuda())
            def projF(x):
                nBatch = x.size(0)
                Q = self.Q.unsqueeze(0).expand(nBatch, nCls, nCls)
                G = self.G.unsqueeze(0).expand(nBatch, nCls, nCls)
                h = self.h.unsqueeze(0).expand(nBatch, nCls)
                A = self.A.unsqueeze(0).expand(nBatch, 1, nCls)
                b = self.b.unsqueeze(0).expand(nBatch, 1)
                x = QPFunction()(Q, -x.double(), G, h, A, b).float()
                x = x.log()
                return x
            self.projF = projF
        else:
            self.projF = F.log_softmax

    def forward(self, x):
        nBatch = x.size(0)

        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.projF(x)

class LenetOptNet(nn.Module):
    def __init__(self, nHidden=50, nineq=200, neq=0, eps=1e-4, dataset='mnist'):
        super(LenetOptNet, self).__init__()
        if dataset == "mnist":
            input_channel = 1
            output_width = 4
        # cifar-10
        else:
            input_channel = 3
            output_width = 5
        self.conv1 = nn.Conv2d(input_channel, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        self.qp_o = nn.Linear(50*output_width*output_width, nHidden)
        self.qp_z0 = nn.Linear(50*output_width*output_width, nHidden)
        self.qp_s0 = nn.Linear(50*output_width*output_width, nineq)

        assert(neq==0)
        self.M = Variable(torch.tril(torch.ones(nHidden, nHidden)).cuda())
        self.L = Parameter(torch.tril(torch.rand(nHidden, nHidden).cuda()))
        self.G = Parameter(torch.Tensor(nineq,nHidden).uniform_(-1,1).cuda())
        # self.z0 = Parameter(torch.zeros(nHidden).cuda())
        # self.s0 = Parameter(torch.ones(nineq).cuda())

        self.nHidden = nHidden
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

    def forward(self, x):
        nBatch = x.size(0)

        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(nBatch, -1)

        L = self.M*self.L
        Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nHidden)).cuda()
        Q = Q.unsqueeze(0).expand(nBatch, self.nHidden, self.nHidden)
        G = self.G.unsqueeze(0).expand(nBatch, self.nineq, self.nHidden)
        z0 = self.qp_z0(x)
        s0 = self.qp_s0(x)
        h = z0.mm(self.G.t())+s0
        e = Variable(torch.Tensor())
        inputs = self.qp_o(x)
        x = QPFunction()(Q, inputs, G, h, e, e)
        x = x[:,:10]
        print(x)
        return F.log_softmax(x)

class FC(nn.Module):
    def __init__(self, nHidden, bn, dataset):
        super().__init__()
        self.bn = bn
        if dataset == "mnist":
            input_number = 784
        # cifar-10
        else:
            input_number = 3 * 32 * 32
        self.fc1 = nn.Linear(input_number, nHidden)
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(nHidden, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        nBatch = x.size(0)

        # FC-ReLU-(BN)-FC-ReLU-(BN)-FC-Softmax
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)
        x = self.fc3(x)
        return F.log_softmax(x)

class OptNet(nn.Module):
    def __init__(self, nHidden, bn, nineq=200, neq=0, eps=1e-4, dataset='mnist', new_init=False):
        super().__init__()
        if dataset == 'mnist':
            self.nFeatures = 28 * 28
            self.nCls = 10
        # cifar-10: 3 channel 32*32
        else:
            self.nFeatures = 3 * 32 * 32
            self.nCls = 10
        self.nHidden = nHidden
        self.bn = bn


        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(self.nCls)

        self.fc1 = nn.Linear(self.nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, self.nCls)

        # self.qp_z0 = nn.Linear(nCls, nCls)
        # self.qp_s0 = nn.Linear(nCls, nineq)

        assert(neq==0)
        self.M = Variable(torch.tril(torch.ones(self.nCls, self.nCls)).cuda())
        if new_init:
            self.L = Parameter(torch.eye(self.nCls).cuda())
        else:
            self.L = Parameter(torch.tril(torch.rand(self.nCls, self.nCls).cuda()))
        self.G = Parameter(torch.Tensor(nineq,self.nCls).uniform_(-1,1).cuda())
        self.z0 = Parameter(torch.zeros(self.nCls).cuda())
        self.s0 = Parameter(torch.ones(nineq).cuda())

        self.nineq = nineq
        self.neq = neq
        self.eps = eps

    def forward(self, x):
        nBatch = x.size(0)

        # FC-ReLU-(BN)-FC-ReLU-(BN)-QP-Softmax
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)

        L = self.M*self.L
        Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nCls)).cuda()
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
        G = self.G.unsqueeze(0).expand(nBatch, self.nineq, self.nCls)
        # z0 = self.qp_z0(x)
        # s0 = self.qp_s0(x)
        z0 = self.z0.unsqueeze(0).expand(nBatch, self.nCls)
        s0 = self.s0.unsqueeze(0).expand(nBatch, self.nineq)
        h = z0.mm(self.G.t())+s0
        e = Variable(torch.Tensor())
        inputs = x
        x = QPFunction(verbose=-1)(
            Q.double(), inputs.double(), G.double(), h.double(), e, e)
        x = x.float()
        # x = x[:,:10].float()

        return F.log_softmax(x)
    
class ResOptNet1(nn.Module):
    def __init__(self, nHidden, bn, nineq=200, neq=0, eps=1e-4, dataset='mnist'):
        super().__init__()
        if dataset == 'mnist':
            self.nFeatures = 28 * 28
            self.nCls = 10
        # cifar-10: 3 channel 32*32
        else:
            self.nFeatures = 3 * 32 * 32
            self.nCls = 10
        self.nHidden = nHidden
        self.bn = bn


        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(self.nCls)

        self.fc1 = nn.Linear(self.nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, self.nCls)

        # self.qp_z0 = nn.Linear(nCls, nCls)
        # self.qp_s0 = nn.Linear(nCls, nineq)

        assert(neq==0)
        self.M = Variable(torch.tril(torch.ones(self.nCls, self.nCls)).cuda())
        self.L = Parameter(torch.tril(torch.rand(self.nCls, self.nCls).cuda()))
        self.G = Parameter(torch.Tensor(nineq,self.nCls).uniform_(-1,1).cuda())
        self.z0 = Parameter(torch.zeros(self.nCls).cuda())
        self.s0 = Parameter(torch.ones(nineq).cuda())

        self.nineq = nineq
        self.neq = neq
        self.eps = eps

    def forward(self, x):
        nBatch = x.size(0)

        # FC-ReLU-(BN)-FC-ReLU-(BN)-QP-Res-Softmax
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)

        L = self.M*self.L
        Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nCls)).cuda()
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
        G = self.G.unsqueeze(0).expand(nBatch, self.nineq, self.nCls)
        # z0 = self.qp_z0(x)
        # s0 = self.qp_s0(x)
        z0 = self.z0.unsqueeze(0).expand(nBatch, self.nCls)
        s0 = self.s0.unsqueeze(0).expand(nBatch, self.nineq)
        h = z0.mm(self.G.t())+s0
        e = Variable(torch.Tensor())
        inputs = x
        x = QPFunction(verbose=-1)(
            Q.double(), inputs.double(), G.double(), h.double(), e, e)
        x = x.float() + inputs.float()
        # x = x[:,:10].float()

        return F.log_softmax(x)
    


class OptNetEq(nn.Module):
    def __init__(self, nHidden, neq, Qpenalty=0.1, eps=1e-4, dataset='mnist'):
        super().__init__()
        if dataset == 'mnist':
            self.nFeatures = 28 * 28
            self.nCls = 10
        # cifar-10: 3 channel 32*32
        else:
            self.nFeatures = 3 * 32 * 32
            self.nCls = 10

        self.nHidden = nHidden


        self.fc1 = nn.Linear(self.nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, self.nCls)

        self.Q = Variable(Qpenalty*torch.eye(nHidden).double().cuda())
        self.G = Variable(-torch.eye(nHidden).double().cuda())
        self.h = Variable(torch.zeros(nHidden).double().cuda())
        self.A = Parameter(torch.rand(neq,nHidden).double().cuda())
        self.b = Variable(torch.ones(self.A.size(0)).double().cuda())

        self.neq = neq

    def forward(self, x):
        nBatch = x.size(0)

        # FC-ReLU-QP-FC-Softmax
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))

        Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1))
        p = -x.view(nBatch,-1)
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        A = self.A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1))
        b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0))

        x = QPFunction(verbose=False)(Q, p.double(), G, h, A, b).float()
        x = self.fc2(x)

        return F.log_softmax(x)

# class Ours(nn.Module):
#     def __init__(self, nFeatures, nHidden, nCls, neq, nineq, Qpenalty=0.1, eps=1e-4):
#         super().__init__()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.nFeatures = nFeatures
#         self.nHidden = nHidden
#         self.nCls = nCls

#         self.fc1 = nn.Linear(nFeatures, nHidden)
#         self.fc2 = nn.Linear(nHidden, nCls)

#         self.Q = Qpenalty*torch.eye(nHidden).double().to(device)
#         # self.G = -torch.eye(nHidden).double().to(device)
#         # self.h = torch.zeros(nHidden).double().to(device)
#         self.G = torch.rand(nineq, nHidden).double().to(device)
#         self.h = torch.rand(self.G  .size(0)).double().to(device)
#         self.A = torch.rand(neq,nHidden).double().to(device)
#         self.b = torch.ones(self.A.size(0)).double().to(device)

#         self.neq = neq
#         self.nineq = nineq

#     def forward(self, x):
#         nBatch = x.size(0)

#         # FC-ReLU-QP-FC-Softmax
#         x = x.view(nBatch, -1)

#         x = F.relu(self.fc1(x))

#         Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1)).double()
#         #p = -x.view(nBatch,-1)
#         G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1)).double()
#         h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0)).double()
#         A = self.A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1)).double()
#         b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0)).double()
        
        

#         x = diff(verbose=False)(Q, x.double(), G, h, A, b).float()

#         x = self.fc2(x)

#         return F.log_softmax(x)
    

# def decode(X_):
#     a = []
#     X = X_
#     # X = X_.cpu().numpy()
#     for i in range(len(X)):
#         a.append(X[i])
#     return a


# def relu(s):
#     ss = s
#     for i in range(len(s)):
#         if s[i] < 0:
#             ss[i] = 0
#     return ss


# def sgn(s):
#     ss = torch.zeros(len(s))
#     for i in range(len(s)):
#         if s[i]<=0:
#             ss[i] = 0
#         else:
#             ss[i] = 1
#     return ss


# def diff(eps=1e-3, verbose=0):
#     class Newlayer(Function):
#         @staticmethod
#         def forward(ctx, Q_, p_, G_, h_, A_, b_):
#             n = p_.shape[1]
#             m = b_.shape[1]
#             d = h_.shape[1]
#             #print(n, m, d)
#             Q = decode(Q_)
#             p = p_
#             G = G_
#             h = h_
#             A = A_
#             b = b_
#             # p = p_.cpu().numpy()
#             # G = G_.cpu().numpy()
#             # h = h_.cpu().numpy()
#             # A = A_.cpu().numpy()
#             # b = b_.cpu().numpy()
#             # Define and solve the CVXPY problem.
#             optimal = []
#             gradient = []

#             for i in range(len(Q)):
#                 begin = time.time()
#                 Qi, pi, Ai, bi, Gi, hi = Q[i], p[i], A[i], b[i], G[i], h[i]
#                 xk = torch.zeros(n)
#                 sk = torch.zeros(d)

#                 lamb = torch.zeros(m)
#                 nu = torch.zeros(d)

#                 dxk = torch.zeros((n, n))
#                 dsk = torch.zeros((d, n))
#                 dlamb = torch.zeros((m, n))
#                 dnu = torch.zeros((d, n))

#                 res = [1000, -100]
#                 #thres = 1e-4
#                 rho = 1
#                 R = - torch.linalg.inv(Qi + rho * Ai.T @ Ai + rho * Gi.T @ Gi)
#                 iters = 0

#                 #for _ in range(iters):
#                 while abs((res[-1] - res[-2]) / res[-2]) > eps:
#                     iters += 1
#                     xk = R @ (pi + Ai.T @ lamb + Gi.T @ nu - rho * Ai.T @ bi + rho * Gi.T @ (sk - hi))
#                     dxk = R @ (torch.eye(n) + Ai.T @ dlamb + Gi.T @ dnu + rho * Gi.T @ dsk)
#                     sk = relu(- (1 / rho) * nu - (Gi @ xk - hi))

#                     dsk = (-1 / rho) * sgn(sk).reshape(d, 1) @ torch.ones((1, n)) * (dnu + rho * Gi @ dxk)

#                     lamb = lamb + rho * (Ai @ xk - bi)
#                     #lamb_all.append(lamb)
#                     dlamb = dlamb + rho * (Ai @ dxk)

#                     nu = nu + rho * (Gi @ xk + sk - hi)
#                     #nu_all.append(nu)
#                     dnu = dnu + rho * (Gi @ dxk + dsk)
#                     #dx_norm.append(np.sum(dxk))
#                     res.append(0.5 * (xk.T @ Qi @ xk) + pi.T @ xk)

#                 end = time.time()
#                 optimal.append(xk)
#                 #print('iterations:', iters)
#                 gradient.append(dxk)


#             ctx.save_for_backward(torch.tensor(torch.array(gradient)))
#             return torch.tensor((torch.array(optimal)))

#         @staticmethod
#         def backward(ctx, grad_output):
#             # only call parameters q
#             grad = ctx.saved_tensors

#             grad_all = torch.zeros((len(grad[0]),200))
#             for i in range(len(grad[0])):
#                 grad_all[i] = grad_output[i] @ grad[0][i]
#             #print(grad_all.shape)
#             return (None, grad_all, None, None, None, None)

#     return Newlayer.apply
