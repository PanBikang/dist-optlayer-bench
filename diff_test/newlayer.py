
import torch
import cvxpy as cp
from torch.autograd import Function
import math
import numpy as np
from cvxpy.atoms.affine.wraps import psd_wrap
import time

def decode(X_):
    a = []
    X = X_.cpu().numpy()
    # X = X_.cpu().numpy()
    for i in range(len(X)):
        a.append(X[i])
    return a


def relu(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = 0
    return ss


def sgn(s):
    ss = np.zeros(len(s))
    for i in range(len(s)):
        if s[i]<=0:
            ss[i] = 0
        else:
            ss[i] = 1
    return ss

def expandParam(X, nBatch, nDim):
    if X.ndimension() in (0, nDim) or X.nelement() == 0:
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")

def diff(eps=1e-3, verbose=0):
    class Newlayer(Function):
        @staticmethod
        def forward(ctx, Q_, p_, G_, h_, A_, b_):
            
            n = p_.shape[1]
            m = b_.shape[1]
            d = h_.shape[1]
            #print(n, m, d)
            Q = decode(Q_)
            # p = p_.numpy()
            # G = G_.numpy()
            # h = h_.numpy()
            # A = A_.numpy()
            # b = b_.numpy()
            p = p_.cpu().numpy()
            G = G_.cpu().numpy()
            h = h_.cpu().numpy()
            A = A_.cpu().numpy()
            b = b_.cpu().numpy()
            # Define and solve the CVXPY problem.
            optimal = []
            gradient = []
            
            # print('A: ', A)
            # print('A_: ', A_)
            

            for i in range(len(Q)):
                begin = time.time()
                Qi, pi, Ai, bi, Gi, hi = Q[i], p[i], A[i], b[i], G[i], h[i]
                xk = np.zeros(n)
                sk = np.zeros(d)

                lamb = np.zeros(m)
                nu = np.zeros(d)

                dxk = np.zeros((n, n))
                dsk = np.zeros((d, n))
                dlamb = np.zeros((m, n))
                dnu = np.zeros((d, n))

                res = [1000, -100]
                #thres = 1e-4
                rho = 1
                R = - np.linalg.inv(Qi + rho * Ai.T @ Ai + rho * Gi.T @ Gi)
                iters = 0
                
                #for _ in range(iters):
                while abs((res[-1] - res[-2]) / res[-2]) > eps:
                    
                    iters += 1
                    xk = R @ (pi + Ai.T @ lamb + Gi.T @ nu - rho * Ai.T @ bi + rho * Gi.T @ (sk - hi))
                    dxk = R @ (np.eye(n) + Ai.T @ dlamb + Gi.T @ dnu + rho * Gi.T @ dsk)
                    sk = relu(- (1 / rho) * nu - (Gi @ xk - hi))

                    dsk = (-1 / rho) * sgn(sk).reshape(d, 1) @ np.ones((1, n)) * (dnu + rho * Gi @ dxk)

                    lamb = lamb + rho * (Ai @ xk - bi)
                    #lamb_all.append(lamb)
                    dlamb = dlamb + rho * (Ai @ dxk)

                    nu = nu + rho * (Gi @ xk + sk - hi)
                    #nu_all.append(nu)
                    dnu = dnu + rho * (Gi @ dxk + dsk)
                    #dx_norm.append(np.sum(dxk))
                    res.append(0.5 * (xk.T @ Qi @ xk) + pi.T @ xk)
                    if (res[-2] == 0):
                        break
                end = time.time()
                optimal.append(xk)
                #print('iterations:', iters)
                gradient.append(dxk)

            # if torch.cuda.is_available():
            #     device = torch.device('cuda')
            # else: device = torch.device('cpu')
            
            ctx.save_for_backward(torch.tensor(np.array(gradient)).to('cpu'))
            return torch.tensor((np.array(optimal))).to('cpu')

        @staticmethod
        def backward(ctx, grad_output):
            # only call parameters q
            grad = ctx.saved_tensors

            grad_all = torch.zeros((len(grad[0]),200))
            for i in range(len(grad[0])):
                grad_all[i] = grad_output[i] @ grad[0][i]
            #print(grad_all.shape)
            
            
            # ??? need to check the primal problem, whether those Q, x, G, h, A, b should also be updated.
            return (None, grad_all, None, None, None, None)

    return Newlayer.apply
