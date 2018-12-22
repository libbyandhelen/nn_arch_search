#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as pl
import torch
from torch.autograd import Variable
# from gaussian_process.wl_kernel import WLKernel
# from NASsearch.wl_kernel import WLKernel
import pickle, scipy, math


def isPSD(A, tol=1e-8):
    E = np.linalg.eigvalsh(A.data.numpy())
    return np.all(E > -tol)


class GaussianProcess:
    def __init__(self, sq_f, l, sq_n, lr):
        self.sq_n = Variable(torch.Tensor([sq_n]), requires_grad=True)  # noise variance.
        self.l = Variable(torch.Tensor([l]), requires_grad=True)
        self.sq_f = Variable(torch.Tensor([sq_f]), requires_grad=True)
        self.X = None
        self.y = None
        self.L = None
        self.dist_mat = None
        self.grads = None
        self.lr = lr
        self.lr_decay = 200
        self.optimizer = torch.optim.Adam([self.sq_f, self.l], lr=lr)

    def square_exponential_kernel(self, dist):
        if len(dist) == len(dist[0]):
            return self.sq_f * np.exp(-0.5 * (1 / self.l ** 2) * dist) + np.eye(len(dist)) * self.sq_n ** 2
        else:
            return self.sq_f * np.exp(-0.5 * (1 / self.l ** 2) * dist)

    def learn(self, iter):
        # cashed value: dist_mat, X, y
        lr = self.lr
        for i in range(iter):
            # y y_mean column vector with shape (n,1)
            sqr_dist_mat = torch.Tensor(self.dist_mat).detach()
            y = torch.Tensor(np.reshape(self.y, (len(self.y), 1))).detach()
            y_mean = torch.Tensor(np.zeros((len(self.y), 1))).detach()
            I = torch.Tensor(np.eye(len(self.y))).detach()
            cov = self.sq_f * torch.exp(-1 / (2 * self.l ** 2) * sqr_dist_mat) + I * self.sq_n ** 2
            if not isPSD(cov):
                print("cov is not PSD")
            # calculate determinent of cov
            det = cov.det()
            # log marginal likelihood, ignore term -(n/2)log(1/2π)
            log_like = -(-0.5 * det.log() - 0.5 * (y - y_mean).transpose(0, 1).mm((cov.inverse())).mm((y - y_mean)))
            self.optimizer.zero_grad()
            log_like.backward()
            if i >= 0:
                print("iter: ", i)
                print("l: ", self.l, " sq_f: ", self.sq_f, " sq_n: ", self.sq_n)
                print("det of K:\n", det.data.numpy())
                print("loglike:\n", -log_like.data.numpy())
                print("grads: ",
                      [-self.l.grad.data.numpy()[0], -self.sq_f.grad.data.numpy()[0], -self.sq_n.grad.data.numpy()[0]])
                print("")
            if torch.isnan(log_like):
                print("grad nan at iter: ", i)
                print("det: ", det.data.numpy())
                break
            '''
            # manually update parameter
            self.l += 100*lr*l.grad.data.numpy()[0]
            self.sq_f += 100*lr*sq_f.grad.data.numpy()[0]
            self.sq_n += lr*sq_n.grad.data.numpy()[0]
            '''
            self.optimizer.step()


def gp_gradient(dist_mat_file, stage_file):
    net_list = []
    acc_list = []
    i = 0
    inputF = open(dist_mat_file, 'rb')
    with open(stage_file) as file:
        for line in file:
            i += 1
            net, acc = line.split(" accuracy: ")
            net_list.append(net)
            acc_list.append(float(acc[:-1]))
            if i == 136 + 256 + 256 + 256:
                break

    y = np.array(acc_list)
    y = (y - y.mean()) / y.std()
    print(y)

    # normalized data for first 136 samples
    print("y len: ", len(y))
    # sq_f,l,sq_n,lr
    model = GaussianProcess(10.5155, 30.5034, 0.39, 0.05)
    print("now fitting model...")
    model.dist_mat = np.array(pickle.load(inputF))[:136 + 256 + 256 + 256, :136 + 256 + 256 + 256]
    print(model.dist_mat)
    print(model.dist_mat.shape)
    model.y = y
    inputF.close()
    print("now learning hp...")
    model.learn(20000)

