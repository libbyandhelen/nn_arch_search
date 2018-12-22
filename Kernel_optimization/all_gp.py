#! /usr/bin/env python3
import numpy as np
import json
import pickle
from Kernel_optimization.all_kernel import all_Kernel


def get_vectors(all_nets, weight_file, vector_file, predict, dump_label):
    dist = np.zeros((len(all_nets), len(all_nets)))
    kernel = all_Kernel(N=3, netlist=all_nets, level=8)
    net_vectors = kernel.run(dump_label)
    # reweight net_vectors
    print('vector shape:', net_vectors.shape)
    if weight_file is not None:
        with open(weight_file, 'rb') as f:
            weight = pickle.load(f)

        # weight = np.tile(weight.data.numpy(), (len(all_nets), 1))
        # weight = weight.data.numpy()
        print('weight shape:', weight.shape)
        if predict:
            len_predict = net_vectors.shape[1]
            len_weight = weight.shape[0]
            avg_weight = np.mean(weight)
            tail = np.full(len_predict - len_weight, avg_weight)
            print("tail shape", tail.shape)
            weight = np.append(weight, tail)

        net_vectors = net_vectors * weight
    print('net_vectors\n', net_vectors.shape)
    print('net_vectors\n', net_vectors)
    # save vectors into file

    if vector_file is not None:
        with open(vector_file,'wb') as f:
            pickle.dump(net_vectors,f)

    return net_vectors


def cal_distance(all_nets, weight_file, vector_file, predict, dump_label):
    dist = np.zeros((len(all_nets), len(all_nets)))
    net_vectors = get_vectors(all_nets, weight_file, vector_file, predict, dump_label)
    for i in range(len(all_nets)):
        for j in range(i + 1, len(all_nets)):
            dist[i, j] = np.sum((net_vectors[i] - net_vectors[j]) ** 2)

    for i in range(len(all_nets)):
        for j in range(0, i):
            dist[i, j] = dist[j, i]
    return dist


class GaussianProcess:
    def __init__(self, kernel_param):
        self.s = 0.00005
        self.X = None
        self.y = None
        self.L = None
        self.kernel_param = kernel_param

    def square_exponential_kernel(self, dist):
        l = 30.5788
        sq_f = 10.6237
        sq_n = 0.39  # -8.0 score on CV.py

        # l = 47.7643
        # sq_f = 7.0309
        # sq_n = 0.8

        # l= 60.1034
        # sq_f = 11.2664
        # sq_n = 0.75

        # l = 79.6804
        # sq_f = 10.2821
        # sq_n = 0.85

        if len(dist) == len(dist[0]):
            return sq_f * np.exp(-0.5 * (1 / l ** 2) * dist) + np.eye(len(dist)) * sq_n ** 2
        else:
            return sq_f * np.exp(-0.5 * (1 / l ** 2) * dist)

    def fit_predict(self, X, Xtest, y, weight_file):
        self.X = X
        self.y = np.array(y)
        all_dist_mat = cal_distance(X + Xtest, weight_file, vector_file=None, predict=True, dump_label=False)
        print(all_dist_mat)
        self.dist_mat = all_dist_mat[:len(X), :len(X)]
        K = self.square_exponential_kernel(self.dist_mat)
        self.L = np.linalg.cholesky(K)
        Lk = np.linalg.solve(self.L, self.square_exponential_kernel(all_dist_mat[:len(X), len(X):]))
        mu = np.dot(Lk.T, np.linalg.solve(self.L, self.y))

        K_ = self.square_exponential_kernel(all_dist_mat[len(X):, len(X):])
        s2 = K_ - np.sum(Lk ** 2, axis=0)
        s = np.sqrt(np.diag(s2))

        return mu, s2, s


def test(stage_file, dist_mat_file, weight_file, vector_file):
    archs = []
    no_of_sample = 136 + 256 + 256 + 256
    no_of_line = 1
    label = []

    with open(stage_file, "r") as f:
        for line in f.readlines():
            archs.append(json.loads(line.split(" accuracy: ")[0]))
            label.append(float(line.split(" accuracy: ")[1][:-1]))
            no_of_line += 1
            if no_of_line > no_of_sample:
                break

    dist_mat = cal_distance(archs, weight_file, vector_file, predict=False, dump_label=True)
    print(dist_mat)
    with open(dist_mat_file, 'wb') as opf:
        pickle.dump(dist_mat, opf)
    counter = 0
    for i in range(dist_mat.shape[0] - 1):
        for j in range(i + 1, dist_mat.shape[0]):
            if np.array_equal(dist_mat[i, :], dist_mat[j, :]):
                counter += 1
                print('identity: ', i, ' and ', j)
    print(counter)


def save_vector_label(stage_file, label_file, dist_mat_file, weight_file, vector_file):
    test(stage_file, dist_mat_file, weight_file, vector_file)
    net_list = []
    acc_list = []
    i = 0
    with open(stage_file) as file:
        for line in file:
            i += 1
            net, acc = line.split(" accuracy: ")
            net_list.append(net)
            acc_list.append(float(acc[:-1]))
            # if i == 136:
            #     break
    y = np.array(acc_list)
    y = (y - y.mean()) / y.std()
    with open(label_file, 'wb') as f:
        pickle.dump(y, f)

