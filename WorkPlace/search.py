import argparse
import cProfile
import io
import json
import pstats
import random

import subprocess

import os

import copy

import numpy as np

from NASsearch.acquisition_func import AcquisitionFunc
from Kernel_optimization.all_gp import cal_distance
from WorkPlace import train_eval

OPERATIONS = ['avg_pool_3x3', 'max_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7',
              'dil_conv_3x3', 'conv_7x1_1x7']

WEIGHT_FILE = "./cache/weight_1.pkl"


def generate_combination(arch, b):
    archs = []
    for op1 in OPERATIONS:
        for op2 in OPERATIONS:
            for op1_conn in range(b + 1):
                for op2_conn in range(b + 1):
                    temp = copy.deepcopy(arch)
                    temp["normal"].append((op1, op1_conn))
                    temp["normal"].append((op2, op2_conn))
                    temp["reduce"].append((op1, op1_conn))
                    temp["reduce"].append((op2, op2_conn))
                    temp["normal_concat"].append(b+1)
                    temp["reduce_concat"].append(b+1)
                    if op1_conn >= 2 and op1_conn in temp["normal_concat"]:
                        idx = temp["normal_concat"].index(op1_conn)
                        del temp["normal_concat"][idx]
                        del temp["reduce_concat"][idx]
                    if op2_conn >= 2 and op2_conn in temp["reduce_concat"]:
                        idx = temp["normal_concat"].index(op2_conn)
                        del temp["normal_concat"][idx]
                        del temp["reduce_concat"][idx]
                    # temp["normal_concat"] = list(range(2, b + 2))
                    # temp["reduce_concat"] = list(range(2, b + 2))
                    archs.append(json.dumps(temp))
    return archs


def generate_136():
    # arch = dict(
    #     normal=[],
    #     normal_concat=[],
    #     reduce=[],
    #     reduce_concat=[],
    # )
    arch = {"normal": [["avg_pool_3x3", 0], ["avg_pool_3x3", 0], ["avg_pool_3x3", 0], ["avg_pool_3x3", 0]], "normal_concat": [2, 3], "reduce": [["avg_pool_3x3", 0], ["avg_pool_3x3", 0], ["avg_pool_3x3", 0], ["avg_pool_3x3", 0]], "reduce_concat": [2, 3]}
    archs = generate_combination(arch, 3)
    eliminate_duplicate(archs)
    print(len(archs))
    with open("b3_arch.txt", "w") as f:
        for arch in archs:
            f.write(arch+"\n")


def read_block_file(file_name):
    train_X = []
    train_y = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            train_X.append(line.split(" accuracy: ")[0])
            train_y.append(float(line.split(" accuracy: ")[1]))

    return train_X, train_y


def normal_acc(file_names):
    archs = []
    accs = []
    for file_name in file_names:
        tmp1, tmp2 = read_block_file(file_name)
        archs += tmp1
        accs += tmp2
    accs = np.array(accs)
    mean = np.mean(accs)
    std = np.std(accs)
    accs = (accs - mean) / std
    return archs, list(accs), mean, std


def run_gp():
    parser = argparse.ArgumentParser()
    # parser.add_argument('arch', type=str, help='structure indicators')
    # parser.add_argument('--train_file', type=str, help='structure indicators')
    # parser.add_argument('--file_name', type=str, help='structure indicators')
    parser.add_argument('--out_file', type=str, help='structure indicators')
    parser.add_argument('--start', type=int, help='structure indicators')
    parser.add_argument('--end', type=int, help='structure indicators')
    args = parser.parse_args()

    train_file = "./cache/stage_1.txt"
    file_name = "./cache/stage_1.txt"

    # train_file
    train_X, train_y, train_mean, train_std = normal_acc([train_file])
    print("train_std", train_std)
    current_optimal = np.max(train_y)
    acquisition = AcquisitionFunc(train_X, train_y, current_optimal, mode="ucb", trade_off=0.01)

    prev_block_archs = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            prev_block_archs.append(line.split(" accuracy: ")[0])

    counter = 0
    with open(args.out_file, "a") as f:
        for arch in prev_block_archs[args.start:args.end]:
            counter += 1
            print("counter", counter)
            new_arch = generate_combination(json.loads(arch), 2)
            length = len(new_arch)
            print(length)
            for i in range(1):
                print("range: ", i)
                start = int(i * length / 1)
                end = int((i + 1) * length / 1)
                acquisition_value, mean, std = acquisition.compute(new_arch[start:end], weight_file=WEIGHT_FILE)

                for i in range(int(length / 1)):
                    f.write(
                        new_arch[i] + " accquisition_value: " + str(acquisition_value[i]) + " mean: " + str(
                            mean[i] * train_std + train_mean) + " std: " + str(std[i] * train_std) + "\n")


def random_select(size):
    total_size = 256 * 820
    sampled_archs_index = []
    for i in range(size):
        sampled_archs_index.append(random.randint(0, total_size - 1))

    arch_list = []
    archs, _ = read_block_file("block3_sample.txt")
    for arch in archs:
        arch_list += generate_combination(json.loads(arch), 4)

    for i in sampled_archs_index:
        with open("block4_sample_arch.txt", "a") as f:
            f.write(arch_list[i] + "\n")


def sort_arch():
    my_list = List(1000)
    # for arch in archs:
    #     my_list.insert(arch)

    with open("b2_predict_ucb_001.txt", "r") as f:
        for line in f.readlines():
            arch = line.split(" accquisition_value: ")[0]
            value = float(line.split(" accquisition_value: ")[1].split(" mean: ")[0])
            mean = float(line.split(" accquisition_value: ")[1].split(" mean: ")[1].split(" std: ")[0])
            std = float(line.split(" accquisition_value: ")[1].split(" mean: ")[1].split(" std: ")[1])

            my_list.insert(dict(
                arch=arch,
                value=value,
                mean=mean,
                std=std,
            ))

    with open("b2_arch_ucb_001.txt", "w") as f:
        for item in my_list.array:
            f.write(
                item["arch"] + " acquisition: " + str(item["value"]) + " mean: " + str(item["mean"]) + " std: " + str(
                    item["std"]) + "\n")


def eliminate_duplicate(archs, arch_items=None):
    dist = cal_distance(archs, weight_file=WEIGHT_FILE, vector_file=None, predict=True, dump_label=False)
    zero_x, zero_y = np.where(dist == 0)
    delete_items = []
    for i in range(len(zero_x)):
        if zero_x[i] != zero_y[i] and zero_x[i] not in delete_items and zero_y[i] not in delete_items:
            delete_items.append(zero_y[i])

    # print(len(delete_items))
    delete_items.sort()
    for item in reversed(delete_items):
        if arch_items is None:
            archs.pop(item)
        else:
            arch_items.pop(item)


def test_eliminate():
    archs = []
    arch_items = []
    with open("b2_arch_ucb_001.txt", "r") as f:
        for line in f.readlines():
            arch = line.split(" acquisition: ")[0]
            archs.append(arch)
            value = float(line.split(" acquisition: ")[1].split(" mean: ")[0])
            mean = float(line.split(" acquisition: ")[1].split(" mean: ")[1].split(" std: ")[0])
            std = float(line.split(" acquisition: ")[1].split(" mean: ")[1].split(" std: ")[1])
            arch_items.append(dict(
                arch=arch,
                value=value,
                mean=mean,
                std=std
            ))
    eliminate_duplicate(archs, arch_items)
    print(len(arch_items))
    with open("b2_arch.txt", "w") as f:
        for item in arch_items[:256]:
            f.write(
                item["arch"] + " acquisition: " + str(item["value"]) + " mean: " + str(item["mean"]) + " std: " + str(
                    item["std"]) + "\n")


class List:
    def __init__(self, size=256):
        self.size = size
        self.array = []

    def insert(self, x):
        flag = 0
        for i in range(len(self.array)):
            if x["value"] > self.array[i]["value"]:
                flag = 1
                self.array.insert(i, x)
                break
        if flag == 0:
            self.array.append(x)

        if len(self.array) > self.size:
            self.array = self.array[:self.size]


def progressive_search():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default="block4_sample_arch.txt", type=str,
                        help='the file contains the architectures for networks to train')
    parser.add_argument('--start', type=int, help='0-256')
    parser.add_argument('--end', type=int, help='0-256')
    parser.add_argument('--index', type=int, help='which gpu to run on')
    parser.add_argument('--outfile', type=str, help='the output file name to store the accuracy')
    args = parser.parse_args()
    archs = []
    with open(args.filename, "r") as f:
        for line in f.readlines():
            arch = line.split(" acquisition: ")[0]
            archs.append(arch)

    for a in archs[args.start:args.end]:
        accr = train_eval.main(a, args.index)
        with open(args.outfile, 'a') as f:
            f.write(a + " accuracy: " + str(accr) + "\n")


def temp():
    counter = 1
    with open("block3_arch_15f.txt", "a") as f2:
        with open("block3_arch_15.txt", "r") as f1:
            for line in f1.readlines():
                if counter <= 7392:
                    f2.write(line)
                    counter += 1
        with open("block3_arch15_254_256.txt", "r") as f1:
            for line in f1.readlines():
                f2.write(line)


if __name__ == "__main__":
    generate_136()
    # progressive_search()
    # temp()
    # run_gp()
    # _, accs, _, _ = normal_acc(["block1.txt"])
    # print(accs)
    # test_eliminate()
    # sort_arch()
    # random_select(256)

