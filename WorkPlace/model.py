from copy import deepcopy

import torch
import torch.nn as nn
from torch.autograd import Variable

from NASsearch.operations import *
from NASsearch.utils import drop_path


class Cell(nn.Module):
    def __init__(self, genotype, c_prev, c_prev_prev, C, reductions):
        super(Cell, self).__init__()

        self.reductions = reductions
        self.reduce_preprocess0 = FactorizedReduce(c_prev_prev, C)
        # self.reduce_preprocess1 = FactorizedReduce(c_prev, C)
        self.normal_preprocess0 = ReLUConvBN(c_prev_prev, C, 1, 1, 0)
        self.normal_preprocess1 = ReLUConvBN(c_prev, C, 1, 1, 0)

        if reductions[-1]:
            op_names, indices = zip(*genotype["reduce"])
            # print(op_names, indices)
            concat = genotype["reduce_concat"]
        else:
            op_names, indices = zip(*genotype["normal"])
            concat = genotype["normal_concat"]
        self._compile(C, op_names, indices, concat, reductions)

    def _compile(self, C, op_names, indices, concat, reductions):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reductions[-1] and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        # s0_normal = self.normal_preprocess0(s0)
        # s0_reduce = self.reduce_preprocess0(s0)
        if len(self.reductions) >= 3 and not self.reductions[-1] and self.reductions[-2] and not self.reductions[-3]:
            s0 = self.reduce_preprocess0(s0)
        else:
            s0 = self.normal_preprocess0(s0)
        s1 = self.normal_preprocess1(s1)
        # s1_reduce = self.reduce_preprocess1(s1)
        # print(s0.shape)
        states = [s0, s1]
        for i in range(self._steps):
            index1 = self._indices[2 * i]
            index2 = self._indices[2 * i + 1]
            h1 = states[index1]
            h2 = states[index2]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        # print(x.view(x.size(0), -1).shape)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.genotype = genotype

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        channels = [C_curr]
        reductions = [False]
        C_curr = C
        self.cells = nn.ModuleList()
        # reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                op1_prev = genotype["reduce"][0][1]+1
                op2_prev = genotype["reduce"][1][1]+1
            else:
                reduction = False
                op1_prev = genotype["normal"][0][1]+1
                op2_prev = genotype["normal"][1][1]+1

            reductions.append(reduction)
            c_prev = channels[-1]
            try:
                c_prev_prev = channels[-2]
            except:
                c_prev_prev = channels[-1]
            cell = Cell(genotype, c_prev, c_prev_prev, C_curr, deepcopy(reductions))

            self.cells += [cell]
            channels.append(C_curr*cell.multiplier)
            # print(channels[-1])
            if i == 2 * layers // 3:
                C_to_auxiliary = channels[-1]

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, input, drop_path_prob):
        logits_aux = None
        s0 = s1 = self.stem(input)
        states = [s0, s1]
        for i, cell in enumerate(self.cells):
            # s0 pre_pre cell
            # s1 pre_cell
            print(states[-1].shape)
            s0 = states[i]
            s1 = states[i+1]
            states.append(cell(s0, s1, drop_path_prob))
            # s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(states[-1])
        out = self.global_pooling(states[-1])
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

