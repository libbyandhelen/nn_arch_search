import argparse
import json

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.distributed as dist
import torch.utils.data.distributed

from WorkPlace.model import NetworkCIFAR
from pretrain import utils

AUXILIARY = False


def dist_main(size, genotype):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('rank', default=0, type=int)
    # parser.add_argument('genotype', default="", type=str)
    # parser.add_argument('size', default=7000, type=int)
    # args = parser.parse_args()

    genotype = json.loads(genotype)
    # torch.cuda.set_device(rank)
    #
    cudnn.benchmark = True
    cudnn.enabled = True

    # dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:7000", world_size=size, rank=rank)
    model = NetworkCIFAR(
        # C=36,
        C=8,
        num_classes=10,
        layers=8,
        auxiliary=False,
        genotype=genotype,
    )
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(size))).cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        # lr=0.025,
        lr=0.01,
        momentum=0.9,
        weight_decay=3e-4,
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(cutout=False, cutout_length=16)
    train_data = dset.CIFAR10(root="data/", train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root="data/", train=False, download=True, transform=valid_transform)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=False, pin_memory=True, num_workers=2)
    # train_data, batch_size=16, shuffle=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=64, shuffle=False, pin_memory=True, num_workers=2)
    # valid_data, batch_size=16, shuffle=False, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
    prev_valid_acc = 0
    max_acc = 0

    for epoch in range(50):
        # train_sampler.set_epoch(epoch)
        scheduler.step()
        # print('epoch {} lr {}'.format(epoch, scheduler.get_lr()[0]))
        print('epoch {}'.format(epoch))
        drop_path_prob = 0.2 * epoch / 50

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, drop_path_prob)
        print('train_acc {}'.format(train_acc))

        valid_acc, valid_obj = infer(valid_queue, model, criterion, drop_path_prob)
        print('valid_acc {}'.format(valid_acc))

        # if prev_valid_acc > valid_acc:
        #     return prev_valid_acc
        # prev_valid_acc = valid_acc
        if valid_acc > max_acc:
            max_acc = valid_acc

        utils.save(model, 'weights.pt')

    # print("type", type(valid_acc))
    return max_acc


def main(genotype, index):
    genotype = json.loads(genotype)
    torch.cuda.set_device(index)
    #
    cudnn.benchmark = True
    cudnn.enabled = True

    model = NetworkCIFAR(
        C=216,
        # C=48,
        num_classes=10,
        layers=14,
        # layers=20,
        auxiliary=False,
        # auxiliary=True,
        genotype=genotype,
    )
    print("parameters number:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        # lr=0.025,
        lr=0.01,
        momentum=0.9,
        weight_decay=3e-4,
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(cutout=False, cutout_length=16)
    train_data = dset.CIFAR10(root="data/", train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root="data/", train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)
    # train_data, batch_size=16, shuffle=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=32, shuffle=False, pin_memory=True, num_workers=2)
    # valid_data, batch_size=16, shuffle=False, num_workers=2)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    prev_valid_acc = 0
    max_acc = 0

    for epoch in range(20):
        scheduler.step()
        print('epoch {} lr {}'.format(epoch, scheduler.get_lr()[0]))
        print('epoch {}'.format(epoch))
        drop_path_prob = 0.2 * epoch / 20

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, drop_path_prob)
        print('train_acc {}'.format(train_acc))

        valid_acc, valid_obj = infer(valid_queue, model, criterion, drop_path_prob)
        print('valid_acc {}'.format(valid_acc))

        # if prev_valid_acc > valid_acc:
        #     return prev_valid_acc
        # prev_valid_acc = valid_acc
        if valid_acc > max_acc:
            max_acc = valid_acc

        # utils.save(model, 'weights.pt')

    # print("type", type(valid_acc))
    return max_acc


def train(train_queue, model, criterion, optimizer, drop_path_prob):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        # with torch.cuda.device(0):
        input = Variable(input).cuda()
        # target = Variable(target).cuda(async=True)
        target = Variable(target).cuda()
        #
        # input = Variable(input)
        # target = Variable(target)

        optimizer.zero_grad()
        logits, logits_aux = model(input, drop_path_prob)
        loss = criterion(logits, target)
        if AUXILIARY:
            loss_aux = criterion(logits_aux, target)
            loss += 0.4 * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        # if step == 1:
        #     return top1.avg, objs.avg
        if step % 50 == 0:
            print(loss.data[0])
            print('train {} {} {} {}'.format(step, objs.avg, top1.avg, top5.avg))

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, drop_path_prob):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        # with torch.cuda.device(0):
        input = Variable(input, volatile=True).cuda()
        # target = Variable(target, volatile=True).cuda(async=True)
        target = Variable(target, volatile=True).cuda()
        #
        # input = Variable(input, volatile=True)
        # target = Variable(target, volatile=True)

        logits, _ = model(input, drop_path_prob)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        # if step == 1:
        #     return top1.avg, objs.avg
        if step % 50 == 0:
            print('valid {} {} {} {}'.format(step, objs.avg, top1.avg, top5.avg))

    return top1.avg, objs.avg

# if __name__ == "__main__":
#     dist_main()

