import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from model import GGNN
from utils.train import train
from utils.test import test
from utils.data.dataset import bAbIDataset
from utils.data.dataloader import bAbIDataloader
from flags import *

parser = argparse.ArgumentParser()
# support change
parser.add_argument('--task_id', type=int, default=16, help='bAbI task id')
parser.add_argument('--state_dim', type=int, default=10, help='GGNN hidden state size')
parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')

# unsupport change
parser.add_argument('--question_id', type=int, default=0, help='question types')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--n_steps', type=int, default=4, help='propogation steps number of GGNN')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')

opt = parser.parse_args()
# print(opt)

# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if task_set_flag:
    opt.task_id, opt.state_dim, opt.niter = task_id_set, state_dim_set, niter_set
opt.train_dataroot = 'babi_data/processed_1/train/%d_graphs.txt' % opt.task_id
opt.test_dataroot = 'babi_data/processed_1/test/%d_graphs.txt' % opt.task_id

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

def main(opt):
    train_dataset = bAbIDataset(opt.train_dataroot, opt.question_id, True)
    train_dataloader = bAbIDataloader(train_dataset, batch_size=opt.batchSize, \
                                      shuffle=False, num_workers=2)

    val_dataset = bAbIDataset(opt.train_dataroot, opt.question_id, False)
    val_dataloader = bAbIDataloader(val_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=2)

    test_dataset = bAbIDataset(opt.test_dataroot, opt.question_id, False)
    test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=2)

    opt.annotation_dim = 1  # for bAbI
    opt.n_edge_types = train_dataset.n_edge_types
    opt.n_node = train_dataset.n_node

    if n_steps_flag:
        opt.n_steps = n_steps_set
    if n_lr_flag:
        opt.lr = n_lr_set
    net = GGNN(opt)
    net.double()
    # print(net)

    criterion = nn.CrossEntropyLoss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    for epoch in range(0, opt.niter):
        train(epoch, train_dataloader, net, criterion, optimizer, opt)
        break
    #     test(val_dataloader, net, criterion, optimizer, opt, test_flag="Val")
    # print("begin test: ")
    # test(test_dataloader, net, criterion, optimizer, opt)

if __name__ == "__main__":
    main(opt)

