import torch
from torch.autograd import Variable
from flags import *
import torch.nn as nn

def train(epoch, dataloader, net, criterion, optimizer, opt):
    net.train()
    for i, (adj_matrix, annotation, target) in enumerate(dataloader, 0):
        # adj_np = adj_matrix.numpy() # give up, believe pytorch is right
        net.zero_grad()

        padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 2)
        if opt.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()

        init_input = Variable(init_input)
        adj_matrix = Variable(adj_matrix)
        annotation = Variable(annotation)
        target = Variable(target)

        output = net(init_input, annotation, adj_matrix)

        loss = criterion(output, target)

        if forward_flag:
            print("end state", net.end_state)
            print("z", net.z)

        x = loss.backward()

        print("", x)
        if grad_flag:
            grad_print(net)

        optimizer.step()

        if updated_weight_flag:
            weight_print(net)

        print('[{}/{}][{}/{}] Loss: {}'.format(epoch, opt.niter, i, len(dataloader), loss.item()))
        if sing_step_flag:
            break
        # if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
