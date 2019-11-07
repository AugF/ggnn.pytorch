import torch
from torch.autograd import Variable
from flags import *

def train(epoch, dataloader, net, criterion, optimizer, opt):
    net.train()
    for i, (adj_matrix, annotation, target) in enumerate(dataloader, 0):
        # print(adj_matrix, "dasd", annotation, "dada", target)
        # if i == 5:
        #     break
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
        # for m in net.modules():
        #     if isinstance(m, nn.Linear):
        #         print("bias", m.bias.detach().numpy())  # print bias
        #         if hasattr(m.bias.grad, "numpy"):
        #             print("grad", m.bias.grad.numpy())

        loss.backward()
        optimizer.step()
        print('[{}/{}][{}/{}] Loss: {}'.format(epoch, opt.niter, i, len(dataloader), loss.item()))
        if sing_step_flag:
            break
        # if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
