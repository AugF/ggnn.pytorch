import torch
import torch.nn as nn
from tools import save_weight
from flags import *

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.state_dim = state_dim
        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim, bias=bias_flag),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim, bias=bias_flag),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim, bias=bias_flag),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        if save_flag:
            gates = [self.reset_gate[0], self.update_gate[0], self.tansform[0]]
            for j in range(3):
                left = j * self.state_dim
                right = (j + 1) * self.state_dim
                save_weight("weight_r_{}".format(j), gates[0].weight.detach().numpy()[:, left:right])
                save_weight("weight_z_{}".format(j), gates[1].weight.detach().numpy()[:, left:right])
                save_weight("weight_h_{}".format(j), gates[2].weight.detach().numpy()[:, left:right])
            if bias_flag:
                save_weight("weight_z_bias", gates[0].bias)
                save_weight("weight_r_bias", gates[1].bias)
                save_weight("weight_h_bias", gates[2].bias)

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()

        assert (opt.state_dim >= opt.annotation_dim,  \
                'state_dim must be no less than annotation_dim')

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim, bias=bias_flag)
            out_fc = nn.Linear(self.state_dim, self.state_dim, bias=bias_flag)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim, bias=bias_flag),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1, bias=bias_flag)
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)  # todo check
                if bias_flag:
                    m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
                if save_flag:
                    save_weight("weight_in_{}".format(i), self.in_fcs[i].weight)
                    save_weight("weight_out_{}".format(i), self.out_fcs[i].weight)
                    if bias_flag:
                        save_weight("weight_in_bias_{}".format(i), self.in_fcs[i].bias)
                        save_weight("weight_out_bias_{}".format(i), self.out_fcs[i].bias)
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A)

        join_state = torch.cat((prop_state, annotation), 2)

        output = self.out(join_state)
        output = output.sum(2)

        if save_flag:
            save_weight("weight_o", self.out[2].weight)
            weight_z_1 = self.out[0].weight
            weight_ho, weight_xo = weight_z_1[:, :self.state_dim], weight_z_1[:, -self.annotation_dim:]
            save_weight("weight_ho", weight_ho)
            save_weight("weight_xo", weight_xo)
            if bias_flag:
                save_weight("weight_o2_bias", self.out[2].bias)
                weight_z_1_bias = self.out[0].bias
                save_weight("weight_o1_bias", weight_z_1_bias)
        return output
