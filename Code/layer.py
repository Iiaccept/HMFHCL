import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class hyperedge_encoder(nn.Module):
    def __init__(self, num_in_edge, num_hidden, dropout, act=torch.tanh):
        super(hyperedge_encoder, self).__init__()
        self.num_in_edge = num_in_edge
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act

        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_edge, self.num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

    def forward(self, H_T):
        z1 = self.act(torch.mm(H_T, self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_edge) + ' -> ' + str(self.num_hidden)


class node_encoder(nn.Module):
    def __init__(self, num_in_node, num_hidden, dropout, act=torch.tanh):
        super(node_encoder, self).__init__()
        self.num_in_node = num_in_node
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_node, self.num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

    def forward(self, H):
        z1 = self.act(H.mm(self.W1) + 2*self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_node) + ' -> ' + str(self.num_hidden)


class decoder2(nn.Module):
    def __init__(self, dropout=0.8, act=torch.sigmoid):
        super(decoder2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z_node, z_hyperedge):
        z_node_ = self.dropout(z_node)
        z_hyperedge_ = self.dropout(z_hyperedge)

        z = self.act(z_node_.mm(z_hyperedge_.t()))
        return z

class decoder1(nn.Module):
    def __init__(self, dropout=0.5, act=torch.sigmoid):
        super(decoder1, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z_node, z_hyperedge):
        z_node_ = z_node
        z_hyperedge_ = z_hyperedge
        z = self.act(z_node_.mm(z_hyperedge_.t()))

        return z

class HGNN_conv1(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):  #
        super(HGNN_conv1, self).__init__()

        self.weight = Parameter(torch.DoubleTensor(in_ft, out_ft))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.linear_x_1 = nn.Linear(in_ft, out_ft).to(torch.double)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):  # x: torch.Tensor, G: torch.Tensor

        # part1
        x = x.double()
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x) + x
        return x



class HGNN1(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, n_node, emb_dim, n_hid_2=128,dropout=0.5):
        super(HGNN1, self).__init__()
        self.dropout = dropout
        #两层是在这里体现的
        self.hgc1 = HGNN_conv1(in_ch, n_hid)
        # self.hgc2 = HGNN_conv1(n_hid, n_hid)
        self.hgc2 = HGNN_conv1(n_hid, n_class)

    def forward(self, x, G):
        G = G + torch.eye(G.shape[0]).cuda()
        x= self.hgc1(x, G)
        x = torch.tanh(x)
        x= self.hgc2(x, G)
        # x = torch.tanh(x)
        # x= self.hgc3(x, G)



        return x






