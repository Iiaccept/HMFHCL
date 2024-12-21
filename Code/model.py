from layer import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self, num_in_node=218, num_in_edge=271, num_hidden1=512, num_out=128):  # 435, 757, 512, 128

        super(Model, self).__init__()


        self.decoder1 = decoder1(act=lambda x: x)
        self.decoder2 = decoder2(act=lambda x: x)




        #超图卷积
        self.hgnn_node2 = HGNN1(num_in_node, num_in_node, num_out, num_in_node, num_in_node)
        self.hgnn_hyperedge2 = HGNN1(num_in_edge, num_in_edge, num_out, num_in_edge, num_in_edge)


    def sample_latent(self, z_node, z_hyperedge):
        # Return the latent normal sample z ~ N(mu, sigma^2)
        self.z_node_mean = self._enc_mu_node(z_node)  # mu
        self.z_node_log_std = self._enc_log_sigma_node(z_node)
        self.z_node_std = torch.exp(self.z_node_log_std)  # sigma
        z_node_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_node_std.size())).double()
        self.z_node_std_ = z_node_std_.cuda()
        self.z_node_ = self.z_node_mean + self.z_node_std.mul(Variable(self.z_node_std_, requires_grad=True))

        self.z_edge_mean = self._enc_mu_hedge(z_hyperedge)
        self.z_edge_log_std = self._enc_log_sigma_hyedge(z_hyperedge)
        self.z_edge_std = torch.exp(self.z_edge_log_std)
        z_edge_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_edge_std.size())).double()
        self.z_edge_std_ = z_edge_std_.cuda()
        self.z_hyperedge_ = self.z_edge_mean + self.z_edge_std.mul(Variable(self.z_edge_std_, requires_grad=True))

        if self.training:
            return self.z_node_, self.z_hyperedge_
        else:
            return self.z_node_mean, self.z_edge_mean



    def forward(self, HMG, HDG, mir_feat, dis_feat, HMD, HDM):


        #卷之后进行解码
        mir_feature_1 = self.hgnn_hyperedge2(mir_feat, HMG)
        dis_feature_1 = self.hgnn_node2(dis_feat, HDG)

        mir_feature_2 = self.hgnn_hyperedge2(mir_feat, HMD)
        dis_feature_2 = self.hgnn_node2(dis_feat, HDM)



        #这个是局部超图
        reconstructionMD = self.decoder1(dis_feature_2, mir_feature_2)

        #这个是全局超图
        reconstructionG = self.decoder1(dis_feature_1, mir_feature_1)


        result_h = (reconstructionG + reconstructionMD ) / 2
        recover = result_h

        return   reconstructionG, reconstructionMD, recover,  mir_feature_1, mir_feature_2, dis_feature_1, dis_feature_2
