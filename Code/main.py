import numpy as np
import copy
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from model import Model
from numpy.core import multiarray
from hypergraph_utils import *
from hypergraph_utils import _generate_G_from_H
import G_similarity
import L_similarity
import os
from kl_loss import kl_loss
from function import create_resultlist
from utils import f1_score_binary,precision_binary,recall_binary, mcc_binary, accuracy_binary
from HMF import *
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import interp
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 设置随机数种子
seed = 47
# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作的确定性
torch.backends.cudnn.benchmark = False  # 禁用卷积算法选择，确保结果可重复

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def contrastive_loss(h1, h2, tau = 0.1):
    sim_matrix = sim(h1, h2)
    f = lambda x: torch.exp(x / tau)
    matrix_t = f(sim_matrix)
    numerator = matrix_t.diag()
    denominator = torch.sum(matrix_t, dim=-1)
    loss = -torch.log(numerator / denominator).mean()
    return loss



def train(epochs):

    auc1 = 0
    aupr1 = 0
    recall1 = 0
    precision1 = 0
    f11 = 0
    mcc1 = 0
    accuracy1 = 0
    if epoch != epochs - 1:
        model.train()


        reconstructionG, reconstructionMD, recover, mir_feature_1, mir_feature_2, dis_feature_1, dis_feature_2 = model(
            HMG, HDG, mir_feat, dis_feat, HMD, HDM)  # 将数据传入模型




        outputs = recover .t().cpu().detach().numpy()
        test_predict = create_resultlist(outputs, testset, Index_PositiveRow, Index_PositiveCol, Index_zeroRow,Index_zeroCol, len(test_p), zero_length, test_f)

        #对初始关联进行掩码
        MA = torch.masked_select(A, train_mask_tensor)
        #两个超图的重构
        reG = torch.masked_select(reconstructionG.t(),train_mask_tensor)
        #两个超图的重构
        reMD = torch.masked_select(reconstructionMD.t(), train_mask_tensor)


        #超图总的重构矩阵
        rec = torch.masked_select(recover.t(), train_mask_tensor)



        loss_c_m = contrastive_loss(mir_feature_2, mir_feature_1)
        loss_c_d = contrastive_loss(dis_feature_2, dis_feature_1)
        #对比损失
        loss_c =  loss_c_m + loss_c_d


        loss_r_h = F.binary_cross_entropy_with_logits(reG.t(), MA,pos_weight=pos_weight)  + F.binary_cross_entropy_with_logits(reMD.t(), MA,pos_weight=pos_weight) + F.binary_cross_entropy_with_logits(rec.t(), MA,pos_weight=pos_weight)



        loss = loss_r_h + loss_c

        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()




        auc_val = roc_auc_score(label, test_predict)
        aupr_val = average_precision_score(label, test_predict)



        print('Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.5f}'.format(loss.data.item()),
              'auc_val: {:.5f}'.format(auc_val),
              'aupr_val: {:.5f}'.format(aupr_val),
              )
        max_f1_score, threshold = f1_score_binary(torch.from_numpy(label).float(),torch.from_numpy(test_predict).float())
        print("//////////max_f1_score",max_f1_score)
        precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////precision:", precision)
        recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////recall:", recall)

        mcc = mcc_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////mcc:", mcc)
        accuracy = accuracy_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("//////////accuracy:", accuracy)
        #
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')



    # 在函数开始处初始化 fpr 和 tpr
    fpr, tpr = [], []
    if epoch == args.epochs - 1:
        auc1 = auc_val
        aupr1 = aupr_val
        recall1 = recall
        precision1 = precision
        f11 =  max_f1_score
        mcc1= mcc
        accuracy1 = accuracy


        print('auc_test: {:.5f}'.format(auc1),
              'aupr_test: {:.5f}'.format(aupr1),
              'precision_test: {:.5f}'.format(precision1),
              'recall_test: {:.5f}'.format(recall1),
              'f1_test: {:.5f}'.format(f11),
              'mcc_test: {:.5f}'.format(mcc1),
              'accuracy_test: {:.5f}'.format(accuracy1),

              )

    return auc1, aupr1, recall1, precision1, f11, mcc1, accuracy1



#circRNA-drug
MD = np.loadtxt("circRNAdrug/association.txt")
MM = np.loadtxt("circRNAdrug/integration_circRNA.txt")
DD = np.loadtxt("circRNAdrug/integration_drug.txt")

[row, col] = np.shape(MD)

indexn = np.argwhere(MD == 0)
Index_zeroRow = indexn[:, 0]
Index_zeroCol = indexn[:, 1]

indexp = np.argwhere(MD == 1)
Index_PositiveRow = indexp[:, 0]
Index_PositiveCol = indexp[:, 1]

totalassociation = np.size(Index_PositiveRow) #7694
fold = int(totalassociation / 5) #1538

zero_length = np.size(Index_zeroRow)#321601

seed = 66

n = 1
hidden1 = 512
hidden2 = 256
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=135, help='Number of epochs to train.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--cv_num', type=int, default=5, help='number of fold')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



#矩阵分解预处理

# 矩阵分解得到两个异构矩阵
GSM = G_similarity.compute_global_similarity_matrix(MM)
# 计算disease的全局相似性
GMM = G_similarity.compute_global_similarity_matrix(DD)
# # #计算局部图推理
# # # 计算circRNA的局部相似性
LSM = L_similarity.row_normalization(MM, 10)
# # 计算disease的 局部相似性
LMM = L_similarity.row_normalization(DD, 10)

a3 = np.hstack((LSM, MD))  # 将参数元组的元素数组按水平方向进行叠加
a4 = np.hstack((np.transpose(MD), LMM))  # 对矩阵b进行转置操作
H_1 = np.vstack((a3, a4))  # 将参数元组的元素数组按垂直方向进行叠加

a2 = np.hstack((GSM, MD))  # 将参数元组的元素数组按水平方向进行叠加
a5 = np.hstack((np.transpose(MD), GMM))  # 对矩阵b进行转置操作
H_2 = np.vstack((a2, a5))  # 将参数元组的元素数组按垂直方向进行叠加

L1 = run_MC(H_1)  # 全局

M_1 = L1[0:MM.shape[0], MM.shape[0]:L1.shape[1]]  # 把补充的关联矩阵原来A位置给取出来
L2 = run_MC(H_2)  # 局部
M_2 = L2[0:MM.shape[0], MM.shape[0]:L2.shape[1]]  # 把补充的关联矩阵原来A位置给取出来


AAuc_list1 = []
f1_score_list1 = []
precision_list1 = []
recall_list1 = []
aupr_list1 = []

auc_sum = 0
aupr_sum = 0
AUC = 0
AUPR = 0
recall_sum = 0
precision_sum = 0
f1_sum = 0
mcc_sum = 0
acc_sum = 0
accuracy_sum = 0
#绘制ROC曲线
tprs=[]
aucs=[]
all_fpr, all_tpr, all_auc = [], [], []
mean_fpr=np.linspace(0,1,100)
#绘制PR曲线
all_precision, all_recall, all_aupr = [], [], []
aupr_sum, time = 0, 0
for time in range(1,n+1):
    Auc_per = []
    f1_score_per = []
    precision_per = []
    recall_per = []
    aupr_per = []
    p = np.random.permutation(totalassociation)
    # print(p)

    AUC = 0
    aupr = 0
    rec = 0
    pre = 0
    f1 = 0
    mcc = 0
    accuracy = 0
    # 5-折
    for f in range(1, args.cv_num + 1):
        print("cross_validation:", '%01d' % (f))

        if f == args.cv_num:
            testset = p[((f - 1) * fold): totalassociation + 1]
        else:
            testset = p[((f - 1) * fold): f * fold]

        all_f = np.random.permutation(np.size(Index_zeroRow))

        test_p = list(testset)

        test_f = all_f[0:len(test_p)]

        difference_set_f = list(set(all_f).difference(set(test_f)))
        train_f = difference_set_f

        train_p = list(set(p).difference(set(testset)))
        #全局
        X_1 = copy.deepcopy(M_1)  # 深拷贝 M_1
        Xn_1 = copy.deepcopy(X_1)  # 初始化 Xn_1
        #局部
        X_2 = copy.deepcopy(M_2)  # 深拷贝 M_2
        Xn_2 = copy.deepcopy(X_2)  # 初始化 Xn_2

        X = copy.deepcopy(MD)
        Xn = copy.deepcopy(X)
        zero_index = []
        for ii in range(len(train_f)):
            zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])

        true_list = multiarray.zeros((len(test_p) + len(test_f), 1))
        for ii in range(len(test_p)):
            Xn[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
            Xn_1[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
            Xn_2[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
            true_list[ii, 0] = 1
        train_mask = np.ones(shape=Xn_2.shape)
        for ii in range(len(test_p)):
            train_mask[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
            train_mask[Index_zeroRow[test_f[ii]], Index_zeroCol[test_f[ii]]] = 0
        train_mask_tensor = torch.from_numpy(train_mask).to(torch.bool)



        label = true_list
        A1 = copy.deepcopy(Xn_1)
        A1T = A1.T
        A2 = copy.deepcopy(Xn_2)
        A2T = A2.T




        HHMG = construct_H_with_KNN(A1)  #全局circRNA-drug关联
        HMG = generate_G_from_H(HHMG)
        HMG = HMG.double()

        HHDG = construct_H_with_KNN(A1T)  #全局drug-circRNA
        HDG = generate_G_from_H(HHDG)
        HDG = HDG.double()

        #局部超图
        HHMD = construct_H_with_KNN(A2)
        HMD = generate_G_from_H(HHMD) #局部 circRNA-drug关联
        HMD = HMD.double()

        HHDM = construct_H_with_KNN(A2T)  # 局部drug-circRNA关联
        HDM = generate_G_from_H(HHDM)
        HDM = HDM.double()

        rr = MD.shape[0]
        cc = MD.shape[1]
        mir_feat = torch.eye(rr)
        dis_feat = torch.eye(cc)
        parameters = [cc, rr]

        model = Model()
        optimizer2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        A = torch.from_numpy(A1)
        AT = torch.from_numpy(A1T)
        XX = copy.deepcopy(Xn)
        XX = torch.from_numpy(XX)
        XXN = A
        pos_weight = float(XXN.shape[0] * XXN.shape[1] - XXN.sum()) / XXN.sum()

        mir_feat, dis_feat = Variable(mir_feat), Variable(dis_feat)

        if args.cuda:
            model.cuda()

            XX = XX.cuda()

            A = A.cuda()
            AT = AT.cuda()

            HMG = HMG.cuda()
            HDG = HDG.cuda()

            HMD = HMD.cuda()
            HDM = HDM.cuda()


            mir_feat = mir_feat.cuda()
            dis_feat = dis_feat.cuda()

            train_mask_tensor = train_mask_tensor.cuda()

        for epoch in range(args.epochs):

            auc1, aupr1, recall1, precision1, f11, mcc1, accuracy1 = train(epoch)
            AUC = AUC + auc1
            aupr = aupr + aupr1
            rec = rec + recall1
            pre = pre + precision1
            f1 = f1 + f11
            mcc = mcc + mcc1
            accuracy = accuracy + accuracy1
        print(auc)
        if f == args.cv_num:
            print('AUC: {:.5f}'.format(AUC/args.cv_num),
                  'aupr: {:.5f}'.format(aupr/args.cv_num),
                  'precision: {:.5f}'.format(pre / args.cv_num),
                  'recall: {:.5f}'.format(rec / args.cv_num),
                  'f1_score: {:.5f}'.format(f1 / args.cv_num),
                  'mcc_score: {:.5f}'.format(mcc / args.cv_num),
                  'accuracy_score: {:.5f}'.format(accuracy / args.cv_num),
                      )

            a = AUC/args.cv_num
            b = aupr/args.cv_num
            c = pre / args.cv_num
            d = rec / args.cv_num
            e = f1 / args.cv_num
            f = mcc / args.cv_num
            g = accuracy / args.cv_num






    auc_sum = auc_sum + a
    aupr_sum = aupr_sum + b
    precision_sum= precision_sum +c
    recall_sum = recall_sum + d
    f1_sum = f1_sum + e
    mcc_sum = mcc_sum + f
    accuracy_sum = accuracy_sum + g


print(
      'auc_ave: {:.5f}'.format(auc_sum/n),
      'aupr_ave: {:.5f}'.format(aupr_sum/n),
      'precision_ave: {:.5f}'.format(precision_sum / n),
      'recall_ave: {:.5f}'.format(recall_sum / n),
      'f1_ave: {:.5f}'.format(f1_sum / n),
      'mcc_ave: {:.5f}'.format(mcc_sum / n),
      'accuracy_ave: {:.5f}'.format(accuracy_sum / n),
                      )














