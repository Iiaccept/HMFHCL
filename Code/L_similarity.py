import numpy as np





# 行归一化(邻居归一化）
def row_normalization(S, k):
    #计算邻居集合N
    N = {}
    for i in range(S.shape[0]):
        neighbors = np.argsort(S[i])[::-1][:k]  # 获取相似性最高的k个邻居的索引
        N[i] = list(neighbors)
    #计算行归一化
    result = np.zeros(S.shape)
    for i in range(S.shape[0]):
        num = 0
        denominator=np.sum(S[i, N[i]]) # 分母计算的是邻居的和

       # denominator= np.sum(S[i])   #分母计算的是一整行
        for j in range(S.shape[1]):
            if j in N[i]:
                if denominator != 0:
                    num = num + 1

                    result[i, j] =  S[i, j] / denominator
                else:
                    result[i, j] = 0  # Set to 0 or handle differently based on your requirements
            else:
                result[i, j] = 0
    return result

