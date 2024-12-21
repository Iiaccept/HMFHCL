import numpy as np

def compute_global_similarity_matrix(M_bar, alpha=0.8):
    """
    计算全局相似性矩阵。

    参数:
    - M_bar (np.ndarray): miRNA 相似性矩阵，要求是 n x n 的矩阵。
    - alpha (float): 阻尼系数，默认为 0.85。

    返回:
    - global_similarity_matrix (np.ndarray): 归一化后的全局相似性矩阵。
    """
    n = M_bar.shape[0]  # 假设 M_bar 是 n x n 的矩阵

    # 构建单位矩阵 I
    I = np.eye(n)

    # 对 M_bar 进行列归一化
    column_sums = M_bar.sum(axis=0)
    M_bar_normalized = M_bar / column_sums

    # 计算 (I - alpha * M_bar_normalized) 的逆矩阵
    inv_matrix = np.linalg.inv(I - alpha * M_bar_normalized)

    # 初始化全局相似性矩阵
    global_similarity_matrix = np.zeros((n, n))

    # 对每个 miRNA 计算全局相似性
    for i in range(n):
        # 构建二值向量 m 对应当前索引的 miRNA
        m = np.zeros((n, 1))
        m[i] = 1

        # 计算该 miRNA 的全局相似性向量 m_tilde
        m_tilde = (1 - alpha) * inv_matrix.dot(m)

        # 将结果存储到全局相似性矩阵的第 i 列
        global_similarity_matrix[:, i] = m_tilde.flatten()

    return global_similarity_matrix


