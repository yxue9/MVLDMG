import numpy as np


def FindDominantSet(W, K):
    m, n = W.shape
    DS = np.zeros((m, n))
    for i in range(m):
        index = np.argsort(W[i, :])[-K:]  # get the closest K neighbors
        DS[i, index] = W[i, index]  # keep only the nearest neighbors

    # normalize by sum
    B = np.sum(DS, axis=1)
    B = B.reshape(len(B), 1)
    DS = DS / B
    return DS


def normalized(W, ALPHA):
    m, n = W.shape
    DS = W - np.diag(np.diag(W))
    B = np.sum(DS, 1)
    p = B != 0
    B = B.reshape(len(B), 1)
    DS[p, :] = DS[p, :] / B[p, :]
    DS = DS + ALPHA * np.identity(m)
    return (DS + np.transpose(DS)) / 2


def SNF(Wall, K, t, ALPHA=1):
    C = len(Wall)
    m, n = Wall[0].shape

    for i in range(C):
        B = np.sum(Wall[i], axis=1)
        len_b = len(B)
        B = B.reshape(len_b, 1)
        Wall[i] = Wall[i] / B
        Wall[i] = (Wall[i] + np.transpose(Wall[i])) / 2

    newW = []

    for i in range(C):
        newW.append(FindDominantSet(Wall[i], K))

    for iteration in range(t):
        Wsum = np.zeros((m, n))
        for i in range(C):
            Wall[i] = normalized(Wall[i], ALPHA)
            Wsum += Wall[i]
        for i in range(C):
            Wall[i] = np.dot(np.dot(newW[i], (Wsum - Wall[i])), np.transpose(newW[i])) / (C - 1)

    Wsum = np.zeros((m, n))
    for i in range(C):
        Wall[i] = normalized(Wall[i], ALPHA)
        Wsum += Wall[i]

    W = Wsum / C
    W = normalized(W, ALPHA)
    return W

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 假设您的CSV文件路径如下
path1 = '../data2/updated_metabolite_fingerprint_similarity.csv'
path2 = 'E:\python\GATCL2CD-main\GATCL2CD\data2\ms_matrix_GIP.csv'
path3 = '../data2/metabolites_jaccard_similarity.csv'

# 读取数据
data1 = pd.read_csv(path1, header=None)
data2 = pd.read_csv(path2, header=None)
data3 = pd.read_csv(path3, header=None)

# 转换为NumPy数组以便后续处理
W1 = data1.values
W2 = data2.values
W3 = data3.values
# 将相似性矩阵放入列表中
Wall = [W1, W2, W3]

# 选择参数
K = 20  # 选择最近的K个邻居
t = 20  # 迭代次数
ALPHA = 1  # ALPHA参数

# 应用SNF算法
W_integrated = SNF(Wall, K, t, ALPHA)

import pandas as pd

# 转换为DataFrame
# 这里假设没有为行和列指定特定的标签，所以我们只用数字索引
df = pd.DataFrame(W_integrated)

# 指定保存文件的路径
save_path = '../data2/MS_ntegrated_matrix.csv'

# 保存DataFrame到CSV
df.to_csv(save_path, index=False, header=False)
