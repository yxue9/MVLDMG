import numpy as np
import pandas as pd

def compute_local_density(W):
    """
    计算每个节点的局部密度。
    """
    density = 1 / np.mean(W, axis=1)
    # 归一化密度值到[0, 1]
    density = (density - np.min(density)) / (np.max(density) - np.min(density))
    return density

def FindDominantSetAdaptive(W, density, K_min, K_max):
    """
    根据局部密度自适应地选择最近的邻居。
    """
    m, n = W.shape
    DS = np.zeros((m, n))
    for i in range(m):
        # 基于密度动态确定K值
        K_dynamic = int(K_min + density[i] * (K_max - K_min))
        indices = np.argpartition(W[i, :], -K_dynamic)[-K_dynamic:]
        DS[i, indices] = W[i, indices]

    # 归一化
    B = np.sum(DS, axis=1, keepdims=True)
    DS = DS / B
    return DS

def normalized(W, ALPHA):
    m, n = W.shape
    DS = W - np.diag(np.diag(W))  # 去除对角线元素
    B = np.sum(DS, axis=1, keepdims=True)  # 沿着列求和，保持二维形状

    # 为了避免维度不匹配的问题，可以直接操作B的一维形式，然后再对DS应用结果
    B = B.squeeze()  # 将B转换为一维数组，以匹配DS的布尔索引
    p = B != 0  # 创建布尔索引数组
    DS[p, :] = DS[p, :] / B[p, np.newaxis]  # 使用np.newaxis保持正确的广播行为

    DS = DS + ALPHA * np.identity(m)  # 添加ALPHA倍的单位矩阵
    return (DS + DS.T) / 2  # 确保结果是对称的


def compute_entropy(W):
    """
    计算给定相似性矩阵每个节点的熵，并返回节点平均熵。
    """
    # 保证矩阵W的每一行和为1
    W_normalized = W / W.sum(axis=1, keepdims=True)
    # 计算熵
    entropy = -np.nansum(W_normalized * np.log(W_normalized + 1e-12), axis=1)  # 防止对0取对数
    # 返回平均熵
    return np.mean(entropy)

def compute_weights(Wall):
    """
    计算每个网络的权重，基于每个网络的节点平均熵。
    """
    entropies = np.array([compute_entropy(W) for W in Wall])
    weights = entropies / entropies.sum()  # 根据熵值分配权重
    return weights

def SNF(Wall, K_min, K_max, t, ALPHA=1):
    """
    执行带有自适应邻居选择和基于熵的加权融合策略的相似性网络融合算法。
    """
    C = len(Wall)
    weights = compute_weights(Wall)  # 计算每个网络的权重

    densities = [compute_local_density(W) for W in Wall]
    newW = [FindDominantSetAdaptive(Wall[i], densities[i], K_min, K_max) for i in range(C)]

    for iteration in range(t):
        Wsum = sum(weights[i] * normalized(W, ALPHA) for i, W in enumerate(Wall))
        for i in range(C):
            Wall[i] = np.dot(np.dot(newW[i], Wsum - weights[i] * normalized(Wall[i], ALPHA)), newW[i].T) / (np.sum(weights) - weights[i])

    Wsum = sum(weights[i] * normalized(W, ALPHA) for i, W in enumerate(Wall))
    W = normalized(Wsum / np.sum(weights), ALPHA)
    return W

def load_data(paths):
    """
    从给定的文件路径中加载数据。
    """
    return [pd.read_csv(path, header=None).values for path in paths]

# 文件路径
paths = [
    '../data2/metabolite_fingerprint_similarity.csv',
    '../data2/ms_matrix_GIP.csv',
    '../data2/metabolites_similarity.csv'
]

# 加载数据
Wall = load_data(paths)

# 参数设置
K_min = 5   # 最小邻居数
K_max = 25  # 最大邻居数，可根据实际情况调整
t = 20
ALPHA = 1

# 应用带有自适应邻居选择的SNF算法
W_integrated = SNF(Wall, K_min, K_max, t, ALPHA)

# 转换为DataFrame
df = pd.DataFrame(W_integrated)

# 保存到CSV文件
save_path = '../data2/MS_integrated_matrix02.csv'
df.to_csv(save_path, index=False, header=False)
