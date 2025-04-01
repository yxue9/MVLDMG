import dgl
import pandas as pd
import torch
import torch.nn as nn
from GAT_layer_v2 import GATv2Conv
import tensorly as tl
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_heterograph(met_disease_matrix, metSimi, disSimi):
    # for met->adj
    matAdj_met = np.where(metSimi > 0.5, 1, 0)
    # for disease->adj
    matAdj_dis = np.where(disSimi > 0.5, 1, 0)
    # Heterogeneous adjacency matrix
    h_adjmat_1 = np.hstack((matAdj_met, met_disease_matrix))
    h_adjmat_2 = np.hstack((met_disease_matrix.transpose(), matAdj_dis))
    Heterogeneous = np.vstack((h_adjmat_1, h_adjmat_2))
    # heterograph
    g = dgl.heterograph(
        data_dict={('met_disease', 'interaction', 'met_disease'): Heterogeneous.nonzero()},
        num_nodes_dict={'met_disease': 1678})
    return g


def test_features_choose(rel_adj_mat, features_embedding):
    # 展平特征
    flattened_features = features_embedding.view(features_embedding.size(0), -1)
    # 标签展平
    flattened_labels = rel_adj_mat.view(-1, 1)
    return flattened_features.to(device), flattened_labels.to(device)


# 修改后的张量交互模块：使用两个秩值，并手动实现低秩交互
class TensorInteraction(nn.Module):
    def __init__(self, in_dim, ranks=(16, 8)):
        """
        Args:
            in_dim: 输入特征维度
            ranks: 二元组 (r1, r2)，其中 r1 * r2 为最终交互特征维度（例如128）
        """
        super(TensorInteraction, self).__init__()
        self.r1, self.r2 = ranks
        # 投影矩阵：分别将代谢物和疾病特征映射到低秩空间
        self.factor1 = nn.Parameter(torch.randn(in_dim, self.r1))
        self.factor2 = nn.Parameter(torch.randn(in_dim, self.r2))
        # 核心张量，用于调制交互，形状为 [r1, r2]
        self.core = nn.Parameter(torch.randn(self.r1, self.r2))

    def forward(self, met_emb, dis_emb):
        """
        Args:
            met_emb: [num_mets, in_dim] 代谢物特征
            dis_emb: [num_dis, in_dim] 疾病特征
        Returns:
            交互特征张量，形状为 [num_mets, num_dis, r1, r2]
        """
        # 分别投影到低秩空间
        met_proj = torch.matmul(met_emb, self.factor1)  # [num_mets, r1]
        dis_proj = torch.matmul(dis_emb, self.factor2)  # [num_dis, r2]
        # 计算外积后与核心张量按元素相乘，得到交互张量
        # 交互结果形状为 [num_mets, num_dis, r1, r2]
        interaction = torch.einsum('mi,nj,ij->mnij', met_proj, dis_proj, self.core)
        return interaction


# 注意力机制模块保持不变
class Attention(nn.Module):
    def __init__(self, feature_dim):
        """
        Args:
            feature_dim: 每个尺度的特征维度（例如128）
        """
        super(Attention, self).__init__()
        self.fc = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, x):
        """
        x: shape [num_mets, num_dis, scales, feature_dim]
        """
        # 对每个尺度的 128 维特征计算一个标量评分
        scores = self.fc(x)  # shape: [num_mets, num_dis, scales, 1]
        weights = torch.softmax(scores, dim=2)  # 在尺度维度上做 softmax
        # 用注意力权重对各尺度特征加权求和，消除尺度维度
        aggregated = torch.sum(weights * x, dim=2)  # shape: [num_mets, num_dis, feature_dim]
        return aggregated


class GATCNNMF(nn.Module):
    torch.cuda.empty_cache()

    def __init__(self, in_metfeat_size, in_disfeat_size, outfeature_size, heads, drop_rate, negative_slope,
                 features_embedding_size, negative_times):
        super(GATCNNMF, self).__init__()
        self.in_circfeat_size = in_metfeat_size
        self.in_disfeat_size = in_disfeat_size
        self.outfeature_size = outfeature_size  # 128
        self.heads = heads
        self.drop_rate = drop_rate
        self.negative_slope = negative_slope
        self.features_embedding_size = features_embedding_size
        self.negative_times = negative_times

        # 图注意层（多头）
        self.att_layer = GATv2Conv(self.outfeature_size, self.outfeature_size, self.heads,
                                   self.drop_rate, self.drop_rate, self.negative_slope)

        # 投影算子，将原始特征映射到公共特征空间（128维）
        self.W_rna = nn.Parameter(torch.zeros(size=(self.in_circfeat_size, self.outfeature_size)))
        self.W_dis = nn.Parameter(torch.zeros(size=(self.in_disfeat_size, self.outfeature_size)))
        nn.init.xavier_uniform_(self.W_rna.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_dis.data, gain=1.414)

        # 张量交互模块，使用两个秩值：例如[16, 8]
        self.tensor_interaction = TensorInteraction(self.outfeature_size, ranks=[16, 8])

        # 注意力机制模块（4个不同尺度的特征）
        self.attention = Attention(128)

        # 定义卷积层的权重初始化函数
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight)

        # 四个不同尺度的二维卷积层搭建
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 1), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 4), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer16 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 16), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer32 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(self.heads, 32), padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)

        self.cnn_layer1.apply(init_weights)
        self.cnn_layer4.apply(init_weights)
        self.cnn_layer16.apply(init_weights)
        self.cnn_layer32.apply(init_weights)

        # 添加全连接层，将各个卷积分支的输出投影到128维
        self.fc_cnn_layer1 = nn.Linear(768, 128)
        self.fc_cnn_layer4 = nn.Linear(750, 128)
        self.fc_cnn_layer16 = nn.Linear(678, 128)
        self.fc_cnn_layer32 = nn.Linear(582, 128)

        self.fc_cnn_layer1.apply(init_weights)
        self.fc_cnn_layer4.apply(init_weights)
        self.fc_cnn_layer16.apply(init_weights)
        self.fc_cnn_layer32.apply(init_weights)

    def forward(self, graph, met_feature_tensor, dis_feature_tensor, rel_matrix, train_model):
        # 投影到高维空间后进行图注意力处理
        met_met_f = met_feature_tensor.mm(self.W_rna)
        dis_dis_f = dis_feature_tensor.mm(self.W_dis)
        N = met_met_f.size(0) + dis_dis_f.size(0)
        h_c_d_feature = torch.cat((met_met_f, dis_dis_f), dim=0)
        res = self.att_layer(graph, h_c_d_feature)  # [N, heads, outfeature_size]
        x = res.view(N, 1, self.heads, -1)

        # 分别通过四个不同尺度的卷积层
        cnn_embedding1 = self.cnn_layer1(x).view(N, -1)
        cnn_embedding4 = self.cnn_layer4(x).view(N, -1)
        cnn_embedding16 = self.cnn_layer16(x).view(N, -1)
        cnn_embedding32 = self.cnn_layer32(x).view(N, -1)

        # 使用全连接层将卷积输出投影到128维
        cnn_embedding1 = self.fc_cnn_layer1(cnn_embedding1)
        cnn_embedding4 = self.fc_cnn_layer4(cnn_embedding4)
        cnn_embedding16 = self.fc_cnn_layer16(cnn_embedding16)
        cnn_embedding32 = self.fc_cnn_layer32(cnn_embedding32)

        print(f"cnn_embedding1 size: {cnn_embedding1.size()}")
        print(f"cnn_embedding4 size: {cnn_embedding4.size()}")
        print(f"cnn_embedding16 size: {cnn_embedding16.size()}")
        print(f"cnn_embedding32 size: {cnn_embedding32.size()}")

        num_mets = met_feature_tensor.size(0)
        num_dis = dis_feature_tensor.size(0)

        met_emb1 = cnn_embedding1[:num_mets]
        dis_emb1 = cnn_embedding1[num_mets:]
        met_emb4 = cnn_embedding4[:num_mets]
        dis_emb4 = cnn_embedding4[num_mets:]
        met_emb16 = cnn_embedding16[:num_mets]
        dis_emb16 = cnn_embedding16[num_mets:]
        met_emb32 = cnn_embedding32[:num_mets]
        dis_emb32 = cnn_embedding32[num_mets:]

        # 张量交互与分解（每个输出形状为 [num_mets, num_dis, 16, 8]）
        interaction_features1 = self.tensor_interaction(met_emb1, dis_emb1)
        interaction_features4 = self.tensor_interaction(met_emb4, dis_emb4)
        interaction_features16 = self.tensor_interaction(met_emb16, dis_emb16)
        interaction_features32 = self.tensor_interaction(met_emb32, dis_emb32)

        # 将每个尺度的交互特征展平：形状变为 [num_mets, num_dis, 128]
        interaction_features1 = interaction_features1.view(num_mets, num_dis, -1)
        interaction_features4 = interaction_features4.view(num_mets, num_dis, -1)
        interaction_features16 = interaction_features16.view(num_mets, num_dis, -1)
        interaction_features32 = interaction_features32.view(num_mets, num_dis, -1)

        # 在新的尺度维度上拼接，得到形状 [num_mets, num_dis, 4, 128]
        all_interaction_features = torch.stack(
            (interaction_features1, interaction_features4, interaction_features16, interaction_features32),
            dim=2
        )

        # 用注意力聚合：输出形状为 [num_mets, num_dis, 128]
        aggregated_features = self.attention(all_interaction_features)

        # 展平为 [num_mets * num_dis, 128]
        flattened = aggregated_features.view(-1, 128)

        test_features_inputs, test_label = test_features_choose(rel_matrix, flattened)
        return test_features_inputs, test_label


if __name__ == '__main__':
    m_d = np.loadtxt('data/matrix.csv', delimiter=',')
    m_m_sim = np.loadtxt('data/all_MS_file.csv', delimiter=',')
    d_d_sim = np.loadtxt('data/all_DS_file.csv', delimiter=',')
    graph = build_heterograph(m_d, m_m_sim, d_d_sim)
    graph = graph.to(device)
    m_m_tensor = torch.from_numpy(m_m_sim).to(torch.float32).to(device)
    d_d_tensor = torch.from_numpy(d_d_sim).to(torch.float32).to(device)
    m_d_tensor = torch.from_numpy(m_d).to(torch.float32).to(device)
    print(graph, m_m_tensor, d_d_tensor, m_d_tensor)

    model = GATCNNMF(1436, 242, 128, 4, 0.1, 0.3, 2048, 2)
    model = model.to(device)
    out = model(graph, m_m_tensor, d_d_tensor, m_d_tensor, True)
    print(out)
    train_features, train_labels = model(graph, m_m_tensor, d_d_tensor, m_d_tensor, True)
    print(train_features, train_labels)
    # 保存为 CSV 文件
    train_features_numpy = train_features.cpu().detach().numpy()
    df_features = pd.DataFrame(train_features_numpy)
    df_features.to_csv('features1.csv', index=False, header=False)
    train_labels_numpy = train_labels.cpu().detach().numpy()
    df_labels = pd.DataFrame(train_labels_numpy)
    df_labels.to_csv('labels1.csv', index=False, header=False)