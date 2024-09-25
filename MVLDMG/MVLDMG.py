import dgl
import pandas as pd
import torch
import torch.nn as nn
from GAT_layer_v2 import GATv2Conv

torch.cuda.empty_cache()

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
        data_dict={
            ('met_disease', 'interaction', 'met_disease'): Heterogeneous.nonzero()},
        num_nodes_dict={
            'met_disease': 1678
        })
    return g


def test_features_choose(rel_adj_mat, features_embedding):
    rna_nums, dis_nums = rel_adj_mat.size()[0], rel_adj_mat.size()[1]
    features_embedding_rna = features_embedding[0:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:features_embedding.size()[0], :]
    test_features_input, test_lable = [], []

    for i in range(rna_nums):
        for j in range(dis_nums):
            test_features_input.append((features_embedding_rna[i, :] * features_embedding_dis[j, :]).unsqueeze(0))
            test_lable.append(rel_adj_mat[i, j])

    test_features_input = torch.cat(test_features_input, dim=0)
    test_lable = torch.FloatTensor(np.array(test_lable)).unsqueeze(1)
    return test_features_input.to(device), test_lable.to(device)

class FeatureTransform(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureTransform, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),  # 可以添加更多的层进行更复杂的变换
        )

    def forward(self, *embeddings):
        # 特征拼接
        concatenated_features = torch.cat(embeddings, dim=1)
        # 特征变换
        transformed_features = self.fc(concatenated_features)
        return transformed_features


class GATCNNMF(nn.Module):
    torch.cuda.empty_cache()
    def __init__(self, in_metfeat_size, in_disfeat_size, outfeature_size, heads, drop_rate, negative_slope,
                 features_embedding_size, negative_times):
        super(GATCNNMF, self).__init__()
        self.in_circfeat_size = in_metfeat_size
        self.in_disfeat_size = in_disfeat_size
        self.outfeature_size = outfeature_size
        self.heads = heads
        self.drop_rate = drop_rate
        self.negative_slope = negative_slope
        self.features_embedding_size = features_embedding_size
        self.negative_times = negative_times
        # 其余初始化保持不变
        # 图注意层（多头）
        self.att_layer = GATv2Conv(self.outfeature_size, self.outfeature_size, self.heads, self.drop_rate,
                                   self.drop_rate, self.negative_slope)

        # W_rna和W_dis为两个权重矩阵（也称为投影算子），
        # 用于将代谢物节点和疾病节点的原始特征投影到一个共同的特征空间中，以便后续处理
        # 定义投影算子
        self.W_rna = nn.Parameter(torch.zeros(size=(self.in_circfeat_size, self.outfeature_size)))
        self.W_dis = nn.Parameter(torch.zeros(size=(self.in_disfeat_size, self.outfeature_size)))
        # 初始化投影算子
        nn.init.xavier_uniform_(self.W_rna.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_dis.data, gain=1.414)

        # 定义卷积层的权重初始化函数
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        # W_rna和W_dis为两个权重矩阵（也称为投影算子），
        # 用于将代谢物节点和疾病节点的原始特征投影到一个共同的特征空间中，以便后续处理
        # 二维卷积层搭建
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
        # 初始化
        self.cnn_layer1.apply(init_weights)
        self.cnn_layer4.apply(init_weights)
        self.cnn_layer16.apply(init_weights)
        self.cnn_layer32.apply(init_weights)


    # 前向传播(forward)
    def forward(self, graph, met_feature_tensor, dis_feature_tensor, rel_matrix, train_model):
        # 通过投影算子将代谢物和疾病特征映射到高维空间，然后将映射后的特征通过GAT层进行图注意力处理。
        met_met_f = met_feature_tensor.mm(self.W_rna)
        dis_dis_f = dis_feature_tensor.mm(self.W_dis)

        N = met_met_f.size()[0] + dis_dis_f.size()[0]  # 异构网络的节点个数,num_circ+num_dis

        # 异构网络节点的特征表达矩阵
        h_c_d_feature = torch.cat((met_met_f, dis_dis_f), dim=0)

        # 特征聚合
        res = self.att_layer(graph, h_c_d_feature)  # size:[nodes,heads,outfeature_size]

        x = res.view(N, 1, self.heads, -1)

        cnn_embedding1 = self.cnn_layer1(x).view(N, -1)
        cnn_embedding4 = self.cnn_layer4(x).view(N, -1)
        cnn_embedding16 = self.cnn_layer16(x).view(N, -1)
        cnn_embedding32 = self.cnn_layer32(x).view(N, -1)

        # 特征变换
        cnn_outputs = cnn_embedding1.size(1) + cnn_embedding4.size(1) + cnn_embedding16.size(1) + cnn_embedding32.size(
            1)
        transform_module = FeatureTransform(cnn_outputs, output_size=cnn_outputs)  # 假定输出大小为512
        transformed_features = transform_module(cnn_embedding1, cnn_embedding4, cnn_embedding16, cnn_embedding32)
        print('features_embedding_size:', transformed_features.size()[1])

        test_features_inputs, test_lable = test_features_choose(rel_matrix, transformed_features)
        return test_features_inputs, test_lable
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    m_d = np.loadtxt('./data/matrix.csv', delimiter=',')
    m_m_sim = np.loadtxt('./data/all_MS_file.csv', delimiter=',')
    d_d_sim = np.loadtxt('./data/all_DS_file.csv', delimiter=',')

    graph = build_heterograph(m_d, m_m_sim, d_d_sim)
    graph = graph.to(device)
    # 转换数据类型并移动到正确的设备
    m_m_tensor = torch.from_numpy(m_m_sim).to(torch.float32)
    d_d_tensor = torch.from_numpy(d_d_sim).to(torch.float32)
    m_d_tensor = torch.from_numpy(m_d).to(torch.float32)
    print(graph,m_m_tensor, d_d_tensor, m_d_tensor)


    model = GATCNNMF(1436, 242, 128, 4, 0.1, 0.3, 2048, 2)
    model = model
    out = model(graph, m_m_tensor, d_d_tensor, m_d_tensor, True)
    print(out)
    train_features, train_labels = model(graph, m_m_tensor, d_d_tensor, m_d_tensor, True)
    print(train_features, train_labels)
    # 如果需要保存为numpy数组格式
    train_features_numpy = train_features.cpu().detach().numpy()
    # 使用Pandas创建DataFrame
    df_features = pd.DataFrame(train_features_numpy)

    # 保存DataFrame为CSV文件
    df_features.to_csv('train_features3.csv', index=False, header=False)
    train_labels_numpy = train_labels.cpu().detach().numpy()

    # 使用Pandas创建DataFrame
    df_labels = pd.DataFrame(train_labels_numpy)

    # 保存DataFrame为CSV文件
    df_labels.to_csv('train_labels3.csv', index=False, header=False)