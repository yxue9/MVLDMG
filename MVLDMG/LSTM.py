import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, auc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import random
import torch.nn.functional as F



def set_seed(seed):
    """设置所有随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(32)

# 加载特征和标签
features_path = 'data-end/balanced_train_features.csv'
labels_path = 'data-end/balanced_train_labels.csv'


features = pd.read_csv(features_path, header=None)
labels = pd.read_csv(labels_path, header=None)

# 数据预处理
scaler = StandardScaler()
features = scaler.fit_transform(features)

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs 的形状: (batch_size, seq_length, hidden_dim)
        energy = self.projection(encoder_outputs)  # (batch_size, seq_length, 1)
        weights = F.softmax(energy.squeeze(-1), dim=1)  # (batch_size, seq_length)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, hidden_dim)
        return outputs, weights

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # 使用双向LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)

        # Dropout层
        self.dropout = nn.Dropout(dropout_prob)

        # 自注意力层
        self.attention = SelfAttention(hidden_dim * 2)  # 注意：因为是双向的，所以是 hidden_dim * 2

        # 由于使用了双向LSTM和自注意力，我们将注意力层的输出用于全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out)

        # 应用自注意力机制
        attn_out, _ = self.attention(out)

        # 使用注意力机制的输出
        out = self.fc(attn_out)
        return out

hidden_dim = 128
layer_dim = 2
output_dim = 1

# 训练模型
num_epochs = 20
kf = KFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.metrics import precision_recall_curve

all_precision = []
all_recall = []
all_aupr = []  # 存储所有fold的AUPR值

# 存储所有FPR和TPR
all_fpr = []
all_tpr = []

# 转换整个数据集为Tensor
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels.values, dtype=torch.float32).view(-1, 1)  # 确保labels是正确的形状
# 初始化用于存储每一折AUC的列表
fold_auc_scores = []

all_auc_scores = []  # 存储所有fold的最后一次AUC值
# 五折交叉验证
fold = 0
for train_index, test_index in kf.split(features_tensor):
    fold += 1
    print(f"Fold {fold}")

    # 分割数据
    X_train, X_test = features_tensor[train_index], features_tensor[test_index]
    y_train, y_test = labels_tensor[train_index], labels_tensor[test_index]

    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    input_dim = X_train.size(1)  # 根据你的特征数量调整
    # 初始化模型、损失函数和优化器
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)  # 增加序列长度维度
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # 评估阶段
        val_losses = []

        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                probabilities = torch.sigmoid(outputs).squeeze().tolist()
                y_pred.extend(probabilities)
                y_true.extend(labels.squeeze().tolist())
        fold_auc = roc_auc_score(y_true, y_pred)
        fold_auc_scores.append(fold_auc)
        print(
            f'Epoch {epoch + 1:03}: 训练损失: {np.mean(train_losses):.4f}, 验证损失: {np.mean(val_losses):.4f}, 验证AUC: {fold_auc:.4f}')

    model.eval()  # 确保在评估模式下使用模型

# 假设features已经是预处理后的numpy数组
features_tensor = torch.tensor(features, dtype=torch.float32)
dataset = TensorDataset(features_tensor)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)  # 不需要打乱数据

all_predictions = []

with torch.no_grad():
    for inputs in data_loader:
        inputs = inputs[0].unsqueeze(1)  # 确保输入符合模型的期望形状
        outputs = model(inputs)
        probabilities = torch.sigmoid(outputs).squeeze().tolist()  # 获取概率值
        all_predictions.extend(probabilities)

all_predictions = np.array(all_predictions)
reshaped_predictions = all_predictions.reshape(1436, 242)

np.savetxt("predicted_values.csv", reshaped_predictions, delimiter=",")

