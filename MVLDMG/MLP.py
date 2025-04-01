import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, precision_score, recall_score,
                             roc_curve, precision_recall_curve)
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
features_path = '../data-end/balanced_features1.csv'
labels_path = '../data-end/balanced_labels1.csv'
# 加载数据

features = pd.read_csv(features_path, header=None).values.astype(np.float32)
labels = pd.read_csv(labels_path, header=None).values.flatten().astype(np.float32)

# 归一化（示例：Min-Max 归一化）
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# 定义带有 BatchNorm 的 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.5):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# 超参数设置
num_epochs = 40
batch_size = 128
learning_rate = 0.001
n_splits = 5
patience = 10  # 提前停止的耐心轮数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_auc = []
all_aupr = []
all_accuracy = []
all_f1 = []
all_precision = []
all_recall = []

fold_auc_curves = []
fold_aupr_curves = []

fold = 0
for train_index, test_index in skf.split(features, labels):
    fold += 1
    print(f"Fold {fold}")
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    X_train_tensor = torch.from_numpy(X_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).unsqueeze(1).to(device)

    # 构建数据集与 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MLP(input_dim=features.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        # 验证（这里简单使用训练集loss作为验证依据，可替换为独立验证集）
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_train_tensor)
            val_loss = criterion(val_outputs, y_train_tensor).item()

        scheduler.step(val_loss)

        # 提前停止逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # 可打印训练日志
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 在测试集上评估
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        y_true = y_test_tensor.cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_true, probs)
    aupr = average_precision_score(y_true, probs)
    accuracy = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)

    all_auc.append(auc)
    all_aupr.append(aupr)
    all_accuracy.append(accuracy)
    all_f1.append(f1)
    all_precision.append(precision)
    all_recall.append(recall)

    print(f"Fold {fold} -- AUC: {auc:.4f}, AUPR: {aupr:.4f}, Accuracy: {accuracy:.4f}, "
          f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Store curves
    fold_auc_curves.append((probs, y_true))
    fold_aupr_curves.append((probs, y_true))

mean_auc, std_auc = np.mean(all_auc), np.std(all_auc)
mean_aupr, std_aupr = np.mean(all_aupr), np.std(all_aupr)
mean_accuracy, std_accuracy = np.mean(all_accuracy), np.std(all_accuracy)
mean_f1, std_f1 = np.mean(all_f1), np.std(all_f1)
mean_precision_score, std_precision = np.mean(all_precision), np.std(all_precision)
mean_recall_score, std_recall = np.mean(all_recall), np.std(all_recall)

print(f"\n{n_splits}-Fold Cross Validation Results:")
print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
print(f"Mean AUPR: {mean_aupr:.4f} ± {std_aupr:.4f}")
print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"Mean Precision: {mean_precision_score:.4f} ± {std_precision:.4f}")
print(f"Mean Recall: {mean_recall_score:.4f} ± {std_recall:.4f}")

# # 绘制 AUC 图
# plt.figure(figsize=(8, 6))
# colors = sns.color_palette("husl", n_splits)
# for i, (probs, y_true) in enumerate(fold_auc_curves):
#     fpr, tpr, _ = roc_curve(y_true, probs)
#     plt.plot(fpr, tpr, color=colors[i], label=f'Fold {i+1} AUC: {roc_auc_score(y_true, probs):.4f}')
# mean_fpr = np.linspace(0, 1, 100)
# mean_tpr = np.mean([np.interp(mean_fpr, roc_curve(y_true, probs)[0], roc_curve(y_true, probs)[1]) for probs, y_true in fold_auc_curves], axis=0)
# plt.plot(mean_fpr, mean_tpr, color='black', linestyle='--', label=f'Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}', lw=2)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('AUC Curve')
# plt.legend()
# plt.tight_layout()
# plt.savefig('auc_per_fold.png')
# plt.show()
#
# # 绘制 AUPR 图
# plt.figure(figsize=(8, 6))
# for i, (probs, y_true) in enumerate(fold_aupr_curves):
#     precision, recall, _ = precision_recall_curve(y_true, probs)
#     plt.plot(recall, precision, color=colors[i], label=f'Fold {i+1} AUPR: {average_precision_score(y_true, probs):.4f}')
# mean_recall = np.linspace(0, 1, 100)
# mean_precision = np.zeros_like(mean_recall)
# for probs, y_true in fold_aupr_curves:
#     precision, recall, _ = precision_recall_curve(y_true, probs)
#     mean_precision += np.interp(mean_recall, recall[::-1], precision[::-1])
# mean_precision /= n_splits
# plt.plot(mean_recall, mean_precision, color='black', linestyle='--', label=f'Mean AUPR: {mean_aupr:.4f} ± {std_aupr:.4f}', lw=2)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()
# plt.tight_layout()
# plt.savefig('aupr_per_fold.png')
# plt.show()
