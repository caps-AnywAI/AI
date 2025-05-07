import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

# 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.cvae import CVAE
from model.utils import soft_condition_vector

# ✅ CPU 전용 설정
device = torch.device("cpu")

# ✅ Load dataset
df = pd.read_csv('data/festivals_with_soft_vectors_final_adjusted_utf8.csv')

# ✅ 테마 컬럼 정의
theme_cols = ['nature', 'urban', 'healing', 'activity', 'traditional', 'new',
              'spectating', 'experience', 'popular', 'hidden', 'quiet', 'lively']

# ✅ 소프트 condition + item vector 생성
cond_vectors, item_vectors = [], []

for _, row in df.iterrows():
    top_k = row[theme_cols].sort_values(ascending=False).head(6).index.tolist()
    selected_keywords = np.random.choice(top_k, size=np.random.choice([4, 5]), replace=False).tolist()
    cond_vec = soft_condition_vector(selected_keywords, all_keywords=theme_cols, sigma=0.5)
    item_vec = torch.tensor(row[theme_cols].astype(float).values, dtype=torch.float32)
    cond_vectors.append(cond_vec)
    item_vectors.append(item_vec)

cond_tensor = torch.stack(cond_vectors)
item_tensor = torch.stack(item_vectors)

# ✅ 3-way split
X_temp_c, X_test_c, X_temp_i, X_test_i = train_test_split(cond_tensor, item_tensor, test_size=0.2, random_state=42)
X_train_c, X_val_c, X_train_i, X_val_i = train_test_split(X_temp_c, X_temp_i, test_size=0.25, random_state=42)

train_loader = DataLoader(TensorDataset(X_train_c, X_train_i), batch_size=32)
val_loader = DataLoader(TensorDataset(X_val_c, X_val_i), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test_c, X_test_i), batch_size=32)

# ✅ 모델 로딩
model = CVAE(cond_dim=12, item_dim=item_tensor.shape[1])
model.load_state_dict(torch.load('weights/cvae_final_12dim.pt', map_location=device))
model.to(device)
model.eval()

# ✅ 평가 함수
def evaluate_rmse(model, loader):
    total_mse, count = 0, 0
    with torch.no_grad():
        for cond, item in loader:
            cond, item = cond.to(device), item.to(device)
            recon, _, _ = model(cond, item)
            total_mse += F.mse_loss(recon, item, reduction='sum').item()
            count += item.size(0)
    return np.sqrt(total_mse / count)

def evaluate_ndcg(model, loader, k=5):
    all_scores, all_targets = [], []
    with torch.no_grad():
        for cond, item in loader:
            cond, item = cond.to(device), item.to(device)
            recon, _, _ = model(cond, item)
            all_scores.extend(recon.cpu().numpy())
            all_targets.extend(item.cpu().numpy())
    return ndcg_score(np.array(all_targets), np.array(all_scores), k=k)

def evaluate_accuracy(model, loader, threshold=0.5):
    match_ratio = []
    with torch.no_grad():
        for cond, item in loader:
            cond, item = cond.to(device), item.to(device)
            recon, _, _ = model(cond, item)
            pred = (recon > threshold).float()
            true = (item > threshold).float()
            match = (pred == true).float().mean(dim=1)
            match_ratio.extend(match.cpu().tolist())
    return np.mean(match_ratio)

# ✅ RMSE 기록 시각화
train_rmse = np.load('data/train/train_rmse.npy')
val_rmse = np.load('data/val/val_rmse.npy')

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_rmse)+1), train_rmse, label='Train RMSE', marker='o', linewidth=2)
plt.plot(range(1, len(val_rmse)+1), val_rmse, label='Validation RMSE', marker='o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Train vs Validation RMSE over Epochs')
plt.xticks(range(1, len(train_rmse)+1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ 출력
print(f"✅ Final Train RMSE      : {train_rmse[-1]:.4f}")
print(f"✅ Final Validation RMSE : {val_rmse[-1]:.4f}")

# ✅ 테스트 평가
rmse = evaluate_rmse(model, test_loader)
ndcg = evaluate_ndcg(model, test_loader, k=5)
acc = evaluate_accuracy(model, test_loader, threshold=0.5)

print(f"✅ Final Test RMSE     : {rmse:.4f}")
print(f"✅ Final Test nDCG@5   : {ndcg:.4f}")
print(f"✅ Final Test Accuracy : {acc:.4f}")