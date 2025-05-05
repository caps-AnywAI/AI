
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model.cvae import CVAE
from model.utils import make_condition_vector

# Load dataset
df = pd.read_csv('data/festivals_with_soft_vectors_final_adjusted_utf8.csv')

theme_cols = ['nature', 'urban', 'healing', 'activity', 'traditional', 'new',
              'food', 'experience', 'popular', 'hidden', 'quiet', 'lively']

cond_vectors = []
item_vectors = []

for _, row in df.iterrows():
    top_k = row[theme_cols].sort_values(ascending=False).head(6).index.tolist()
    selected_keywords = random.sample(top_k, k=random.choice([4, 5]))
    cond_vec = make_condition_vector(selected_keywords)
    item_vec = torch.tensor(row[theme_cols].astype(float).values, dtype=torch.float32)

    cond_vectors.append(cond_vec)
    item_vectors.append(item_vec)

cond_tensor = torch.stack(cond_vectors)
item_tensor = torch.stack(item_vectors)

# Test set only
X_temp_cond, X_test_cond, X_temp_item, X_test_item = train_test_split(
    cond_tensor, item_tensor, test_size=0.2, random_state=42
)

test_loader = DataLoader(TensorDataset(X_test_cond, X_test_item), batch_size=32)

# Load trained model
model = CVAE(cond_dim=12, item_dim=item_tensor.shape[1])
model.load_state_dict(torch.load('weights/cvae_final_12dim.pt'))
model.eval()

# Evaluation metrics
def evaluate_rmse(model, loader):
    model.eval()
    total_mse = 0
    count = 0
    with torch.no_grad():
        for cond, item in loader:
            recon, _, _ = model(cond, item)
            total_mse += F.mse_loss(recon, item, reduction='sum').item()
            count += item.size(0)
    return np.sqrt(total_mse / count)

def evaluate_ndcg(model, loader, k=5):
    model.eval()
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for cond, item in loader:
            recon, _, _ = model(cond, item)
            all_scores.extend(recon.numpy())
            all_targets.extend(item.numpy())
    return ndcg_score(np.array(all_targets), np.array(all_scores), k=k)

def evaluate_accuracy(model, loader, threshold=0.5):
    model.eval()
    match_ratio = []
    with torch.no_grad():
        for cond, item in loader:
            recon, _, _ = model(cond, item)
            pred = (recon > threshold).float()
            true = (item > threshold).float()
            match = (pred == true).float().mean(dim=1)
            match_ratio.extend(match.tolist())
    return np.mean(match_ratio)

# Final evaluation
rmse = evaluate_rmse(model, test_loader)
ndcg = evaluate_ndcg(model, test_loader, k=5)
acc = evaluate_accuracy(model, test_loader, threshold=0.5)

print(f"✅ Final Test RMSE     : {rmse:.4f}")
print(f"✅ Final Test nDCG@5   : {ndcg:.4f}")
print(f"✅ Final Test Accuracy : {acc:.4f}")
