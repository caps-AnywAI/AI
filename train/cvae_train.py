
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model.utils import load_embedding_model, build_embedding_dict
from model.cvae import CVAE
from model.utils import soft_condition_vector, ALL_KEYWORDS

theme_cols = ALL_KEYWORDS.copy()
# ✅ 임베딩 모델 및 딕셔너리 로드
tokenizer, model = load_embedding_model()
embedding_dict = build_embedding_dict(ALL_KEYWORDS, tokenizer, model)
# ✅ Load dataset
df = pd.read_csv('data/festivals_with_soft_vectors_final_adjusted_utf8.csv')


# ✅ Prepare condition and item vectors (soft cond_vector via RBF)
cond_vectors = []
item_vectors = []

for _, row in df.iterrows():
    top_5 = row[theme_cols].sort_values(ascending=False).head(5).index.tolist()
    selected_keywords = top_5
    cond_vec = soft_condition_vector(selected_keywords, embedding_dict, ALL_KEYWORDS, sigma=0.5)    
    item_vec = torch.tensor(row[theme_cols].astype(float).values, dtype=torch.float32)
    cond_vectors.append(cond_vec)
    item_vectors.append(item_vec)

cond_tensor = torch.stack(cond_vectors)
item_tensor = torch.stack(item_vectors)

# ✅ 3-way split
X_temp_cond, X_test_cond, X_temp_item, X_test_item = train_test_split(
    cond_tensor, item_tensor, test_size=0.2, random_state=42
)
X_train_cond, X_val_cond, X_train_item, X_val_item = train_test_split(
    X_temp_cond, X_temp_item, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train_cond, X_train_item)
val_dataset = TensorDataset(X_val_cond, X_val_item)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ✅ Model, loss, optimizer
model = CVAE(cond_dim=12, item_dim=item_tensor.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss(reduction='sum')

# ✅ Training loop
EPOCHS = 50
train_rmses, val_rmses = [], []

model.train()
for epoch in range(EPOCHS):
    total_train_loss = 0
    for cond, item in train_loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(cond, item)
        recon_loss = loss_fn(recon, item)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld
        loss.backward()
        optimizer.step()
        total_train_loss += torch.sum((recon - item) ** 2).item()

    train_rmse = np.sqrt(total_train_loss / len(train_dataset))
    train_rmses.append(train_rmse)

    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for cond, item in val_loader:
            recon, _, _ = model(cond, item)
            total_val_loss += torch.sum((recon - item) ** 2).item()

    val_rmse = np.sqrt(total_val_loss / len(val_dataset))
    val_rmses.append(val_rmse)

    print(f"Epoch {epoch+1:02d} - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
    model.train()

# ✅ Save RMSE histories for plotting
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)
np.save('data/train/train_rmse.npy', train_rmses)
np.save('data/val/val_rmse.npy', val_rmses)

# ✅ Save model
torch.save(model.state_dict(), 'weights/cvae_final_12dim.pt')
print("✅ Model saved to weights/cvae_final_12dim.pt")
