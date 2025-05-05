
import numpy as np
import matplotlib.pyplot as plt

# ✅ Load RMSE histories
train_rmse = np.load('data/train/train_rmse.npy')
val_rmse = np.load('data/val/val_rmse.npy')

# ✅ Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_rmse) + 1), train_rmse, label='Train RMSE', marker='o', linewidth=2)
plt.plot(range(1, len(val_rmse) + 1), val_rmse, label='Validation RMSE', marker='o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Train vs Validation RMSE over Epochs')
plt.xticks(range(1, len(train_rmse) + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ Final values print
print(f"✅ Final Train RMSE      : {train_rmse[-1]:.4f}")
print(f"✅ Final Validation RMSE : {val_rmse[-1]:.4f}")
