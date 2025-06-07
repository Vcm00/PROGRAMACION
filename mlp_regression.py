import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import TensorDataset, DataLoader

# synthetic data
X = np.array([
    258.0, 270.0, 294.0, 320.0, 342.0,
    368.0, 396.0, 446.0, 480.0, 586.0
])[:, None]

y = np.array([
    236.4, 234.4, 252.8, 298.6, 314.2,
    342.2, 360.8, 368.0, 391.2, 390.8
])

# simple linear regression for comparison
lr = LinearRegression()
lr.fit(X, y)
X_range = np.arange(250, 600, 10)[:, None]
y_linear = lr.predict(X_range)

# normalize data for the MLP
x_mean, x_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()
X_norm = (X - x_mean) / x_std
y_norm = (y - y_mean) / y_std

# dataset and dataloader
features = torch.tensor(X_norm, dtype=torch.float32)
targets = torch.tensor(y_norm, dtype=torch.float32)
dataset = TensorDataset(features, targets)
dataloader = DataLoader(dataset)

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 1),
            torch.nn.ReLU(),
            torch.nn.Linear(1, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = MLP()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

loss_history = []
for epoch in range(200):
    for f, t in dataloader:
        out = model(f)
        loss = loss_fn(out.squeeze(), t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_history.append(loss.item())

# evaluate
model.eval()
X_range_norm = (X_range - x_mean) / x_std
X_range_norm = torch.tensor(X_range_norm, dtype=torch.float32)
y_mlp_norm = model(X_range_norm).detach().numpy().astype(float)
y_mlp = y_mlp_norm * y_std + y_mean

# plot results
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("loss")

plt.subplot(1, 2, 2)
plt.scatter(X, y, label="Datos")
plt.plot(X_range, y_linear, label="Regresión lineal", linestyle="--", color="green")
plt.plot(X_range, y_mlp, label="MLP", color="orange")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()
