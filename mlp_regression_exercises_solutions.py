# Solutions for MLP regression exercises using PyTorch
# Each section can be copied into a Colab notebook cell.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error


# --------------------------------------------------
# Ejercicio 1: Ajustar una red que consiga error cero
# --------------------------------------------------

# Datos de ejemplo
X1 = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:, None]
y1 = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])

plt.scatter(X1, y1)
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Normalización
x1_mean, x1_std = X1.mean(), X1.std()
y1_mean, y1_std = y1.mean(), y1.std()
X1_norm = (X1 - x1_mean) / x1_std
y1_norm = (y1 - y1_mean) / y1_std

# TensorDataset y DataLoader
X1_tensor = torch.tensor(X1_norm, dtype=torch.float32)
y1_tensor = torch.tensor(y1_norm, dtype=torch.float32)
train_ds1 = TensorDataset(X1_tensor, y1_tensor)
train_dl1 = DataLoader(train_ds1, shuffle=True)

# MLP lo bastante grande para aprender los 10 puntos
class PerfectMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

model1 = PerfectMLP()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.01)
loss_history1 = []

for epoch in range(2000):
    for feats, targets in train_dl1:
        preds = model1(feats)
        loss = loss_fn(preds.squeeze(), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_history1.append(loss.item())
    if loss.item() < 1e-6:
        break

plt.plot(loss_history1)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Predicción y desnormalización
model1.eval()
with torch.no_grad():
    preds_norm = model1(X1_tensor).squeeze().numpy()
y_pred1 = preds_norm * y1_std + y1_mean

plt.scatter(X1, y1, label="Datos")
plt.plot(X1, y_pred1, label="MLP", color="orange")
plt.legend()
plt.show()


# --------------------------------------------------
# Ejercicio 2: Ajustar un seno con un MLP
# --------------------------------------------------

cycles = 2
n = 100
length = 2 * np.pi * cycles
x2 = np.arange(0, length, length / n)
y2 = np.sin(x2)

plt.plot(x2, y2, "-")
plt.show()

# Normalización
x2_mean, x2_std = x2.mean(), x2.std()
y2_mean, y2_std = y2.mean(), y2.std()
x2_norm = (x2 - x2_mean) / x2_std
y2_norm = (y2 - y2_mean) / y2_std

x2_tensor = torch.tensor(x2_norm[:, None], dtype=torch.float32)
y2_tensor = torch.tensor(y2_norm, dtype=torch.float32)
ds2 = TensorDataset(x2_tensor, y2_tensor)
dl2 = DataLoader(ds2, shuffle=True)

class SineMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model2 = SineMLP()
loss_fn2 = torch.nn.MSELoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)

for epoch in range(1000):
    for feats, targets in dl2:
        preds = model2(feats)
        loss = loss_fn2(preds.squeeze(), targets)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

# Predicciones para visualizar
model2.eval()
with torch.no_grad():
    pred_norm = model2(x2_tensor).squeeze().numpy()
y_pred2 = pred_norm * y2_std + y2_mean

plt.plot(x2, y2, label="seno real")
plt.plot(x2, y_pred2, label="MLP", color="orange")
plt.legend()
plt.show()


# --------------------------------------------------
# Ejercicio 3: Regresión con datos de felicidad
# --------------------------------------------------

data = pd.read_csv("https://raw.githubusercontent.com/mcstllns/UNIR2024/main/data-happiness.csv")
data = data.dropna()

X3 = data.drop("Life.Ladder", axis=1)
y3 = data["Life.Ladder"]

# Normalización
a3_mean, a3_std = X3.mean(), X3.std()
y3_mean, y3_std = y3.mean(), y3.std()
X3_norm = (X3 - a3_mean) / a3_std
y3_norm = (y3 - y3_mean) / y3_std

X3_tensor = torch.tensor(X3_norm.values, dtype=torch.float32)
y3_tensor = torch.tensor(y3_norm.values, dtype=torch.float32)
ds3 = TensorDataset(X3_tensor, y3_tensor)
dl3 = DataLoader(ds3, shuffle=True)

class HappinessMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(9, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model3 = HappinessMLP()
loss_fn3 = torch.nn.MSELoss()
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.01)
mse_history = []

for epoch in range(500):
    for feats, targets in dl3:
        preds = model3(feats).squeeze()
        loss = loss_fn3(preds, targets)
        optimizer3.zero_grad()
        loss.backward()
        optimizer3.step()
    mse_history.append(loss.item())

# Evaluación final
model3.eval()
with torch.no_grad():
    predictions = model3(X3_tensor).squeeze()
    mse = mean_squared_error(y3, predictions.numpy() * y3_std + y3_mean)
print("MSE final:", mse)
