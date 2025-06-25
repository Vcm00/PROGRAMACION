"""MLP regression - Exercise 2

This script fits a Multilayer Perceptron (MLP) to sine wave data using
PyTorch. It closely follows the instructions from the practice notebook.

Steps:
1. Generate sine wave points.
2. Normalize the data for stable training.
3. Create a PyTorch dataset and dataloader.
4. Define a simple MLP with two hidden layers.
5. Train the model using mean squared error loss.
6. Plot the original data and the predictions of the trained MLP.

The code is self contained so it can be copied directly into a Colab
notebook cell and executed.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader


# --------------------------------------------------
# 1. Generate sine data
# --------------------------------------------------
cycles = 2  # number of periods
n_points = 100  # number of samples

length = 2 * np.pi * cycles
x = np.arange(0, length, length / n_points)
y = np.sin(x)

# Visualization of the true sine curve
plt.plot(x, y, label="Sine true")
plt.title("Original sine curve")
plt.show()


# --------------------------------------------------
# 2. Normalize the data
# --------------------------------------------------
x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()

x_norm = (x - x_mean) / x_std
y_norm = (y - y_mean) / y_std


# --------------------------------------------------
# 3. Create dataset and dataloader
# --------------------------------------------------
# Convert arrays to tensors. x needs to be (n, 1).
tensor_x = torch.tensor(x_norm[:, None], dtype=torch.float32)
tensor_y = torch.tensor(y_norm, dtype=torch.float32)

# Dataset and dataloader for training
sine_ds = TensorDataset(tensor_x, tensor_y)
sine_dl = DataLoader(sine_ds, shuffle=True)


# --------------------------------------------------
# 4. Define the MLP model
# --------------------------------------------------
class SineMLP(torch.nn.Module):
    """Small MLP to approximate the sine function."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = SineMLP()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# --------------------------------------------------
# 5. Training loop
# --------------------------------------------------
num_epochs = 1000
for epoch in range(num_epochs):
    for features, targets in sine_dl:
        preds = model(features).squeeze()
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# --------------------------------------------------
# 6. Evaluate and plot results
# --------------------------------------------------
model.eval()
with torch.no_grad():
    pred_norm = model(tensor_x).squeeze().numpy()

y_pred = pred_norm * y_std + y_mean

plt.plot(x, y, label="Sine true")
plt.plot(x, y_pred, label="MLP", color="orange")
plt.title("MLP approximation")
plt.legend()
plt.show()
