import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def generate_data(seq_len=100, num_samples=2000):
    """Generate random binary sequences where the target is the first element."""
    X = np.random.randint(0, 2, size=(num_samples, seq_len)).astype(np.float32)
    y = X[:, 0:1]
    return X[..., None], y


class SimpleRNN(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)


class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)


def train_model(model, loader, epochs=10):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            preds = (model(batch_x) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def main():
    seq_len = 100  # long sequence where RNN forgets the first element
    X, y = generate_data(seq_len)
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    rnn = SimpleRNN()
    train_model(rnn, loader)
    rnn_acc = evaluate(rnn, loader)

    lstm = SimpleLSTM()
    train_model(lstm, loader)
    lstm_acc = evaluate(lstm, loader)

    print(f"RNN accuracy: {rnn_acc:.2f}")
    print(f"LSTM accuracy: {lstm_acc:.2f}")


if __name__ == "__main__":
    main()
