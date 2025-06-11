import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
import optuna
from torcheval.metrics import MulticlassAccuracy


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_p: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def get_dataloaders(batch_size: int):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_size = int(0.8 * len(train_data))
    dev_size = len(train_data) - train_size
    train_dataset, dev_dataset = random_split(train_data, [train_size, dev_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, dev_loader, test_loader


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    metric = MulticlassAccuracy(num_classes=10)
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            metric.update(outputs, y_batch)
    return metric.compute().item()


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    dropout_p = trial.suggest_float("dropout_p", 0.2, 0.5)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256])

    train_loader, dev_loader, _ = get_dataloaders(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(28 * 28, hidden_size, dropout_p).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(3):
        train_epoch(model, train_loader, loss_fn, optimizer, device)

    acc = evaluate(model, dev_loader, device)
    return acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best params:", study.best_params)

    batch_size = study.best_params.get("batch_size", 64)
    dropout_p = study.best_params.get("dropout_p", 0.3)
    hidden_size = study.best_params.get("hidden_size", 128)
    lr = study.best_params.get("lr", 1e-3)

    train_loader, dev_loader, test_loader = get_dataloaders(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(28 * 28, hidden_size, dropout_p).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        train_epoch(model, train_loader, loss_fn, optimizer, device)
        dev_acc = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1}, dev accuracy: {dev_acc:.4f}")

    test_acc = evaluate(model, test_loader, device)
    print("Test accuracy:", test_acc)
