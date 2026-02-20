import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse

from models.kan import KAN
from models.mlp import MLP
from data.dataloader import get_loaders, get_device
from utils import count_parameters


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "kan"], default="kan")
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], default="MNIST")
    args = parser.parse_args()

    device = get_device()
    train_loader, test_loader = get_loaders(args.dataset)

    layers = [784, 64, 10]

    if args.model == "mlp":
        model = MLP(layers).to(device)
    else:
        model = KAN(layers, grid_size=args.grid).to(device)

    print("Parameters:", count_parameters(model))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(10):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Test Acc: {val_acc:.2f}%")

    print("Total Time:", round(time.time() - start, 2), "s")


if __name__ == "__main__":
    main()