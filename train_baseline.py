import time
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from dataset import get_dataloaders, N_CLASSES
from models import create_resnet18


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.squeeze().long().to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 1e-3
    batch_size = 64
    optimizer_name = "adam"
    num_epochs = 5

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    model = create_resnet18(num_classes=N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=lr) if optimizer_name == "adam" \
                                            else SGD(model.parameters(), lr=lr, momentum=0.9)

    start = time.time()
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        best_acc = max(best_acc, val_acc)

        print(f"Epoch {epoch}: Train {train_acc:.4f}, Val {val_acc:.4f}")

    print("Best Acc =", best_acc)
    print("Time =", (time.time() - start) / 60, "min")


if __name__ == "__main__":
    main()
