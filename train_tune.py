import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from ray import air, tune
from ray.tune.schedulers import ASHAScheduler

from dataset import get_dataloaders, N_CLASSES
from models import create_resnet18


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    correct, total, loss_sum = 0, 0, 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.squeeze().long().to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss_sum += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return loss_sum / total, correct / total


def train_tune(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(batch_size=config["batch_size"])

    model = create_resnet18(N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=config["lr"]) \
                if config["optimizer"] == "adam" \
                else SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    best_acc = 0

    for epoch in range(config["epochs"]):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, val_acc = evaluate(model, val_loader, criterion, device)

        best_acc = max(best_acc, val_acc)

        tune.report({"accuracy": val_acc, "best_accuracy": best_acc})



if __name__ == "__main__":
    param_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "optimizer": tune.choice(["adam", "sgd"]),
        "epochs": 8
    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        grace_period=3,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
    tune.with_resources(train_tune, {"cpu": 4, "gpu": 1}),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=12,
    ),
    run_config = air.RunConfig(
    storage_path="/global/u2/y/yuning88/rl-hyper-tuning/ray_results",
    name="pathmnist_resnet18_tune",
    ),
    param_space=param_space,
)


    results = tuner.fit()
    best = results.get_best_result(metric="accuracy", mode="max")
    print("Best Config:", best.config)
