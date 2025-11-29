import torch.nn as nn
from torchvision.models import resnet18

def create_resnet18(num_classes):
    model = resnet18(weights=None)   # 可换成 weights="IMAGENET1K_V1"
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
