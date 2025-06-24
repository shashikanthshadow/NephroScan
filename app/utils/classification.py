 # app/utils/classification.py
import torch
import torch.nn as nn
from torchvision import models

class ResNetWithDropout(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.base.parameters():
            param.requires_grad = False
        for param in self.base.layer3.parameters():
            param.requires_grad = True
        for param in self.base.layer4.parameters():
            param.requires_grad = True
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base(x)

def load_classifier(model_path="models/ResNet18_Optimized_AntiOverfit.pth"):
    model = ResNetWithDropout()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model
