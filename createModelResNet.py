import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, pretrained=True):  # in_channels=1: Graustufen, in_channels=3: RGB
        super().__init__()

        # pretrained durch weights
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.model = models.resnet18(weights=weights)

        # Ursprüngliche erste Conv-Schicht speichern
        orig_conv = self.model.conv1

        # Erste Conv-Schicht anpassen auf in_channels
        self.model.conv1 = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None
        )

        # Falls pretrained und Kanäle ≠ 3 → Gewichte anpassen
        if pretrained and in_channels != 3:
            with torch.no_grad():
                w = orig_conv.weight.data
                if in_channels == 1:
                    # Für Graustufen: Mittelwert über RGB-Kanäle
                    w = w.mean(dim=1, keepdim=True)
                else:
                    # Für andere Kanalzahlen: ersten in_channels-Kanäle nehmen
                    w = w[:, :in_channels, :, :].clone()
                self.model.conv1.weight.copy_(w)

        # Letzte Fully-Connected-Schicht anpassen
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
