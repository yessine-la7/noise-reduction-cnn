import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, pretrained=True): # in_channels=1: Graustufen, in_channels=3: RGB
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)

        orig_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None
        )

        if pretrained and in_channels != 3:
            with torch.no_grad():
                w = orig_conv.weight.data
                if in_channels == 1:
                    # Für Graustufen: Gewichtemittelung über Kanäle
                    w = w.mean(dim=1, keepdim=True)
                else:
                    # Für andere Kanalzahlen: kopiere ein RGB-Kanalgewicht
                    w = w[:, :in_channels, :, :].clone()
                self.model.conv1.weight.copy_(w)

        # Letzte Fully-Connected-Schicht anpassen
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)











################################# Graustufen
# import torch
# import torch.nn as nn
# import torchvision.models as models

# class ResNet18Gray(nn.Module):
#     def __init__(self, num_classes=2, pretrained=True):
#         super().__init__()
#         self.model = models.resnet18(pretrained=pretrained)

#         # Alte RGB-Konv durch neue Graustufen-Konv ersetzen
#         orig_conv = self.model.conv1
#         self.model.conv1 = nn.Conv2d(
#             1,
#             orig_conv.out_channels,
#             kernel_size=orig_conv.kernel_size,
#             stride=orig_conv.stride,
#             padding=orig_conv.padding,
#             bias=orig_conv.bias is not None
#         )

#         # Vortrainierte Gewichte adaptieren → Mittelung über RGB-Kanäle
#         with torch.no_grad():
#             w = orig_conv.weight.data
#             w = w.mean(dim=1, keepdim=True)
#             self.model.conv1.weight.copy_(w)

#         # Letzte FC-Schicht anpassen
#         num_ftrs = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_ftrs, num_classes)

#     def forward(self, x):
#         return self.model(x)
