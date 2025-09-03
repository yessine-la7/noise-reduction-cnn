"""
U-Net für Denoising von STFT-Spektrogramm-Tiles.
- GroupNorm (bevorzugt 8).
- Finale Aktivierung: Identity (linear).
- 5-stufig (Faktor 32), Down/Up in H und W.
- Unterstützt in_channels=1 (Graustufen) oder 3 (RGB). out_channels default = in_channels.

Hinweis:
Eingaben sollten in H und W durch 32 teilbar sein (z. B. 512×256).
"""

from typing import Optional
import torch
import torch.nn as nn


# ---------------------------
# Hilfsbausteine
# ---------------------------

def _gn(num_channels: int) -> nn.GroupNorm:
    """
    GroupNorm mit sinnvoller Gruppenzahl (bevorzugt 8),
    fällt auf 4, 2 oder 1 zurück, falls nicht teilbar.
    """
    for g in (8, 4, 2):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class DoubleConv(nn.Module):
    """
    Zwei 3x3-Convs (ohne Bias) + GroupNorm + ReLU.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """ MaxPool(2) gefolgt von DoubleConv """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    ConvTranspose2d (x2 Upsampling) -> Concat mit Skip -> DoubleConv
    Cropping ist NICHT nötig (Größen sind Vielfache von 2^n).
    """
    def __init__(self, in_ch: int, out_ch: int):
        """
        in_ch: Kanäle aus dem Decoder-Pfad (vor Up)
        Nach dem Up hat der Tensor out_ch Kanäle und wird mit dem Skip (out_ch) concateniert.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch=out_ch * 2, out_ch=out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Erwartet identische HxW; mit Vielfachen von 2^n gegeben.
        assert x.shape[-2:] == skip.shape[-2:], f"Shape mismatch: {x.shape[-2:]} vs {skip.shape[-2:]}"
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """ 1x1-Conv zur Kanälereduktion auf out_channels """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# -------------
# U-Net (5 Stufen)
# -------------
class UNetCustom(nn.Module):
    """
    U-Net (5 Stufen) mit fester Norm (GroupNorm) und finaler Aktivierung (Identity).

    Args:
        in_channels:   1 (Graustufen) oder 3 (RGB)
        out_channels:  Default = in_channels
        base_channels: Startkanäle (64 üblich; 32 spart Speicher)

        Für L1/L2-Rekonstruktion auf normalisierten Spektrogrammen: Identity-Output.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: Optional[int] = None,
        base_channels: int = 64,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        # Encoder
        self.inc   = DoubleConv(in_channels, base_channels)           # C
        self.down1 = Down(base_channels, base_channels * 2)           # 2C
        self.down2 = Down(base_channels * 2, base_channels * 4)       # 4C
        self.down3 = Down(base_channels * 4, base_channels * 8)       # 8C
        self.down4 = Down(base_channels * 8, base_channels * 16)      # 16C (Bottleneck)

        # Decoder
        self.up1   = Up(base_channels * 16, base_channels * 8)
        self.up2   = Up(base_channels * 8,  base_channels * 4)
        self.up3   = Up(base_channels * 4,  base_channels * 2)
        self.up4   = Up(base_channels * 2,  base_channels)

        self.outc  = OutConv(base_channels, out_channels)

        self._init_weights()

    def _init_weights(self):
        # Kaiming-Init für Convs, konstante Init für Normen
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # (B,  C,  H,   W)
        x2 = self.down1(x1)    # (B, 2C,  H/2, W/2)
        x3 = self.down2(x2)    # (B, 4C,  H/4, W/4)
        x4 = self.down3(x3)    # (B, 8C,  H/8, W/8)
        x5 = self.down4(x4)    # (B,16C,  H/16,W/16)

        # Decoder
        u1 = self.up1(x5, x4)  # (B, 8C,  H/8, W/8)
        u2 = self.up2(u1, x3)  # (B, 4C,  H/4, W/4)
        u3 = self.up3(u2, x2)  # (B, 2C,  H/2, W/2)
        u4 = self.up4(u3, x1)  # (B,  C,  H,   W)

        # Finale Aktivierung: Identity → einfach linear ausgeben
        return self.outc(u4)


# ---------------------------
# Test
# ---------------------------
if __name__ == "__main__":
    # Graustufen
    net = UNetCustom(in_channels=1, out_channels=1, base_channels=64)
    x = torch.zeros(2, 1, 512, 256)
    y = net(x)
    print("Gray:", x.shape, "->", y.shape)

    # RGB
    net_rgb = UNetCustom(in_channels=3, out_channels=3, base_channels=64)
    x3 = torch.zeros(2, 3, 512, 256)
    y3 = net_rgb(x3)
    print("RGB :", x3.shape, "->", y3.shape)
