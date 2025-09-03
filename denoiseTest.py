# -*- coding: utf-8 -*-
"""
denoiseTest.py

Nimmt 5 zufällige Tiles aus dem Denoising-Testset (STFT-Paare noisy/clean),
berechnet die denoised Ausgabe mit dem gespeicherten U-Net
und speichert eine 5x3 Bildtafel:
    [Noisy | Denoised (mit Bewertung) | Clean]

Bewertung unter "Denoised": PSNR [dB], SSIM, L1 (alle auf denormalisierter [0,1]-Skala).

Voraussetzungen:
- results_denoising/best_model.pth existiert (aus denoisingTrain.py)
- Dataset-Struktur wie in loadData.get_data_loaders()
"""

import os
import random
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from createModelUnet import UNetCustom
from loadData import get_data_loaders


# -------------------------------
# Utils: Denormalisieren & Metriken
# -------------------------------
def denorm_to_01(x: torch.Tensor, mean=0.5, std=0.5) -> torch.Tensor:
    """Inverse von Normalize(mean=0.5, std=0.5): [-1,1] -> [0,1]"""
    x = x * std + mean
    return x.clamp(0.0, 1.0)

def gaussian_window(kernel_size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0)  # 1x1xKxK
    window = window.repeat(channels, 1, 1, 1)  # Cx1xKxK (depthwise)
    return window

def compute_psnr(pred_01: torch.Tensor, target_01: torch.Tensor, eps=1e-8) -> float:
    """PSNR in dB (auf [0,1])."""
    mse = torch.mean((pred_01 - target_01) ** 2).item()
    if mse <= eps:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))

def compute_ssim(pred_01: torch.Tensor, target_01: torch.Tensor, window: torch.Tensor,
                 C1=0.01**2, C2=0.03**2) -> float:
    """SSIM auf [0,1]; depthwise Faltung, erwartet BxCxHxW."""
    window = window.to(pred_01.device, dtype=pred_01.dtype)
    C = pred_01.shape[1]
    mu_x = torch.nn.functional.conv2d(pred_01, window, padding=window.shape[-1]//2, groups=C)
    mu_y = torch.nn.functional.conv2d(target_01, window, padding=window.shape[-1]//2, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = torch.nn.functional.conv2d(pred_01 * pred_01, window, padding=window.shape[-1]//2, groups=C) - mu_x2
    sigma_y2 = torch.nn.functional.conv2d(target_01 * target_01, window, padding=window.shape[-1]//2, groups=C) - mu_y2
    sigma_xy = torch.nn.functional.conv2d(pred_01 * target_01, window, padding=window.shape[-1]//2, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return float(ssim_map.mean().item())


# -------------------------------
# Plot-Helfer
# -------------------------------
def tensor_to_img2d(x01: torch.Tensor) -> np.ndarray:
    """
    x01: (C,H,W) auf [0,1]. Gibt 2D-Array (H,W) für imshow zurück.
    - Bei C=1: Kanal 0
    - Bei C=3: Mittelwert der Kanäle (praktische Visualisierung)
    """
    if x01.ndim != 3:
        raise ValueError(f"Erwarte (C,H,W), bekam {tuple(x01.shape)}")
    C = x01.shape[0]
    if C == 1:
        return x01[0].cpu().numpy()
    else:
        return x01.mean(dim=0).cpu().numpy()


# -------------------------------
# Main
# -------------------------------
def main():
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pfade
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results_denoising")
    model_path = os.path.join(results_dir, "best_model_baseCh_32_batch_8.pth")
    out_png = os.path.join(results_dir, "denoise_test.png")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}. Bitte erst denoisingTrain.py ausführen.")

    # Loader
    DATASET_PATH = os.path.abspath(os.path.join(base_dir, "..", "Dataset"))
    loaders = get_data_loaders(
        DATASET_PATH,
        batch_size=1,
        num_workers=0,
        enable_classification=False,
        enable_denoising=True,
        dn_in_channels=1,             # 1=Graustufen
        dn_tile_h=512,
        dn_tile_w=256,
        dn_stride_w=256,
        dn_val_ratio=0.2,
        dn_seed=SEED,
    )
    _, _, test_loader = loaders["denoising"]
    test_ds = test_loader.dataset

    # Kanalzahl aus erstem Sample ableiten
    C = test_ds[0][0].shape[0]

    # Modell laden
    model = UNetCustom(in_channels=C, out_channels=C, base_channels=32).to(device)  # base_channels egal fürs Laden
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # SSIM-Fenster
    win = gaussian_window(kernel_size=11, sigma=1.5, channels=C).to(device)

    # 5 zufällige Indizes
    num_examples = 5
    total = len(test_ds)
    if total == 0:
        raise RuntimeError("Test-Dataset ist leer.")
    idxs = random.sample(range(total), k=min(num_examples, total))

    # Plot vorbereiten
    fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 3 * len(idxs)))
    if len(idxs) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(idxs):
        # (C,H,W) normalisiert ~[-1,1]
        x_noisy, y_clean = test_ds[idx]
        with torch.no_grad():
            y_hat = model(x_noisy.unsqueeze(0).to(device)).squeeze(0).cpu()

        # Für Metriken/Anzeige auf [0,1] denormalisieren
        nz01 = denorm_to_01(x_noisy)
        pr01 = denorm_to_01(y_hat)
        gt01 = denorm_to_01(y_clean)

        # Kennzahlen (auf [0,1])
        # BxC: füge Batch-Dim hinzu
        psnr = compute_psnr(pr01.unsqueeze(0), gt01.unsqueeze(0))
        ssim = compute_ssim(pr01.unsqueeze(0), gt01.unsqueeze(0), win)
        l1   = torch.mean(torch.abs(pr01 - gt01)).item()

        # 2D für imshow
        n_img = tensor_to_img2d(nz01)
        d_img = tensor_to_img2d(pr01)
        c_img = tensor_to_img2d(gt01)

        # Plot: Noisy
        ax = axes[row, 0]
        im = ax.imshow(n_img, origin='upper', aspect='auto', cmap='magma')
        ax.set_title("Noisy")
        ax.axis('off')

        # Plot: Denoised + Bewertung in Titel
        ax = axes[row, 1]
        ax.imshow(d_img, origin='upper', aspect='auto', cmap='magma')
        ax.set_title(f"Denoised\nPSNR={psnr:.2f} dB • SSIM={ssim:.3f} • L1={l1:.4f}")
        ax.axis('off')

        # Plot: Clean
        ax = axes[row, 2]
        ax.imshow(c_img, origin='upper', aspect='auto', cmap='magma')
        ax.set_title("Clean")
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(out_png, dpi=120)
    plt.close(fig)

    print(f"[OK] Gespeichert: {out_png}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
