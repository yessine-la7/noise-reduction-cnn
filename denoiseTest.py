import os
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from createModelUnet import UNetCustom
from loadData import get_data_loaders


# -------------------------------
# Metriken
# -------------------------------
def denorm_to_01(x: torch.Tensor, mean=0.5, std=0.5) -> torch.Tensor:
    """Inverse von Normalize."""
    x = x * std + mean
    return x.clamp(0.0, 1.0)

def gaussian_window(kernel_size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.repeat(channels, 1, 1, 1)
    return window

def compute_psnr(pred_01: torch.Tensor, target_01: torch.Tensor, eps=1e-8) -> float:
    """PSNR in dB auf [0,1]."""
    if pred_01.ndim == 3:
        pred_01 = pred_01.unsqueeze(0)
        target_01 = target_01.unsqueeze(0)
    mse = torch.mean((pred_01 - target_01) ** 2).item()
    if mse <= eps:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))

def compute_ssim(pred_01: torch.Tensor, target_01: torch.Tensor, window: torch.Tensor,
                 C1=0.01**2, C2=0.03**2) -> float:
    """SSIM auf [0,1]."""
    if pred_01.ndim == 3:
        pred_01 = pred_01.unsqueeze(0)
        target_01 = target_01.unsqueeze(0)
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
# Visualisierung & Inferenz
# -------------------------------
def tensor_to_img2d(x01: torch.Tensor) -> np.ndarray:
    """
    (C,H,W) auf [0,1]. Gibt 2D-Array (H,W) für imshow zurück.
    """
    if x01.ndim != 3:
        raise ValueError(f"Erwarte (C,H,W), bekam {tuple(x01.shape)}")
    C = x01.shape[0]
    if C == 1:
        return x01[0].cpu().numpy()
    else:
        return x01.mean(dim=0).cpu().numpy()

@torch.no_grad()
def tile_infer_full_simple(
    model: torch.nn.Module,
    x_noisy_norm: torch.Tensor,
    tile_w: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Tiled-Inferenz über die gesamte Breite mit Hann-Overlap-Add.
    """
    model.eval()
    x_noisy_norm = x_noisy_norm.to(device)
    _, C, H, W = x_noisy_norm.shape
    assert H == 512, f"Erwarte Höhe 512 (DC entfernt), bekam {H}"

    pad_right = max(0, tile_w - W)
    if pad_right:
        x_pad = torch.zeros((1, C, H, tile_w), dtype=x_noisy_norm.dtype, device=device)
        x_pad[..., :W] = x_noisy_norm
        Wp = tile_w
    else:
        x_pad = x_noisy_norm
        Wp = W

    starts = list(range(0, max(Wp - tile_w, 0) + 1, stride))
    if not starts or starts[-1] != Wp - tile_w:
        starts.append(Wp - tile_w)

    hann = torch.hann_window(tile_w, periodic=True, dtype=x_pad.dtype, device=device).view(1,1,1,tile_w)

    out = torch.zeros_like(x_pad)
    wgt = torch.zeros_like(x_pad)

    for x0 in starts:
        pred = model(x_pad[..., x0:x0+tile_w])
        out[..., x0:x0+tile_w] += pred * hann
        wgt[..., x0:x0+tile_w] += hann

    y_hat = out / (wgt + 1e-8)
    return y_hat[..., :W]


# -------------------------------
# Main
# -------------------------------
def main():
    SEED = 33
    SAMPLES_NUM = 5
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pfade
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results_denoising")

    model_path = os.path.join(results_dir, "best_model_baseCh_32_batch_8_full_data_aug_60_epochen.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}. Bitte erst denoisingTrain.py ausführen.")

    out_png = os.path.join(results_dir, "denoise_test.png")

    # Loader
    DATASET_PATH = os.path.abspath(os.path.join(base_dir, "..", "Dataset"))
    loaders = get_data_loaders(
        DATASET_PATH,
        batch_size=1,
        num_workers=0,
        enable_classification=False,
        enable_denoising=True,
        dn_in_channels=1,             # 1: Graustufen
        dn_tile_h=512,
        dn_tile_w=256,
        dn_stride_w=128,
        dn_val_ratio=0.2,
        dn_seed=SEED,
    )
    _, _, test_loader = loaders["denoising"]
    test_ds = test_loader.dataset

    # Kanalzahl aus Dataset ableiten (C=1 bei Graustufen)
    in_ch = getattr(test_ds, "in_channels", 1)

    # Modell laden (base_channels: probiere 64, dann 32)
    for base_ch_try in (64, 32):
        try:
            model = UNetCustom(in_channels=in_ch, out_channels=in_ch, base_channels=base_ch_try).to(device)
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            break
        except Exception as e:
            model = None
            last_err = e
    if model is None:
        raise RuntimeError(f"Modell konnte nicht geladen werden (64/32 base_channels getestet): {last_err}")

    model.eval()

    # SSIM-Fenster
    win = gaussian_window(kernel_size=11, sigma=1.5, channels=in_ch).to(device)

    # Zufällige Bilder aus den Paaren
    n_pairs = len(test_ds.pairs)
    if n_pairs == 0:
        raise RuntimeError("Test-Dataset hat keine Paare.")
    pick = min(SAMPLES_NUM, n_pairs)
    pids = random.sample(range(n_pairs), k=pick)

    # Plot vorbereiten
    fig, axes = plt.subplots(len(pids), 3, figsize=(9, 3 * len(pids)))
    if len(pids) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, pid in enumerate(pids):
        # Vollbilder laden
        noisy_img, clean_img = test_ds._load_pair(pid)
        # auf richtigen Modus bringen
        noisy_img = noisy_img.convert("L") if in_ch == 1 else noisy_img.convert("RGB")
        clean_img = clean_img.convert("L") if in_ch == 1 else clean_img.convert("RGB")

        # Obere Zeile entfernen -> H=512
        noisy_img = test_ds._remove_dc_row(noisy_img)
        clean_img = test_ds._remove_dc_row(clean_img)

        # Preprocessing wie im Training -> Tensor [-1,1], Form (1,C,512,W)
        x_noisy = test_ds.pre(noisy_img).unsqueeze(0)
        x_clean = test_ds.pre(clean_img).unsqueeze(0)

        # Tiled-Inferenz über die gesamte Breite
        y_hat = tile_infer_full_simple(
            model, x_noisy, tile_w=test_ds.tile_w, stride=test_ds.stride_w, device=device
        )

        nz01 = denorm_to_01(x_noisy[0].cpu())
        pr01 = denorm_to_01(y_hat[0].cpu())
        gt01 = denorm_to_01(x_clean[0].cpu())

        psnr_dn = compute_psnr(pr01, gt01)
        ssim_dn = compute_ssim(pr01, gt01, win)
        l1_dn   = torch.mean(torch.abs(pr01 - gt01)).item()

        psnr_n = compute_psnr(nz01, gt01)
        ssim_n = compute_ssim(nz01, gt01, win)

        # 2D für imshow
        n_img = tensor_to_img2d(nz01)
        d_img = tensor_to_img2d(pr01)
        c_img = tensor_to_img2d(gt01)

        # Plot: Noisy
        ax = axes[row, 0]
        ax.imshow(n_img, origin='upper', aspect='auto', cmap='magma', vmin=0.0, vmax=1.0)
        ax.set_title(f"Noisy\nPSNR={psnr_n:.2f} dB • SSIM={ssim_n:.3f}")
        ax.axis('off')

        # Plot: Denoised + Bewertung
        ax = axes[row, 1]
        ax.imshow(d_img, origin='upper', aspect='auto', cmap='magma', vmin=0.0, vmax=1.0)
        ax.set_title(f"Denoised\nPSNR={psnr_dn:.2f} dB • SSIM={ssim_dn:.3f}")
        ax.axis('off')

        # Plot: Clean
        ax = axes[row, 2]
        ax.imshow(c_img, origin='upper', aspect='auto', cmap='magma', vmin=0.0, vmax=1.0)
        ax.set_title("Clean\n")
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(out_png, dpi=120)
    plt.close(fig)

    print(f"[OK] Gespeichert: {out_png}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
