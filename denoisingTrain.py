import os
import time
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt

from createModelUnet import UNetCustom
from loadData import get_data_loaders
from earlyStopping import EarlyStopping
from trainLogging import setup_logging


# -------------------------------
# Determinismus
# -------------------------------
DETERMINISTIC = False
GLOBAL_SEED = 42
if DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)

def set_global_determinism(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)

set_global_determinism(GLOBAL_SEED, DETERMINISTIC)


# -------------------------------
# Logging: Hyperparameter
# -------------------------------
def log_hyperparams(logger, *,
                    lr, batch_size, in_channels, base_channels,
                    tile_h, tile_w, stride_w,
                    patience, adapt_start_epoch, num_epochs,
                    deterministic, num_workers,
                    step_size, gamma, val_ratio):
    logger.info("======== Denoising Hyperparameter ========")
    logger.info(f"Learning Rate:                    {lr}")
    logger.info(f"Batch Size:                       {batch_size}")
    logger.info(f"in_channels (1=Gray, 3=RGB):      {in_channels}")
    logger.info(f"U-Net base_channels:              {base_channels}")
    logger.info(f"Tile (H x W, stride_w):           {tile_h} x {tile_w} (stride {stride_w})")
    logger.info(f"EarlyStopping patience:           {patience}")
    logger.info(f"EarlyStopping adapt_start_epoch:  {adapt_start_epoch}")
    logger.info(f"Num epochs:                       {num_epochs}")
    logger.info(f"LR Scheduler:                     step_size={step_size}, gamma={gamma}")
    logger.info(f"Val split ratio:                  {val_ratio}")
    logger.info(f"Deterministic:                    {deterministic}")
    logger.info(f"DataLoader num_workers:           {num_workers}")
    logger.info("==========================================")


# -------------------------------
# Hilfsfunktionen: Denorm & Metriken
# -------------------------------
def denorm_to_01(x: torch.Tensor, mean=0.5, std=0.5) -> torch.Tensor:
    """Inverse von Normalize(mean=0.5, std=0.5): [-1,1] -> [0,1]"""
    x = x * std + mean
    return x.clamp(0.0, 1.0)

def compute_psnr(pred_01: torch.Tensor, target_01: torch.Tensor, eps=1e-8) -> float:
    """PSNR auf [0,1]-Skala (max_val=1)."""
    mse = torch.mean((pred_01 - target_01) ** 2).item()
    if mse <= eps:
        return 99.0
    psnr = 10.0 * math.log10(1.0 / mse)
    return float(psnr)

def gaussian_window(kernel_size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0)  # 1x1xKxK
    window = window.repeat(channels, 1, 1, 1)  # Cx1xKxK (depthwise)
    return window

def compute_ssim(pred_01: torch.Tensor, target_01: torch.Tensor, window: torch.Tensor,
                 C1=0.01**2, C2=0.03**2) -> float:
    """SSIM auf [0,1]-Skala, depthwise Faltung. Erwartet BxCxHxW."""
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
# Visualisierung einiger Beispiele
# -------------------------------
def save_sample_triplets(noisy: torch.Tensor, denoised: torch.Tensor, clean: torch.Tensor,
                         out_path: str, max_items: int = 5):
    """
    Speichert ein Grid mit (noisy | denoised | clean) auf [0,1]-Skala.
    Erwartet Tensoren in [0,1].
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    b = min(noisy.size(0), max_items)
    C = noisy.size(1)

    def to_np(img):
        if C == 1:
            arr = img[0].cpu().numpy()
        else:
            arr = img.mean(dim=0).cpu().numpy()
        return arr

    fig, axes = plt.subplots(b, 3, figsize=(9, 3*b))
    if b == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(b):
        n = to_np(noisy[i])
        d = to_np(denoised[i])
        c = to_np(clean[i])

        for j, (title, arr) in enumerate([("Noisy", n), ("Denoised", d), ("Clean", c)]):
            ax = axes[i, j]
            ax.imshow(arr, origin='upper', aspect='auto', cmap='magma')
            ax.set_title(title)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# -------------------------------
# Plots (Kurven & Histogramm)
# -------------------------------
def save_curves(log_dir: str, history: Dict[str, List[float]]):
    os.makedirs(log_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=(8,5))
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"],   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("L1 Loss"); plt.title("Loss over Epochs"); plt.legend()
    plt.savefig(os.path.join(log_dir, "curves_loss.png")); plt.close()

    # PSNR
    plt.figure(figsize=(8,5))
    plt.plot(epochs, history["train_psnr"], label="Train")
    plt.plot(epochs, history["val_psnr"],   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)"); plt.title("PSNR over Epochs"); plt.legend()
    plt.savefig(os.path.join(log_dir, "curves_psnr.png")); plt.close()

    # SSIM
    plt.figure(figsize=(8,5))
    plt.plot(epochs, history["train_ssim"], label="Train")
    plt.plot(epochs, history["val_ssim"],   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("SSIM"); plt.title("SSIM over Epochs"); plt.legend()
    plt.savefig(os.path.join(log_dir, "curves_ssim.png")); plt.close()

    # LR
    plt.figure(figsize=(8,5))
    plt.plot(epochs, history["lr"], label="LR")
    plt.xlabel("Epoch"); plt.ylabel("LR"); plt.yscale("log"); plt.title("Learning Rate over Epochs"); plt.legend()
    plt.savefig(os.path.join(log_dir, "curves_lr.png")); plt.close()


@torch.no_grad()
def save_val_residual_hist(model: nn.Module, loader, device, out_path: str, bins: int = 60):
    """Histogramm der Residuen (denormiert, [0,1]) auf dem Val-Set."""
    model.eval()
    errs = []
    for x_noisy, y_clean in tqdm(loader, desc="Residual histogram", unit="batch"):
        x_noisy = x_noisy.to(device)
        y_clean = y_clean.to(device)
        y_hat = model(x_noisy)
        e = denorm_to_01(y_hat) - denorm_to_01(y_clean)
        errs.append(e.detach().cpu().flatten().numpy())
    if not errs:
        return
    errs = np.concatenate(errs, axis=0)

    plt.figure(figsize=(7,5))
    plt.hist(errs, bins=bins, alpha=0.85, edgecolor="black")
    plt.title("Residual Distribution (Val, denormalized [0,1])")
    plt.xlabel("Prediction - Target"); plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


# -------------------------------
# Training + validation
# -------------------------------
def train(model: nn.Module,
          train_loader,
          val_loader,
          device,
          results_dir: str,
          *,
          num_epochs: int,
          lr: float,
          patience: int,
          adapt_start_epoch: int,
          step_size: int,
          gamma: float,
          logger):

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    early_stopper = EarlyStopping(patience=patience, min_delta=None, verbose=True,
                                  adapt_start_epoch=adapt_start_epoch)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_psnr": [], "val_psnr": [],
        "train_ssim": [], "val_ssim": [],
        "lr": [],
    }

    # SSIM-Fenster einmal vorab bestimmen
    try:
        C = train_loader.dataset[0][0].shape[0]  # (C,H,W)
    except Exception as e:
        raise RuntimeError(
            f"Kann Kanalzahl nicht aus train_loader.dataset[0] lesen: {e}"
        )
    win = gaussian_window(kernel_size=11, sigma=1.5, channels=C).to(device)

    best_val_loss = float("inf")
    start_time = time.time()
    logger.info("=== Start Denoising-Training ===")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        # ---------- TRAIN ----------
        model.train()
        train_loss_sum = 0.0
        train_psnr_sum = 0.0
        train_ssim_sum = 0.0
        n_train_batches = 0

        for x_noisy, y_clean in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
            x_noisy = x_noisy.to(device, non_blocking=True)
            y_clean = y_clean.to(device, non_blocking=True)

            optimizer.zero_grad()
            y_hat = model(x_noisy)
            loss = criterion(y_hat, y_clean)
            loss.backward()
            optimizer.step()

            # Metriken (auf [0,1])
            y_hat_01 = denorm_to_01(y_hat)
            y_ref_01 = denorm_to_01(y_clean)
            psnr = compute_psnr(y_hat_01, y_ref_01)
            ssim = compute_ssim(y_hat_01, y_ref_01, win)

            train_loss_sum += loss.item()
            train_psnr_sum += psnr
            train_ssim_sum += ssim
            n_train_batches += 1

        train_loss = train_loss_sum / max(1, n_train_batches)
        train_psnr = train_psnr_sum / max(1, n_train_batches)
        train_ssim = train_ssim_sum / max(1, n_train_batches)

        # ---------- VAL ----------
        model.eval()
        val_loss_sum = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for x_noisy, y_clean in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
                x_noisy = x_noisy.to(device, non_blocking=True)
                y_clean = y_clean.to(device, non_blocking=True)

                y_hat = model(x_noisy)
                loss = criterion(y_hat, y_clean)

                y_hat_01 = denorm_to_01(y_hat)
                y_ref_01 = denorm_to_01(y_clean)
                psnr = compute_psnr(y_hat_01, y_ref_01)
                ssim = compute_ssim(y_hat_01, y_ref_01, win)

                val_loss_sum += loss.item()
                val_psnr_sum += psnr
                val_ssim_sum += ssim
                n_val_batches += 1

        val_loss = val_loss_sum / max(1, n_val_batches)
        val_psnr = val_psnr_sum / max(1, n_val_batches)
        val_ssim = val_ssim_sum / max(1, n_val_batches)

        # ---------- Logging ----------
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_psnr"].append(train_psnr)
        history["val_psnr"].append(val_psnr)
        history["train_ssim"].append(train_ssim)
        history["val_ssim"].append(val_ssim)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        logger.info(
            f"Epoch {epoch:02d} | "
            f"Train: L1={train_loss:.5f}, PSNR={train_psnr:.2f}, SSIM={train_ssim:.4f} | "
            f"Val:   L1={val_loss:.5f}, PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f} | "
            f"LR={history['lr'][-1]:.2e}"
        )

        # ---------- Bestes Modell sichern + Beispiele ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            logger.info(f"Saved best_model.pth (val L1 = {best_val_loss:.6f})")

            # Beispielvisualisierung auf Val (denorm)
            try:
                x_noisy_sample, y_clean_sample = next(iter(val_loader))
                x_noisy_sample = x_noisy_sample.to(device)
                y_clean_sample = y_clean_sample.to(device)
                with torch.no_grad():
                    y_hat_sample = model(x_noisy_sample)
                nz = denorm_to_01(x_noisy_sample)
                pr = denorm_to_01(y_hat_sample)
                gt = denorm_to_01(y_clean_sample)
                out_img = os.path.join(results_dir, f"samples_epoch{epoch:02d}.png")
                save_sample_triplets(nz, pr, gt, out_img, max_items=5)
                logger.info(f"Saved sample triplets: {out_img}")
            except StopIteration:
                logger.warning("Val loader leer – keine Beispielvisualisierung erzeugt.")

        # ---------- EarlyStopping & Scheduler ----------
        if early_stopper(val_loss):
            logger.warning(f"Early stopping after epoch {epoch}.")
            break

        scheduler.step()

    minutes = (time.time() - start_time) / 60.0
    logger.info(f"Training finished in {minutes:.2f} min")
    logger.info(f"Best validation L1: {best_val_loss:.6f}")

    # Kurven speichern (PNG)
    save_curves(results_dir, history)

    return history



# -------------------------------
# Main
# -------------------------------
def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results_denoising")
    os.makedirs(results_dir, exist_ok=True)
    logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

    # ===== Hyperparameter =====
    in_channels = 1           # 1=Graustufen; 3=RGB
    base_channels = 64        # 32 spart Speicher; 64 ist Standard
    lr = 1e-4
    batch_size = 4
    num_epochs = 30
    patience = 6
    adapt_start_epoch = 5
    step_size = 5
    gamma = 0.1
    dn_tile_h = 512
    dn_tile_w = 256
    dn_stride_w = 256
    dn_val_ratio = 0.2
    # ==========================

    # Für strikten Determinismus: Single-process Loading
    num_workers = 0 if DETERMINISTIC else 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Hyperparameter ins Log schreiben
    log_hyperparams(
        logger,
        lr=lr, batch_size=batch_size, in_channels=in_channels, base_channels=base_channels,
        tile_h=dn_tile_h, tile_w=dn_tile_w, stride_w=dn_stride_w,
        patience=patience, adapt_start_epoch=adapt_start_epoch, num_epochs=num_epochs,
        deterministic=DETERMINISTIC, num_workers=num_workers,
        step_size=step_size, gamma=gamma, val_ratio=dn_val_ratio
    )

    # Daten
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset"))
    loaders = get_data_loaders(
        DATASET_PATH,
        batch_size=batch_size,
        num_workers=num_workers,
        enable_classification=False,
        enable_denoising=True,
        dn_in_channels=in_channels,
        dn_tile_h=dn_tile_h,
        dn_tile_w=dn_tile_w,
        dn_stride_w=dn_stride_w,
        dn_val_ratio=dn_val_ratio,
        dn_seed=GLOBAL_SEED,
    )
    train_loader, val_loader, test_loader = loaders["denoising"]

    # Modell
    model = UNetCustom(in_channels=in_channels, out_channels=in_channels, base_channels=base_channels).to(device)

    # Training + validation
    history = train(
        model, train_loader, val_loader, device, results_dir,
        num_epochs=num_epochs, lr=lr,
        patience=patience, adapt_start_epoch=adapt_start_epoch,
        step_size=step_size, gamma=gamma, logger=logger
    )

    # Bestes Modell laden & Test evaluieren
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
    criterion = nn.L1Loss()
    model.eval()
    test_loss_sum = 0.0
    test_psnr_sum = 0.0
    test_ssim_sum = 0.0
    n_test_batches = 0
    win_test = None
    with torch.no_grad():
        for x_noisy, y_clean in tqdm(test_loader, desc="Test", unit="batch"):
            x_noisy = x_noisy.to(device)
            y_clean = y_clean.to(device)
            if win_test is None:
                C = x_noisy.shape[1]
                win_test = gaussian_window(kernel_size=11, sigma=1.5, channels=C).to(device)

            y_hat = model(x_noisy)
            loss = criterion(y_hat, y_clean)

            y_hat_01 = denorm_to_01(y_hat)
            y_ref_01 = denorm_to_01(y_clean)
            psnr = compute_psnr(y_hat_01, y_ref_01)
            ssim = compute_ssim(y_hat_01, y_ref_01, win_test)

            test_loss_sum += loss.item()
            test_psnr_sum += psnr
            test_ssim_sum += ssim
            n_test_batches += 1

    test_loss = test_loss_sum / max(1, n_test_batches)
    test_psnr = test_psnr_sum / max(1, n_test_batches)
    test_ssim = test_ssim_sum / max(1, n_test_batches)
    logger.info(f"[TEST] L1={test_loss:.5f}, PSNR={test_psnr:.2f}, SSIM={test_ssim:.4f}")

    # Residuen-Histogramm speichern
    save_val_residual_hist(model, val_loader, device, os.path.join(results_dir, "val_residual_hist.png"), bins=60)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
