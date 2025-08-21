################ custom split tensorboard MIL mit seed und loss regularization (tiles visualization)
# ==== Set cuBLAS determinism BEFORE importing torch ====
import os
DETERMINISTIC = True  # <- False für schnellere, nicht exakt reproduzierbare Runs
if DETERMINISTIC:
    # Muss gesetzt werden, bevor ein CUDA-Kontext entsteht
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "42"

import time
from collections import defaultdict
from typing import Dict, List, Tuple

import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO

from loadData import get_data_loaders
from createModelResNet import ResNet18Custom
from earlyStopping import EarlyStopping
from trainLogging import setup_logging


# -------------------------------
# Global seeding & determinism
# -------------------------------
GLOBAL_SEED = 42

def set_global_determinism(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False  # muss False sein für Determinismus
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)

set_global_determinism(GLOBAL_SEED, DETERMINISTIC)


# -------------------------------
# Utils
# -------------------------------
def fig_to_tensorboard_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    return image


def log_hyperparams(logger, writer, *, lr, batch_size, in_channels,
                    tile_h, tile_w, stride_w, patience, adapt_start_epoch, num_epochs,
                    weight_decay, deterministic, num_workers):
    logger.info("======== Hyperparameter ========")
    logger.info(f"Learning Rate:                    {lr}")
    logger.info(f"Batch Size:                       {batch_size}")
    logger.info(f"in_channels (1=Gray, 3=RGB):      {in_channels}")
    logger.info(f"MIL tile_h x tile_w (stride_w):   {tile_h} x {tile_w} (stride {stride_w})")
    logger.info(f"EarlyStopping patience:           {patience}")
    logger.info(f"EarlyStopping adapt_start_epoch:  {adapt_start_epoch}")
    logger.info(f"Num epochs:                       {num_epochs}")
    logger.info(f"Weight Decay (L2):                {weight_decay}")
    logger.info(f"Deterministic:                    {deterministic}")
    logger.info(f"DataLoader num_workers:           {num_workers}")
    logger.info("=================================")

    writer.add_text("config/learning_rate", str(lr))
    writer.add_text("config/batch_size", str(batch_size))
    writer.add_text("config/in_channels", str(in_channels))
    writer.add_text("config/tile", f"{tile_h}x{tile_w}, stride={stride_w}")
    writer.add_text("config/early_stopping_patience", str(patience))
    writer.add_text("config/early_stopping_adapt_start_epoch", str(adapt_start_epoch))
    writer.add_text("config/num_epochs", str(num_epochs))
    writer.add_text("config/weight_decay", str(weight_decay))
    writer.add_text("config/deterministic", str(deterministic))
    writer.add_text("config/num_workers", str(num_workers))
    try:
        writer.add_hparams(
            {
                "lr": lr,
                "batch_size": batch_size,
                "in_channels": in_channels,
                "tile_h": tile_h,
                "tile_w": tile_w,
                "stride_w": stride_w,
                "patience": patience,
                "adapt_start_epoch": adapt_start_epoch,
                "num_epochs": num_epochs,
                "weight_decay": weight_decay,
                "deterministic": int(deterministic),
                "num_workers": num_workers,
            },
            {}
        )
    except Exception as e:
        logger.warning(f"add_hparams konnte nicht geschrieben werden: {e}")


# -------------------------------
# MIL Tiling Visualization
# -------------------------------
def visualize_mil_tiling(
    dataset,
    results_dir: str,
    writer: SummaryWriter,
    num_files: int = 3,
    tag_prefix: str = "Tiling"
):
    """
    Zeichnet für die ersten `num_files` Dateien im Dataset die MIL-Tiles als
    Rechtecke ins Spektrogramm. Speichert PNGs und loggt in TensorBoard.
    """
    logger = logging.getLogger()
    os.makedirs(os.path.join(results_dir, "tiling"), exist_ok=True)

    files = dataset.files  # Liste von (path, label)
    if len(files) == 0:
        logger.warning("[Tiling] Dataset leer – keine Visualisierung erzeugt.")
        return

    # Nimm deterministisch die ersten num_files Dateien (statt zufällig)
    chosen_fids = list(range(min(num_files, len(files))))

    # Baue pro Datei die Liste der x-Offsets aus dem globalen Index
    index_by_fid: Dict[int, List[int]] = defaultdict(list)
    for fid, x_left in dataset.index:
        index_by_fid[fid].append(x_left)

    for fid in chosen_fids:
        path, label = files[fid]

        # Original laden
        with Image.open(path) as im:
            if dataset.in_channels == 1:
                im = im.convert("L")
                cmap = "gray"
            else:
                im = im.convert("RGB")
                cmap = None

            W, H = im.size
            img_np = np.array(im)

        # Plot
        fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
        if dataset.in_channels == 1:
            ax.imshow(img_np, cmap=cmap, origin='upper', aspect='auto')
        else:
            ax.imshow(img_np, origin='upper', aspect='auto')
        ax.set_title(f"File: {os.path.basename(path)} | Label: {'Noisy' if label==1 else 'Clean'}")
        ax.axis('off')

        # Tiles als Rechtecke einzeichnen
        xs = index_by_fid.get(fid, [])
        for k, x_left in enumerate(xs):
            rect = patches.Rectangle(
                (x_left, 0),
                dataset.tile_w,
                dataset.tile_h,
                linewidth=1.0,
                edgecolor='r' if k == 0 else ('orange' if k == len(xs)-1 else 'lime'),
                facecolor='none',
                alpha=0.9
            )
            ax.add_patch(rect)

        # Datei speichern & TensorBoard
        out_png = os.path.join(results_dir, "tiling", f"tiling_f{fid}_{os.path.basename(path)}.png")
        fig.savefig(out_png, bbox_inches='tight', pad_inches=0)
        tb_img = fig_to_tensorboard_image(fig)
        writer.add_images(f"{tag_prefix}/{os.path.basename(path)}", tb_img, 0)
        plt.close(fig)

    logger.info(f"[Tiling] MIL-Visualisierung für {len(chosen_fids)} Dateien erzeugt.")


# -------------------------------
# Training (Tile-Level)
# -------------------------------
def train(model, train_loader, val_loader, device, results_dir,
          num_epochs=50, lr=1e-4, patience=10, adapt_start_epoch=10,
          weight_decay=1e-4):
    logger = logging.getLogger()
    writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # L2 regularization
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopper = EarlyStopping(patience=patience, min_delta=None, verbose=True,
                                  adapt_start_epoch=adapt_start_epoch)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    start_time = time.time()

    logger.info("Starting training")
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        # Train
        model.train()
        running_loss = 0.0
        for inputs, targets, _fid in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)     # logits
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        # Val loss (tile-level, nur fürs EarlyStopping)
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _fid in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        # Bestes Modell sichern
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

        elapsed = time.time() - epoch_start
        logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

        if early_stopper(val_loss):
            logger.warning(f"Early stopping at epoch {epoch}")
            break

        scheduler.step()

    total_minutes = (time.time() - start_time) / 60
    logger.info(f"Training completed in {total_minutes:.2f} minutes")
    writer.close()

    # Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('Loss Curves with EarlyStopping')
    plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
    plt.close()


# -------------------------------
# MIL Evaluation (File-Level)
# -------------------------------
@torch.no_grad()
def _collect_file_logits(model, loader, device) -> Tuple[Dict[int, list], Dict[int, int]]:
    file_logits: Dict[int, list] = defaultdict(list)
    file_labels: Dict[int, int] = {}

    model.eval()
    for x, y, fid in tqdm(loader, desc="Collect logits (MIL)", unit="batch"):
        x = x.to(device)
        logits = model(x)  # (B,2)
        for i in range(x.size(0)):
            fid_i = int(fid[i].item())
            file_logits[fid_i].append(logits[i].detach().cpu())
            if fid_i not in file_labels:
                file_labels[fid_i] = int(y[i].item())
    return file_logits, file_labels


def _aggregate_logits_to_probs(file_logits: Dict[int, list]) -> Dict[int, float]:
    probs: Dict[int, float] = {}
    for fid, logit_list in file_logits.items():
        L = torch.stack(logit_list, dim=0)       # (N_tiles, 2)
        L_mean = L.mean(dim=0, keepdim=True)     # (1,2)
        p = torch.softmax(L_mean, dim=1)[0, 1].item()
        probs[fid] = p
    return probs


def _find_best_threshold(y_true: List[int], y_score: List[float]) -> Tuple[float, Dict]:
    best_f1, best_th = -1.0, 0.5
    for th in [i / 100.0 for i in range(0, 101)]:
        y_pred = [1 if s >= th else 0 for s in y_score]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th, {"best_f1": best_f1}


def _report_and_plot(name: str, y_true: List[int], y_pred: List[int],
                     results_dir: str, writer: SummaryWriter = None):
    logger = logging.getLogger()
    report = classification_report(y_true, y_pred, digits=4)
    logger.info(f"\n{name} (FILE / MIL) Classification Report:\n{report}\n")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Noisy'], yticklabels=['Clean', 'Noisy'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{name} Confusion Matrix (File/MIL)')
    out_path = os.path.join(results_dir, f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
    fig.savefig(out_path); plt.close(fig)

    if writer is not None:
        img_tensor = fig_to_tensorboard_image(fig)
        writer.add_images(f"{name}/Confusion_Matrix_FileMIL", img_tensor, 0)


def evaluate_file_mil(model, loader, device, name, results_dir, threshold: float,
                      writer: SummaryWriter = None):
    file_logits, file_labels = _collect_file_logits(model, loader, device)
    file_probs = _aggregate_logits_to_probs(file_logits)

    fids = sorted(file_labels.keys())
    y_true = [file_labels[f] for f in fids]
    y_score = [file_probs[f] for f in fids]
    y_pred = [1 if s >= threshold else 0 for s in y_score]

    _report_and_plot(name, y_true, y_pred, results_dir, writer)


def tune_threshold_on_val(model, val_loader, device) -> float:
    file_logits, file_labels = _collect_file_logits(model, val_loader, device)
    file_probs = _aggregate_logits_to_probs(file_logits)

    fids = sorted(file_labels.keys())
    y_true = [file_labels[f] for f in fids]
    y_score = [file_probs[f] for f in fids]

    best_th, info = _find_best_threshold(y_true, y_score)
    logging.getLogger().info(f"[Threshold] Best on Val: th={best_th:.2f}, F1={info['best_f1']:.4f}")
    return best_th


# -------------------------------
# Main
# -------------------------------
def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

    # ==== Hyperparameter ====
    lr = 1e-4
    batch_size = 16
    in_channels = 1   # Graustufen
    tile_h = 128
    tile_w = 256
    stride_w = 128
    patience = 7
    adapt_start_epoch = 5
    num_epochs = 25
    cls_val_ratio = 0.2  # 20% validation
    weight_decay = 1e-4  # L2 regularization
    # ========================

    # Für strikten Determinismus: Single-process Loading
    num_workers = 0 if DETERMINISTIC else 4

    writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    log_hyperparams(
        logger, writer,
        lr=lr, batch_size=batch_size, in_channels=in_channels,
        tile_h=tile_h, tile_w=tile_w, stride_w=stride_w,
        patience=patience, adapt_start_epoch=adapt_start_epoch, num_epochs=num_epochs,
        weight_decay=weight_decay, deterministic=DETERMINISTIC, num_workers=num_workers
    )

    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))

    loaders = get_data_loaders(
        DATASET_PATH,
        batch_size=batch_size,
        num_workers=num_workers,
        cls_in_channels=in_channels,
        cls_tile_h=tile_h,
        cls_tile_w=tile_w,
        cls_stride_w=stride_w,
        cls_val_ratio=cls_val_ratio,
        cls_seed=GLOBAL_SEED,
        enable_classification=True,
        enable_denoising=False,
    )
    train_loader, val_loader, test_loader = loaders["classification"]

    # === MIL-Tiling Visualisierung (vor dem Training) ===
    visualize_mil_tiling(train_loader.dataset, results_dir, writer, num_files=2, tag_prefix="Tiling/Train")
    visualize_mil_tiling(val_loader.dataset,   results_dir, writer, num_files=2, tag_prefix="Tiling/Val")

    # Modell
    model = ResNet18Custom(num_classes=2, in_channels=in_channels, pretrained=True).to(device)

    # Training (Tile-Level)
    train(model, train_loader, val_loader, device, results_dir,
          num_epochs=num_epochs, lr=lr, patience=patience,
          adapt_start_epoch=adapt_start_epoch, weight_decay=weight_decay)

    # Bestes Modell laden
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))

    # Threshold auf Validation optimieren (File/MIL)
    best_th = tune_threshold_on_val(model, val_loader, device)

    # Nur File/MIL-Reports
    evaluate_file_mil(model, train_loader, device, name="Train", results_dir=results_dir, threshold=best_th, writer=writer)
    evaluate_file_mil(model, val_loader, device, name="Validation", results_dir=results_dir, threshold=best_th, writer=writer)
    evaluate_file_mil(model, test_loader, device, name="Test", results_dir=results_dir, threshold=best_th, writer=writer)

    writer.close()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()














# ################ custom split tensorboard MIL mit seed und loss regularization
# # ==== Set cuBLAS determinism BEFORE importing torch ====
# import os
# DETERMINISTIC = True  # <- set False for speed / HPO runs
# if DETERMINISTIC:
#     # Must be set before any CUDA context is created
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#     os.environ["PYTHONHASHSEED"] = "42"

# import time
# from collections import defaultdict
# from typing import Dict, List, Tuple

# import random
# import numpy as np
# from PIL import Image
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix, f1_score
# from torch.utils.tensorboard import SummaryWriter
# from io import BytesIO

# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging


# # -------------------------------
# # Global seeding & determinism
# # -------------------------------
# GLOBAL_SEED = 42

# def set_global_determinism(seed: int = 42, deterministic: bool = True):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     # cuDNN switches
#     torch.backends.cudnn.deterministic = bool(deterministic)
#     torch.backends.cudnn.benchmark = False  # must be False for determinism

#     # PyTorch-level determinism (may error if a non-det op is used)
#     torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)

# set_global_determinism(GLOBAL_SEED, DETERMINISTIC)


# # -------------------------------
# # Utils
# # -------------------------------
# def fig_to_tensorboard_image(fig):
#     buf = BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     image = Image.open(buf).convert('RGB')
#     image = np.array(image).astype(np.float32) / 255.0
#     image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
#     return image


# def log_hyperparams(logger, writer, *, lr, batch_size, in_channels,
#                     tile_h, tile_w, stride_w, patience, adapt_start_epoch, num_epochs,
#                     weight_decay, deterministic, num_workers):
#     logger.info("======== Hyperparameter ========")
#     logger.info(f"Learning Rate:                    {lr}")
#     logger.info(f"Batch Size:                       {batch_size}")
#     logger.info(f"in_channels (1=Gray, 3=RGB):      {in_channels}")
#     logger.info(f"MIL tile_h x tile_w (stride_w):   {tile_h} x {tile_w} (stride {stride_w})")
#     logger.info(f"EarlyStopping patience:           {patience}")
#     logger.info(f"EarlyStopping adapt_start_epoch:  {adapt_start_epoch}")
#     logger.info(f"Num epochs:                       {num_epochs}")
#     logger.info(f"Weight Decay (L2):                {weight_decay}")
#     logger.info(f"Deterministic:                    {deterministic}")
#     logger.info(f"DataLoader num_workers:           {num_workers}")
#     logger.info("=================================")

#     writer.add_text("config/learning_rate", str(lr))
#     writer.add_text("config/batch_size", str(batch_size))
#     writer.add_text("config/in_channels", str(in_channels))
#     writer.add_text("config/tile", f"{tile_h}x{tile_w}, stride={stride_w}")
#     writer.add_text("config/early_stopping_patience", str(patience))
#     writer.add_text("config/early_stopping_adapt_start_epoch", str(adapt_start_epoch))
#     writer.add_text("config/num_epochs", str(num_epochs))
#     writer.add_text("config/weight_decay", str(weight_decay))
#     writer.add_text("config/deterministic", str(deterministic))
#     writer.add_text("config/num_workers", str(num_workers))
#     try:
#         writer.add_hparams(
#             {
#                 "lr": lr,
#                 "batch_size": batch_size,
#                 "in_channels": in_channels,
#                 "tile_h": tile_h,
#                 "tile_w": tile_w,
#                 "stride_w": stride_w,
#                 "patience": patience,
#                 "adapt_start_epoch": adapt_start_epoch,
#                 "num_epochs": num_epochs,
#                 "weight_decay": weight_decay,
#                 "deterministic": int(deterministic),
#                 "num_workers": num_workers,
#             },
#             {}
#         )
#     except Exception as e:
#         logger.warning(f"add_hparams konnte nicht geschrieben werden: {e}")


# # -------------------------------
# # Training (Tile-Level)
# # -------------------------------
# def train(model, train_loader, val_loader, device, results_dir,
#           num_epochs=50, lr=1e-4, patience=10, adapt_start_epoch=10,
#           weight_decay=1e-4):
#     logger = logging.getLogger()
#     writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # L2 regularization
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#     early_stopper = EarlyStopping(patience=patience, min_delta=None, verbose=True,
#                                   adapt_start_epoch=adapt_start_epoch)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info("Starting training")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

#         # Train
#         model.train()
#         running_loss = 0.0
#         for inputs, targets, _fid in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)     # logits
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)
#         writer.add_scalar("Loss/Train", train_loss, epoch)

#         # Val loss (tile-level, nur fürs EarlyStopping)
#         model.eval()
#         running_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets, _fid in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 running_loss += loss.item() * inputs.size(0)
#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)
#         writer.add_scalar("Loss/Val", val_loss, epoch)

#         # Bestes Modell sichern
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.warning(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training completed in {total_minutes:.2f} minutes")
#     writer.close()

#     # Loss Curves
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
#     plt.title('Loss Curves with EarlyStopping')
#     plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
#     plt.close()


# # -------------------------------
# # MIL Evaluation (File-Level)
# # -------------------------------
# @torch.no_grad()
# def _collect_file_logits(model, loader, device) -> Tuple[Dict[int, list], Dict[int, int]]:
#     file_logits: Dict[int, list] = defaultdict(list)
#     file_labels: Dict[int, int] = {}

#     model.eval()
#     for x, y, fid in tqdm(loader, desc="Collect logits (MIL)", unit="batch"):
#         x = x.to(device)
#         logits = model(x)  # (B,2)
#         for i in range(x.size(0)):
#             fid_i = int(fid[i].item())
#             file_logits[fid_i].append(logits[i].detach().cpu())
#             if fid_i not in file_labels:
#                 file_labels[fid_i] = int(y[i].item())
#     return file_logits, file_labels


# def _aggregate_logits_to_probs(file_logits: Dict[int, list]) -> Dict[int, float]:
#     probs: Dict[int, float] = {}
#     for fid, logit_list in file_logits.items():
#         L = torch.stack(logit_list, dim=0)       # (N_tiles, 2)
#         L_mean = L.mean(dim=0, keepdim=True)     # (1,2)
#         p = torch.softmax(L_mean, dim=1)[0, 1].item()
#         probs[fid] = p
#     return probs


# def _find_best_threshold(y_true: List[int], y_score: List[float]) -> Tuple[float, Dict]:
#     best_f1, best_th = -1.0, 0.5
#     for th in [i / 100.0 for i in range(0, 101)]:
#         y_pred = [1 if s >= th else 0 for s in y_score]
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#         if f1 > best_f1:
#             best_f1, best_th = f1, th
#     return best_th, {"best_f1": best_f1}


# def _report_and_plot(name: str, y_true: List[int], y_pred: List[int],
#                      results_dir: str, writer: SummaryWriter = None):
#     logger = logging.getLogger()
#     report = classification_report(y_true, y_pred, digits=4)
#     logger.info(f"\n{name} (FILE / MIL) Classification Report:\n{report}\n")

#     cm = confusion_matrix(y_true, y_pred)
#     fig, ax = plt.subplots(figsize=(5, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean', 'Noisy'], yticklabels=['Clean', 'Noisy'], ax=ax)
#     ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{name} Confusion Matrix (File/MIL)')
#     out_path = os.path.join(results_dir, f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
#     fig.savefig(out_path); plt.close(fig)

#     if writer is not None:
#         img_tensor = fig_to_tensorboard_image(fig)
#         writer.add_images(f"{name}/Confusion_Matrix_FileMIL", img_tensor, 0)


# def evaluate_file_mil(model, loader, device, name, results_dir, threshold: float,
#                       writer: SummaryWriter = None):
#     file_logits, file_labels = _collect_file_logits(model, loader, device)
#     file_probs = _aggregate_logits_to_probs(file_logits)

#     fids = sorted(file_labels.keys())
#     y_true = [file_labels[f] for f in fids]
#     y_score = [file_probs[f] for f in fids]
#     y_pred = [1 if s >= threshold else 0 for s in y_score]

#     _report_and_plot(name, y_true, y_pred, results_dir, writer)


# def tune_threshold_on_val(model, val_loader, device) -> float:
#     file_logits, file_labels = _collect_file_logits(model, val_loader, device)
#     file_probs = _aggregate_logits_to_probs(file_logits)

#     fids = sorted(file_labels.keys())
#     y_true = [file_labels[f] for f in fids]
#     y_score = [file_probs[f] for f in fids]

#     best_th, info = _find_best_threshold(y_true, y_score)
#     logging.getLogger().info(f"[Threshold] Best on Val: th={best_th:.2f}, F1={info['best_f1']:.4f}")
#     return best_th


# # -------------------------------
# # Main
# # -------------------------------
# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     os.makedirs(results_dir, exist_ok=True)
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     # ==== Hyperparameter ====
#     lr = 1e-4
#     batch_size = 16
#     in_channels = 1   # Graustufen
#     tile_h = 128
#     tile_w = 256
#     stride_w = 128
#     patience = 8
#     adapt_start_epoch = 5
#     num_epochs = 25
#     weight_decay = 1e-4  # L2 regularization
#     # ========================

#     # Deterministic choice affects num_workers
#     num_workers = 0 if DETERMINISTIC else 4

#     writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     log_hyperparams(
#         logger, writer,
#         lr=lr, batch_size=batch_size, in_channels=in_channels,
#         tile_h=tile_h, tile_w=tile_w, stride_w=stride_w,
#         patience=patience, adapt_start_epoch=adapt_start_epoch, num_epochs=num_epochs,
#         weight_decay=weight_decay, deterministic=DETERMINISTIC, num_workers=num_workers
#     )

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))

#     loaders = get_data_loaders(
#         DATASET_PATH,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         cls_in_channels=in_channels,
#         cls_tile_h=tile_h,
#         cls_tile_w=tile_w,
#         cls_stride_w=stride_w,
#         cls_resize_height=True,
#         cls_val_ratio=0.2,
#         cls_seed=GLOBAL_SEED,
#         enable_classification=True,
#         enable_denoising=False,
#     )
#     train_loader, val_loader, test_loader = loaders["classification"]

#     # Modell
#     model = ResNet18Custom(num_classes=2, in_channels=in_channels, pretrained=True).to(device)

#     # Training (Tile-Level)
#     train(model, train_loader, val_loader, device, results_dir,
#           num_epochs=num_epochs, lr=lr, patience=patience,
#           adapt_start_epoch=adapt_start_epoch, weight_decay=weight_decay)

#     # Bestes Modell laden
#     model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))

#     # Threshold auf Validation optimieren (File/MIL)
#     best_th = tune_threshold_on_val(model, val_loader, device)

#     # Nur File/MIL-Reports
#     evaluate_file_mil(model, train_loader, device, name="Train", results_dir=results_dir, threshold=best_th, writer=writer)
#     evaluate_file_mil(model, val_loader, device, name="Validation", results_dir=results_dir, threshold=best_th, writer=writer)
#     evaluate_file_mil(model, test_loader, device, name="Test", results_dir=results_dir, threshold=best_th, writer=writer)

#     writer.close()


# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()









# ################ custom split tensorboard MIL mit seed
# import os
# import time
# import random
# from collections import defaultdict
# from typing import List, Tuple, Dict

# import numpy as np
# from PIL import Image
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix, f1_score
# from io import BytesIO

# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging


# # -------------------------------
# # Repro: Global determinism
# # -------------------------------
# def set_global_determinism(seed: int = 42):
#     """
#     Stellt (so gut wie möglich) reproduzierbare Ergebnisse sicher:
#     - Seeds für Python, NumPy, PyTorch (CPU/GPU)
#     - Deterministische/cuDNN-Einstellungen
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # no-op auf CPU

#     # Für deterministisches Verhalten in cuDNN
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # Optional (PyTorch >= 1.8), kann minimalen Overhead erzeugen
#     try:
#         torch.use_deterministic_algorithms(True)
#     except Exception:
#         pass


# # -------------------------------
# # Utils
# # -------------------------------
# def fig_to_tensorboard_image(fig):
#     buf = BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     image = Image.open(buf).convert('RGB')
#     image = np.array(image).astype(np.float32) / 255.0
#     image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
#     return image


# def log_hyperparams(logger, writer, *, lr, batch_size, in_channels,
#                     tile_h, tile_w, stride_w, patience, adapt_start_epoch,
#                     num_epochs, seed):
#     logger.info("======== Hyperparameter ========")
#     logger.info(f"Learning Rate:                    {lr}")
#     logger.info(f"Batch Size:                       {batch_size}")
#     logger.info(f"in_channels (1=Gray, 3=RGB):      {in_channels}")
#     logger.info(f"MIL tile_h x tile_w (stride_w):   {tile_h} x {tile_w} (stride {stride_w})")
#     logger.info(f"EarlyStopping patience:           {patience}")
#     logger.info(f"EarlyStopping adapt_start_epoch:  {adapt_start_epoch}")
#     logger.info(f"Num epochs:                       {num_epochs}")
#     logger.info(f"Global/Loader Seed:               {seed}")
#     logger.info("=================================")

#     # TB Text
#     writer.add_text("config/learning_rate", str(lr))
#     writer.add_text("config/batch_size", str(batch_size))
#     writer.add_text("config/in_channels", str(in_channels))
#     writer.add_text("config/tile", f"{tile_h}x{tile_w}, stride={stride_w}")
#     writer.add_text("config/early_stopping_patience", str(patience))
#     writer.add_text("config/early_stopping_adapt_start_epoch", str(adapt_start_epoch))
#     writer.add_text("config/num_epochs", str(num_epochs))
#     writer.add_text("config/seed", str(seed))
#     try:
#         writer.add_hparams(
#             {
#                 "lr": lr,
#                 "batch_size": batch_size,
#                 "in_channels": in_channels,
#                 "tile_h": tile_h,
#                 "tile_w": tile_w,
#                 "stride_w": stride_w,
#                 "patience": patience,
#                 "adapt_start_epoch": adapt_start_epoch,
#                 "num_epochs": num_epochs,
#                 "seed": seed,
#             },
#             {}
#         )
#     except Exception as e:
#         logger.warning(f"add_hparams konnte nicht geschrieben werden: {e}")


# # -------------------------------
# # Training (Tile-Level)
# # -------------------------------
# def train(model, train_loader, val_loader, device, results_dir,
#           num_epochs=50, lr=1e-4, patience=10, adapt_start_epoch=10):
#     logger = logging.getLogger()
#     writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#     early_stopper = EarlyStopping(patience=patience, min_delta=None, verbose=True,
#                                   adapt_start_epoch=adapt_start_epoch)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info("Starting training")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

#         # Train
#         model.train()
#         running_loss = 0.0
#         for inputs, targets, _fid in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)     # logits
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)
#         writer.add_scalar("Loss/Train", train_loss, epoch)

#         # Val loss (tile-level, nur fürs EarlyStopping)
#         model.eval()
#         running_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets, _fid in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 running_loss += loss.item() * inputs.size(0)
#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)
#         writer.add_scalar("Loss/Val", val_loss, epoch)

#         # Bestes Modell auf Val-Loss sichern (stabil und günstig)
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.warning(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training completed in {total_minutes:.2f} minutes")
#     writer.close()

#     # Loss Curves
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
#     plt.title('Loss Curves with EarlyStopping')
#     plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
#     plt.close()


# # -------------------------------
# # MIL Evaluation (File-Level)
# # -------------------------------
# @torch.no_grad()
# def _collect_file_logits(model, loader, device) -> Tuple[Dict[int, list], Dict[int, int]]:
#     """
#     Führt Inferenz über Tiles aus und sammelt pro Datei (file_id) die Logits-Liste.
#     Gibt (file_logits_dict, file_label_dict) zurück.
#     """
#     file_logits: Dict[int, list] = defaultdict(list)
#     file_labels: Dict[int, int] = {}

#     model.eval()
#     for x, y, fid in tqdm(loader, desc="Collect logits (MIL)", unit="batch"):
#         x = x.to(device)
#         logits = model(x)  # (B,2)
#         for i in range(x.size(0)):
#             fid_i = int(fid[i].item())
#             file_logits[fid_i].append(logits[i].detach().cpu())
#             # gleiche Datei hat überall denselben Label
#             if fid_i not in file_labels:
#                 file_labels[fid_i] = int(y[i].item())
#     return file_logits, file_labels


# def _aggregate_logits_to_probs(file_logits: Dict[int, list]) -> Dict[int, float]:
#     """
#     Aggregation: mean der Logits pro Datei → Softmax → p(noisy).
#     """
#     probs: Dict[int, float] = {}
#     for fid, logit_list in file_logits.items():
#         # (N_tiles, 2)
#         L = torch.stack(logit_list, dim=0)       # auf CPU
#         L_mean = L.mean(dim=0, keepdim=True)     # (1,2)
#         p = torch.softmax(L_mean, dim=1)[0, 1].item()
#         probs[fid] = p
#     return probs


# def _find_best_threshold(y_true: List[int], y_score: List[float]) -> Tuple[float, Dict]:
#     """
#     Sweep von 0.00..1.00 (Step 0.01) und wähle Threshold mit maximalem F1.
#     """
#     best_f1, best_th = -1.0, 0.5
#     for th in [i / 100.0 for i in range(0, 101)]:
#         y_pred = [1 if s >= th else 0 for s in y_score]
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#         if f1 > best_f1:
#             best_f1, best_th = f1, th
#     return best_th, {"best_f1": best_f1}


# def _report_and_plot(name: str, y_true: List[int], y_pred: List[int],
#                      results_dir: str, writer: SummaryWriter = None):
#     logger = logging.getLogger()
#     report = classification_report(y_true, y_pred, digits=4)
#     logger.info(f"\n{name} (FILE / MIL) Classification Report:\n{report}\n")

#     cm = confusion_matrix(y_true, y_pred)
#     fig, ax = plt.subplots(figsize=(5, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean', 'Noisy'], yticklabels=['Clean', 'Noisy'], ax=ax)
#     ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{name} Confusion Matrix (File/MIL)')
#     out_path = os.path.join(results_dir, f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
#     fig.savefig(out_path); plt.close(fig)

#     if writer is not None:
#         img_tensor = fig_to_tensorboard_image(fig)
#         writer.add_images(f"{name}/Confusion_Matrix_FileMIL", img_tensor, 0)


# def evaluate_file_mil(model, loader, device, name, results_dir, threshold: float,
#                       writer: SummaryWriter = None):
#     """
#     Eval auf Datei-Ebene (MIL) mit vorgegebenem Threshold.
#     """
#     file_logits, file_labels = _collect_file_logits(model, loader, device)
#     file_probs = _aggregate_logits_to_probs(file_logits)

#     # konsistente Reihenfolge
#     fids = sorted(file_labels.keys())
#     y_true = [file_labels[f] for f in fids]
#     y_score = [file_probs[f] for f in fids]
#     y_pred = [1 if s >= threshold else 0 for s in y_score]

#     _report_and_plot(name, y_true, y_pred, results_dir, writer)


# def tune_threshold_on_val(model, val_loader, device) -> float:
#     """
#     Sammelt Val-File-Probs und bestimmt bestes Threshold (max F1).
#     """
#     file_logits, file_labels = _collect_file_logits(model, val_loader, device)
#     file_probs = _aggregate_logits_to_probs(file_logits)

#     fids = sorted(file_labels.keys())
#     y_true = [file_labels[f] for f in fids]
#     y_score = [file_probs[f] for f in fids]

#     best_th, info = _find_best_threshold(y_true, y_score)
#     logging.getLogger().info(f"[Threshold] Best on Val: th={best_th:.2f}, F1={info['best_f1']:.4f}")
#     return best_th


# # -------------------------------
# # Main
# # -------------------------------
# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     os.makedirs(results_dir, exist_ok=True)
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     # ==== Hyperparameter ====
#     seed = 42
#     lr = 1e-4
#     batch_size = 8
#     in_channels = 1   # Graustufen
#     tile_h = 128
#     tile_w = 256
#     stride_w = 128
#     patience = 8
#     adapt_start_epoch = 5
#     num_epochs = 50
#     # ========================

#     # Global determinism
#     set_global_determinism(seed)

#     writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     log_hyperparams(
#         logger, writer,
#         lr=lr, batch_size=batch_size, in_channels=in_channels,
#         tile_h=tile_h, tile_w=tile_w, stride_w=stride_w,
#         patience=patience, adapt_start_epoch=adapt_start_epoch,
#         num_epochs=num_epochs, seed=seed
#     )

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))

#     # Loader aus neuem loadData.py (enthält worker-seeding + deterministischen Split)
#     loaders = get_data_loaders(
#         DATASET_PATH,
#         batch_size=batch_size,
#         num_workers=4,
#         cls_in_channels=in_channels,
#         cls_tile_h=tile_h,
#         cls_tile_w=tile_w,
#         cls_stride_w=stride_w,
#         cls_resize_height=True,
#         cls_val_ratio=0.2,
#         cls_seed=seed,
#         enable_classification=True,
#         enable_denoising=False,
#     )
#     train_loader, val_loader, test_loader = loaders["classification"]

#     # Modell
#     model = ResNet18Custom(num_classes=2, in_channels=in_channels, pretrained=True).to(device)

#     # Training (Tile-Level)
#     train(
#         model, train_loader, val_loader, device, results_dir,
#         num_epochs=num_epochs, lr=lr, patience=patience, adapt_start_epoch=adapt_start_epoch
#     )

#     # Bestes Modell laden
#     model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))

#     # Threshold auf Validation optimieren (File/MIL)
#     best_th = tune_threshold_on_val(model, val_loader, device)

#     # Nur File/MIL-Reports
#     evaluate_file_mil(model, train_loader, device, name="Train", results_dir=results_dir, threshold=best_th, writer=writer)
#     evaluate_file_mil(model, val_loader, device, name="Validation", results_dir=results_dir, threshold=best_th, writer=writer)
#     evaluate_file_mil(model, test_loader, device, name="Test", results_dir=results_dir, threshold=best_th, writer=writer)

#     writer.close()


# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()
















# ################ custom split tensorboard MIL
# import os
# import time
# from collections import defaultdict
# from typing import List, Tuple, Dict

# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix, f1_score
# from torch.utils.tensorboard import SummaryWriter
# from PIL import Image
# import numpy as np
# from io import BytesIO

# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging


# # -------------------------------
# # Utils
# # -------------------------------
# def fig_to_tensorboard_image(fig):
#     buf = BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     image = Image.open(buf).convert('RGB')
#     image = np.array(image).astype(np.float32) / 255.0
#     image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
#     return image


# def log_hyperparams(logger, writer, *, lr, batch_size, in_channels,
#                     tile_h, tile_w, stride_w, patience, adapt_start_epoch, num_epochs):
#     logger.info("======== Hyperparameter ========")
#     logger.info(f"Learning Rate:                    {lr}")
#     logger.info(f"Batch Size:                       {batch_size}")
#     logger.info(f"in_channels (1=Gray, 3=RGB):      {in_channels}")
#     logger.info(f"MIL tile_h x tile_w (stride_w):   {tile_h} x {tile_w} (stride {stride_w})")
#     logger.info(f"EarlyStopping patience:           {patience}")
#     logger.info(f"EarlyStopping adapt_start_epoch:  {adapt_start_epoch}")
#     logger.info(f"Num epochs:                        {num_epochs}")
#     logger.info("=================================")

#     # TB Text
#     writer.add_text("config/learning_rate", str(lr))
#     writer.add_text("config/batch_size", str(batch_size))
#     writer.add_text("config/in_channels", str(in_channels))
#     writer.add_text("config/tile", f"{tile_h}x{tile_w}, stride={stride_w}")
#     writer.add_text("config/early_stopping_patience", str(patience))
#     writer.add_text("config/early_stopping_adapt_start_epoch", str(adapt_start_epoch))
#     writer.add_text("config/num_epochs", str(num_epochs))
#     try:
#         writer.add_hparams(
#             {
#                 "lr": lr,
#                 "batch_size": batch_size,
#                 "in_channels": in_channels,
#                 "tile_h": tile_h,
#                 "tile_w": tile_w,
#                 "stride_w": stride_w,
#                 "patience": patience,
#                 "adapt_start_epoch": adapt_start_epoch,
#                 "num_epochs": num_epochs,
#             },
#             {}
#         )
#     except Exception as e:
#         logger.warning(f"add_hparams konnte nicht geschrieben werden: {e}")


# # -------------------------------
# # Training (Tile-Level)
# # -------------------------------
# def train(model, train_loader, val_loader, device, results_dir,
#           num_epochs=50, lr=1e-4, patience=10, adapt_start_epoch=10):
#     logger = logging.getLogger()
#     writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#     early_stopper = EarlyStopping(patience=patience, min_delta=None, verbose=True,
#                                   adapt_start_epoch=adapt_start_epoch)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info("Starting training")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

#         # Train
#         model.train()
#         running_loss = 0.0
#         for inputs, targets, _fid in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)     # logits
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)
#         writer.add_scalar("Loss/Train", train_loss, epoch)

#         # Val loss (tile-level, nur fürs EarlyStopping)
#         model.eval()
#         running_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets, _fid in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 running_loss += loss.item() * inputs.size(0)
#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)
#         writer.add_scalar("Loss/Val", val_loss, epoch)

#         # Bestes Modell auf Val-Loss sichern (stabil und günstig)
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.warning(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training completed in {total_minutes:.2f} minutes")
#     writer.close()

#     # Loss Curves
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
#     plt.title('Loss Curves with EarlyStopping')
#     plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
#     plt.close()


# # -------------------------------
# # MIL Evaluation (File-Level)
# # -------------------------------
# @torch.no_grad()
# def _collect_file_logits(model, loader, device) -> Tuple[Dict[int, list], Dict[int, int]]:
#     """
#     Führt Inferenz über Tiles aus und sammelt pro Datei (file_id) die Logits-Liste.
#     Gibt (file_logits_dict, file_label_dict) zurück.
#     """
#     file_logits: Dict[int, list] = defaultdict(list)
#     file_labels: Dict[int, int] = {}

#     model.eval()
#     for x, y, fid in tqdm(loader, desc="Collect logits (MIL)", unit="batch"):
#         x = x.to(device)
#         logits = model(x)  # (B,2)
#         for i in range(x.size(0)):
#             fid_i = int(fid[i].item())
#             file_logits[fid_i].append(logits[i].detach().cpu())
#             # gleiche Datei hat überall denselben Label
#             if fid_i not in file_labels:
#                 file_labels[fid_i] = int(y[i].item())
#     return file_logits, file_labels


# def _aggregate_logits_to_probs(file_logits: Dict[int, list]) -> Dict[int, float]:
#     """
#     Aggregation: mean der Logits pro Datei → Softmax → p(noisy).
#     """
#     probs: Dict[int, float] = {}
#     for fid, logit_list in file_logits.items():
#         # (N_tiles, 2)
#         L = torch.stack(logit_list, dim=0)       # auf CPU
#         L_mean = L.mean(dim=0, keepdim=True)     # (1,2)
#         p = torch.softmax(L_mean, dim=1)[0, 1].item()
#         probs[fid] = p
#     return probs


# def _find_best_threshold(y_true: List[int], y_score: List[float]) -> Tuple[float, Dict]:
#     """
#     Sweep von 0.00..1.00 (Step 0.01) und wähle Threshold mit maximalem F1.
#     """
#     best_f1, best_th = -1.0, 0.5
#     for th in [i / 100.0 for i in range(0, 101)]:
#         y_pred = [1 if s >= th else 0 for s in y_score]
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#         if f1 > best_f1:
#             best_f1, best_th = f1, th
#     return best_th, {"best_f1": best_f1}


# def _report_and_plot(name: str, y_true: List[int], y_pred: List[int],
#                      results_dir: str, writer: SummaryWriter = None):
#     logger = logging.getLogger()
#     report = classification_report(y_true, y_pred, digits=4)
#     logger.info(f"\n{name} (FILE / MIL) Classification Report:\n{report}\n")

#     cm = confusion_matrix(y_true, y_pred)
#     fig, ax = plt.subplots(figsize=(5, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean', 'Noisy'], yticklabels=['Clean', 'Noisy'], ax=ax)
#     ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{name} Confusion Matrix (File/MIL)')
#     out_path = os.path.join(results_dir, f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
#     fig.savefig(out_path); plt.close(fig)

#     if writer is not None:
#         img_tensor = fig_to_tensorboard_image(fig)
#         writer.add_images(f"{name}/Confusion_Matrix_FileMIL", img_tensor, 0)


# def evaluate_file_mil(model, loader, device, name, results_dir, threshold: float,
#                       writer: SummaryWriter = None):
#     """
#     Eval auf Datei-Ebene (MIL) mit vorgegebenem Threshold.
#     """
#     file_logits, file_labels = _collect_file_logits(model, loader, device)
#     file_probs = _aggregate_logits_to_probs(file_logits)

#     # konsistente Reihenfolge
#     fids = sorted(file_labels.keys())
#     y_true = [file_labels[f] for f in fids]
#     y_score = [file_probs[f] for f in fids]
#     y_pred = [1 if s >= threshold else 0 for s in y_score]

#     _report_and_plot(name, y_true, y_pred, results_dir, writer)


# def tune_threshold_on_val(model, val_loader, device) -> float:
#     """
#     Sammelt Val-File-Probs und bestimmt bestes Threshold (max F1).
#     """
#     file_logits, file_labels = _collect_file_logits(model, val_loader, device)
#     file_probs = _aggregate_logits_to_probs(file_logits)

#     fids = sorted(file_labels.keys())
#     y_true = [file_labels[f] for f in fids]
#     y_score = [file_probs[f] for f in fids]

#     best_th, info = _find_best_threshold(y_true, y_score)
#     logging.getLogger().info(f"[Threshold] Best on Val: th={best_th:.2f}, F1={info['best_f1']:.4f}")
#     return best_th


# # -------------------------------
# # Main
# # -------------------------------
# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     os.makedirs(results_dir, exist_ok=True)
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     # ==== Hyperparameter ====
#     lr = 1e-4
#     batch_size = 8
#     in_channels = 1   # Graustufen
#     tile_h = 128
#     tile_w = 256
#     stride_w = 128
#     patience = 8
#     adapt_start_epoch = 5
#     num_epochs = 50
#     # ========================

#     writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     log_hyperparams(logger, writer, lr=lr, batch_size=batch_size, in_channels=in_channels,
#                     tile_h=tile_h, tile_w=tile_w, stride_w=stride_w,
#                     patience=patience, adapt_start_epoch=adapt_start_epoch, num_epochs=num_epochs)

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))

#     loaders = get_data_loaders(
#         DATASET_PATH,
#         batch_size=batch_size,
#         num_workers=4,
#         cls_in_channels=in_channels,
#         cls_tile_h=tile_h,
#         cls_tile_w=tile_w,
#         cls_stride_w=stride_w,
#         cls_resize_height=True,
#         cls_val_ratio=0.2,
#         cls_seed=42,
#         enable_classification=True,
#         enable_denoising=False,
#     )
#     train_loader, val_loader, test_loader = loaders["classification"]

#     # Modell
#     model = ResNet18Custom(num_classes=2, in_channels=in_channels, pretrained=True).to(device)

#     # Training (Tile-Level)
#     train(model, train_loader, val_loader, device, results_dir,
#           num_epochs=num_epochs, lr=lr, patience=patience, adapt_start_epoch=adapt_start_epoch)

#     # Bestes Modell laden
#     model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))

#     # Threshold auf Validation optimieren (File/MIL)
#     best_th = tune_threshold_on_val(model, val_loader, device)

#     # Nur File/MIL-Reports (keine Tile-Reports mehr)
#     evaluate_file_mil(model, train_loader, device, name="Train", results_dir=results_dir, threshold=best_th, writer=writer)
#     evaluate_file_mil(model, val_loader, device, name="Validation", results_dir=results_dir, threshold=best_th, writer=writer)
#     evaluate_file_mil(model, test_loader, device, name="Test", results_dir=results_dir, threshold=best_th, writer=writer)

#     writer.close()


# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()













# ################ custom split tensorboard
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix
# from torch.utils.tensorboard import SummaryWriter
# from PIL import Image
# import numpy as np
# from io import BytesIO

# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging


# def fig_to_tensorboard_image(fig):
#     """Konvertiere matplotlib-Figure zu TensorBoard-kompatiblem Tensor."""
#     buf = BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     image = Image.open(buf).convert('RGB')
#     image = np.array(image).astype(np.float32) / 255.0
#     image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
#     return image


# def log_hyperparams(logger, writer, *, lr, batch_size, image_size, in_channels, patience, adapt_start_epoch):
#     """Logge Hyperparameter in Terminal/Datei und TensorBoard."""
#     logger.info("======== Hyperparameter ========")
#     logger.info(f"Learning Rate:           {lr}")
#     logger.info(f"Batch Size:                   {batch_size}")
#     logger.info(f"Image Size:                   {image_size}x{image_size}")
#     logger.info(f"in_channels (1=Gray,3=RGB):   {in_channels}")
#     logger.info(f"EarlyStopping patience:       {patience}")
#     logger.info(f"EarlyStopping adapt_start_epoch: {adapt_start_epoch}")
#     logger.info("================================")

#     # Im TensorBoard als Text ablegen (gut sichtbar im SCALARS/TEXT)
#     writer.add_text("config/learning_rate", str(lr))
#     writer.add_text("config/batch_size", str(batch_size))
#     writer.add_text("config/image_size", f"{image_size}x{image_size}")
#     writer.add_text("config/in_channels", str(in_channels))
#     writer.add_text("config/early_stopping_patience", str(patience))
#     writer.add_text("config/early_stopping_adapt_start_epoch", str(adapt_start_epoch))

#     # Optional: HParams-Tab in TensorBoard
#     try:
#         writer.add_hparams(
#             {
#                 "lr": lr,
#                 "batch_size": batch_size,
#                 "image_size": image_size,
#                 "in_channels": in_channels,
#                 "patience": patience,
#                 "adapt_start_epoch": adapt_start_epoch,
#             },
#             {}
#         )
#     except Exception as e:
#         logger.warning(f"add_hparams konnte nicht geschrieben werden: {e}")


# def train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4,
#           patience=10, adapt_start_epoch=10):
#     logger = logging.getLogger()
#     writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#     early_stopper = EarlyStopping(patience=patience, min_delta=None, verbose=True, adapt_start_epoch=adapt_start_epoch)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info("Starting training")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

#         model.train()
#         running_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)
#         writer.add_scalar("Loss/Train", train_loss, epoch)

#         model.eval()
#         running_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 running_loss += loss.item() * inputs.size(0)

#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)
#         writer.add_scalar("Loss/Val", val_loss, epoch)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.warning(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training completed in {total_minutes:.2f} minutes")
#     writer.close()

#     # Plot Loss Curves
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Loss Curves with EarlyStopping')
#     plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
#     plt.close()


# def evaluate(model, loader, device, name, results_dir, writer=None):
#     logger = logging.getLogger()
#     model.eval()
#     all_preds, all_targets = [], []

#     with torch.no_grad():
#         for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())

#     report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
#     df = pd.DataFrame(report_dict).transpose()
#     logger.info(f"\n{name} Classification Report:\n{df.to_string()}\n")

#     if writer is not None:
#         # Log Klassenmetriken als Scalars
#         for cls in ['0', '1']:
#             writer.add_scalar(f"{name}/Precision_class_{cls}", report_dict[cls]['precision'], 0)
#             writer.add_scalar(f"{name}/Recall_class_{cls}", report_dict[cls]['recall'], 0)
#             writer.add_scalar(f"{name}/F1_class_{cls}", report_dict[cls]['f1-score'], 0)
#         # Macro/Weighted
#         writer.add_scalar(f"{name}/F1_macro", report_dict["macro avg"]["f1-score"], 0)
#         writer.add_scalar(f"{name}/F1_weighted", report_dict["weighted avg"]["f1-score"], 0)
#         writer.add_scalar(f"{name}/Accuracy", report_dict["accuracy"], 0)

#     cm = confusion_matrix(all_targets, all_preds)
#     fig, ax = plt.subplots(figsize=(5, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean', 'Noisy'], yticklabels=['Clean', 'Noisy'], ax=ax)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_title(f'{name} Confusion Matrix')

#     cm_path = os.path.join(results_dir, f'{name.lower()}_confusion_matrix.png')
#     fig.savefig(cm_path)
#     plt.close(fig)

#     if writer is not None:
#         img_tensor = fig_to_tensorboard_image(fig)
#         writer.add_images(f"{name}/Confusion_Matrix", img_tensor, 0)


# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     os.makedirs(results_dir, exist_ok=True)
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     # ==== Hyperparameter ====
#     lr = 1e-4
#     batch_size = 8
#     image_size = 256
#     in_channels = 1           # 1 = Graustufen, 3 = RGB
#     patience = 10
#     adapt_start_epoch = 10
#     num_epochs = 50
#     # ==========================================

#     writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     # Hyperparameter in Terminal/Datei + TensorBoard loggen
#     log_hyperparams(
#         logger, writer,
#         lr=lr, batch_size=batch_size, image_size=image_size, in_channels=in_channels,
#         patience=patience, adapt_start_epoch=adapt_start_epoch
#     )

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, batch_size=batch_size, image_size=image_size, in_channels=in_channels)
#     train_loader, val_loader, test_loader = loaders["classification"]

#     model = ResNet18Custom(num_classes=2, in_channels=in_channels, pretrained=True).to(device)
#     train(model, train_loader, val_loader, device, results_dir,
#           num_epochs=num_epochs, lr=lr, patience=patience, adapt_start_epoch=adapt_start_epoch)

#     model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
#     evaluate(model, train_loader, device, name="Train", results_dir=results_dir, writer=writer)
#     evaluate(model, val_loader, device, name="Validation", results_dir=results_dir, writer=writer)
#     evaluate(model, test_loader, device, name="Test", results_dir=results_dir, writer=writer)

#     writer.close()


# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()









# ################## custom split ohne early stopping
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix

# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from trainLogging import setup_logging

# def train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4):
#     logger = logging.getLogger()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info("Starting training")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

#         model.train()
#         running_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)

#         model.eval()
#         running_loss = 0.0
#         for inputs, targets in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             running_loss += loss.item() * inputs.size(0)

#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training completed in {total_minutes:.2f} minutes")

#     plt.figure(figsize=(8,5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Loss Curves (No EarlyStopping)')
#     plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
#     plt.close()

# def evaluate(model, loader, device, name, results_dir):
#     logger = logging.getLogger()
#     model.eval()
#     all_preds, all_targets = [], []
#     for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_targets.extend(targets.cpu().numpy())

#     report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
#     df = pd.DataFrame(report_dict).transpose()
#     logger.info(f"\n{name} Classification Report:\n{df.to_string()}\n")

#     cm = confusion_matrix(all_targets, all_preds)
#     plt.figure(figsize=(5,4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean','Noisy'], yticklabels=['Clean','Noisy'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(f'{name} Confusion Matrix')
#     plt.savefig(os.path.join(results_dir, f'{name.lower()}_confusion_matrix.png'))
#     plt.close()

# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, batch_size=16, image_size=224)
#     train_loader, val_loader, test_loader = loaders["classification"]

#     model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=True).to(device)
#     train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4)

#     model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
#     evaluate(model, train_loader, device, name="Train", results_dir=results_dir)
#     evaluate(model, val_loader, device, name="Validation", results_dir=results_dir)
#     evaluate(model, test_loader, device, name="Test", results_dir=results_dir)

# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()






# ############### custom cross validation mit average
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np

# # Deine Module
# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging

# def train_fold(model, train_loader, val_loader, device, results_dir, fold_idx, num_epochs=50, lr=1e-4):
#     logger = logging.getLogger()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     early_stopper = EarlyStopping(patience=5, min_delta=None, verbose=True)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info(f"Starting training Fold {fold_idx}")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Fold {fold_idx} Epoch {epoch}/{num_epochs} ---")

#         model.train()
#         running_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Train Fold {fold_idx}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)

#         model.eval()
#         running_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets in tqdm(val_loader, desc=f"Val Fold {fold_idx}", unit="batch", leave=False):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 running_loss += loss.item() * inputs.size(0)

#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             model_path = os.path.join(results_dir, f"best_model_fold{fold_idx}.pth")
#             torch.save(model.state_dict(), model_path)
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.error(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training Fold {fold_idx} completed in {total_minutes:.2f} minutes")

#     # Save loss curves
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title(f'Loss Curves Fold {fold_idx}')
#     plt.savefig(os.path.join(results_dir, f'loss_curves_fold{fold_idx}.png'))
#     plt.close()

#     return best_val_loss


# def evaluate(model, loader, device, name, results_dir, fold_idx):
#     logger = logging.getLogger()
#     model.eval()
#     all_preds, all_targets = [], []
#     with torch.no_grad():
#         for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())

#     # Classification Report
#     report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
#     df = pd.DataFrame(report_dict).transpose()
#     logger.info(f"\n{name} Classification Report (Fold {fold_idx}):\n{df.to_string()}\n")

#     # Confusion Matrix
#     cm = confusion_matrix(all_targets, all_preds)
#     plt.figure(figsize=(5, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean','Noisy'], yticklabels=['Clean','Noisy'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(f'{name} Confusion Matrix Fold {fold_idx}')
#     plt.savefig(os.path.join(results_dir, f'{name.lower()}_confusion_matrix_fold{fold_idx}.png'))
#     plt.close()

#     return df


# def aggregate_fold_results(fold_metrics, results_dir):
#     """
#     Aggregiert die Ergebnisse über alle Folds.
#     """
#     logger = logging.getLogger()
#     logger.info("\n" + "="*50)
#     logger.info("           CROSS-VALIDATION RESULTS")
#     logger.info("="*50)

#     # Sammle Metriken
#     f1_clean = [df.loc['0', 'f1-score'] for df in fold_metrics]
#     f1_noisy = [df.loc['1', 'f1-score'] for df in fold_metrics]
#     acc = [df.loc['accuracy', 'precision'] for df in fold_metrics]

#     # Berechne Mittelwert und Standardabweichung
#     def log_metric(name, values):
#         mean = np.mean(values)
#         std = np.std(values)
#         logger.info(f"{name} - Mean: {mean:.4f} ± {std:.4f}")

#     log_metric("F1-Score (Clean)", f1_clean)
#     log_metric("F1-Score (Noisy)", f1_noisy)
#     log_metric("Accuracy", acc)

#     # Bestes Modell identifizieren (nach Val-F1)
#     val_f1_scores = [df.loc['macro avg', 'f1-score'] for df in fold_metrics]
#     best_fold_idx = np.argmax(val_f1_scores) + 1
#     logger.info(f"\nBest Validation F1-Score: Fold {best_fold_idx}")
#     return best_fold_idx


# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results_cv")
#     os.makedirs(results_dir, exist_ok=True)
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, batch_size=16, image_size=224)

#     folds = loaders["classification"]["folds"]
#     test_loader = loaders["classification"]["test"]

#     # Speichern der Metriken und Modelle
#     fold_metrics = []

#     for fold_idx, (train_loader, val_loader) in enumerate(folds, start=1):
#         logger.info(f"\n{'='*20} Fold {fold_idx}/5 {'='*20}")

#         model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=True).to(device)

#         #  KORREKT: 'device' wird jetzt übergeben!
#         train_fold(model, train_loader, val_loader, device, results_dir, fold_idx, num_epochs=50, lr=1e-4)

#         # Lade bestes Modell des Folds
#         best_model_path = os.path.join(results_dir, f"best_model_fold{fold_idx}.pth")
#         model.load_state_dict(torch.load(best_model_path, map_location=device))

#         # Evaluate auf allen Sets
#         train_metrics = evaluate(model, train_loader, device, "Train", results_dir, fold_idx)
#         val_metrics = evaluate(model, val_loader, device, "Validation", results_dir, fold_idx)
#         test_metrics = evaluate(model, test_loader, device, "Test", results_dir, fold_idx)

#         fold_metrics.append(val_metrics)

#     # --- Aggregation der Ergebnisse ---
#     best_fold_idx = aggregate_fold_results(fold_metrics, results_dir)

#     # --- Finale Evaluation mit bestem Modell ---
#     logger.info(f"\n\n{'='*20} FINAL EVALUATION {'='*20}")
#     best_model_path = os.path.join(results_dir, f"best_model_fold{best_fold_idx}.pth")
#     final_model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=True).to(device)
#     final_model.load_state_dict(torch.load(best_model_path, map_location=device))

#     final_test_metrics = evaluate(final_model, test_loader, device, "Final Test", results_dir, fold_idx="BEST")
#     logger.info(f"Final Test Accuracy (Best Model): {final_test_metrics.loc['accuracy', 'precision']:.4f}")


# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()







# ############### custom cross validation
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix

# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging

# def train(model, train_loader, val_loader, device, results_dir, fold_idx, num_epochs=50, lr=1e-4):
#     logger = logging.getLogger()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     early_stopper = EarlyStopping(patience=5, min_delta=None, verbose=True)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info(f"Starting training Fold {fold_idx}")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Fold {fold_idx} Epoch {epoch}/{num_epochs} ---")

#         model.train()
#         running_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Train Fold {fold_idx}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)

#         model.eval()
#         running_loss = 0.0
#         for inputs, targets in tqdm(val_loader, desc=f"Val Fold {fold_idx}", unit="batch", leave=False):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             running_loss += loss.item() * inputs.size(0)

#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             model_path = os.path.join(results_dir, f"best_model_fold{fold_idx}.pth")
#             torch.save(model.state_dict(), model_path)
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.error(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training Fold {fold_idx} completed in {total_minutes:.2f} minutes")

#     # Save loss curves
#     plt.figure(figsize=(8,5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title(f'Loss Curves Fold {fold_idx}')
#     plt.savefig(os.path.join(results_dir, f'loss_curves_fold{fold_idx}.png'))
#     plt.close()

# def evaluate(model, loader, device, name, results_dir, fold_idx):
#     logger = logging.getLogger()
#     model.eval()
#     all_preds, all_targets = [], []
#     for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_targets.extend(targets.cpu().numpy())

#     report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
#     df = pd.DataFrame(report_dict).transpose()
#     logger.info(f"\n{name} Classification Report (Fold {fold_idx}):\n{df.to_string()}\n")

#     cm = confusion_matrix(all_targets, all_preds)
#     plt.figure(figsize=(5,4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean','Noisy'], yticklabels=['Clean','Noisy'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(f'{name} Confusion Matrix Fold {fold_idx}')
#     plt.savefig(os.path.join(results_dir, f'{name.lower()}_confusion_matrix_fold{fold_idx}.png'))
#     plt.close()

# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     os.makedirs(results_dir, exist_ok=True)
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, batch_size=16, image_size=224)

#     folds = loaders["classification"]["folds"]
#     test_loader = loaders["classification"]["test"]

#     for fold_idx, (train_loader, val_loader) in enumerate(folds, start=1):
#         logger.info(f"==== Fold {fold_idx} ====")
#         model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=True).to(device)
#         train(model, train_loader, val_loader, device, results_dir, fold_idx=fold_idx, num_epochs=50, lr=1e-4)

#         best_model_path = os.path.join(results_dir, f"best_model_fold{fold_idx}.pth")
#         model.load_state_dict(torch.load(best_model_path, map_location=device))

#         evaluate(model, train_loader, device, name="Train", results_dir=results_dir, fold_idx=fold_idx)
#         evaluate(model, val_loader, device, name="Validation", results_dir=results_dir, fold_idx=fold_idx)
#         evaluate(model, test_loader, device, name="Test", results_dir=results_dir, fold_idx=fold_idx)

# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()
















# ################## custom split
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix

# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging

# def train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4):
#     logger = logging.getLogger()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     early_stopper = EarlyStopping(patience=5, min_delta=None, verbose=True)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info("Starting training")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

#         model.train()
#         running_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)

#         model.eval()
#         running_loss = 0.0
#         for inputs, targets in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             running_loss += loss.item() * inputs.size(0)

#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.error(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training completed in {total_minutes:.2f} minutes")

#     plt.figure(figsize=(8,5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Loss Curves with EarlyStopping')
#     plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
#     plt.close()

# def evaluate(model, loader, device, name, results_dir):
#     logger = logging.getLogger()
#     model.eval()
#     all_preds, all_targets = [], []
#     for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_targets.extend(targets.cpu().numpy())

#     report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
#     df = pd.DataFrame(report_dict).transpose()
#     logger.info(f"\n{name} Classification Report:\n{df.to_string()}\n")

#     cm = confusion_matrix(all_targets, all_preds)
#     plt.figure(figsize=(5,4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean','Noisy'], yticklabels=['Clean','Noisy'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(f'{name} Confusion Matrix')
#     plt.savefig(os.path.join(results_dir, f'{name.lower()}_confusion_matrix.png'))
#     plt.close()

# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, batch_size=16, image_size=224)
#     train_loader, val_loader, test_loader = loaders["classification"]

#     model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=True).to(device)
#     train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4)

#     model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
#     evaluate(model, train_loader, device, name="Train", results_dir=results_dir)
#     evaluate(model, val_loader, device, name="Validation", results_dir=results_dir)
#     evaluate(model, test_loader, device, name="Test", results_dir=results_dir)

# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()
