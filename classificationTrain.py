# ############################## resize no mil
# import os
# DETERMINISTIC = True  # <- False für schnellere, nicht reproduzierbare Runs
# if DETERMINISTIC:
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#     os.environ["PYTHONHASHSEED"] = "42"
# import time
# import random
# import numpy as np

# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix, f1_score

# from loadData import get_data_loaders
# from createModelResNet import ResNet18Custom
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

#     torch.backends.cudnn.deterministic = bool(deterministic)
#     torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)

# set_global_determinism(GLOBAL_SEED, DETERMINISTIC)


# # -------------------------------
# # Logging: Hyperparameter
# # -------------------------------
# def log_hyperparams(logger, *, image_size, lr, batch_size, in_channels,
#                     patience, adapt_start_epoch,
#                     step_size, gamma, num_epochs,
#                     weight_decay, deterministic, num_workers,
#                     use_early_stopping: bool):
#     logger.info("======== Klassifizierung Hyperparameter ========")
#     logger.info(f"Image Size:                       {image_size}")
#     logger.info(f"Learning Rate:                    {lr}")
#     logger.info(f"Batch Size:                       {batch_size}")
#     logger.info(f"in_channels (1=Gray, 3=RGB):      {in_channels}")
#     logger.info(f"LR Scheduler:                     step_size={step_size}, gamma={gamma}")
#     logger.info(f"Num epochs:                       {num_epochs}")
#     logger.info(f"Weight Decay (L2):                {weight_decay}")
#     logger.info(f"Deterministic:                    {deterministic}")
#     logger.info(f"DataLoader num_workers:           {num_workers}")
#     logger.info(f"EarlyStopping enabled:            {use_early_stopping}")
#     if use_early_stopping:
#         logger.info(f"patience:                     {patience}")
#         logger.info(f"adapt_start_epoch:            {adapt_start_epoch}")
#     logger.info("=================================")


# def train(model, train_loader, val_loader, device, results_dir,
#           num_epochs=25, lr=1e-4, patience=10, adapt_start_epoch=5,
#           step_size=5, gamma=0.1, weight_decay=1e-4, use_early_stopping: bool = True):
#     logger = logging.getLogger()

#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # L2 regularization
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

#     early_stopper = None
#     if use_early_stopping:
#         early_stopper = EarlyStopping(
#             patience=patience,
#             min_delta=None,
#             verbose=True,
#             adapt_start_epoch=adapt_start_epoch
#         )

#     # Tracking
#     train_losses, val_losses = [], []
#     train_accs, val_accs = [], []
#     val_f1s = []
#     lrs_epoch = []

#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info("Starting training")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

#         # ---------- Train ----------
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         for inputs, targets in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()           # Gradienten zurücksetzen
#             outputs = model(inputs)         # logits
#             loss = criterion(outputs, targets)
#             loss.backward()                 # Gradienten berechnen
#             optimizer.step()                # Gewichte aktualisieren

#             running_loss += loss.item() * inputs.size(0)

#             # Accuracy (Train)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == targets).sum().item()
#             total   += targets.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_acc  = correct / max(1, total)
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)

#         # ---------- Val ----------
#         model.eval()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         all_preds = []
#         all_targets = []

#         with torch.no_grad():
#             for inputs, targets in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 running_loss += loss.item() * inputs.size(0)

#                 preds = outputs.argmax(dim=1)
#                 correct += (preds == targets).sum().item()
#                 total   += targets.size(0)

#                 all_preds.append(preds.cpu())
#                 all_targets.append(targets.cpu())

#         val_loss = running_loss / len(val_loader.dataset)
#         val_acc  = correct / max(1, total)
#         val_losses.append(val_loss)
#         val_accs.append(val_acc)

#         # F1 (binary) auf Val (Tile-Level)
#         if len(all_targets) > 0:
#             y_true = torch.cat(all_targets).numpy().tolist()
#             y_pred = torch.cat(all_preds).numpy().tolist()
#             f1 = f1_score(y_true, y_pred, zero_division=0)
#         else:
#             f1 = 0.0
#         val_f1s.append(f1)

#         # Bestes Modell sichern (nach Val)
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         # Lernrate loggen (vor Step)
#         lrs_epoch.append(optimizer.param_groups[0]["lr"])

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch:02d} — "
#                     f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
#                     f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
#                     f"Val F1: {f1:.4f}, LR: {lrs_epoch[-1]:.2e}, "
#                     f"Time: {elapsed:.1f}s")

#         # ---------- EarlyStopping & Scheduler ----------
#         if use_early_stopping and early_stopper is not None:
#             if early_stopper(val_loss):
#                 logger.warning(f"Early stopping after epoch {epoch}.")
#                 break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training completed in {total_minutes:.2f} minutes")

#     # ---------- Kurven speichern ----------
#     # Loss Curves
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
#     plt.title('Loss Curves')
#     plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
#     plt.close()

#     # Accuracy Curves
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Acc')
#     plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Acc')
#     plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
#     plt.title('Accuracy over Epochs')
#     plt.savefig(os.path.join(results_dir, 'accuracy_curves.png'))
#     plt.close()

#     # F1 (Val) Curve
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(val_f1s) + 1), val_f1s, label='Val F1')
#     plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend()
#     plt.title('Validation F1 over Epochs')
#     plt.savefig(os.path.join(results_dir, 'val_f1_curve.png'))
#     plt.close()

#     # LR Curve
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(lrs_epoch) + 1), lrs_epoch, label='Learning Rate')
#     plt.xlabel('Epoch'); plt.ylabel('LR'); plt.yscale('log'); plt.legend()
#     plt.title('Learning Rate over Epochs')
#     plt.savefig(os.path.join(results_dir, 'lr_curve.png'))
#     plt.close()

#     return

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
#     results_dir = os.path.join(os.path.dirname(__file__), "results_classification")
#     os.makedirs(results_dir, exist_ok=True)
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     # ==== Hyperparameter ====
#     image_size = 128
#     lr = 1e-4
#     batch_size = 16
#     in_channels = 1   # 1: Graustufen, 3: RGB
#     patience = 10
#     adapt_start_epoch = 5
#     step_size = 5
#     gamma = 0.1
#     num_epochs = 25
#     cls_val_ratio = 0.2  # 20% validation
#     weight_decay = 1e-4  # L2 regularization
#     # ========================

#     USE_EARLY_STOPPING = True  # auf False setzen, um ohne EarlyStopping zu trainieren

#     # Für strikten Determinismus: Single-process Loading
#     num_workers = 0 if DETERMINISTIC else 4

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     log_hyperparams(
#         logger,
#         image_size=image_size, lr=lr, batch_size=batch_size, in_channels=in_channels,
#         patience=patience, adapt_start_epoch=adapt_start_epoch,
#         step_size=step_size, gamma=gamma, num_epochs=num_epochs,
#         weight_decay=weight_decay, deterministic=DETERMINISTIC, num_workers=num_workers,
#         use_early_stopping=USE_EARLY_STOPPING
#     )

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))

#     loaders = get_data_loaders(DATASET_PATH, batch_size=16, image_size=image_size, cls_val_ratio=cls_val_ratio,
#                                enable_classification=True, enable_denoising=False)

#     train_loader, val_loader, test_loader = loaders["classification"]

#     # Modell
#     model = ResNet18Custom(num_classes=2, in_channels=in_channels, pretrained=True).to(device)

#     # Training + Kurven speichern
#     train(model, train_loader, val_loader, device, results_dir,
#           num_epochs=num_epochs, lr=lr, patience=patience,
#           step_size=step_size, gamma=gamma, adapt_start_epoch=adapt_start_epoch,
#           weight_decay=weight_decay, use_early_stopping=USE_EARLY_STOPPING)

#     # Bestes Modell laden
#     model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
#     evaluate(model, train_loader, device, name="Train", results_dir=results_dir)
#     evaluate(model, val_loader, device, name="Validation", results_dir=results_dir)
#     evaluate(model, test_loader, device, name="Test", results_dir=results_dir)

# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()














############### custom split MIL mit seed, L2 und alle visualization
import os
DETERMINISTIC = True  # <- False für schnellere, nicht reproduzierbare Runs
if DETERMINISTIC:
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

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
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)

set_global_determinism(GLOBAL_SEED, DETERMINISTIC)


# -------------------------------
# Logging: Hyperparameter
# -------------------------------
def log_hyperparams(logger, *, lr, batch_size, in_channels,
                    tile_h, tile_w, stride_w, patience, adapt_start_epoch,
                    step_size, gamma, num_epochs,
                    weight_decay, deterministic, num_workers,
                    use_early_stopping: bool):
    logger.info("======== Klassifizierung Hyperparameter ========")
    logger.info(f"Learning Rate:                    {lr}")
    logger.info(f"Batch Size:                       {batch_size}")
    logger.info(f"in_channels (1=Gray, 3=RGB):      {in_channels}")
    logger.info(f"MIL tile_h x tile_w (stride_w):   {tile_h} x {tile_w} (stride {stride_w})")
    logger.info(f"LR Scheduler:                     step_size={step_size}, gamma={gamma}")
    logger.info(f"Num epochs:                       {num_epochs}")
    logger.info(f"Weight Decay (L2):                {weight_decay}")
    logger.info(f"Deterministic:                    {deterministic}")
    logger.info(f"DataLoader num_workers:           {num_workers}")
    logger.info(f"EarlyStopping enabled:            {use_early_stopping}")
    if use_early_stopping:
        logger.info(f"patience:                     {patience}")
        logger.info(f"adapt_start_epoch:            {adapt_start_epoch}")
    logger.info("=================================")


# -------------------------------
# MIL Tiling Visualization (nur PNG-Dateien)
# -------------------------------
def visualize_mil_tiling(
    dataset,
    results_dir: str,
    num_files: int = 3,
    subdir: str = "tiling",
    tag_prefix: str = "Tiling"
):
    """
    Zeichnet für die ersten `num_files` Dateien im Dataset die MIL-Tiles als
    Rechtecke ins Spektrogramm. Speichert PNGs.
    """
    logger = logging.getLogger()
    out_dir = os.path.join(results_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    files = dataset.files  # Liste von (path, label)
    if len(files) == 0:
        logger.warning("[Tiling] Dataset leer – keine Visualisierung erzeugt.")
        return

    # deterministisch: ersten num_files
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
        fig, ax = plt.subplots(figsize=(max(6, W/100), max(3, H/100)), dpi=100)
        if dataset.in_channels == 1:
            ax.imshow(img_np, cmap=cmap, origin='upper', aspect='auto')
        else:
            ax.imshow(img_np, origin='upper', aspect='auto')
        ax.set_title(f"{tag_prefix}: {os.path.basename(path)} | Label: {'Noisy' if label==1 else 'Clean'}")
        ax.axis('off')

        # Tiles einzeichnen
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

        out_png = os.path.join(out_dir, f"{tag_prefix}_f{fid}_{os.path.basename(path)}.png")
        fig.savefig(out_png, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    logger.info(f"[Tiling] MIL-Visualisierung für {len(chosen_fids)} Dateien erzeugt (unter {out_dir}).")


# -------------------------------
# Training (Tile-Level) + Kurven
# -------------------------------
def train(model, train_loader, val_loader, device, results_dir,
          num_epochs=25, lr=1e-4, patience=7, adapt_start_epoch=5,
          step_size=5, gamma=0.1, weight_decay=1e-4, use_early_stopping: bool = True):
    logger = logging.getLogger()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # L2 regularization
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    early_stopper = None
    if use_early_stopping:
        early_stopper = EarlyStopping(
            patience=patience,
            min_delta=None,
            verbose=True,
            adapt_start_epoch=adapt_start_epoch
        )

    # Tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_f1s = []
    lrs_epoch = []

    best_val_loss = float('inf')
    start_time = time.time()

    logger.info("Starting training")
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        # ---------- Train ----------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets, _fid in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()           # Gradienten zurücksetzen
            outputs = model(inputs)         # logits
            loss = criterion(outputs, targets)
            loss.backward()                 # Gradienten berechnen
            optimizer.step()                # Gewichte aktualisieren

            running_loss += loss.item() * inputs.size(0)

            # Accuracy (Train)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc  = correct / max(1, total)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---------- Val ----------
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets, _fid in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total   += targets.size(0)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        val_loss = running_loss / len(val_loader.dataset)
        val_acc  = correct / max(1, total)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # F1 (binary) auf Val (Tile-Level)
        if len(all_targets) > 0:
            y_true = torch.cat(all_targets).numpy().tolist()
            y_pred = torch.cat(all_preds).numpy().tolist()
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            f1 = 0.0
        val_f1s.append(f1)

        # Bestes Modell sichern (nach Val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

        # Lernrate loggen (vor Step)
        lrs_epoch.append(optimizer.param_groups[0]["lr"])

        elapsed = time.time() - epoch_start
        logger.info(f"Epoch {epoch:02d} — "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                    f"Val F1: {f1:.4f}, LR: {lrs_epoch[-1]:.2e}, "
                    f"Time: {elapsed:.1f}s")

        # ---------- EarlyStopping & Scheduler ----------
        if use_early_stopping and early_stopper is not None:
            if early_stopper(val_loss):
                logger.warning(f"Early stopping after epoch {epoch}.")
                break

        scheduler.step()

    total_minutes = (time.time() - start_time) / 60
    logger.info(f"Training completed in {total_minutes:.2f} minutes")

    # ---------- Kurven speichern ----------
    # Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('Loss Curves')
    plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
    plt.close()

    # Accuracy Curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Acc')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.title('Accuracy over Epochs')
    plt.savefig(os.path.join(results_dir, 'accuracy_curves.png'))
    plt.close()

    # F1 (Val) Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(val_f1s) + 1), val_f1s, label='Val F1 (tile-level)')
    plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.legend()
    plt.title('Validation F1 over Epochs')
    plt.savefig(os.path.join(results_dir, 'val_f1_curve.png'))
    plt.close()

    # LR Curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(lrs_epoch) + 1), lrs_epoch, label='Learning Rate')
    plt.xlabel('Epoch'); plt.ylabel('LR'); plt.yscale('log'); plt.legend()
    plt.title('Learning Rate over Epochs')
    plt.savefig(os.path.join(results_dir, 'lr_curve.png'))
    plt.close()

    return


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


def _report_and_plot(name: str, y_true: List[int], y_pred: List[int],
                     results_dir: str):
    logger = logging.getLogger()
    report = classification_report(y_true, y_pred, digits=4)
    logger.info(f"\n{name} Classification Report:\n{report}\n")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Noisy'], yticklabels=['Clean', 'Noisy'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{name} Confusion Matrix')
    out_path = os.path.join(results_dir, f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
    fig.savefig(out_path); plt.close(fig)


def evaluate_file_mil(model, loader, device, name, results_dir, threshold: float):
    file_logits, file_labels = _collect_file_logits(model, loader, device)
    file_probs = _aggregate_logits_to_probs(file_logits)

    fids = sorted(file_labels.keys())
    y_true = [file_labels[f] for f in fids]
    y_score = [file_probs[f] for f in fids]
    y_pred = [1 if s >= threshold else 0 for s in y_score]

    _report_and_plot(name, y_true, y_pred, results_dir)


def tune_threshold_on_val(model, val_loader, device) -> float:
    file_logits, file_labels = _collect_file_logits(model, val_loader, device)
    file_probs = _aggregate_logits_to_probs(file_logits)

    fids = sorted(file_labels.keys())
    y_true = [file_labels[f] for f in fids]
    y_score = [file_probs[f] for f in fids]

    best_f1, best_th = -1.0, 0.5
    for th in [i / 100.0 for i in range(0, 101)]:
        y_pred = [1 if s >= th else 0 for s in y_score]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    logging.getLogger().info(f"[Threshold] Best on Val: th={best_th:.2f}, F1={best_f1:.4f}")
    return best_th



# -------------------------------
# Main
# -------------------------------
def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results_classification")
    os.makedirs(results_dir, exist_ok=True)
    logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

    # ==== Hyperparameter ====
    lr = 1e-4
    batch_size = 8
    in_channels = 1   # 1: Graustufen, 3: RGB
    tile_h = 128
    tile_w = 256
    stride_w = 128
    patience = 10
    adapt_start_epoch = 5
    step_size = 10
    gamma = 0.5
    num_epochs = 25
    cls_val_ratio = 0.2  # 20% validation
    weight_decay = 1e-4  # L2 regularization
    # ========================

    USE_EARLY_STOPPING = True  # auf False setzen, um ohne EarlyStopping zu trainieren

    # Für strikten Determinismus: Single-process Loading
    num_workers = 0 if DETERMINISTIC else 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    log_hyperparams(
        logger,
        lr=lr, batch_size=batch_size, in_channels=in_channels,
        tile_h=tile_h, tile_w=tile_w, stride_w=stride_w,
        patience=patience, adapt_start_epoch=adapt_start_epoch,
        step_size=step_size, gamma=gamma, num_epochs=num_epochs,
        weight_decay=weight_decay, deterministic=DETERMINISTIC, num_workers=num_workers,
        use_early_stopping=USE_EARLY_STOPPING
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
    visualize_mil_tiling(train_loader.dataset, results_dir, num_files=2, subdir="tiling_train", tag_prefix="Tiling_Train")
    visualize_mil_tiling(val_loader.dataset,   results_dir, num_files=2, subdir="tiling_val",   tag_prefix="Tiling_Val")

    # Modell
    model = ResNet18Custom(num_classes=2, in_channels=in_channels, pretrained=True).to(device)

    # Training (Tile-Level) + Kurven speichern
    train(model, train_loader, val_loader, device, results_dir,
          num_epochs=num_epochs, lr=lr, patience=patience,
          step_size=step_size, gamma=gamma, adapt_start_epoch=adapt_start_epoch,
          weight_decay=weight_decay, use_early_stopping=USE_EARLY_STOPPING)

    # Bestes Modell laden
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))

    # Threshold auf Validation optimieren (File/MIL)
    best_th = tune_threshold_on_val(model, val_loader, device)

    # Nur File/MIL-Reports
    evaluate_file_mil(model, train_loader, device, name="Train", results_dir=results_dir, threshold=best_th)
    evaluate_file_mil(model, val_loader, device, name="Validation", results_dir=results_dir, threshold=best_th)
    evaluate_file_mil(model, test_loader, device, name="Test", results_dir=results_dir, threshold=best_th)


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
#     plt.title('Loss Curves')
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
#     logger.info(f"\n{name} Classification Report:\n{report}\n")

#     cm = confusion_matrix(y_true, y_pred)
#     fig, ax = plt.subplots(figsize=(5, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean', 'Noisy'], yticklabels=['Clean', 'Noisy'], ax=ax)
#     ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{name} Confusion Matrix')
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
#     plt.title('Loss Curves')
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
