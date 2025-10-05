import os
import random
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch

from loadData import get_data_loaders
from createModelResNet import ResNet18Custom

# =========================
# Reproduzierbarkeit
# =========================
SEED = 55

def set_global_seed(seed: int = 42):
    """Seeds & deterministische Flags für Python, NumPy, PyTorch, cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


# =========================
# Klassifikations-Parameter
# =========================
IN_CHANNELS   = 1
TILE_H        = 128
TILE_W        = 256
STRIDE_W      = 128
NORM_MEAN     = (0.5,)
NORM_STD      = (0.5,)
PLOT_DPI      = 120

BEST_TH = 0.09


# =========================
# Bild → Tensor
# =========================
def pil_to_training_tensor(pil_img: Image.Image, in_channels: int = IN_CHANNELS) -> torch.Tensor:
    """Spiegelt die Preprocessing-Schritte aus dem Loader (Graustufe, [0..1], Normalize)."""
    if in_channels == 1:
        pil_img = pil_img.convert("L")
    else:
        pil_img = pil_img.convert("RGB")

    arr = np.array(pil_img, dtype=np.float32) / 255.0
    if in_channels == 1:
        if arr.ndim == 2:
            arr = arr[:, :, None]  # H,W,1
        t = torch.from_numpy(arr).permute(2, 0, 1)  # 1,H,W
        t = (t - NORM_MEAN[0]) / (NORM_STD[0] + 1e-12)
    else:
        t = torch.from_numpy(arr).permute(2, 0, 1)  # 3,H,W
        mean = torch.tensor(NORM_MEAN).view(-1, 1, 1)
        std  = torch.tensor(NORM_STD).view(-1, 1, 1)
        t = (t - mean) / (std + 1e-12)
    return t


def tile_along_width(t_img: torch.Tensor, tile_w: int, stride_w: int) -> List[torch.Tensor]:
    """
    Kachelt das Bild entlang der Breite; letztes Tile rechtsbündig.
    Erwartet Shape (C,H,W) mit H==128 (Mel).
    """
    _, H, W = t_img.shape
    if H != TILE_H:
        raise ValueError(f"Erwarte Höhe {TILE_H}, bekam {H} (evtl. falsches PNG).")

    tiles: List[torch.Tensor] = []
    if W <= tile_w:
        if W < tile_w:
            tiles.append(t_img[:, :, max(0, W - tile_w):W])
        else:
            tiles.append(t_img)
        return tiles

    x = 0
    while True:
        tiles.append(t_img[:, :, x:x + tile_w])
        x += stride_w
        if x + tile_w >= W:
            break
    return tiles


@torch.no_grad()
def predict_from_full_png(model: torch.nn.Module, png_path: str, device: torch.device) -> Tuple[int, float]:
    """
    Lädt das komplette Spektrogramm-PNG, zerlegt in Tiles (wie Training),
    mittelt die Logits und gibt (pred_label, confidence in %) zurück.
    """
    img = Image.open(png_path)
    t_img = pil_to_training_tensor(img, in_channels=IN_CHANNELS)  # (1,128,W)
    tiles = tile_along_width(t_img, TILE_W, STRIDE_W)

    model.eval()
    logits_all = []
    for i in range(0, len(tiles), 32):
        batch = torch.stack(tiles[i:i+32], dim=0).to(device)  # (B,1,128,256)
        out = model(batch)  # (B,2)
        logits_all.append(out.detach().cpu())
    L = torch.cat(logits_all, dim=0)  # (N_tiles,2)
    L_mean = L.mean(dim=0, keepdim=True)
    p = torch.softmax(L_mean, dim=1)[0]
    # pred = int(p.argmax().item())
    # conf = float(p[pred].item()) * 100.0

    # Entscheidung basierend auf manuellem Threshold
    p_noisy = float(p[1].item())  # Wahrscheinlichkeit für "Noisy"
    pred = 1 if p_noisy >= BEST_TH else 0
    conf = p_noisy * 100.0

    return pred, conf


def plot_full_png_grid(examples: List[Tuple[str, int, int, float]], out_path: str):
    """
    Plottet je Beispiel 2 Reihen:
      oben: True-Label
      unten: Prediction + Confidence
    """
    n = len(examples)
    fig_w = 3.2 * n
    fig_h = 6.4
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(fig_w, fig_h))

    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, (png_path, t_lbl, p_lbl, conf) in enumerate(examples):
        img = Image.open(png_path).convert("L")
        arr = np.array(img)

        axes[0, i].imshow(arr, cmap="magma", aspect="auto")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"T: {'Noisy' if t_lbl == 1 else 'Clean'}")

        axes[1, i].imshow(arr, cmap="magma", aspect="auto")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"P: {'Noisy' if p_lbl == 1 else 'Clean'}\n{conf:.1f}%")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


def predict_and_plot_full_spectrograms(
    model: torch.nn.Module,
    test_dataset,
    device: torch.device,
    results_dir: str,
    num_samples: int = 6,
    seed: int = SEED,
):
    """
    Wählt deterministisch num_samples **Dateien** (nicht Tiles),
    nutzt den PNG-Pfad direkt aus test_dataset.files[fid][0],
    macht Vorhersagen über das gesamte Bild und plottet es.
    """
    model.eval()

    # Wir ziehen Beispiele als **Datei-IDs** (fids), nicht als Tile-Indizes
    n_files = len(test_dataset.files)
    rng = random.Random(seed)
    fids = rng.sample(range(n_files), k=min(num_samples, n_files))

    examples: List[Tuple[str, int, int, float]] = []

    for fid in fids:
        png_path, true_label = test_dataset.files[fid]
        pred_label, conf = predict_from_full_png(model, png_path, device)
        examples.append((png_path, int(true_label), pred_label, conf))

    outpath = os.path.join(results_dir, "predictions_test.png")
    plot_full_png_grid(examples, outpath)
    print(f"Gesamtspektrogramme gespeichert unter: {outpath}")


def main():
    # 1) Seeds/Deterministik
    set_global_seed(SEED)

    # 2) Ordner/Logging
    here = os.path.dirname(__file__)
    results_dir = os.path.join(here, "results_classification")
    os.makedirs(results_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("predictTest")
    logger.info(f"Seed gesetzt: {SEED}")

    # 3) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 4) Loader (Testset der Klassifikation/Mel)
    DATASET_PATH = os.path.abspath(os.path.join(here, "..", "Dataset"))
    loaders = get_data_loaders(
        DATASET_PATH,
        batch_size=1,
        num_workers=0,
        cls_in_channels=IN_CHANNELS,
        cls_tile_h=TILE_H,
        cls_tile_w=TILE_W,
        cls_stride_w=STRIDE_W,
        cls_val_ratio=0.2,
        cls_seed=SEED,
        enable_classification=True,
        enable_denoising=False,
    )
    _, _, test_loader = loaders["classification"]
    test_dataset = test_loader.dataset

    # 5) Modell laden
    model = ResNet18Custom(num_classes=2, in_channels=IN_CHANNELS, pretrained=False).to(device)
    best_model_path = os.path.join(results_dir, "best_model_MIL_batch_16_regul_step_7.pth")
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)
    logger.info(f"Model geladen: {best_model_path}")

    # 6) Vorhersagen & Plot kompletter Spektrogramme
    predict_and_plot_full_spectrograms(
        model=model,
        test_dataset=test_dataset,
        device=device,
        results_dir=results_dir,
        num_samples=6,
        seed=SEED,
    )


if __name__ == "__main__":
    main()










# import os
# import random
# import logging
# import numpy as np

# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# from loadData import get_data_loaders
# from createModelResNet import ResNet18Custom

# # =========================
# # Reproduzierbarkeit
# # =========================
# SEED = 42

# def set_global_seed(seed: int = 42):
#     """Setzt Seeds und deterministische Flags für Python, NumPy und PyTorch."""
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)

#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     try:
#         torch.use_deterministic_algorithms(True)
#     except Exception:
#         pass

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


# def predict_and_plot(model, dataset, device, results_dir, num_samples=10, seed: int = 42):
#     """Wählt deterministisch num_samples Indizes, macht Vorhersagen und speichert ein Raster."""
#     model.eval()

#     # deterministische Stichprobe aus dem Dataset
#     rng = random.Random(seed)
#     indices = rng.sample(range(len(dataset)), num_samples)

#     fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(3 * num_samples, 6))

#     for i, idx in enumerate(indices):
#         img, label, _ = dataset[idx]   # (Bild, Label, file_id)
#         inp = img.unsqueeze(0).to(device)

#         with torch.no_grad():
#             out = model(inp)
#             probs = F.softmax(out, dim=1).cpu().numpy()[0]
#             pred = int(probs.argmax())
#             conf = float(probs[pred])

#         # Obere Reihe: True Label
#         axes[0, i].imshow(img.squeeze(), cmap='magma')
#         axes[0, i].axis('off')
#         axes[0, i].set_title(f"T: {'Noisy' if label==1 else 'Clean'}")

#         # Untere Reihe: Prediction + Confidence
#         axes[1, i].imshow(img.squeeze(), cmap='magma')
#         axes[1, i].axis('off')
#         axes[1, i].set_title(f"P: {'Noisy' if pred==1 else 'Clean'}\n{conf*100:.1f}%")

#     plt.tight_layout()
#     outpath = os.path.join(results_dir, "predictions.png")
#     plt.savefig(outpath, dpi=120)
#     plt.close()
#     print(f"Vorhersagen gespeichert unter: {outpath}")


# def main():
#     # Deterministik vor allen Initialisierungen setzen
#     set_global_seed(SEED)

#     results_dir = os.path.join(os.path.dirname(__file__), "results_classification")
#     os.makedirs(results_dir, exist_ok=True)

#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("predictTest")
#     logger.info(f"Seed gesetzt: {SEED}")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset"))

#     # Lade das Test-Set
#     loaders = get_data_loaders(
#         DATASET_PATH,
#         batch_size=1,
#         num_workers=0,
#         cls_in_channels=1,
#         cls_tile_h=128,
#         cls_tile_w=256,
#         cls_stride_w=128,
#         cls_val_ratio=0.2,
#         cls_seed=SEED,
#         enable_classification=True,
#         enable_denoising=False,
#     )
#     _, _, test_loader = loaders["classification"]
#     test_dataset = test_loader.dataset

#     # Modell laden
#     model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=False).to(device)
#     best_model_path = os.path.join(results_dir, "best_model_MIL_batch_16_seed_regul.pth")
#     state = torch.load(best_model_path, map_location=device)
#     model.load_state_dict(state)
#     logger.info(f"Loaded model weights from {best_model_path}")

#     # Vorhersage für 10 deterministisch gewählte Bilder
#     predict_and_plot(model, test_dataset, device, results_dir, num_samples=10, seed=SEED)


# if __name__ == "__main__":
#     main()
