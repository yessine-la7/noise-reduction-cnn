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


SEED = 50

def set_global_seed(seed: int = 42):
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
# Klassifikation-Parameter
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
    if in_channels == 1:
        pil_img = pil_img.convert("L")
    else:
        pil_img = pil_img.convert("RGB")

    arr = np.array(pil_img, dtype=np.float32) / 255.0
    if in_channels == 1:
        if arr.ndim == 2:
            arr = arr[:, :, None]
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
    Kachelt das Bild entlang der Breite.
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
def predict_from_full_png(model: torch.nn.Module, png_path: str, device: torch.device) -> Tuple[int, float, float]:
    """
    Lädt das Spektrogramm-PNG, zerlegt in Tiles (wie Training).
    """
    img = Image.open(png_path)
    t_img = pil_to_training_tensor(img, in_channels=IN_CHANNELS)  # (1,128,W)
    tiles = tile_along_width(t_img, TILE_W, STRIDE_W)

    model.eval()
    logits_all = []
    for i in range(0, len(tiles), 32):
        batch = torch.stack(tiles[i:i+32], dim=0).to(device)  # (B,1,128,256)
        out = model(batch)
        logits_all.append(out.detach().cpu())
    L = torch.cat(logits_all, dim=0)
    L_mean = L.mean(dim=0, keepdim=True)
    p = torch.softmax(L_mean, dim=1)[0]

    p_noisy = float(p[1].item())  # Wahrscheinlichkeit für "Noisy"
    p_clean = float(p[0].item())  # Wahrscheinlichkeit für "Clean"

    # Entscheidung basierend auf Threshold
    pred = 1 if p_noisy >= BEST_TH else 0

    # Wahrscheinlichkeit der vorhergesagten Klasse
    conf = p_noisy * 100.0 if pred == 1 else p_clean * 100.0

    return pred, conf, p_noisy


def plot_full_png_grid(examples: List[Tuple[str, int, int, float, float]], out_path: str):
    """
    Plottet Bilder und Beschriftungen darunter.
    """
    n = len(examples)
    fig_w = 4 * n
    fig_h = 5

    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(fig_w, fig_h))

    if n == 1:
        axes = [axes]

    for i, (png_path, t_lbl, p_lbl, conf, p_noisy) in enumerate(examples):
        img = Image.open(png_path).convert("L")
        arr = np.array(img)

        # Bild anzeigen
        axes[i].imshow(arr, cmap="magma", aspect="auto")
        axes[i].axis("off")

        # Dateiname extrahieren
        filename = os.path.splitext(os.path.basename(png_path))[0]

        # Titel erstellen
        title_text = f"{filename}\nTrue: {'Noisy' if t_lbl == 1 else 'Clean'}\nPred: {'Noisy' if p_lbl == 1 else 'Clean'}\nProb: {conf:.1f}%"
        axes[i].set_title(title_text, fontsize=10, pad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches='tight')
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
    Wählt deterministisch num_samples Dateien,
    nutzt den PNG-Pfad direkt aus test_dataset,
    macht Vorhersagen über das gesamte Bild
    """
    model.eval()

    n_files = len(test_dataset.files)
    rng = random.Random(seed)
    fids = rng.sample(range(n_files), k=min(num_samples, n_files))

    examples: List[Tuple[str, int, int, float, float]] = []

    for fid in fids:
        png_path, true_label = test_dataset.files[fid]
        pred_label, conf, p_noisy = predict_from_full_png(model, png_path, device)
        examples.append((png_path, int(true_label), pred_label, conf, p_noisy))

    outpath = os.path.join(results_dir, "predictions_test.png")
    plot_full_png_grid(examples, outpath)
    print(f"Spektrogramme gespeichert unter: {outpath}")

    print("\nVorhersageergebnisse:")
    for i, (png_path, t_lbl, p_lbl, conf, p_noisy) in enumerate(examples):
        filename = os.path.basename(png_path)
        print(f"  {i+1}: {filename}")
        print(f"     True: {'Noisy' if t_lbl == 1 else 'Clean'}")
        print(f"     Pred: {'Noisy' if p_lbl == 1 else 'Clean'} (Prob: {conf:.1f}%)")
        print(f"     Noisy Probability: {p_noisy:.3f}")
        print()


def main():
    # 1) Seeds
    set_global_seed(SEED)

    # 2) Ordner
    here = os.path.dirname(__file__)
    results_dir = os.path.join(here, "results_classification")
    os.makedirs(results_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("predictTest")
    logger.info(f"Seed gesetzt: {SEED}")

    # 3) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 4) Daten laden
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

    # 6) Vorhersagen & Plot
    predict_and_plot_full_spectrograms(
        model=model,
        test_dataset=test_dataset,
        device=device,
        results_dir=results_dir,
        num_samples=5,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
