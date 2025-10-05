import os
import io
from typing import List, Tuple
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from PIL import Image

from createModelResNet import ResNet18Custom
from trainLogging import setup_logging

# ======= Parameter wie im Training (MIL / Mel) =======
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

IN_CHANNELS = 1
TILE_W = 256
STRIDE_W = 128

# gleiche Normalisierung wie im Loader
NORM_MEAN = (0.5,)
NORM_STD  = (0.5,)

# Rendering: gleiche Pipeline wie beim PNG-Erzeugen (kein cmap angegeben => Default "viridis")
RENDER_DPI = 100
# =====================================================

BEST_TH = 0.09

def load_mel_spectrogram(audio_path: str) -> Tuple[np.ndarray, int, float]:
    """Audio -> Mel (dB)."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_mels=N_MELS, power=2.0, center=True,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB, sr, duration


def render_mel_to_pil(S_dB: np.ndarray, dpi: int = RENDER_DPI) -> Image.Image:
    """
    Rendere S_dB mit derselben Matplotlib/Librosa-Pipeline wie beim Erzeugen der Trainings-PNGs.
    Kein cmap gesetzt => Default (viridis). Achsen aus, tight layout.
    Ergebnis: RGB-PNG in-memory -> PIL.Image.
    """
    H, W = S_dB.shape
    fig_w = W / dpi
    fig_h = H / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    librosa.display.specshow(S_dB, x_axis=None, y_axis=None, hop_length=HOP_LENGTH)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert('RGB')
    return im


def pil_to_training_tensor(pil_img: Image.Image, in_channels: int = IN_CHANNELS) -> torch.Tensor:
    """
    Spiegel die Loader-Preprocessing-Schritte:
      - Grayscale(1) (falls IN_CHANNELS=1)
      - ToTensor()  -> skaliert [0..255] nach [0..1]
      - Normalize((0.5,), (0.5,))
    """
    if in_channels == 1:
        pil_img = pil_img.convert('L')  # Grayscale
    else:
        pil_img = pil_img.convert('RGB')

    # ToTensor
    arr = np.array(pil_img, dtype=np.uint8)
    if in_channels == 1:
        if arr.ndim == 2:
            arr = arr[:, :, None]  # H,W,1
        arr = arr.astype(np.float32) / 255.0  # [0,1]
        t = torch.from_numpy(arr).permute(2, 0, 1)  # 1,H,W
        # Normalize
        t = (t - NORM_MEAN[0]) / NORM_STD[0]
    else:
        arr = arr.astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)  # 3,H,W
        mean = torch.tensor(NORM_MEAN).view(-1, 1, 1)
        std = torch.tensor(NORM_STD).view(-1, 1, 1)
        t = (t - mean) / (std + 1e-12)

    return t  # (C,H,W)


def tile_along_width(t_img: torch.Tensor, tile_w: int, stride_w: int) -> List[torch.Tensor]:
    """Wie im Dataset: entlang der Breite sliden, letztes Tile anliegenden Abschluss."""
    _, H, W = t_img.shape

     # Prüfe die Höhe
    if t_img.shape[1] != 128:
        raise ValueError(f"Erwartete Höhe 128, aber erhielt {t_img.shape[1]}")

    tiles = []
    if W <= tile_w:
        if W < tile_w:
            # rechtsbündig croppen (wie im Loader)
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
def predict_mil(model: torch.nn.Module, tiles: List[torch.Tensor], device: torch.device) -> Tuple[float, int]:
    """Mean-Logits → Softmax (wie im Training)."""
    model.eval()
    logits_all = []
    for i in range(0, len(tiles), 32):
        batch = torch.stack(tiles[i:i+32], dim=0).to(device)  # (B,1,128,256)
        out = model(batch)  # (B,2)
        logits_all.append(out.cpu())
    L = torch.cat(logits_all, dim=0)  # (N_tiles,2)
    L_mean = L.mean(dim=0, keepdim=True)
    p = torch.softmax(L_mean, dim=1)[0]
    p_noisy = float(p[1].item())
    # pred = int(p.argmax().item())
    return p_noisy


def plot_mel(S_dB: np.ndarray, sr: int, out_path: str, title: str):
    plt.figure(figsize=(10, 2.5))
    librosa.display.specshow(S_dB, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel")
    plt.colorbar(label="Amplitude [dB]")
    if title: plt.title(title)
    plt.xlabel("Zeit [s]"); plt.ylabel("Frequenz [Hz]")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def main():
    here = os.path.dirname(__file__)
    results_dir = os.path.join(here, "results_classification")
    audios_dir = os.path.join(here, "audios")

    best_model = "best_model_MIL_batch_16_regul_step_7.pth"
    model_path = os.path.join(results_dir, f"{best_model}")
    audio_name = "tomaten_noisy"
    audio_extension = ".wav"
    audio_path = os.path.join(audios_dir, f"{audio_name}{audio_extension}")

    logger = setup_logging(results_dir, log_file=f"prediction_output_{audio_name}.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1) Audio -> Mel
    S_dB, sr, duration = load_mel_spectrogram(audio_path)
    H, W = S_dB.shape
    logger.info(f"Audio: {audio_path}")
    logger.info(f"Samplerate: {sr} Hz, Dauer: {duration:.2f} s, Mel-Shape: {H}x{W}")

    # 2) Render wie Training -> PIL
    pil_img = render_mel_to_pil(S_dB, dpi=RENDER_DPI)       # RGB wie png-Erzeugung

    # PIL -> Tensor + Normalize
    t_img = pil_to_training_tensor(pil_img, in_channels=IN_CHANNELS)  # (1,128, W')

    # 3) Tiles erzeugen
    tiles = tile_along_width(t_img, TILE_W, STRIDE_W)
    logger.info(f"Tiles erzeugt: {len(tiles)}  | Beispiel-Tile-Shape: {tiles[0].shape if tiles else None}")

    # 4) Modell laden
    model = ResNet18Custom(num_classes=2, in_channels=IN_CHANNELS, pretrained=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"{best_model} geladen.")

    # 5) Prediction (MIL)
    p_noisy = predict_mil(model, tiles, device)
    is_noisy = p_noisy >= BEST_TH
    label = "Noisy" if is_noisy else "Clean"
    logger.info(f"Prediction (th={BEST_TH:.3f}): {label}  (p_noisy = {p_noisy:.4f})")
    # label = "Noisy" if pred == 1 else "Clean"
    # logger.info(f"Prediction: {label}  (p_noisy = {p_noisy:.4f})")

    # 6) Plot Spektrogramm
    plot_path = os.path.join(results_dir, f"extern_prediction_{audio_name}.png")
    plot_title = f"{audio_name}{audio_extension}, Duration: {duration:.2f}s, \n Threshold: {BEST_TH}, Prediction: {label}, p(noisy)={p_noisy:.4f}"
    plot_mel(S_dB, sr, plot_path, plot_title)
    logger.info(f"Plot gespeichert: {plot_path}")


if __name__ == "__main__":
    main()
