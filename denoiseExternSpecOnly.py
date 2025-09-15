"""
Externes (Handy-)Audio -> STFT -> specshow -> Bild -> Graustufe -> DC-Zeile weg -> Tiles
-> U-Net -> Hann-Overlap-Stitching -> denoised Bild speichern
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image

from createModelUnet import UNetCustom
from trainLogging import setup_logging

# =========================
#   Pfade
# =========================
audios_dir      = os.path.join(os.path.dirname(__file__), "audios")
audio_name      = "noisy2"
audio_extension = ".m4a"
AUDIO_PATH      = os.path.join(audios_dir, f"{audio_name}{audio_extension}")

RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "results_denoising")
MODEL_PATH      = os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8_patience_20.pth")
LOG_FILE        = "denoising_output.txt"

# =========================
#   STFT-Parameter
# =========================
N_FFT      = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
SR         = None  # sr=None -> keine Resampling-Änderung

# Rendering (Pixelgenau wie convertDataSpec.py, default colormap)
DPI = 100  # Bildgröße = (W/DPI, H/DPI)

# =========================
#   Tiling / Modell
# =========================
TILE_W  = 256
STRIDE  = 256
IN_CH   = 1     # Graustufen
BASE_CH = 32

# =========================
#   Helfer
# =========================
def render_spec_like_training_to_gray01(S_dB, sr):
    """
    Rendert S_dB mit librosa.display.specshow -> RGB-Canvas -> Graustufe (PIL "L").
    Rückgabe:
      gray01: (H, W) in [0,1]
    """
    H, W = S_dB.shape
    fig_w = W / DPI
    fig_h = H / DPI
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    ax.set_position([0, 0, 1, 1])  # vollflächig
    im = librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax)  # AxesImage
    ax.axis('off')

    fig.canvas.draw()  # rendern

    # RGB aus Canvas lesen
    w_px, h_px = fig.canvas.get_width_height()  # (W, H)
    rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rgb = rgb.reshape(h_px, w_px, 3)  # (H,W, 3)
    plt.close(fig)

    # RGB -> Graustufe (PIL "L")
    gray = np.array(Image.fromarray(rgb).convert("L"), dtype=np.float32) / 255.0  # (H,W) in [0,1]
    return gray

def audio_to_gray01_training_like(wav_path):
    """
    1) Audio laden (sr=None, mono=True).
    2) STFT -> |S| -> dB (ref=max pro Datei).
    3) Rendern zu Gray [0,1] via specshow (wie im Dataset).
    4) DC-Zeile entfernen (erste Pixelzeile) -> Höhe 512.

    Rückgabe:
      gray01_wo_dc: (512, W) in [0,1]
      y:            geladene Wellenform (1D)
      sr_ret:       Samplerate
    """
    y, sr_ret = librosa.load(wav_path, sr=SR, mono=True)

    S = librosa.stft(
        y=y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    S_mag = np.abs(S)                # (513, W)
    amp_ref = float(np.max(S_mag) + 1e-12)

    # dB relativ zu amp_ref (Peak -> 0 dB)
    S_db = librosa.amplitude_to_db(S_mag, ref=amp_ref)  # (513, W), Werte in [negativ, 0]

    # Bild wie im Training erzeugen
    gray01_full = render_spec_like_training_to_gray01(S_db, sr_ret)  # (513, W)
    # DC-Zeile weg
    gray01_wo_dc = gray01_full[1:, :]  # (512, W)
    return gray01_wo_dc, y, sr_ret

def hann1d_w(tile_w: int) -> np.ndarray:
    if tile_w <= 1:
        return np.ones((1,), dtype=np.float32)
    return np.hanning(tile_w).astype(np.float32)

def to_model_space(tile01: torch.Tensor) -> torch.Tensor:
    # [0,1] -> [-1,1]
    return tile01 * 2.0 - 1.0

def to_img01_space(tile_norm: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (tile_norm + 1.0) * 0.5

@torch.no_grad()
def tiled_infer_and_stitch(model: nn.Module, img01: np.ndarray, device: torch.device,
                           tile_w: int = TILE_W, stride: int = STRIDE) -> np.ndarray:
    """
    Horizontal Tilen, Inferenz, Hann-Overlap-Stitch.
    Rückgabe: denoised in [0,1], gleiche Form wie img01.
    """
    H, W = img01.shape
    need_pad = W < tile_w
    W_pad = max(W, tile_w)
    if not need_pad and ((W - tile_w) % stride != 0):
        n_steps = (W - tile_w + stride - 1) // stride
        last_x = n_steps * stride
        W_pad = last_x + tile_w
        need_pad = W_pad > W

    if need_pad:
        img01_pad = np.zeros((H, W_pad), dtype=np.float32)
        img01_pad[:, :W] = img01
    else:
        img01_pad = img01

    win_w = hann1d_w(tile_w)
    win_w = np.maximum(win_w, 1e-6)
    win = torch.from_numpy(win_w.reshape(1, 1, 1, tile_w))

    out_sum = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)
    w_sum   = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)

    xs = list(range(0, max(1, W_pad - tile_w + 1), stride))
    if xs[-1] != W_pad - tile_w:
        xs.append(W_pad - tile_w)

    model.eval()
    with torch.no_grad():
        for x_left in xs:
            tile = img01_pad[:, x_left:x_left + tile_w]      # (H, tile_w)
            tile_t = torch.from_numpy(tile).unsqueeze(0)      # (1,H,Wt)
            tile_t = tile_t.unsqueeze(0).to(device)           # (1,1,H,Wt)

            pred = model(to_model_space(tile_t))              # [-1,1]
            pred01 = to_img01_space(pred).clamp(0.0, 1.0)

            pred_w = pred01 * win.to(device)
            out_sum[:, :, :, x_left:x_left + tile_w] += pred_w
            w_sum[:, :, :, x_left:x_left + tile_w]   += win.to(device)

    out01 = (out_sum / (w_sum + 1e-8)).squeeze(0).squeeze(0).cpu().numpy()  # (H,W_pad)
    return out01[:, :W]

def save_spectrogram_png(img2d: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(img2d.shape[1] / DPI, img2d.shape[0] / DPI), dpi=DPI)
    plt.imshow(img2d, origin='upper', aspect='auto', cmap="magma")
    plt.axis('off'); plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_waveform(y, sr, out_path, title="Waveform"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, color='b')
    plt.title(title)
    plt.xlabel("Zeit (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def mean_abs_change01(noisy01: np.ndarray, denoised01: np.ndarray) -> float:
    return float(np.mean(np.abs(denoised01 - noisy01)))


# =========================
#   MAIN
# =========================
def main():
    logger = setup_logging(RESULTS_DIR, log_file=LOG_FILE, level="INFO")

    if not os.path.isfile(AUDIO_PATH):
        logger.error(f"Audio nicht gefunden: {AUDIO_PATH}"); return
    if not os.path.isfile(MODEL_PATH):
        logger.error(f"Modell nicht gefunden: {MODEL_PATH}"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 70)
    logger.info("Externes Audio denoisen (training-like rendering + Audio via iSTFT/Original-Phase)")
    logger.info(f"Device: {device}")
    logger.info(f"Datei:  {AUDIO_PATH}")

    # Modell
    model = UNetCustom(in_channels=IN_CH, out_channels=IN_CH, base_channels=BASE_CH).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)

    # 1) Audio -> S_dB -> Bild -> Gray01 -> DC-Bin entfernen (H=512) + STFT + y
    gray01, y, sr_ret = audio_to_gray01_training_like(AUDIO_PATH)
    H, W = gray01.shape
    dur_s = len(y) / float(sr_ret)
    logger.info(f"Audio-Laenge: {dur_s:.2f} s, SR: {sr_ret} Hz")
    logger.info(f"Spektrogramm (nach DC-Entfernung): H={H}, W={W}")
    logger.info(f"STFT: n_fft={N_FFT}, hop={HOP_LENGTH}, win={WIN_LENGTH}")

    # 2) Tiled Inferenz & Stitch (Grauraum [0,1])
    denoised01 = tiled_infer_and_stitch(model, gray01, device, tile_w=TILE_W, stride=STRIDE)
    tiles_est = 1 if W <= TILE_W else 1 + (W - TILE_W + STRIDE - 1) // STRIDE
    logger.info(f"Tiling: tile_w={TILE_W}, stride={STRIDE}, Tiles ~= {tiles_est}")

    # 3) Bilder speichern
    base = os.path.splitext(os.path.basename(AUDIO_PATH))[0]
    out_png_noisy  = os.path.join(RESULTS_DIR, f"extern_noisy_{base}.png")
    out_png_deno   = os.path.join(RESULTS_DIR, f"extern_denoised_{base}.png")
    save_spectrogram_png(gray01,     out_png_noisy)
    save_spectrogram_png(denoised01, out_png_deno)
    logger.info(f"Spektrogramme gespeichert: {out_png_noisy}, {out_png_deno}")

    # 4) Heuristik (Bildraum)
    mac = mean_abs_change01(gray01, denoised01)
    logger.info(f"Mittlere absolute Aenderung (Grauraum [0,1]): {mac:.6f}")

    # 5) Wellenform-Darstellung (Zeitbereich)
    out_wave_noisy = os.path.join(RESULTS_DIR, f"wave_noisy_{base}.png")
    out_wave_deno  = os.path.join(RESULTS_DIR, f"wave_denoised_{base}.png")
    save_waveform(y, sr_ret, out_wave_deno, title="Noisy Audio (Time Domain)")


    logger.info("=" * 70)
    logger.info("Fertig.")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
