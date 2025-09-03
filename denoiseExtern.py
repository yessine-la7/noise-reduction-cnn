##################################### mit flag
"""
denoiseExtern.py

Externes Audio (M4A) -> STFT -> U-Net-Denoising (Tiles) -> iSTFT -> WAV
Zwei Eingabemodi:
- "NUMERIC"   : numerische dB-Pipeline (empfohlen, invertierbar)
- "PNG_COMPAT": Bild-Pipeline ähnlich convertDataSpec.py (annähernd invertierbar; Warnung beachten)

Speichert:
- results_extern/denoised_spectrogram.png
- results_extern/denoised_audio.wav
- results_extern/denoising_output.txt  (Log-Metadaten)

Voraussetzungen:
- results_denoising/best_model.pth (aus Training)
- createModelUnet.UNetCustom
- trainLogging.setup_logging
"""

import os
import math
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image
import soundfile as sf

from createModelUnet import UNetCustom
from trainLogging import setup_logging


# =========================
# --- USER: INPUT PFAD  ---
# =========================
audios_dir = os.path.join(os.path.dirname(__file__), "audios")
audio_name = "noisy2"
audio_extension = ".m4a"
INPUT_M4A = os.path.join(audios_dir, f"{audio_name}{audio_extension}")

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results_denoising")
BEST_MODEL_PATH   = os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8.pth")

# Umschalten des Eingabemodus: "NUMERIC" (empfohlen) oder "PNG_COMPAT"
EXTERN_MODE = "NUMERIC"   # "NUMERIC" | "PNG_COMPAT"

# -------------------------------
# STFT- & Tiling-Parameter
# -------------------------------
N_FFT      = 1024         # -> 513 Bins (inkl. DC)
HOP_LENGTH = 256
WIN_LENGTH = 1024
WINDOW     = "hann"

TILE_W  = 256             # Zeit-Breite (Frames)
STRIDE  = 128             # Überlapp (Frames)
MODEL_H = 512             # wir entfernen DC -> 512

# NUMERIC-Modus: dB-Fenster (fix & invertierbar)
DB_MIN, DB_MAX = -60.0, 0.0

# Normalisierung wie im Training:
def norm_01_to_model(x01: torch.Tensor) -> torch.Tensor:
    # [0,1] -> [-1,1]
    return (x01 - 0.5) / 0.5

def model_to_norm_01(xm: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (xm * 0.5) + 0.5


# -------------------------------
# Hilfen: SSIM (nur für Heuristik)
# -------------------------------
def gaussian_window(kernel_size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0)  # 1x1xKxK
    window = window.repeat(channels, 1, 1, 1)  # Cx1xKxK (depthwise)
    return window

@torch.no_grad()
def compute_psnr_01(pred_01: torch.Tensor, target_01: torch.Tensor, eps=1e-8) -> float:
    mse = torch.mean((pred_01 - target_01) ** 2).item()
    if mse <= eps:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


# -------------------------------
# Modell-Infos aus State
# -------------------------------
def infer_in_base_channels_from_state(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """Liest in_channels und base_channels aus dem ersten Conv-Weight."""
    key = None
    for k in ["inc.block.0.weight", "inc.0.weight", "inc.conv.weight"]:
        if k in state_dict:
            key = k
            break
    if key is None:
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim == 4:
                key = k
                break
    if key is None:
        raise RuntimeError("Konnte erste Conv-Schicht im State Dict nicht finden.")
    w = state_dict[key]
    base_channels = int(w.shape[0])
    in_channels = int(w.shape[1])
    return in_channels, base_channels


# -------------------------------
# NUMERIC-Pipeline
# -------------------------------
def audio_to_spec_db01_NUMERIC(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    y -> STFT; gibt (S_db01_wo_dc [0,1], phase (513,T), amp_ref)
    dB-Fenster: DB_MIN..DB_MAX (invertierbar)
    """
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW, center=True)
    mag = np.abs(S)                       # (513, T)
    phase = np.angle(S)                   # (513, T)
    amp_ref = float(np.max(mag) + 1e-12)

    S_db = librosa.amplitude_to_db(mag, ref=amp_ref)  # typ. [-80,0]..[-5,0]
    S_db = np.clip(S_db, DB_MIN, DB_MAX)
    S_db01 = (S_db - DB_MIN) / (DB_MAX - DB_MIN)

    S_db01_wo_dc = S_db01[1:, :]         # (512, T)
    return S_db01_wo_dc.astype(np.float32), phase, amp_ref

def db01_wo_dc_to_audio_NUMERIC(den_db01_wo_dc: np.ndarray, phase: np.ndarray, amp_ref: float, length: int) -> np.ndarray:
    """[0,1] -> dB -> Amplitude (mit fixem dB-Fenster) -> iSTFT"""
    db = den_db01_wo_dc * (DB_MAX - DB_MIN) + DB_MIN
    mag = librosa.db_to_amplitude(db, ref=amp_ref)     # (512, T)
    dc_mag = np.zeros((1, mag.shape[1]), dtype=np.float32)
    mag_full = np.vstack([dc_mag, mag])                # (513,T)

    S_complex = mag_full * np.exp(1j * phase)
    y_hat = librosa.istft(S_complex, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW, length=length)
    return y_hat.astype(np.float32)


# -------------------------------
# PNG_COMPAT-Pipeline (annähernd)
# -------------------------------
def render_spec_rgb(S_db: np.ndarray, dpi: int = 100, cmap: str = "magma", vmin=None, vmax=None) -> np.ndarray:
    """
    Rendert S_db (H,B) als RGB-Array (H,B,3) wie specshow -> PNG, ohne zu speichern.
    vmin/vmax: Skalen. Wenn None, werden min/max des S_db verwendet (PNG-kompatibles Verhalten).
    """
    H, B = S_db.shape
    if vmin is None: vmin = float(S_db.min())
    if vmax is None: vmax = float(S_db.max())
    fig_w = B / dpi
    fig_h = H / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # randlos
    img = librosa.display.specshow(S_db, x_axis=None, y_axis=None, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)
    ax.set_axis_off()
    fig.canvas.draw()
    # RGBA aus dem Canvas lesen
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rgb = buf.reshape(int(fig.canvas.get_width_height()[1]), int(fig.canvas.get_width_height()[0]), 3)
    plt.close(fig)
    # Form sollte (H,B,3) sein
    if rgb.shape[0] != H or rgb.shape[1] != B:
        # zur Sicherheit, falls Backends runden
        rgb = np.array(Image.fromarray(rgb).resize((B, H), resample=Image.NEAREST))
    rgb = rgb.astype(np.float32) / 255.0
    return rgb

def rgb_to_gray_luma(rgb: np.ndarray) -> np.ndarray:
    """sRGB -> Luma (ITU-R BT.601) ~ PIL 'L'."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b

def build_gray_inverse_LUT(cmap_name: str = "magma", levels: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baut eine LUT: grau(z) ~ gray(colormap(z)) -> invertierbar via Interpolation.
    Rückgabe: (z_vals, gray_vals), monotone Kurven (typisch für 'magma').
    """
    cm = plt.get_cmap(cmap_name)
    z = np.linspace(0, 1, levels, dtype=np.float32)
    rgb = cm(z)[..., :3]  # (L,3)
    gray = rgb_to_gray_luma(rgb)
    # Für 'magma' ist gray(z) streng monoton steigend -> direkt invertierbar
    return z, gray

def audio_to_spec_img01_PNGCOMPAT(y: np.ndarray, cmap: str = "magma") -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    y -> STFT -> dB -> spekshow-RGB -> (optional) Grau -> [0,1]
    Rückgabe:
      img01_wo_dc  : (C,H=512,W) in [0,1] (C=1 oder 3, je nach späterem Modell)
      phase        : (513,W)
      amp_ref      : float
      vmin, vmax   : Skalierung, falls wir invertieren wollen
    """
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW, center=True)
    mag = np.abs(S)
    phase = np.angle(S)
    amp_ref = float(np.max(mag) + 1e-12)

    S_db = librosa.amplitude_to_db(mag, ref=amp_ref)  # auto range (z. B. [-84.3, -3.1])
    vmin, vmax = float(S_db.min()), float(S_db.max())
    rgb = render_spec_rgb(S_db, dpi=100, cmap=cmap, vmin=vmin, vmax=vmax)  # (H=513,W,3)

    # DC-Zeile entfernen -> (512,W,3)
    rgb_wo_dc = rgb[1:, :, :]

    return rgb_wo_dc.transpose(2,0,1).astype(np.float32), phase, amp_ref, vmin, vmax  # (3,512,W)

def invert_img01_to_db_PNGCOMPAT(img01: np.ndarray, vmin: float, vmax: float, in_channels: int, cmap: str = "magma") -> np.ndarray:
    """
    Invertiert ein Bild (C,512,W) in [0,1] zurück in dB (512,W), näherungsweise.
    - Bei C=1: Annahme: grayscale ≈ luma(cmap(z))  -> invertiere via LUT
    - Bei C=3: Annahme: RGB ≈ cmap(z)             -> invertiere via NN auf der Colormap
    """
    C, H, W = img01.shape
    assert H == MODEL_H
    if in_channels == 1:
        # LUT gray(z) -> z invertieren via Interpolation
        z_vals, gray_vals = build_gray_inverse_LUT(cmap_name=cmap, levels=2048)
        # Sicherheit: monotone Steigung erzwingen (für Interp)
        idx = np.argsort(gray_vals)
        gray_vals_sorted = gray_vals[idx]
        z_sorted = z_vals[idx]

        gray = img01[0]  # (H,W)
        # Interpolation (grau->z), clamp in [0,1]
        z = np.interp(gray.flatten(), gray_vals_sorted, z_sorted).reshape(H, W)
    else:
        # RGB -> z: wähle den z mit minimaler Distanz in RGB (NN auf feiner Tabelle)
        cm = plt.get_cmap(cmap)
        z_vals = np.linspace(0,1,4096, dtype=np.float32)
        rgb_lut = cm(z_vals)[..., :3]  # (L,3)
        img = img01.transpose(1,2,0)   # (H,W,3)
        img_f = img.reshape(-1,3)
        # Squared distances: brute force (H*W x L) kann groß sein; deshalb chunken
        L = rgb_lut.shape[0]
        out_z = np.empty((img_f.shape[0],), dtype=np.float32)
        bs = 65536
        for i in range(0, img_f.shape[0], bs):
            chunk = img_f[i:i+bs][:, None, :]             # (B,1,3)
            d2 = np.sum((chunk - rgb_lut[None, :, :])**2, axis=2)  # (B,L)
            idx = np.argmin(d2, axis=1)
            out_z[i:i+bs] = z_vals[idx]
        z = out_z.reshape(H, W)

    # z \in [0,1] entspricht S_db normiert auf [vmin,vmax]
    S_db = z * (vmax - vmin) + vmin
    return S_db.astype(np.float32)


# -------------------------------
# Tiled-Inference (beide Modi)
# -------------------------------
@torch.no_grad()
def denoise_tiles_model_space(model: nn.Module, img01: np.ndarray, device: torch.device,
                              tile_w: int = TILE_W, stride: int = STRIDE) -> np.ndarray:
    """
    img01: (C,512,W) in [0,1]  ->  Modellraum [-1,1] -> zurück [0,1]
    Overlap-Add mit 1D-Hann über Zeit.
    """
    C, H, W = img01.shape
    assert H == MODEL_H
    pad_right = 0
    if W < tile_w:
        pad_right = tile_w - W
        Wp = tile_w
    else:
        # last tile alignment
        last_start = (max(0, (W - tile_w + stride - 1) // stride) * stride)
        if last_start + tile_w > W:
            pad_right = (last_start + tile_w) - W
        Wp = W + pad_right

    if pad_right > 0:
        pad = np.pad(img01, ((0,0),(0,0),(0,pad_right)), mode="edge")
    else:
        pad = img01

    hann = np.hanning(tile_w).astype(np.float32)      # (tile_w,)
    hann2d = np.repeat(hann[np.newaxis, :], H, axis=0)

    out_acc = np.zeros((C, H, Wp), dtype=np.float32)
    w_acc   = np.zeros((H, Wp), dtype=np.float32)

    x01 = torch.from_numpy(pad).unsqueeze(0).to(device)     # (1,C,H,Wp)
    for x in range(0, Wp - tile_w + 1, stride):
        tile01 = x01[:, :, :, x:x+tile_w]                   # (1,C,H,tile_w)
        tile_m = norm_01_to_model(tile01)                   # [-1,1]
        pred_m = model(tile_m)
        pred01 = model_to_norm_01(pred_m).squeeze(0).detach().cpu().numpy()  # (C,H,tile_w)
        pred01 = np.clip(pred01, 0.0, 1.0)

        out_acc[:, :, x:x+tile_w] += (pred01 * hann2d[None, :, :])
        w_acc[:,       x:x+tile_w] += hann2d

    w_acc[w_acc < 1e-8] = 1.0
    out = out_acc / w_acc[None, :, :]

    return out[:, :, :W]   # (C,512,W)


# -------------------------------
# Heuristiken ohne Clean
# -------------------------------
def spectral_flatness_median(mag: np.ndarray) -> float:
    sfm = librosa.feature.spectral_flatness(S=np.maximum(1e-12, mag**2))
    return float(np.median(sfm))

def hf_lf_energy_ratio(mag: np.ndarray, sr: int, split_hz: float = 4000.0) -> float:
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    idx = np.where(freqs >= split_hz)[0]
    if len(idx) == 0:
        return 0.0
    hf = np.sum(mag[idx, :])
    lf = np.sum(mag[:idx[0], :]) + 1e-12
    return float(hf / lf)


# -------------------------------
# Hauptprogramm
# -------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger = setup_logging(RESULTS_DIR, log_file="denoising_output.txt", level="INFO")

    if not os.path.isfile(INPUT_M4A):
        logger.error(f"Eingabedatei nicht gefunden: {INPUT_M4A}")
        return
    if not os.path.isfile(BEST_MODEL_PATH):
        logger.error(f"Modell nicht gefunden: {BEST_MODEL_PATH}")
        return

    # Audio laden
    y, sr = librosa.load(INPUT_M4A, sr=None, mono=True)
    length_sec = len(y) / float(sr)
    logger.info(f"Input: {os.path.basename(INPUT_M4A)} | SR={sr} Hz | Dauer={length_sec:.2f} s | Samples={len(y)}")
    logger.info(f"EXTERN_MODE = {EXTERN_MODE}")
    if EXTERN_MODE.upper() == "PNG_COMPAT":
        logger.warning("PNG_COMPAT gewählt: Audio-Inversion ist nur näherungsweise. Für beste Qualität NUMERIC verwenden.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modell rekonstruieren & laden
    state = torch.load(BEST_MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    in_ch, base_ch = infer_in_base_channels_from_state(state)
    model = UNetCustom(in_channels=in_ch, out_channels=in_ch, base_channels=base_ch).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    logger.info(f"Model: in_channels={in_ch}, base_channels={base_ch}")

    # Vorwärtsweg je nach Modus
    if EXTERN_MODE.upper() == "NUMERIC":
        # --- NUMERIC: [0,1] (fixes dB-Fenster) ---
        spec01_wo_dc, phase, amp_ref = audio_to_spec_db01_NUMERIC(y)
        img01_in = spec01_wo_dc[None, :, :] if in_ch == 1 else np.repeat(spec01_wo_dc[None, :, :], repeats=3, axis=0)
        den01 = denoise_tiles_model_space(model, img01_in, device, TILE_W, STRIDE)  # (C,512,W)
        den_db01_wo_dc = den01[0] if in_ch == 1 else den01.mean(axis=0)  # (512,W)
        # PNG speichern
        H, W = den_db01_wo_dc.shape
        fig, ax = plt.subplots(figsize=(W/20, H/50), dpi=100)
        ax.imshow(den_db01_wo_dc, origin="upper", aspect="auto", cmap="magma")
        ax.set_title("Denoised STFT (NUMERIC, [0,1], DC removed)"); ax.axis("off")
        out_png = os.path.join(RESULTS_DIR, "denoised_spectrogram_flag.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close(fig)
        logger.info(f"Spektrogramm gespeichert: {out_png}")

        # Audio
        y_hat = db01_wo_dc_to_audio_NUMERIC(den_db01_wo_dc, phase, amp_ref, length=len(y))

    else:
        # --- PNG_COMPAT: spekshow-Bild (RGB oder Grau) in [0,1] ---
        rgbC, phase, amp_ref, vmin, vmax = audio_to_spec_img01_PNGCOMPAT(y, cmap="magma")  # (3,512,W)
        if in_ch == 1:
            gray = rgb_to_gray_luma(rgbC.transpose(1,2,0)).transpose(1,0)  # (512,W)
            img01_in = gray[None, :, :]                                     # (1,512,W)
        else:
            img01_in = rgbC                                                 # (3,512,W)

        den01 = denoise_tiles_model_space(model, img01_in, device, TILE_W, STRIDE)  # (C,512,W)

        # PNG speichern (direkt das Modell-Output-Intensitätsbild)
        if in_ch == 1:
            to_show = den01[0]
        else:
            to_show = den01.transpose(1,2,0)  # (H,W,3)
        H, W = den01.shape[1], den01.shape[2]
        fig, ax = plt.subplots(figsize=(W/20, H/50), dpi=100)
        ax.imshow(to_show if in_ch == 3 else to_show, origin="upper", aspect="auto", cmap=("magma" if in_ch==1 else None))
        ax.set_title("Denoised STFT (PNG_COMPAT, model-output)"); ax.axis("off")
        out_png = os.path.join(RESULTS_DIR, "denoised_spectrogram_flag.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close(fig)
        logger.info(f"Spektrogramm gespeichert: {out_png}")

        # Annähernde Inversion zurück in dB
        S_db_wo_dc = invert_img01_to_db_PNGCOMPAT(den01, vmin=vmin, vmax=vmax, in_channels=in_ch, cmap="magma")
        # dB -> Amplitude
        mag = librosa.db_to_amplitude(S_db_wo_dc, ref=amp_ref)   # (512,W)
        dc_mag = np.zeros((1, mag.shape[1]), dtype=np.float32)
        mag_full = np.vstack([dc_mag, mag])
        S_complex = mag_full * np.exp(1j * phase)
        y_hat = librosa.istft(S_complex, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW, length=len(y))

    # WAV speichern
    out_wav = os.path.join(RESULTS_DIR, "denoised_audio_flag.wav")
    sf.write(out_wav, y_hat.astype(np.float32), sr)
    logger.info(f"Denoised Audio gespeichert: {out_wav}")

    # Heuristiken (ohne Clean)
    S_noisy = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW, center=True)
    mag_noisy = np.abs(S_noisy)
    if EXTERN_MODE.upper() == "NUMERIC":
        db = den_db01_wo_dc * (DB_MAX - DB_MIN) + DB_MIN
        mag_denoised_wo_dc = librosa.db_to_amplitude(db, ref=np.max(mag_noisy) + 1e-12)
    else:
        mag_denoised_wo_dc = mag  # bereits aus PNG_COMPAT-Zweig
    mag_denoised = np.vstack([np.zeros((1, mag_denoised_wo_dc.shape[1]), dtype=np.float32), mag_denoised_wo_dc])

    flat_before = spectral_flatness_median(mag_noisy)
    flat_after  = spectral_flatness_median(mag_denoised)
    def hf_lf(m):
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        idx = np.where(freqs >= 4000.0)[0]
        if len(idx) == 0: return 0.0
        hf = float(np.sum(m[idx, :])); lf = float(np.sum(m[:idx[0], :]) + 1e-12); return hf/lf
    ratio_before = hf_lf(mag_noisy)
    ratio_after  = hf_lf(mag_denoised)

    logger.info("=== Zusammenfassung ===")
    logger.info(f"Mode: {EXTERN_MODE}")
    logger.info(f"Input: {os.path.basename(INPUT_M4A)} | Dauer={length_sec:.2f}s @ {sr} Hz")
    logger.info(f"Tiles: W={TILE_W}, stride={STRIDE}")
    logger.info(f"Model: in_ch={in_ch}, base_ch={base_ch}")
    logger.info(f"Heuristik: spectral flatness median (↓ gut): before={flat_before:.4f}, after={flat_after:.4f}")
    logger.info(f"Heuristik: HF/LF energy ratio (↓ gut):      before={ratio_before:.4f}, after={ratio_after:.4f}")
    if EXTERN_MODE.upper() == "PNG_COMPAT":
        logger.warning("PNG_COMPAT: Die Audio-Rekonstruktion ist nur näherungsweise und kann hörbare Artefakte enthalten. "
                       "Für robuste Inversion bitte NUMERIC verwenden.")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()















# ##################################### mit db_min/db_max with audio
# """
# Externes Audio (M4A) -> STFT -> U-Net-Denoising (Tiles) -> iSTFT -> WAV
# Speichert:
# - results_extern/denoised_spectrogram.png
# - results_extern/denoised_audio.wav
# - results_extern/denoising_output.txt (Logging)
# Voraussetzungen:
# - results_denoising/best_model.pth (aus dem Training)
# - createModelUnet.UNetCustom
# - trainLogging.setup_logging (oder das gleiche Setup hier)
# """

# import os
# import math
# import random
# from typing import Tuple, Dict

# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import librosa
# import librosa.display
# import soundfile as sf

# from createModelUnet import UNetCustom
# from trainLogging import setup_logging


# # =========================
# # --- USER: INPUT PFAD  ---
# # =========================
# audios_dir = os.path.join(os.path.dirname(__file__), "audios")
# audio_name = "noisy2"
# audio_extension = ".m4a"
# INPUT_M4A = os.path.join(audios_dir, f"{audio_name}{audio_extension}")

# RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results_denoising")
# BEST_MODEL_PATH   = os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8.pth")

# # -------------------------------
# # Pipeline- und STFT-Parameter
# # -------------------------------
# # Diese Parameter sind konsistent zu einem üblichen Setup mit H=513 (n_fft=1024):
# N_FFT      = 1024         # -> 513 Frequenz-Bins (inkl. DC)
# HOP_LENGTH = 256
# WIN_LENGTH = 1024
# WINDOW     = "hann"

# # Tiling wie im Training:
# TILE_W  = 256             # Zeit-Breite (Frames)
# STRIDE  = 128             # Überlapp (Frames)
# # Modell erwartet Höhe 512 (DC-Zeile entfernt):
# MODEL_H = 512

# DB_MIN = 150.0

# # Normalisierung wie im Training:
# # Im Loader: ToTensor() -> [0,1], danach Normalize(mean=0.5, std=0.5) -> [-1,1]
# def norm_01_to_model(x01: torch.Tensor) -> torch.Tensor:
#     # [0,1] -> [-1,1]
#     return (x01 - 0.5) / 0.5

# def model_to_norm_01(xm: torch.Tensor) -> torch.Tensor:
#     # [-1,1] -> [0,1]
#     return (xm * 0.5) + 0.5


# # -------------------------------
# # SSIM-Helfer (für Heuristik/Reporting – optional)
# # -------------------------------
# def gaussian_window(kernel_size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
#     coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
#     g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
#     g = g / g.sum()
#     window = g[:, None] @ g[None, :]
#     window = window / window.sum()
#     window = window.unsqueeze(0).unsqueeze(0)  # 1x1xKxK
#     window = window.repeat(channels, 1, 1, 1)  # Cx1xKxK (depthwise)
#     return window

# @torch.no_grad()
# def compute_psnr_01(pred_01: torch.Tensor, target_01: torch.Tensor, eps=1e-8) -> float:
#     mse = torch.mean((pred_01 - target_01) ** 2).item()
#     if mse <= eps:
#         return 99.0
#     return float(10.0 * math.log10(1.0 / mse))

# @torch.no_grad()
# def compute_ssim_01(pred_01: torch.Tensor, target_01: torch.Tensor, window: torch.Tensor,
#                     C1=0.01**2, C2=0.03**2) -> float:
#     window = window.to(pred_01.device, dtype=pred_01.dtype)
#     C = pred_01.shape[1]
#     mu_x = torch.nn.functional.conv2d(pred_01, window, padding=window.shape[-1]//2, groups=C)
#     mu_y = torch.nn.functional.conv2d(target_01, window, padding=window.shape[-1]//2, groups=C)
#     mu_x2 = mu_x * mu_x
#     mu_y2 = mu_y * mu_y
#     mu_xy = mu_x * mu_y

#     sigma_x2 = torch.nn.functional.conv2d(pred_01 * pred_01, window, padding=window.shape[-1]//2, groups=C) - mu_x2
#     sigma_y2 = torch.nn.functional.conv2d(target_01 * target_01, window, padding=window.shape[-1]//2, groups=C) - mu_y2
#     sigma_xy = torch.nn.functional.conv2d(pred_01 * target_01, window, padding=window.shape[-1]//2, groups=C) - mu_xy

#     ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
#     return float(ssim_map.mean().item())


# # -------------------------------
# # Utility: Modell rekonstruieren
# # -------------------------------
# def infer_in_base_channels_from_state(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
#     """
#     Liest aus dem state_dict die Kanalbreiten der ersten Conv (inc.block.0.weight)
#     -> (in_channels, base_channels)
#     """
#     # Typische Schlüssel: 'inc.block.0.weight' mit Shape (base_ch, in_ch, 3, 3)
#     # Fallback: finde den ersten 4D-Conv-Gewichtseintrag
#     key = None
#     for k in ["inc.block.0.weight", "inc.0.weight", "inc.conv.weight"]:
#         if k in state_dict:
#             key = k
#             break
#     if key is None:
#         # Suche das erste 4D-Weight
#         for k, v in state_dict.items():
#             if isinstance(v, torch.Tensor) and v.ndim == 4:
#                 key = k
#                 break
#     if key is None:
#         raise RuntimeError("Konnte die erste Conv-Schicht im State Dict nicht finden.")
#     w = state_dict[key]
#     base_channels = int(w.shape[0])
#     in_channels = int(w.shape[1])
#     return in_channels, base_channels


# # -------------------------------
# # Audio -> STFT -> normiertes Bild
# # -------------------------------
# def audio_to_spec_db01(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, float]:
#     """
#     y: 1D-Audio
#     Rückgabe:
#       - S_db01_wo_dc: (512, T) in [0,1] (DC-Zeile entfernt)
#       - phase: (513, T) Winkel der Noisy-STFT
#       - amp_ref: Skalenreferenz (Peak-Amplitude), um dB zurück zu Amplitude zu konvertieren
#     """
#     S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW, center=True)
#     mag = np.abs(S)                       # (513, T)
#     phase = np.angle(S)                   # (513, T)
#     amp_ref = float(np.max(mag) + 1e-12)  # Peak als Referenz für dB-Skalierung

#     # dB ([-∞, 0]) relativ zu amp_ref≈max
#     S_db = librosa.amplitude_to_db(mag, ref=amp_ref)       # typ. [-80, 0]
#     # Auf [0,1] normalisieren (−80..0 -> 0..1)
#     S_db01 = np.clip((S_db + DB_MIN) / DB_MIN, 0.0, 1.0)

#     # DC-Zeile (Index 0) entfernen -> Höhe 512 (1..512)
#     S_db01_wo_dc = S_db01[1:, :]          # (512, T)

#     return S_db01_wo_dc.astype(np.float32), phase, amp_ref


# # -------------------------------
# # Tiled-Inference: [0,1] -> [-1,1] -> Model -> [0,1]
# # -------------------------------
# @torch.no_grad()
# def denoise_spectrogram_db01(model: nn.Module, spec_db01_wo_dc: np.ndarray, device: torch.device,
#                              tile_w: int = TILE_W, stride: int = STRIDE) -> np.ndarray:
#     """
#     spec_db01_wo_dc: (512, T) in [0,1]
#     Rückgabe: gleiche Form, denoised in [0,1]
#     """
#     H, W = spec_db01_wo_dc.shape
#     assert H == MODEL_H, f"Erwarte Höhe {MODEL_H}, bekam {H}"

#     # Optionales Padding in der Breite, falls zu klein
#     pad_right = 0
#     if W < tile_w:
#         pad_right = tile_w - W
#     else:
#         # so paddden, dass die letzte Kachel auch genau passt
#         last_start = (max(0, (W - tile_w + stride - 1) // stride) * stride)
#         if last_start + tile_w > W:
#             pad_right = (last_start + tile_w) - W

#     if pad_right > 0:
#         spec_pad = np.pad(spec_db01_wo_dc, ((0,0), (0, pad_right)), mode="edge")
#         Wp = spec_pad.shape[1]
#     else:
#         spec_pad = spec_db01_wo_dc
#         Wp = W

#     # Overlap-Add mit Hann-1D über die Zeitachse
#     hann = np.hanning(tile_w).astype(np.float32)  # (tile_w,)
#     hann2d = np.repeat(hann[np.newaxis, :], H, axis=0)  # (H, tile_w)

#     out_acc = np.zeros_like(spec_pad, dtype=np.float32)
#     w_acc   = np.zeros_like(spec_pad, dtype=np.float32)

#     x01 = torch.from_numpy(spec_pad).unsqueeze(0).unsqueeze(0)  # (1,1,H,Wp)
#     x01 = x01.to(device)

#     # Sliding Window
#     for x in range(0, Wp - tile_w + 1, stride):
#         tile01 = x01[:, :, :, x:x+tile_w]              # (1,1,H,tile_w)
#         tile_m = norm_01_to_model(tile01)              # [-1,1]
#         pred_m = model(tile_m)                         # [-1,1]
#         pred01 = model_to_norm_01(pred_m).squeeze(0).squeeze(0)  # (H,tile_w), [0,1]
#         pred01 = torch.clamp(pred01, 0.0, 1.0).detach().cpu().numpy()

#         out_acc[:, x:x+tile_w] += (pred01 * hann2d)
#         w_acc[:,   x:x+tile_w] += hann2d

#     # vermeiden von Division durch 0
#     w_acc[w_acc < 1e-8] = 1.0
#     out = out_acc / w_acc

#     # auf Originalbreite zurückcroppen
#     out = out[:, :W]
#     return out


# # -------------------------------
# # Spektrogramm [0,1] -> Amplitude -> iSTFT
# # -------------------------------
# def db01_wo_dc_to_audio(den_db01_wo_dc: np.ndarray, phase: np.ndarray, amp_ref: float, length: int) -> np.ndarray:
#     """
#     den_db01_wo_dc: (512, T) in [0,1]
#     phase: (513, T) aus Noisy
#     amp_ref: skalierung für dB->Amplitude (Peak)
#     length: Ziel-Länge in Samples (für librosa.istft)
#     """
#     # [0,1] -> dB (−80..0)
#     db = den_db01_wo_dc * DB_MIN - DB_MIN
#     # dB -> Amplitude, relativ zu amp_ref
#     mag = librosa.db_to_amplitude(db, ref=amp_ref)     # (512, T)
#     # DC-Zeile wieder einfügen: nimm die DC aus dem Noisy (robuster als 0)
#     # phase hat Form (513, T), wir brauchen |S| auch (513,T)
#     dc_mag = np.maximum(1e-8, np.ones((1, mag.shape[1]), dtype=np.float32) * 0.0)  # Default 0
#     # Option: DC von Noisy herstellen? – wir haben sie nicht mehr hier;
#     # Wenn du die Noisy-DC behalten willst, gib sie aus audio_to_spec_db01 zurück.
#     # Für einfache Rekonstruktion reicht DC≈0.
#     mag_full = np.vstack([dc_mag, mag])                # (513, T)

#     # Komplexes Spektrogramm aus Magnitude + Noisy-Phase
#     S_complex = mag_full * np.exp(1j * phase)
#     # iSTFT
#     y_hat = librosa.istft(S_complex, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW, length=length)
#     return y_hat.astype(np.float32)


# # -------------------------------
# # Heuristische „Qualitäts“-Infos ohne Clean
# # -------------------------------
# def spectral_flatness_median(mag: np.ndarray) -> float:
#     """
#     Spektrale Flachheit (0 = tonaler/strukturierter, 1 = rauschhaft).
#     Erwartet Magnitude (nicht dB, nicht [0,1]) mit Form (F,T).
#     """
#     # librosa.feature.spectral_flatness erwartet Power-Spektrum; wir nehmen mag**2
#     sfm = librosa.feature.spectral_flatness(S=np.maximum(1e-12, mag**2))
#     # Median über Zeit
#     return float(np.median(sfm))

# def hf_lf_energy_ratio(mag: np.ndarray, sr: int, split_hz: float = 4000.0) -> float:
#     """
#     Verhältnis High-Frequency (>=split_hz) zu Low-Frequency Energie.
#     Erwartet Magnitude (F,T).
#     """
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)  # (513,)
#     idx = np.where(freqs >= split_hz)[0]
#     if len(idx) == 0:
#         return 0.0
#     hf = np.sum(mag[idx, :])
#     lf = np.sum(mag[:idx[0], :]) + 1e-12
#     return float(hf / lf)


# # -------------------------------
# # Hauptfunktion
# # -------------------------------
# def main():
#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     logger = setup_logging(RESULTS_DIR, log_file="denoising_output.txt", level="INFO")

#     if not os.path.isfile(INPUT_M4A):
#         logger.error(f"Eingabedatei nicht gefunden: {INPUT_M4A}")
#         return
#     if not os.path.isfile(BEST_MODEL_PATH):
#         logger.error(f"Modell nicht gefunden: {BEST_MODEL_PATH}")
#         return

#     # Audio laden (Mono)
#     # librosa nutzt audioread/ffmpeg – m4a wird meist unterstützt, wenn ffmpeg installiert ist.
#     y, sr = librosa.load(INPUT_M4A, sr=None, mono=True)
#     length_sec = len(y) / float(sr)
#     logger.info(f"Input: {os.path.basename(INPUT_M4A)} | SR={sr} Hz | Dauer={length_sec:.2f} s | Samples={len(y)}")

#     # Spektrogramm -> [0,1] + Phase
#     spec01_wo_dc, phase, amp_ref = audio_to_spec_db01(y, sr)
#     H, W = spec01_wo_dc.shape
#     logger.info(f"STFT: shape wo DC = {H}x{W} (H=freq bins, W=time frames), N_FFT={N_FFT}, hop={HOP_LENGTH}")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Modell rekonstruieren (in_channels & base_channels aus State ableiten)
#     state = torch.load(BEST_MODEL_PATH, map_location=device)
#     if isinstance(state, dict) and "state_dict" in state:
#         state = state["state_dict"]
#     in_ch, base_ch = infer_in_base_channels_from_state(state)
#     logger.info(f"Model config inferred: in_channels={in_ch}, base_channels={base_ch}")

#     model = UNetCustom(in_channels=in_ch, out_channels=in_ch, base_channels=base_ch).to(device)
#     model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
#     model.eval()

#     # Denoising (Tiles, Overlap-Add)
#     den01_wo_dc = denoise_spectrogram_db01(model, spec01_wo_dc, device, tile_w=TILE_W, stride=STRIDE)

#     # PNG des denoised Spektrogramms speichern (auf [0,1])
#     fig, ax = plt.subplots(figsize=(W/20, H/50), dpi=100)
#     ax.imshow(den01_wo_dc, origin="upper", aspect="auto", cmap="magma")
#     ax.set_title("Denoised STFT (magnitude, [0,1], DC removed)")
#     ax.axis("off")
#     out_png = os.path.join(RESULTS_DIR, "denoised_spectrogram.png")
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=120)
#     plt.close(fig)
#     logger.info(f"Spektrogramm gespeichert: {out_png}")

#     # Heuristische „Qualität“ (ohne Clean)
#     # Vergleiche Metriken auf Amplitudenbasis (nicht dB)
#     # Rekonstruiere Magnitude (Noisy) für Heuristiken
#     S_noisy = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW, center=True)
#     mag_noisy = np.abs(S_noisy)              # (513,T)
#     # Denormierte Denoised-Magnitude (ohne DC -> fügen 0 an)
#     db_pred = den01_wo_dc * DB_MIN - DB_MIN
#     mag_pred_wo_dc = librosa.db_to_amplitude(db_pred, ref=amp_ref)
#     mag_pred = np.vstack([np.zeros((1, mag_pred_wo_dc.shape[1]), dtype=np.float32), mag_pred_wo_dc])

#     flat_before = spectral_flatness_median(mag_noisy)
#     flat_after  = spectral_flatness_median(mag_pred)
#     ratio_before = hf_lf_energy_ratio(mag_noisy, sr, split_hz=4000.0)
#     ratio_after  = hf_lf_energy_ratio(mag_pred,  sr, split_hz=4000.0)
#     mean_abs_change01 = float(np.mean(np.abs(den01_wo_dc - spec01_wo_dc)))

#     logger.info(f"Heuristik: spectral flatness median (↓ gut): before={flat_before:.4f}, after={flat_after:.4f}")
#     logger.info(f"Heuristik: HF/LF energy ratio (↓ gut):      before={ratio_before:.4f}, after={ratio_after:.4f}")
#     logger.info(f"Heuristik: mean |Δ| in [0,1] (Info):        {mean_abs_change01:.4f}")

#     # Audio synthetisieren (iSTFT) – Noisy-Phase + denoised Magnitude
#     y_hat = db01_wo_dc_to_audio(den01_wo_dc, phase, amp_ref, length=len(y))
#     out_wav = os.path.join(RESULTS_DIR, f"denoised_audio_{DB_MIN}.wav")
#     sf.write(out_wav, y_hat, sr)
#     logger.info(f"Denoised Audio gespeichert: {out_wav}")

#     # Zusammenfassung
#     logger.info("=== Zusammenfassung ===")
#     logger.info(f"Input: {os.path.basename(INPUT_M4A)}")
#     logger.info(f"Länge: {length_sec:.2f} s @ {sr} Hz")
#     logger.info(f"Tiles (W,stride): {TILE_W}, {STRIDE}")
#     logger.info(f"Modell: in_ch={in_ch}, base_ch={base_ch}")
#     logger.info(f"Ausgaben: {os.path.basename(out_png)}, {os.path.basename(out_wav)}")
#     logger.info("Heuristik-Interpretation: kleinere Flatness & kleiner HF/LF-Ratio deuten auf gelungene Rauschunterdrückung hin.")


# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()











# ##################################### mit db_min/db_max no audio
# """
# Externe (Handy-)Aufnahme -> STFT -> specshow (wie convertDataSpec.py) -> Bild -> Graustufe -> DC-Zeile weg -> Tiles
# -> U-Net -> Hann-Overlap-Stitching -> denoised Bild speichern (ohne Achsen).
# Logging über trainLogging.setup_logging nach results_denoising/denoising_output.txt.

# WICHTIG: Diese Pipeline vermeidet feste db_min/db_max und spiegelt damit dein Trainings-Preprocessing.
# """

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import librosa
# import librosa.display
# from PIL import Image

# from createModelUnet import UNetCustom
# from trainLogging import setup_logging

# # =========================
# #   Pfaden
# # =========================
# audios_dir = os.path.join(os.path.dirname(__file__), "audios")
# audio_name = "noisy2"
# audio_extension = ".m4a"
# AUDIO_PATH = os.path.join(audios_dir, f"{audio_name}{audio_extension}")

# RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results_denoising")
# MODEL_PATH   = os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8.pth")
# LOG_FILE     = "denoising_output.txt"


# # STFT-Parameter (wie convertDataSpec.py)
# N_FFT      = 1024
# HOP_LENGTH = 256
# WIN_LENGTH = 1024
# SR         = None  # wie convertDataSpec.py (sr=None) -> keine Resampling-Änderung

# # Rendering (Pixelgenau wie convertDataSpec.py, default cmap)
# DPI = 100  # Bildgröße = (W/DPI, H/DPI)

# # Tiling
# TILE_W  = 256
# STRIDE  = 128
# IN_CH   = 1     # Graustufen
# BASE_CH = 32

# # =========================
# #   Helfer
# # =========================
# def render_spec_like_training_to_gray01(S_dB, sr):
#     """
#     Rendert S_dB mit librosa.display.specshow -> RGB-Canvas -> Graustufe (Luma 0.299/0.587/0.114).
#     Keine vmin/vmax -> identisch zum Konverter (per-Image-Norm durch Matplotlib).
#     Gibt ein 2D-Array [0,1] mit Höhe=H von S_dB und Breite=W zurück.
#     """
#     H, W = S_dB.shape
#     fig_w = W / DPI
#     fig_h = H / DPI
#     fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
#     # Vollflächig, ohne Ränder
#     ax.set_position([0, 0, 1, 1])
#     librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax)
#     ax.axis('off')
#     fig.canvas.draw()  # rendern
#     # RGB aus Canvas lesen
#     w_px, h_px = fig.canvas.get_width_height()  # (W, H)
#     rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     rgb = rgb.reshape(h_px, w_px, 3)  # (H, W, 3)
#     plt.close(fig)
#     # RGB -> Graustufe (wie PIL-Luma grob)
#     # gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
#     # gray = np.clip(gray, 0.0, 1.0).astype(np.float32)  # (H, W)
#     gray = np.array(Image.fromarray(rgb).convert("L"), dtype=np.float32) / 255.0
#     return gray

# def audio_to_gray01_training_like(wav_path):
#     """
#     1) Audio laden (sr=None, mono=True).
#     2) STFT -> |S| -> dB (ref=max).
#     3) Rendern zu Gray [0,1] via specshow (wie im Dataset).
#     4) DC-Zeile entfernen (erste Pixelzeile) -> Höhe 512.
#     """
#     y, sr_ret = librosa.load(wav_path, sr=SR, mono=True)
#     dur_s = len(y) / sr_ret

#     S = librosa.stft(
#         y=y,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         win_length=WIN_LENGTH,
#         window="hann",
#         center=True,
#         pad_mode="reflect",
#     )
#     S_mag = np.abs(S)                     # (513, W)
#     S_dB  = librosa.amplitude_to_db(S_mag, ref=np.max)

#     # Bild wie im Training erzeugen
#     gray01 = render_spec_like_training_to_gray01(S_dB, sr_ret)  # (513, W)
#     # DC-Zeile weg
#     gray01 = gray01[1:, :]  # (512, W)
#     return gray01, float(dur_s), int(sr_ret)

# def hann1d_w(tile_w: int) -> np.ndarray:
#     if tile_w <= 1:
#         return np.ones((1,), dtype=np.float32)
#     return np.hanning(tile_w).astype(np.float32)

# def to_model_space(tile01: torch.Tensor) -> torch.Tensor:
#     # [0,1] -> [-1,1]
#     return tile01 * 2.0 - 1.0

# def to_img01_space(tile_norm: torch.Tensor) -> torch.Tensor:
#     # [-1,1] -> [0,1]
#     return (tile_norm + 1.0) * 0.5

# def tiled_infer_and_stitch(model: nn.Module, img01: np.ndarray, device: torch.device,
#                            tile_w: int = TILE_W, stride: int = STRIDE) -> np.ndarray:
#     """
#     Horizontal Tilen, Inferenz, Hann-Overlap-Stitch.
#     """
#     H, W = img01.shape
#     need_pad = W < tile_w
#     W_pad = max(W, tile_w)
#     if not need_pad and ((W - tile_w) % stride != 0):
#         n_steps = (W - tile_w + stride - 1) // stride
#         last_x = n_steps * stride
#         W_pad = last_x + tile_w
#         need_pad = W_pad > W

#     if need_pad:
#         img01_pad = np.zeros((H, W_pad), dtype=np.float32)
#         img01_pad[:, :W] = img01
#     else:
#         img01_pad = img01

#     win_w = hann1d_w(tile_w)
#     win_w = np.maximum(win_w, 1e-6)
#     win = torch.from_numpy(win_w.reshape(1, 1, 1, tile_w))

#     out_sum = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)
#     w_sum   = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)

#     xs = list(range(0, max(1, W_pad - tile_w + 1), stride))
#     if xs[-1] != W_pad - tile_w:
#         xs.append(W_pad - tile_w)

#     model.eval()
#     with torch.no_grad():
#         for x_left in xs:
#             tile = img01_pad[:, x_left:x_left + tile_w]  # (H, tile_w)
#             tile_t = torch.from_numpy(tile).unsqueeze(0)  # (1,H,Wt)
#             tile_t = tile_t.unsqueeze(0).to(device)       # (1,1,H,Wt)

#             pred = model(to_model_space(tile_t))          # [-1,1]
#             pred01 = to_img01_space(pred).clamp(0.0, 1.0)

#             pred_w = pred01 * win.to(device)
#             out_sum[:, :, :, x_left:x_left + tile_w] += pred_w
#             w_sum[:, :, :, x_left:x_left + tile_w]   += win.to(device)

#     out01 = (out_sum / (w_sum + 1e-8)).squeeze(0).squeeze(0).cpu().numpy()  # (H,W_pad)
#     return out01[:, :W]

# def mean_abs_change01(noisy01: np.ndarray, denoised01: np.ndarray) -> float:
#     return float(np.mean(np.abs(denoised01 - noisy01)))

# def save_spectrogram_png(img2d: np.ndarray, out_path: str):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.figure(figsize=(img2d.shape[1] / DPI, img2d.shape[0] / DPI), dpi=DPI)
#     plt.imshow(img2d, origin='upper', aspect='auto')
#     plt.axis('off'); plt.tight_layout(pad=0)
#     plt.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
#     plt.close()

# # =========================
# #   MAIN
# # =========================
# def main():
#     logger = setup_logging(RESULTS_DIR, log_file=LOG_FILE, level="INFO")

#     if not os.path.isfile(AUDIO_PATH):
#         logger.error(f"Audio nicht gefunden: {AUDIO_PATH}"); return
#     if not os.path.isfile(MODEL_PATH):
#         logger.error(f"Modell nicht gefunden: {MODEL_PATH}"); return

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info("=" * 70)
#     logger.info("Externes Audio denoisen (training-like rendering, keine festen dB-Grenzen)")
#     logger.info(f"Device: {device}")
#     logger.info(f"Datei: {AUDIO_PATH}")

#     # Modell
#     model = UNetCustom(in_channels=IN_CH, out_channels=IN_CH, base_channels=BASE_CH).to(device)
#     state = torch.load(MODEL_PATH, map_location=device)
#     model.load_state_dict(state)

#     # 1) Audio -> S_dB -> Bild -> Gray01 -> DC-Bin entfernen (H=512)
#     gray01, dur_s, sr_ret = audio_to_gray01_training_like(AUDIO_PATH)
#     H, W = gray01.shape
#     logger.info(f"Audio-Länge: {dur_s:.2f} s, SR: {sr_ret} Hz")
#     logger.info(f"Spektrogrammgröße (nach DC-Entfernung): H={H}, W={W}")
#     logger.info(f"STFT: n_fft={N_FFT}, hop={HOP_LENGTH}, win={WIN_LENGTH}")

#     # 2) Tiled Inferenz & Stitch
#     denoised01 = tiled_infer_and_stitch(model, gray01, device, tile_w=TILE_W, stride=STRIDE)
#     logger.info(f"Tiling: tile_w={TILE_W}, stride={STRIDE}, Tiles ~ {1 if W<=TILE_W else 1 + (W - TILE_W + STRIDE - 1)//STRIDE}")

#     # 3) Bild speichern
#     base = os.path.splitext(os.path.basename(AUDIO_PATH))[0]
#     out_png = os.path.join(RESULTS_DIR, f"extern_denoised_{base}.png")
#     save_spectrogram_png(denoised01, out_png)
#     logger.info(f"Ausgabebild: {out_png}")

#     # 4) Heuristik
#     mac = mean_abs_change01(gray01, denoised01)
#     logger.info(f"Mittlere absolute Änderung: {mac:.6f}")
#     logger.info("=" * 70)

# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()












# ################################ mit db_min/db_max
# """
# Nimmt ein externes (verrauschtes) Audio, wandelt es in ein STFT-Spektrogramm (H=513 -> DC-Zeile entfernt -> 512),
# zerlegt es in Tiles, führt U-Net-Inference mit dem gespeicherten best_model.pth durch, stitched die Tiles mit
# Hann-Überlappungsgewichtung und speichert NUR das bereinigte Spektrogramm als Bild (ohne Achsen, Default-cmap).

# Wichtige Infos (Dateiname, Länge, STFT-Parameter, Tile-Infos, einfache Heuristik) werden in
# results_denoising/denoising_output.txt geloggt.
# """

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import librosa

# from createModelUnet import UNetCustom
# from trainLogging import setup_logging

# # =========================
# #   Pfaden
# # =========================
# audios_dir = os.path.join(os.path.dirname(__file__), "audios")
# audio_name = "noisy2"
# audio_extension = ".m4a"
# AUDIO_PATH = os.path.join(audios_dir, f"{audio_name}{audio_extension}")

# RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results_denoising")
# MODEL_PATH   = os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8.pth")
# LOG_FILE     = "denoising_output.txt"

# # STFT-Parameter
# SR          = 48000
# N_FFT       = 1024             # 1024 -> 513 Bins; nach Entfernen Zeile 0 -> 512
# HOP_LENGTH  = 256
# WIN_LENGTH  = 1024
# DB_MIN, DB_MAX = -70.0, 0.0

# # Tiling für Inferenz
# TILE_W      = 256
# STRIDE      = 128
# IN_CHANNELS = 1                # 1=Graustufen
# BASE_CHANNELS = 32


# # =========================
# #   HILFSFUNKTIONEN
# # =========================
# def audio_to_stft_img01(
#     wav_path: str,
#     sr: int = SR,
#     n_fft: int = N_FFT,
#     hop_length: int = HOP_LENGTH,
#     win_length: int = WIN_LENGTH,
#     db_min: float = DB_MIN,
#     db_max: float = DB_MAX,
# ):
#     """
#     Lädt Audio -> STFT -> |Mag| -> dB -> auf [0,1] skalieren.
#     Entfernt DC-Zeile: Höhe 513 -> 512.

#     Returns:
#         img01: 2D np.ndarray, shape (512, T) auf [0,1]
#         dur_s: Dauer des Audios in Sekunden
#         sr:    Abtastrate (ggf. resampled)
#         T:     Anzahl Zeitframes (Breite)
#     """
#     y, sr_ret = librosa.load(wav_path, sr=sr, mono=True)
#     dur_s = len(y) / sr_ret

#     # STFT Magnitude
#     S = librosa.stft(
#         y,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         win_length=win_length,
#         window="hann",
#         center=True,
#         pad_mode="reflect",
#     )
#     mag = np.abs(S)  # (513, T)

#     # dB-Skalierung
#     mag_db = librosa.amplitude_to_db(mag, ref=np.max)  # typischerweise in [-200, 0]
#     # DC-Zeile entfernen
#     mag_db = mag_db[1:, :]  # (512, T)

#     # Clip & auf [0,1]
#     mag_db = np.clip(mag_db, db_min, db_max)
#     img01 = (mag_db - db_min) / (db_max - db_min)  # [-200..0] -> [0..1]

#     return img01.astype(np.float32), float(dur_s), int(sr_ret), int(img01.shape[1])


# def hann1d_w(tile_w: int) -> np.ndarray:
#     """1D-Hann-Fenster entlang der Breite (Zeitachse)."""
#     if tile_w <= 1:
#         return np.ones((1,), dtype=np.float32)
#     return np.hanning(tile_w).astype(np.float32)


# def to_model_space(tile01: torch.Tensor) -> torch.Tensor:
#     """
#     [0,1] -> [-1,1] (wie Normalize mean=0.5, std=0.5)
#     Erwartet Shape: (1, H, W)
#     """
#     return tile01 * 2.0 - 1.0


# def to_img01_space(tile_norm: torch.Tensor) -> torch.Tensor:
#     """
#     [-1,1] -> [0,1]
#     """
#     return (tile_norm + 1.0) * 0.5


# def tiled_infer_and_stitch(
#     model: nn.Module,
#     img01: np.ndarray,            # (H=512, W)
#     device: torch.device,
#     tile_w: int = TILE_W,
#     stride: int = STRIDE,
# ) -> np.ndarray:
#     """
#     Zerschneidet img01 in horizontale Tiles, führt Inferenz durch und stitcht mit Hann-Gewichtung.

#     Returns:
#         out01: np.ndarray (H, W) auf [0,1]
#     """
#     H, W = img01.shape
#     if W <= 0:
#         raise ValueError("Leeres Spektrogramm (W=0).")

#     # ggf. rechts auffüllen für den letzten Tile
#     need_pad = W < tile_w
#     W_pad = max(W, tile_w)
#     if not need_pad and ((W - tile_w) % stride != 0):
#         # Stelle sicher, dass der letzte Tile W_pad - tile_w exakt trifft
#         n_steps = (W - tile_w + stride - 1) // stride
#         last_x = n_steps * stride
#         W_pad = last_x + tile_w
#         need_pad = W_pad > W

#     if need_pad:
#         img01_pad = np.zeros((H, W_pad), dtype=np.float32)
#         img01_pad[:, :W] = img01
#     else:
#         img01_pad = img01

#     # Hann-Gewichtung entlang W
#     win_w = hann1d_w(tile_w)  # (tile_w,)
#     win_w = np.maximum(win_w, 1e-6)
#     win = torch.from_numpy(win_w.reshape(1, 1, 1, tile_w))  # (1,1,1,Wt)

#     # Akkumulatoren
#     out_sum = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)
#     w_sum   = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)

#     # Sliding-Window-Positionen
#     xs = list(range(0, max(1, W_pad - tile_w + 1), stride))
#     if xs[-1] != W_pad - tile_w:
#         xs.append(W_pad - tile_w)  # letzter Tile deckt Ende ab

#     model.eval()
#     with torch.no_grad():
#         for x_left in xs:
#             tile = img01_pad[:, x_left:x_left + tile_w]  # (H, tile_w)
#             tile_t = torch.from_numpy(tile).unsqueeze(0).to(device)  # (1, H, Wt)
#             tile_t = tile_t.unsqueeze(0)  # (B=1, C=1, H, Wt)
#             # in Modellraum
#             tile_norm = to_model_space(tile_t)

#             # Vorhersage
#             pred_norm = model(tile_norm)  # (1,1,H,Wt)
#             pred01 = to_img01_space(pred_norm).clamp(0.0, 1.0)  # (1,1,H,Wt)

#             # Hann-Gewichtung anwenden
#             pred_w = pred01 * win.to(device)  # (1,1,H,Wt)

#             out_sum[:, :, :, x_left:x_left + tile_w] += pred_w
#             w_sum[:, :, :, x_left:x_left + tile_w]   += win.to(device)

#     out01 = (out_sum / (w_sum + 1e-8)).squeeze(0).squeeze(0).cpu().numpy()  # (H, W_pad)
#     out01 = out01[:, :W]  # auf Originalbreite zurückschneiden
#     return out01


# def mean_abs_change01(noisy01: np.ndarray, denoised01: np.ndarray) -> float:
#     """Mittlere absolute Änderung auf [0,1]."""
#     return float(np.mean(np.abs(denoised01 - noisy01)))


# def save_spectrogram_png(img2d: np.ndarray, out_path: str):
#     """Speichert das Bild."""
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.figure(figsize=(img2d.shape[1] / 100, img2d.shape[0] / 100), dpi=100)
#     plt.imshow(img2d, origin='upper', aspect='auto')
#     plt.axis('off')
#     plt.tight_layout(pad=0)
#     plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
#     plt.close()


# # =========================
# #   MAIN
# # =========================
# def main():
#     logger = setup_logging(RESULTS_DIR, log_file=LOG_FILE, level="INFO")

#     if not os.path.isfile(AUDIO_PATH):
#         logger.error(f"Audio nicht gefunden: {AUDIO_PATH}")
#         return
#     if not os.path.isfile(MODEL_PATH):
#         logger.error(f"Modell nicht gefunden: {MODEL_PATH}")
#         return

#     logger.info("=" * 70)
#     logger.info("Externes Audio denoisen (STFT -> Tiles -> U-Net)")

#     # Device & Modell laden
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Device: {device}")

#     model = UNetCustom(in_channels=IN_CHANNELS, out_channels=IN_CHANNELS, base_channels=BASE_CHANNELS).to(device)
#     state = torch.load(MODEL_PATH, map_location=device)
#     model.load_state_dict(state)

#     # 1) Audio -> STFT-Bild [0,1], H=512
#     img01, dur_s, sr_ret, T = audio_to_stft_img01(
#         AUDIO_PATH, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
#     )  # (512, T)
#     logger.info(f"Datei: {AUDIO_PATH}")
#     logger.info(f"Audio-Länge: {dur_s:.2f} s, Sample Rate: {sr_ret} Hz")
#     logger.info(f"STFT: n_fft={N_FFT}, hop={HOP_LENGTH}, win={WIN_LENGTH}")
#     logger.info(f"Spektrogrammgröße: H=512, W={T}")

#     # 2) Tiled Inferenz & Stitching
#     denoised01 = tiled_infer_and_stitch(
#         model, img01, device, tile_w=TILE_W, stride=STRIDE
#     )  # (512, T)
#     logger.info(f"Tiling: tile_w={TILE_W}, stride={STRIDE}")

#     # 3) Bild speichern
#     base = os.path.splitext(os.path.basename(AUDIO_PATH))[0]
#     out_png = os.path.join(RESULTS_DIR, f"extern_denoised_{base}.png")
#     save_spectrogram_png(denoised01, out_png)
#     logger.info(f"Ausgabebild: {out_png}")

#     # 4) Heuristik loggen
#     mac = mean_abs_change01(img01, denoised01)
#     logger.info(f"Mittlere absolute Änderung: {mac:.6f}")

#     logger.info("=" * 70)


# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()
