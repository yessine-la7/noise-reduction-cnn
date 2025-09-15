import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

from createModelUnet import UNetCustom
from trainLogging import setup_logging

# --- Pfade ---
INPUT_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audios")
INPUT_AUDIO_NAME = "noisy3.m4a"
INPUT_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, INPUT_AUDIO_NAME)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_denoising")
MODEL_FILENAME = "best_model_baseCh_32_batch_8_patience_20.pth"
MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_FILENAME)

OUTPUT_AUDIO_NAME = "denoised3.wav"
OUTPUT_AUDIO_PATH = os.path.join(RESULTS_DIR, OUTPUT_AUDIO_NAME)

NOISY_SPEC_PNG   = os.path.join(RESULTS_DIR, f"{os.path.splitext(INPUT_AUDIO_NAME)[0]}_spec.png")
DENOISED_SPEC_PNG= os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_spec.png")
WAVEFORMS_PNG    = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_waveforms.png")

# --- STFT ---
SR_TARGET    = None
N_FFT        = 1024
HOP          = 256
WIN_LENGTH   = 1024
WINDOW       = "hann"

# Datenvorbereitung: "numeric" (vollflächig) oder "image_tiled" (Tiles)
DATA_PREP_METHOD = "numeric"   # "numeric" | "image_tiled"

# Tile-Inferenz nur im image_tiled-Pfad
DPI      = 100
TILE_W   = 256
STRIDE   = 128

# Down/Up-Faktor nur für numeric-Padding
DOWNSAMPLE_FACTOR = 32

BASE_CHANNELS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ausgabepegel
NORMALIZE_METHOD = "peak"        # "peak" | "hard" | "none"
PEAK_TARGET      = 0.2           # Zielpeak für Peak-Norm bei "peak"
HARD_GAIN_FACTOR = 0.5           # Verstärkungsfaktor bei "hard"


# ------------------------------ Utilities (numeric) ------------------------------
def pad_to_multiple_reflect(x: np.ndarray, mult_h: int, mult_w: int) -> Tuple[np.ndarray, Tuple[int,int]]:
    H, W = x.shape
    target_H = int(np.ceil(H / mult_h) * mult_h)
    target_W = int(np.ceil(W / mult_w) * mult_w)
    pad_h = target_H - H
    pad_w = target_W - W
    if pad_h == 0 and pad_w == 0:
        return x, (H, W)

    def _reflect_pad_1d(vec, pad):
        if pad <= 0: return vec
        if len(vec) < 2: return np.pad(vec, (0, pad), mode="edge")
        ref = vec[1:-1][::-1]
        reps = int(np.ceil(pad / len(ref))) if len(ref) > 0 else pad
        pad_block = np.tile(ref, reps)[:pad] if len(ref) > 0 else np.repeat(vec[-1:], pad)
        return np.concatenate([vec, pad_block])

    right = (np.stack([_reflect_pad_1d(x[i, :], pad_w) for i in range(H)], axis=0) if pad_w > 0 else x)
    if pad_h > 0:
        if H < 2:
            pad_block = np.repeat(right[-1:, :], pad_h, axis=0)
        else:
            top = right[1:-1, :][::-1, :]
            reps = int(np.ceil(pad_h / top.shape[0])) if top.shape[0] > 0 else pad_h
            pad_block = np.tile(top, (reps, 1))[:pad_h, :] if top.shape[0] > 0 else np.repeat(right[-1:, :], pad_h, axis=0)
        x_pad = np.concatenate([right, pad_block], axis=0)
    else:
        x_pad = right
    return x_pad.astype(np.float32), (H, W)

def crop_to_original(x_pad: np.ndarray, orig_hw: Tuple[int,int]) -> np.ndarray:
    H, W = orig_hw
    return x_pad[:H, :W]

def to_model_space(img01: torch.Tensor) -> torch.Tensor:
    return img01 * 2.0 - 1.0

def to_img01_space(img_norm: torch.Tensor) -> torch.Tensor:
    return (img_norm + 1.0) * 0.5


# ------------------------------ Utilities (image tiled) ------------------------------
@torch.no_grad()
def tiled_image_inference(model: torch.nn.Module, img01: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    H, W = img01.shape
    if W < TILE_W:
        img01 = np.pad(img01, ((0, 0), (0, TILE_W - W)), mode="constant")
    H, W_pad = img01.shape

    win = torch.from_numpy(np.hanning(TILE_W).astype(np.float32)).to(device).view(1, 1, 1, -1)
    out_sum = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)
    w_sum  = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)

    for x0 in range(0, W_pad - TILE_W + 1, STRIDE):
        tile = img01[:, x0:x0+TILE_W]
        tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(to_model_space(tile_t))
        pred01 = to_img01_space(pred).clamp(0.0, 1.0)
        out_sum[..., x0:x0+TILE_W] += pred01 * win
        w_sum[...,  x0:x0+TILE_W] += win

    if (W_pad - TILE_W) % STRIDE != 0:
        x0 = W_pad - TILE_W
        tile = img01[:, x0:x0+TILE_W]
        tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(to_model_space(tile_t))
        pred01 = to_img01_space(pred).clamp(0.0, 1.0)
        out_sum[..., x0:x0+TILE_W] += pred01 * win
        w_sum[...,  x0:x0+TILE_W] += win

    out01 = (out_sum / (w_sum + 1e-8)).squeeze().detach().cpu().numpy()
    return out01[:, :W]


# ------------------------------ Pegel-Optionen ------------------------------
def apply_output_level(y: np.ndarray, method: str, peak_target: float, hard_gain: float) -> np.ndarray:
    if method == "none":
        return y.astype(np.float32)
    if method == "peak":
        peak = float(np.max(np.abs(y)) + 1e-12)
        return (peak_target / peak) * y
    if method == "hard":
        y2 = y * hard_gain
        peak2 = float(np.max(np.abs(y2)) + 1e-12)
        if peak2 > 1.0:
            y2 = (peak_target / peak2) * y2
        return y2.astype(np.float32)
    return y.astype(np.float32)


# ------------------------------ Utilities (plots) ------------------------------
def plot_spectrogram_db(S_db: np.ndarray, sr: int, hop_length: int, out_path: str, title: str = ""):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear")
    plt.colorbar(label="Amplitude [dB]")
    if title: plt.title(title)
    plt.xlabel("Zeit [s]"); plt.ylabel("Frequenz [Hz]")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()

def save_waveform(y_noisy, y_denoised_final, sr, out_path):
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(y_noisy, sr=sr, alpha=0.6, label='Noisy', color='blue')
    librosa.display.waveshow(y_denoised_final, sr=sr, alpha=0.8, label='Denoised', color='red')
    plt.title('Wellenform'); plt.xlabel('Zeit (s)'); plt.ylabel('Amplitude')
    plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


# ------------------------------ Metrics (SNR, no-ref, VAD-basiert) ------------------------------
def estimate_snr_db_vad(y: np.ndarray, sr: int, frame_length: int = WIN_LENGTH, hop_length: int = HOP, top_db: float = 40.0) -> float:
    """
    Schätzt SNR [dB] ohne Referenz:
      - Sprachregionen via librosa.effects.split (nicht-still)
      - 'Noise' = (vermutete) stille Frames (oder unteres 10%-Quantil der RMS, falls keine Stille)
      - SNR = 10*log10( (E_speech - E_noise) / E_noise )
    """
    # Frame-RMS
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True, pad_mode="reflect")[0]
    n_frames = len(rms)
    frame_centers = np.arange(n_frames) * hop_length + frame_length // 2

    # Nicht-stille Intervalle (samples)
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    speech_mask = np.zeros(n_frames, dtype=bool)
    for s, e in intervals:
        speech_mask |= (frame_centers >= s) & (frame_centers < e)

    # Noise-Frames: komplementäre Frames oder fallback auf unteres 10%-Quantil
    if np.any(~speech_mask):
        noise_rms = rms[~speech_mask]
    else:
        k = max(1, int(0.1 * n_frames))
        noise_rms = np.sort(rms)[:k]

    speech_rms = rms[speech_mask] if np.any(speech_mask) else rms

    noise_power  = float(np.mean(noise_rms ** 2) + 1e-12)
    speech_power = float(max(np.mean(speech_rms ** 2) - noise_power, 1e-12))
    return 10.0 * np.log10(speech_power / noise_power)


# --------------------------------- Hauptpipeline ---------------------------------
def denoise_audio(
    noisy_audio_path: str,
    output_audio_path: str,
    model_path: str,
    noisy_spec_png: str,
    denoised_spec_png: str,
    waveforms_png: str,
    n_fft: int = N_FFT,
    hop_length: int = HOP,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
    sr_target: Optional[int] = SR_TARGET,
    device: str = DEVICE,
):
    # Logger
    logger = setup_logging(RESULTS_DIR, log_file="denoise_output.txt", level="INFO")

    Path(os.path.dirname(output_audio_path)).mkdir(parents=True, exist_ok=True)

    # 1) Audio laden
    y_noisy, sr = librosa.load(noisy_audio_path, sr=sr_target)
    duration_s = len(y_noisy) / sr

    logger.info(f"Audio: {os.path.basename(noisy_audio_path)}")
    logger.info(f"Sample Rate: {sr} Hz | Dauer: {duration_s:.3f} s | Samples: {len(y_noisy)}")

    # 2) STFT
    S_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    S_noisy_mag   = np.abs(S_noisy)
    S_noisy_phase = np.angle(S_noisy)
    S_db          = librosa.amplitude_to_db(S_noisy_mag, ref=np.max)

    n_bins, n_frames = S_db.shape
    time_res = hop_length / sr
    logger.info(f"STFT: n_fft={n_fft}, hop={hop_length}, win={win_length}, window={window}")
    logger.info(f"S_db shape: {S_db.shape} (Freq-Bins x Frames) | Zeitauflösung: {time_res:.6f} s/Frame")

    # 3) per-Sample Norm in [0,1]
    db_min = float(S_db.min())
    db_max = float(S_db.max())
    S_db01 = ((S_db - db_min) / (db_max - db_min + 1e-12)).astype(np.float32)

    # 4) Modell laden
    model = UNetCustom(in_channels=1, out_channels=1, base_channels=BASE_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Model: UNetCustom(base_channels={BASE_CHANNELS}) | Device: {device}")
    logger.info(f"Gewichte: {os.path.basename(model_path)}")

    # 5) Inferenz
    if DATA_PREP_METHOD == "numeric":
        # numeric: Padding-Infos loggen
        S_db01_pad, orig_hw = pad_to_multiple_reflect(S_db01, DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR)
        pad_h = S_db01_pad.shape[0] - orig_hw[0]
        pad_w = S_db01_pad.shape[1] - orig_hw[1]
        logger.info(f"[numeric] DOWNSAMPLE_FACTOR={DOWNSAMPLE_FACTOR} | Original HxW={orig_hw} | Padded HxW={S_db01_pad.shape} | pad_h={pad_h}, pad_w={pad_w}")

        with torch.no_grad():
            x = torch.from_numpy(S_db01_pad[None, None, :, :]).to(device)
            x = to_model_space(x)
            y_hat = model(x)
            y_hat01_pad = to_img01_space(y_hat).clamp(0, 1)
            S_denoised01_pad = y_hat01_pad[0, 0].cpu().numpy()
        S_denoised01 = crop_to_original(S_denoised01_pad, orig_hw)
        S_denoised_db_full = S_denoised01 * (db_max - db_min) + db_min

    else:  # "image_tiled"
        # image_tiled: Tiles-Infos loggen
        S_db01_wo_dc = S_db01[1:, :].astype(np.float32)   # (512, T)
        W = S_db01_wo_dc.shape[1]
        W_pad = max(W, TILE_W)
        n_main = ((W_pad - TILE_W) // STRIDE) + 1
        extra  = 1 if ((W_pad - TILE_W) % STRIDE) != 0 else 0
        n_tiles = n_main + extra
        logger.info(f"[image_tiled] TILE_W={TILE_W}, STRIDE={STRIDE}, H=512, W={W} -> Tiles={n_tiles} (inkl. Rest={extra})")

        S_denoised01_wo_dc = tiled_image_inference(model, S_db01_wo_dc, device)
        S_denoised_db_wo_dc = S_denoised01_wo_dc * (db_max - db_min) + db_min
        S_denoised_db_full = np.vstack([S_db[0:1, :], S_denoised_db_wo_dc])

    # 6) zurück zu linearer Magnitude — ref = max(|S_noisy|)
    ref_val = float(np.max(S_noisy_mag))
    S_denoised_mag = librosa.db_to_amplitude(S_denoised_db_full, ref=ref_val)

    # 7) Rekombiniere mit Original-Phase & iSTFT
    S_denoised_complex = S_denoised_mag * np.exp(1j * S_noisy_phase)
    y_denoised = librosa.istft(
        S_denoised_complex, hop_length=hop_length, win_length=win_length, window=window, length=len(y_noisy)
    )

    # 8) Pegel
    y_denoised = apply_output_level(y_denoised, method=NORMALIZE_METHOD, peak_target=PEAK_TARGET, hard_gain=HARD_GAIN_FACTOR)
    logger.info(f"Output-Pegel: method={NORMALIZE_METHOD}, peak_target={PEAK_TARGET}, hard_gain={HARD_GAIN_FACTOR}")

    # 9) SNR (no-ref) schätzen & loggen
    snr_noisy    = estimate_snr_db_vad(y_noisy,   sr, frame_length=WIN_LENGTH, hop_length=HOP, top_db=40.0)
    snr_denoised = estimate_snr_db_vad(y_denoised, sr, frame_length=WIN_LENGTH, hop_length=HOP, top_db=40.0)
    logger.info(f"SNR noisy (VAD, no-ref):    {snr_noisy:.2f} dB")
    logger.info(f"SNR denoised (VAD, no-ref): {snr_denoised:.2f} dB")
    logger.info(f"SNR-Differenz (denoised - noisy):    {snr_denoised - snr_noisy:+.2f} dB")

    # 10) Speichern + Plots
    sf.write(output_audio_path, y_denoised.astype(np.float32), sr)
    plot_spectrogram_db(S_db,               sr, hop_length, noisy_spec_png,    title="Noisy Spektrogramm")
    plot_spectrogram_db(S_denoised_db_full, sr, hop_length, denoised_spec_png, title="Denoised Spektrogramm")
    save_waveform(y_noisy, y_denoised, sr, WAVEFORMS_PNG)

    logger.info(f"[OK] Denoised WAV:     {output_audio_path}")
    logger.info(f"[OK] Noisy Spec PNG:   {noisy_spec_png}")
    logger.info(f"[OK] Denoised Spec PNG:{denoised_spec_png}")
    logger.info(f"[OK] Waveforms PNG:    {WAVEFORMS_PNG}")


# --------------------------------- Ausführen ---------------------------------
if __name__ == "__main__":
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    denoise_audio(
        noisy_audio_path=INPUT_AUDIO_PATH,
        output_audio_path=OUTPUT_AUDIO_PATH,
        model_path=MODEL_PATH,
        noisy_spec_png=NOISY_SPEC_PNG,
        denoised_spec_png=DENOISED_SPEC_PNG,
        waveforms_png=WAVEFORMS_PNG,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN_LENGTH,
        window=WINDOW,
        sr_target=SR_TARGET,
        device=DEVICE,
    )















# ###################################### versuch 12
# import os
# from pathlib import Path
# from typing import Optional, Tuple

# import numpy as np
# import torch
# import librosa
# import librosa.display
# import soundfile as sf
# import matplotlib.pyplot as plt

# from createModelUnet import UNetCustom

# # --- Pfade ---
# INPUT_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audios")
# INPUT_AUDIO_NAME = "p232_005_noisy.wav"
# INPUT_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, INPUT_AUDIO_NAME)

# RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_denoising")
# MODEL_FILENAME = "best_model_baseCh_32_batch_8_patience_20.pth"
# MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_FILENAME)

# OUTPUT_AUDIO_NAME = "p232_005_denoised.wav"
# OUTPUT_AUDIO_PATH = os.path.join(RESULTS_DIR, OUTPUT_AUDIO_NAME)

# NOISY_SPEC_PNG = os.path.join(RESULTS_DIR, f"{os.path.splitext(INPUT_AUDIO_NAME)[0]}_spec.png")
# DENOISED_SPEC_PNG = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_spec.png")
# WAVEFORMS_PNG = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_waveforms.png")

# # --- STFT ---
# SR_TARGET    = None
# N_FFT        = 1024
# HOP          = 256
# WIN_LENGTH   = 1024
# WINDOW       = "hann"

# # Datenvorbereitung: "numeric" (vollflächig) oder "image_tiled" (Tiles)
# DATA_PREP_METHOD = "numeric"   # "numeric" | "image_tiled"

# # Tile-Inferenz nur im image_tiled-Pfad
# DPI      = 100
# TILE_W   = 256
# STRIDE   = 128

# # Down/Up-Faktor nur für numeric-Padding
# DOWNSAMPLE_FACTOR = 32

# BASE_CHANNELS = 32
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Ausgabepegel
# NORMALIZE_METHOD = "peak"        # "peak" | "hard" | "none"
# PEAK_TARGET      = 0.2           # Zielpeak für Peak-Norm bei "peak"
# HARD_GAIN_FACTOR = 0.5           # Verstärkungsfaktor bei "hard"


# # ------------------------------ Utilities (numeric) ------------------------------
# def pad_to_multiple_reflect(x: np.ndarray, mult_h: int, mult_w: int) -> Tuple[np.ndarray, Tuple[int,int]]:
#     H, W = x.shape                                      # aktuelle Höhe/Breite des Spektrogramms
#     target_H = int(np.ceil(H / mult_h) * mult_h)        # Zielhöhe
#     target_W = int(np.ceil(W / mult_w) * mult_w)        # Zielbreite
#     pad_h = target_H - H
#     pad_w = target_W - W
#     if pad_h == 0 and pad_w == 0:
#         return x, (H, W)                                # schon passend → nichts tun

#     def _reflect_pad_1d(vec, pad):
#         if pad <= 0: return vec
#         if len(vec) < 2: return np.pad(vec, (0, pad), mode="edge")
#         ref = vec[1:-1][::-1]
#         reps = int(np.ceil(pad / len(ref))) if len(ref) > 0 else pad
#         pad_block = np.tile(ref, reps)[:pad] if len(ref) > 0 else np.repeat(vec[-1:], pad)
#         return np.concatenate([vec, pad_block])         # Original + reflektierte Kacheln

#     # Spalten auffüllen
#     right = (np.stack([_reflect_pad_1d(x[i, :], pad_w) for i in range(H)], axis=0) if pad_w > 0 else x)
#     if pad_h > 0:
#         if H < 2:
#             pad_block = np.repeat(right[-1:, :], pad_h, axis=0)
#         else:
#             top = right[1:-1, :][::-1, :]
#             reps = int(np.ceil(pad_h / top.shape[0])) if top.shape[0] > 0 else pad_h
#             pad_block = np.tile(top, (reps, 1))[:pad_h, :] if top.shape[0] > 0 else np.repeat(right[-1:, :], pad_h, axis=0)
#         x_pad = np.concatenate([right, pad_block], axis=0)
#     else:
#         x_pad = right
#     return x_pad.astype(np.float32), (H, W)             # gepaddetes Bild + Originalmaß

# def crop_to_original(x_pad: np.ndarray, orig_hw: Tuple[int,int]) -> np.ndarray:
#     H, W = orig_hw
#     return x_pad[:H, :W]                                # nach Inferenz zurück auf Originalmaß

# def to_model_space(img01: torch.Tensor) -> torch.Tensor:
#     return img01 * 2.0 - 1.0                            # [0,1] → [-1,1]

# def to_img01_space(img_norm: torch.Tensor) -> torch.Tensor:
#     return (img_norm + 1.0) * 0.5                       # [-1,1] → [0,1]


# # ------------------------------ Utilities (image tiled) ------------------------------
# @torch.no_grad()
# def tiled_image_inference(model: torch.nn.Module, img01: np.ndarray, device: str) -> np.ndarray:
#     model.eval()
#     H, W = img01.shape                                  # H=Frequenz-Bins, W=Zeit-Frames
#     # Falls das Bild schmaler als eine Kachel ist
#     if W < TILE_W:
#         img01 = np.pad(img01, ((0, 0), (0, TILE_W - W)), mode="constant")
#     H, W_pad = img01.shape

#     win = torch.from_numpy(np.hanning(TILE_W).astype(np.float32)).to(device).view(1, 1, 1, -1)
#     out_sum = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)
#     w_sum  = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)

#     # Über alle Kachel-Startpositionen mit Schrittweite STRIDE iterieren
#     for x0 in range(0, W_pad - TILE_W + 1, STRIDE):
#         tile = img01[:, x0:x0+TILE_W]                   # (H, TILE_W) Ausschnitt
#         tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
#         pred = model(to_model_space(tile_t))            # [-1,1]
#         pred01 = to_img01_space(pred).clamp(0.0, 1.0)   # [0,1]
#         out_sum[..., x0:x0+TILE_W] += pred01 * win      # gewichtete Summe der Vorhersage
#         w_sum[...,  x0:x0+TILE_W] += win                # Summe der Gewichte

#     # Falls am Ende eine Restzone bleibt
#     if (W_pad - TILE_W) % STRIDE != 0:
#         x0 = W_pad - TILE_W
#         tile = img01[:, x0:x0+TILE_W]
#         tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
#         pred = model(to_model_space(tile_t))
#         pred01 = to_img01_space(pred).clamp(0.0, 1.0)
#         out_sum[..., x0:x0+TILE_W] += pred01 * win
#         w_sum[...,  x0:x0+TILE_W] += win

#     # gewichtetes Mittel bilden
#     out01 = (out_sum / (w_sum + 1e-8)).squeeze().detach().cpu().numpy()
#     return out01[:, :W]                                 # auf ursprüngliche Breite W zurückschneiden


# # ------------------------------ Pegel-Optionen ------------------------------
# def apply_output_level(y: np.ndarray, method: str, peak_target: float, hard_gain: float) -> np.ndarray:
#     if method == "none":
#         return y.astype(np.float32)

#     if method == "peak":
#         peak = float(np.max(np.abs(y)) + 1e-12)
#         return (peak_target / peak) * y

#     if method == "hard":
#         y2 = y * hard_gain
#         peak2 = float(np.max(np.abs(y2)) + 1e-12)
#         if peak2 > 1.0:
#             y2 = (peak_target / peak2) * y2
#         return y2.astype(np.float32)

#     return y.astype(np.float32)


# # ------------------------------ Utilities (plots) ------------------------------
# def plot_spectrogram_db(S_db: np.ndarray, sr: int, hop_length: int, out_path: str, title: str = ""):
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear")
#     plt.colorbar(label="Amplitude [dB]")
#     if title: plt.title(title)
#     plt.xlabel("Zeit [s]"); plt.ylabel("Frequenz [Hz]")
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()

# def save_waveform(y_noisy, y_denoised_final, sr, out_path):
#     plt.figure(figsize=(15, 5))
#     librosa.display.waveshow(y_noisy, sr=sr, alpha=0.6, label='Noisy', color='blue')
#     librosa.display.waveshow(y_denoised_final, sr=sr, alpha=0.8, label='Denoised', color='red')
#     plt.title('Wellenform'); plt.xlabel('Zeit (s)'); plt.ylabel('Amplitude')
#     plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


# # --------------------------------- Hauptpipeline ---------------------------------
# def denoise_audio(
#     noisy_audio_path: str,
#     output_audio_path: str,
#     model_path: str,
#     noisy_spec_png: str,
#     denoised_spec_png: str,
#     waveforms_png: str,
#     n_fft: int = N_FFT,
#     hop_length: int = HOP,
#     win_length: int = WIN_LENGTH,
#     window: str = WINDOW,
#     sr_target: Optional[int] = SR_TARGET,
#     device: str = DEVICE,
# ):
#     Path(os.path.dirname(output_audio_path)).mkdir(parents=True, exist_ok=True)

#     # 1) Audio laden
#     y_noisy, sr = librosa.load(noisy_audio_path, sr=sr_target)

#     # 2) STFT
#     S_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
#     S_noisy_mag = np.abs(S_noisy)
#     S_noisy_phase = np.angle(S_noisy)
#     S_db = librosa.amplitude_to_db(S_noisy_mag, ref=np.max)

#     # 3) per-Sample Norm in [0,1]
#     db_min = float(S_db.min())
#     db_max = float(S_db.max())
#     S_db01 = ((S_db - db_min) / (db_max - db_min + 1e-12)).astype(np.float32)

#     # 4) Modell laden
#     model = UNetCustom(in_channels=1, out_channels=1, base_channels=BASE_CHANNELS).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     # 5) Padding und cropping
#     # ===========================
#     # Pfad A: NUMERIC
#     # ===========================
#     if DATA_PREP_METHOD == "numeric":
#         # 1) auf Vielfache des Downsample-Faktors padden
#         S_db01_pad, orig_hw = pad_to_multiple_reflect(S_db01, DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR)
#         with torch.no_grad():
#             # 2) zu Tensor (N=1, C=1, H, W)
#             x = torch.from_numpy(S_db01_pad[None, None, :, :]).to(device)
#             # 3) in Modell-Skala [-1,1] normalisieren
#             x = to_model_space(x)
#             # 4) Inferenz durch das UNet
#             y_hat = model(x)
#             # 5) zurück in [0,1] + harte Begrenzung
#             y_hat01_pad = to_img01_space(y_hat).clamp(0, 1)
#             # 6) wieder zu NumPy, Batch/Channel entfernen
#             S_denoised01_pad = y_hat01_pad[0, 0].cpu().numpy()
#         # 7) gepaddete Ränder wegschneiden → exakt Originalgröße
#         S_denoised01 = crop_to_original(S_denoised01_pad, orig_hw)
#         # 8) dieselbe dB-Skala zurück anwenden
#         S_denoised_db_full = S_denoised01 * (db_max - db_min) + db_min

#     # ===========================
#     # Pfad B: IMAGE_TILED
#     # ===========================
#     else:  # "image_tiled"
#         # 1) DC-Bin entfernen, damit H=512 für das Modell
#         S_db01_wo_dc = S_db01[1:, :].astype(np.float32)   # (512, T), Werte in [0,1]
#         # 2) Tile-Inferenz mit Hann-Blending
#         S_denoised01_wo_dc = tiled_image_inference(model, S_db01_wo_dc, device)
#         # 3) Rückskalierung in dB (ohne DC):
#         S_denoised_db_wo_dc = S_denoised01_wo_dc * (db_max - db_min) + db_min
#         # 4) DC-Zeile wieder anfügen (aus dem NOISY-Spektrogramm in dB):
#         S_denoised_db_full = np.vstack([S_db[0:1, :], S_denoised_db_wo_dc])

#     # 6) zurück zu linearer Magnitude — ref = max(|S_noisy|)
#     ref_val = float(np.max(S_noisy_mag))
#     S_denoised_mag = librosa.db_to_amplitude(S_denoised_db_full, ref=ref_val)

#     # 7) Rekombiniere mit Original-Phase & iSTFT
#     S_denoised_complex = S_denoised_mag * np.exp(1j * S_noisy_phase)
#     y_denoised = librosa.istft(
#         S_denoised_complex, hop_length=hop_length, win_length=win_length, window=window, length=len(y_noisy)
#     )

#     # 8) Pegel
#     y_denoised = apply_output_level(y_denoised, method=NORMALIZE_METHOD, peak_target=PEAK_TARGET, hard_gain=HARD_GAIN_FACTOR)

#     # 9) Speichern + Plots
#     sf.write(output_audio_path, y_denoised.astype(np.float32), sr)
#     plot_spectrogram_db(S_db,                 sr, hop_length, noisy_spec_png,    title="Noisy Spektrogramm")
#     plot_spectrogram_db(S_denoised_db_full,   sr, hop_length, denoised_spec_png, title="Denoised Spektrogramm")
#     save_waveform(y_noisy, y_denoised, sr, WAVEFORMS_PNG)

#     print(f"[OK] Denoised WAV:       {output_audio_path}")
#     print(f"[OK] Noisy Spec:         {noisy_spec_png}")
#     print(f"[OK] Denoised Spec:      {denoised_spec_png}")
#     print(f"[OK] Waveforms:          {WAVEFORMS_PNG}")


# # --------------------------------- Ausführen ---------------------------------
# if __name__ == "__main__":
#     Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
#     denoise_audio(
#         noisy_audio_path=INPUT_AUDIO_PATH,
#         output_audio_path=OUTPUT_AUDIO_PATH,
#         model_path=MODEL_PATH,
#         noisy_spec_png=NOISY_SPEC_PNG,
#         denoised_spec_png=DENOISED_SPEC_PNG,
#         waveforms_png=WAVEFORMS_PNG,
#         n_fft=N_FFT,
#         hop_length=HOP,
#         win_length=WIN_LENGTH,
#         window=WINDOW,
#         sr_target=SR_TARGET,
#         device=DEVICE,
#     )




















# ################################## versuch 1
# import os
# from pathlib import Path
# from typing import Optional, Tuple

# import numpy as np
# import torch
# import librosa
# import librosa.display
# import soundfile as sf
# import matplotlib.pyplot as plt

# from createModelUnet import UNetCustom

# # --- Pfade ---
# # Eingabedatei im Unterordner "audios"
# INPUT_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audios")
# INPUT_AUDIO_NAME = "p232_005_noisy.wav"
# INPUT_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, INPUT_AUDIO_NAME)

# # Ausgabedateien im Unterordner "results"
# RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_denoising")
# MODEL_FILENAME = "best_model_baseCh_32_batch_8_patience_20.pth"
# MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_FILENAME)

# OUTPUT_AUDIO_NAME = "p232_005_denoised.wav"
# OUTPUT_AUDIO_PATH = os.path.join(RESULTS_DIR, OUTPUT_AUDIO_NAME)

# # Abgeleitete Plot-Pfade
# NOISY_SPEC_PNG = os.path.join(
#     RESULTS_DIR, f"{os.path.splitext(INPUT_AUDIO_NAME)[0]}_spec.png"
# )
# DENOISED_SPEC_PNG = os.path.join(
#     RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_spec.png"
# )
# WAVEFORMS_PNG = os.path.join(
#     RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_waveforms.png"
# )

# # --- STFT/Modell/Optionen ---
# SR_TARGET    = None
# N_FFT        = 1024
# HOP          = 256
# WIN_LENGTH   = 1024
# WINDOW       = "hann"

# # Padding auf Vielfache (typisch 32 für 5x Down/Up bei UNet)
# DOWNSAMPLE_FACTOR = 32

# # Muss wie beim Training sein
# BASE_CHANNELS = 32

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # --- Ausgabepegel-Flags ---
# # "peak" = Peak-Norm auf Ziel-Peak, "hard" = fester Gain, "none" = unverändert
# NORMALIZE_METHOD = "hard"        # "peak" | "hard" | "none"
# PEAK_TARGET      = 0.2           # Zielpeak für Peak-Norm
# HARD_GAIN_FACTOR = 0.5           # fester Verstärkungsfaktor bei "hard"



# # ------------------------------ Utility-Funktionen ------------------------------
# def pad_to_multiple_reflect(x: np.ndarray, mult_h: int, mult_w: int) -> Tuple[np.ndarray, Tuple[int,int]]:
#     """
#     Reflektiertes Padding von (H,W) auf Vielfache (mult_h, mult_w).
#     Rückgabe:
#       x_pad, (orig_H, orig_W)
#     """
#     H, W = x.shape
#     target_H = int(np.ceil(H / mult_h) * mult_h)
#     target_W = int(np.ceil(W / mult_w) * mult_w)

#     pad_h = target_H - H
#     pad_w = target_W - W
#     if pad_h == 0 and pad_w == 0:
#         return x, (H, W)

#     def _reflect_pad_1d(vec, pad):
#         if pad <= 0:
#             return vec
#         if len(vec) < 2:
#             return np.pad(vec, (0, pad), mode="edge")
#         ref = vec[1:-1][::-1]
#         reps = int(np.ceil(pad / len(ref))) if len(ref) > 0 else pad
#         pad_block = np.tile(ref, reps)[:pad] if len(ref) > 0 else np.repeat(vec[-1:], pad)
#         return np.concatenate([vec, pad_block])

#     # erst Spalten (W) auffüllen
#     if pad_w > 0:
#         right = np.stack([_reflect_pad_1d(x[i, :], pad_w) for i in range(H)], axis=0)  # (H, W+pad_w)
#     else:
#         right = x
#     # dann Zeilen (H) auffüllen
#     if pad_h > 0:
#         if H < 2:
#             pad_block = np.repeat(right[-1:, :], pad_h, axis=0)
#         else:
#             top = right[1:-1, :][::-1, :]
#             reps = int(np.ceil(pad_h / top.shape[0])) if top.shape[0] > 0 else pad_h
#             pad_block = np.tile(top, (reps, 1))[:pad_h, :] if top.shape[0] > 0 else np.repeat(right[-1:, :], pad_h, axis=0)
#         x_pad = np.concatenate([right, pad_block], axis=0)
#     else:
#         x_pad = right

#     return x_pad.astype(np.float32), (H, W)


# def crop_to_original(x_pad: np.ndarray, orig_hw: Tuple[int,int]) -> np.ndarray:
#     H, W = orig_hw
#     return x_pad[:H, :W]


# def to_model_space(img01: torch.Tensor) -> torch.Tensor:
#     """ [0,1] -> [-1,1] """
#     return img01 * 2.0 - 1.0


# def to_img01_space(img_norm: torch.Tensor) -> torch.Tensor:
#     """ [-1,1] -> [0,1] """
#     return (img_norm + 1.0) * 0.5


# def plot_spectrogram_db(S_db: np.ndarray, sr: int, hop_length: int, out_path: str, title: str = ""):
#     """
#     Zeichnet ein dB-Spektrogramm mit x-Achse in Sekunden und y-Achse in Hz.
#     """
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(
#         S_db,
#         sr=sr,
#         hop_length=hop_length,
#         x_axis="time",
#         y_axis="linear"
#     )
#     plt.colorbar(label="Amplitude [dB]")
#     if title:
#         plt.title(title)
#     plt.xlabel("Zeit [s]")
#     plt.ylabel("Frequenz [Hz]")
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=120)
#     plt.close()


# def save_waveform(y_noisy, y_denoised_final, sr, out_path):
#     """Speichert einen Vergleich der Wellenformen."""
#     plt.figure(figsize=(15, 5))
#     librosa.display.waveshow(y_noisy, sr=sr, alpha=0.6, label='Noisy', color='blue')
#     librosa.display.waveshow(y_denoised_final, sr=sr, alpha=0.8, label='Denoised', color='red')
#     plt.title('Wellenform')
#     plt.xlabel('Zeit (s)')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     # plt.grid()
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=120)
#     plt.close()
#     print(f"Wellenform-Vergleich gespeichert: {out_path}")


# # ------------------------------ Pegel-Optionen ------------------------------
# def apply_output_level(
#     y: np.ndarray,
#     method: str,
#     peak_target: float,
#     hard_gain: float,
# ) -> np.ndarray:
#     """
#       - method="peak":   Peak-Norm auf 'peak_target'
#       - method="hard":   fester Gain
#       - method="none":   unverändert
#     """

#     if method == "none":
#         return y.astype(np.float32)

#     if method == "peak":
#         peak = float(np.max(np.abs(y)))
#         return (peak_target / peak) * y

#     if method == "hard":
#         amplified_y = y * hard_gain
#         peak2 = float(np.max(np.abs(amplified_y)))
#         if  peak2 > 1.0:
#             amplified_y = (peak_target / peak2) * amplified_y
#         return amplified_y

#     return y.astype(np.float32)


# # --------------------------------- Hauptpipeline ---------------------------------
# def denoise_audio(
#     noisy_audio_path: str,
#     output_audio_path: str,
#     model_path: str,
#     noisy_spec_png: str,
#     denoised_spec_png: str,
#     waveforms_png: str,
#     n_fft: int = N_FFT,
#     hop_length: int = HOP,
#     win_length: int = WIN_LENGTH,
#     window: str = WINDOW,
#     sr_target: Optional[int] = SR_TARGET,
#     device: str = DEVICE,
# ):
#     # 0) Ordner anlegen
#     Path(os.path.dirname(output_audio_path)).mkdir(parents=True, exist_ok=True)

#     # 1) Audio laden
#     y_noisy, sr = librosa.load(noisy_audio_path, sr=sr_target)

#     # 2) STFT
#     S_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
#     S_noisy_mag = np.abs(S_noisy)                 # (513, T)
#     S_noisy_phase = np.angle(S_noisy)

#     # 3) dB (ref am echten Maximum der Input-Magnitude)
#     S_db = librosa.amplitude_to_db(S_noisy_mag, ref=np.max)  # max -> 0 dB
#     # min-max je Sample -> [0,1]
#     db_min = float(S_db.min())
#     db_max = float(S_db.max())
#     S_db01 = ((S_db - db_min) / (db_max - db_min + 1e-12)).astype(np.float32)

#     # 4) (H,W) auf Vielfache von 32 padden
#     S_db01_pad, orig_hw = pad_to_multiple_reflect(S_db01, DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR)

#     # 5) Modell laden & Inferenz
#     model = UNetCustom(in_channels=1, out_channels=1, base_channels=BASE_CHANNELS).to(device)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()
#     with torch.no_grad():
#         x = torch.from_numpy(S_db01_pad[None, None, :, :]).to(device)  # (1,1,H,W)
#         x = to_model_space(x)
#         y_hat = model(x)                        # (1,1,H,W)
#         y_hat01_pad = to_img01_space(y_hat).clamp(0, 1)
#         S_denoised01_pad = y_hat01_pad[0, 0].cpu().numpy()

#     # 6) auf Originalgröße zurück
#     S_denoised01 = crop_to_original(S_denoised01_pad, orig_hw)

#     # 7) zurück zu dB mit gleicher per-Sample Skala
#     S_denoised_db = S_denoised01 * (db_max - db_min) + db_min

#     # 8) zurück zu linearer Magnitude
#     ref_val = float(np.max(S_noisy_mag))
#     S_denoised_mag = librosa.db_to_amplitude(S_denoised_db, ref=ref_val)

#     # 9) Rekombiniere mit Original-Phase
#     S_denoised_complex = S_denoised_mag * np.exp(1j * S_noisy_phase)

#     # 10) iSTFT (mit length für exakte Dauer)
#     y_denoised = librosa.istft(
#         S_denoised_complex, hop_length=hop_length, win_length=win_length, window=window, length=len(y_noisy)
#     )

#     # 12) Pegel-Strategie anwenden (Peak-Norm / Hard Gain / None)
#     y_denoised = apply_output_level(
#         y_denoised,
#         method=NORMALIZE_METHOD,
#         peak_target=PEAK_TARGET,
#         hard_gain=HARD_GAIN_FACTOR,
#     )

#     # 13) Speichern (Audio + Plots)
#     sf.write(output_audio_path, y_denoised.astype(np.float32), sr)

#     # Spektrogramm-Bilder
#     plot_spectrogram_db(S_db, sr, hop_length, noisy_spec_png, title="Noisy Spektrogramm")
#     plot_spectrogram_db(S_denoised_db, sr, hop_length, denoised_spec_png, title="Denoised Spektrogramm")

#     # Waveform-Vergleich
#     save_waveform(y_noisy, y_denoised, sr, waveforms_png)

#     print(f"[OK] Denoised WAV:       {output_audio_path}")
#     print(f"[OK] Noisy dB Spec:      {noisy_spec_png}")
#     print(f"[OK] Denoised dB Spec:   {denoised_spec_png}")
#     print(f"[OK] Waveforms:          {waveforms_png}")


# # --------------------------------- Ausführen ---------------------------------
# if __name__ == "__main__":
#     Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
#     denoise_audio(
#         noisy_audio_path=INPUT_AUDIO_PATH,
#         output_audio_path=OUTPUT_AUDIO_PATH,
#         model_path=MODEL_PATH,
#         noisy_spec_png=NOISY_SPEC_PNG,
#         denoised_spec_png=DENOISED_SPEC_PNG,
#         waveforms_png=WAVEFORMS_PNG,
#         n_fft=N_FFT,
#         hop_length=HOP,
#         win_length=WIN_LENGTH,
#         window=WINDOW,
#         sr_target=SR_TARGET,
#         device=DEVICE,
#     )





















# ################################ versuch 2
# import os
# import torch
# import torch.nn.functional as F
# import numpy as np
# import librosa
# import librosa.display
# import soundfile as sf
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from PIL import Image

# from createModelUnet import UNetCustom

# # =========================================================================
# #
# #   KONFIGURATION
# #
# # =========================================================================

# # --- Pfade ---
# # Eingabedatei im Unterordner "audios"
# INPUT_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audios")
# INPUT_AUDIO_NAME = "p232_005_noisy.wav"
# INPUT_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, INPUT_AUDIO_NAME)

# # Ausgabedateien im Unterordner "results"
# RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_denoising")
# MODEL_FILENAME = "best_model_baseCh_32_batch_8_patience_20.pth"
# MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_FILENAME)
# OUTPUT_AUDIO_NAME = "p232_005_denoised.wav"
# OUTPUT_AUDIO_PATH = os.path.join(RESULTS_DIR, OUTPUT_AUDIO_NAME)


# # --- STFT-Parameter ---
# TARGET_SR = 48000
# N_FFT = 1024
# HOP_LENGTH = 256
# WIN_LENGTH = 1024

# # --- Modell-Parameter ---
# # WICHTIG: Muss mit dem trainierten Modell übereinstimmen!
# BASE_CHANNELS = 32
# # Parameter für das Tiling bei der Bildverarbeitung
# TILE_W = 256
# STRIDE = 128

# # --- Bild-Rendering ---
# DPI = 100 # Hält die Bildauflösung konsistent

# # --- Lautstärke-Steuerung ---
# APPLY_AMPLIFICATION = True
# GAIN_FACTOR = 6


# # =========================================================================
# #
# #   HELFERFUNKTIONEN
# #
# # =========================================================================

# def render_spec_to_gray_image(S_dB, sr):
#     """Rendert ein dB-Spektrogramm in ein Graustufenbild (Format [0,1])."""
#     H, W = S_dB.shape
#     fig_w, fig_h = W / DPI, H / DPI
#     fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
#     ax.set_position([0, 0, 1, 1])
#     librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax)
#     ax.axis('off')
#     fig.canvas.draw()

#     rgb_buf = fig.canvas.buffer_rgba()
#     rgb_array = np.asarray(rgb_buf)
#     plt.close(fig)

#     # RGBA -> RGB -> Graustufe (PIL "L" Formel)
#     gray_img = Image.fromarray(rgb_array).convert("L")
#     return np.array(gray_img, dtype=np.float32) / 255.0

# def to_model_space(tile01: torch.Tensor):
#     """Konvertiert Bilddaten vom Bereich [0,1] nach [-1,1] für das Modell."""
#     return tile01 * 2.0 - 1.0

# def to_img01_space(tile_norm: torch.Tensor):
#     """Konvertiert Modelldaten vom Bereich [-1,1] zurück nach [0,1]."""
#     return (tile_norm + 1.0) * 0.5

# @torch.no_grad()
# def tiled_image_inference(model, img01, device):
#     """Führt eine geteilte Inferenz auf dem Bild durch und setzt es wieder zusammen."""
#     model.eval()
#     H, W = img01.shape

#     # Padding, falls das Bild schmaler als ein Tile ist
#     if W < TILE_W:
#         pad_amount = TILE_W - W
#         img01 = np.pad(img01, ((0, 0), (0, pad_amount)), mode='constant')

#     H, W_pad = img01.shape

#     win = torch.from_numpy(np.hanning(TILE_W).astype(np.float32)).to(device)
#     win = win.view(1, 1, 1, -1)

#     out_sum = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)
#     w_sum = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)

#     for x_left in range(0, W_pad - TILE_W + 1, STRIDE):
#         tile = img01[:, x_left:x_left + TILE_W]
#         tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)

#         pred = model(to_model_space(tile_t))
#         pred01 = to_img01_space(pred).clamp(0.0, 1.0)

#         out_sum[..., x_left:x_left + TILE_W] += pred01 * win
#         w_sum[..., x_left:x_left + TILE_W] += win

#     # Überlappende Reste am Ende behandeln
#     if (W_pad - TILE_W) % STRIDE != 0:
#         x_left = W_pad - TILE_W
#         tile = img01[:, x_left:x_left + TILE_W]
#         tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)

#         pred = model(to_model_space(tile_t))
#         pred01 = to_img01_space(pred).clamp(0.0, 1.0)

#         out_sum[..., x_left:x_left + TILE_W] += pred01 * win
#         w_sum[..., x_left:x_left + TILE_W] += win

#     out01 = (out_sum / (w_sum + 1e-8)).squeeze().cpu().numpy()
#     return out01[:, :W] # Zurückschneiden auf Originalbreite

# def save_spectrogram_png(img2d, out_path):
#     """Speichert das 2D-Array als pixelgenaues Spektrogramm-Bild."""
#     plt.figure(figsize=(img2d.shape[1] / DPI, img2d.shape[0] / DPI), dpi=DPI)
#     plt.imshow(img2d, origin='upper', aspect='auto', cmap="magma")
#     plt.axis('off')
#     plt.tight_layout(pad=0)
#     plt.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
#     plt.close()

# # =========================================================================
# #
# #   HELFERFUNKTIONEN (für Audio-Rekonstruktion & Wellenform)
# #
# # =========================================================================

# def amplify_audio(y: np.ndarray, gain_factor: float):
#     """Verstärkt Audio und schützt vor Clipping."""
#     amplified_y = y * gain_factor
#     if np.max(np.abs(amplified_y)) > 1.0:
#         print(f"Warnung: Clipping bei Verstärkung erkannt. Signal wird begrenzt.")
#         amplified_y = np.clip(amplified_y, -1.0, 1.0)
#     return amplified_y

# def save_waveform_comparison(y_noisy, y_denoised_final, sr, out_path):
#     """Speichert einen Vergleich der Wellenformen."""
#     plt.figure(figsize=(15, 5))
#     librosa.display.waveshow(y_noisy, sr=sr, alpha=0.6, label='Original verrauscht', color='blue')
#     librosa.display.waveshow(y_denoised_final, sr=sr, alpha=0.8, label='Finales Ergebnis', color='red')
#     plt.title('Wellenform-Vergleich')
#     plt.xlabel('Zeit (s)')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.grid(True, linestyle='--')
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#     print(f"Wellenform-Vergleich gespeichert: {out_path}")

# # =========================================================================
# #
# #   HAUPTPROZESS
# #
# # =========================================================================

# def denoise_hybrid_process():
#     # --- 0. Vorbereitung ---
#     if not os.path.exists(INPUT_AUDIO_PATH):
#         print(f"FEHLER: Eingabedatei nicht gefunden: {INPUT_AUDIO_PATH}"); return
#     if not os.path.exists(MODEL_PATH):
#         print(f"FEHLER: Modelldatei nicht gefunden: {MODEL_PATH}"); return
#     os.makedirs(RESULTS_DIR, exist_ok=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Verwende Gerät: {device}")

#     # --- 1. Laden & STFT ---
#     print(f"Lade Audio: {INPUT_AUDIO_PATH}")
#     y_noisy, sr = librosa.load(INPUT_AUDIO_PATH, sr=None, mono=True)
#     if sr != TARGET_SR:
#         print(f"Resampling von {sr} Hz auf {TARGET_SR} Hz...")
#         y_noisy = librosa.resample(y=y_noisy, orig_sr=sr, target_sr=TARGET_SR)
#         sr = TARGET_SR

#     print("Führe STFT durch...")
#     S_noisy_complex = librosa.stft(y=y_noisy, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
#     S_noisy_mag = np.abs(S_noisy_complex)
#     noisy_phase = np.angle(S_noisy_complex)
#     S_noisy_db = librosa.amplitude_to_db(S_noisy_mag, ref=np.max)

#     # --- 2. Modell laden ---
#     print(f"Lade Modell: {MODEL_PATH}")
#     model = UNetCustom(in_channels=1, out_channels=1, base_channels=BASE_CHANNELS).to(device)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()

#     # ===========================================================
#     #   PFAD A: BILD-BASIERTES DENOISING FÜR VISUALISIERUNG
#     # ===========================================================
#     print("\n--- Prozess A: Erstelle Spektrogramm-Bilder ---")
#     print("Rendere verrauschtes Spektrogramm zu einem Bild...")
#     gray_img_full = render_spec_to_gray_image(S_noisy_db, sr)
#     gray_img_noisy = gray_img_full[1:, :] # DC-Zeile entfernen

#     out_path_noisy_spec = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_spectrogram_noisy.png")
#     save_spectrogram_png(gray_img_noisy, out_path_noisy_spec)
#     print(f"Verrauschtes Spektrogramm-Bild gespeichert: {out_path_noisy_spec}")

#     print("Führe Tiled-Inferenz auf dem Bild durch...")
#     denoised_gray_img = tiled_image_inference(model, gray_img_noisy, device)

#     out_path_denoised_spec = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_spectrogram_denoised.png")
#     save_spectrogram_png(denoised_gray_img, out_path_denoised_spec)
#     print(f"Bereinigtes Spektrogramm-Bild gespeichert: {out_path_denoised_spec}")

#     # ===========================================================
#     #   PFAD B: DATEN-BASIERTES DENOISING FÜR AUDIO-REKONSTRUKTION
#     # ===========================================================
#     print("\n--- Prozess B: Rekonstruiere Audio-Datei ---")
#     S_noisy_db_data = S_noisy_db[1:, :] # DC-Zeile entfernen (nur Daten)
#     original_height, original_width = S_noisy_db_data.shape

#     # Normalisieren
#     db_min, db_max = S_noisy_db_data.min(), S_noisy_db_data.max()
#     S_norm = (S_noisy_db_data - db_min) / (db_max - db_min + 1e-7)
#     spec_tensor = torch.from_numpy(S_norm).float().unsqueeze(0).unsqueeze(0).to(device)

#     # Padding für das Modell
#     pad_right = (STRIDE - original_width % STRIDE) % STRIDE
#     spec_tensor_padded = F.pad(spec_tensor, (0, pad_right, 0, 0), "constant", 0)

#     print("Führe Inferenz auf Spektrogramm-Daten durch...")
#     with torch.no_grad():
#         denoised_padded = model(spec_tensor_padded)

#     # Cropping und Denormalisierung
#     denoised_tensor = denoised_padded[:, :, :, :original_width]
#     denoised_norm = denoised_tensor.squeeze().cpu().numpy()
#     denoised_db = denoised_norm * (db_max - db_min + 1e-7) + db_min

#     # DC-Zeile wieder hinzufügen für iSTFT
#     denoised_db_full = np.vstack([np.zeros((1, original_width)), denoised_db])

#     print("Kombiniere mit originaler Phase und führe inverse STFT durch...")
#     ref_val = np.max(S_noisy_mag)
#     S_denoised_mag = librosa.db_to_amplitude(denoised_db_full, ref=ref_val)
#     # S_denoised_mag = librosa.db_to_amplitude(denoised_db_full)
#     S_denoised_complex = S_denoised_mag * np.exp(1j * noisy_phase)
#     y_denoised = librosa.istft(S_denoised_complex, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, length=len(y_noisy))

#     # Verstärkung anwenden
#     y_final = amplify_audio(y_denoised, GAIN_FACTOR) if APPLY_AMPLIFICATION else y_denoised

#     # Audio speichern
#     sf.write(OUTPUT_AUDIO_PATH, y_final, sr)
#     print(f"Finale Audiodatei gespeichert: {OUTPUT_AUDIO_PATH}")

#     # --- Finale Wellenform-Visualisierung ---
#     out_path_wave = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_waveforms.png")
#     save_waveform_comparison(y_noisy, y_final, sr, out_path_wave)

#     print("\nAlle Prozesse abgeschlossen.")

# if __name__ == '__main__':
#     denoise_hybrid_process()





























# ################################# complexe magnitude und phase + max volume + darstellungen + base_channel fixed + best audio
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import librosa
# import librosa.display
# import soundfile as sf
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from PIL import Image

# from createModelUnet import UNetCustom

# # =========================================================================
# #
# #   KONFIGURATION
# #
# # =========================================================================

# # --- Pfade ---
# INPUT_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audios")
# INPUT_AUDIO_NAME = "p232_032_noisy.wav"
# INPUT_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, INPUT_AUDIO_NAME)

# RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_denoising")
# MODEL_FILENAME = "best_model_baseCh_32_batch_8_patience_20.pth"
# MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_FILENAME)
# OUTPUT_AUDIO_NAME = f"{INPUT_AUDIO_NAME}_denoised.wav"
# OUTPUT_AUDIO_PATH = os.path.join(RESULTS_DIR, OUTPUT_AUDIO_NAME)


# # --- STFT-Parameter ---
# TARGET_SR = 48000
# N_FFT = 1024
# HOP_LENGTH = 256
# WIN_LENGTH = 1024

# # --- Modell-Parameter ---
# BASE_CHANNELS = 32
# TILE_W = 256
# STRIDE = 128

# # --- Bild-Rendering ---
# DPI = 100

# # --- Lautstärke-Steuerung ---
# APPLY_AMPLIFICATION = True
# GAIN_FACTOR = 2


# # =========================================================================
# #
# #   HELFERFUNKTIONEN (Unverändert)
# #
# # =========================================================================

# def render_spec_to_gray_image(S_dB, sr):
#     H, W = S_dB.shape
#     fig_w, fig_h = W / DPI, H / DPI
#     fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
#     ax.set_position([0, 0, 1, 1])
#     librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax)
#     ax.axis('off')
#     fig.canvas.draw()

#     rgb_buf = fig.canvas.buffer_rgba()
#     rgb_array = np.asarray(rgb_buf)
#     plt.close(fig)

#     gray_img = Image.fromarray(rgb_array).convert("L")
#     return np.array(gray_img, dtype=np.float32) / 255.0

# def to_model_space(tile01: torch.Tensor):
#     return tile01 * 2.0 - 1.0

# def to_img01_space(tile_norm: torch.Tensor):
#     return (tile_norm + 1.0) * 0.5

# @torch.no_grad()
# def tiled_image_inference(model, img01, device):
#     model.eval()
#     H, W = img01.shape

#     if W < TILE_W:
#         pad_amount = TILE_W - W
#         img01 = np.pad(img01, ((0, 0), (0, pad_amount)), mode='constant')

#     H, W_pad = img01.shape

#     win = torch.from_numpy(np.hanning(TILE_W).astype(np.float32)).to(device)
#     win = win.view(1, 1, 1, -1)

#     out_sum = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)
#     w_sum = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)

#     for x_left in range(0, W_pad - TILE_W + 1, STRIDE):
#         tile = img01[:, x_left:x_left + TILE_W]
#         tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
#         pred = model(to_model_space(tile_t))
#         pred01 = to_img01_space(pred).clamp(0.0, 1.0)
#         out_sum[..., x_left:x_left + TILE_W] += pred01 * win
#         w_sum[..., x_left:x_left + TILE_W] += win

#     if (W_pad - TILE_W) % STRIDE != 0:
#         x_left = W_pad - TILE_W
#         tile = img01[:, x_left:x_left + TILE_W]
#         tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
#         pred = model(to_model_space(tile_t))
#         pred01 = to_img01_space(pred).clamp(0.0, 1.0)
#         out_sum[..., x_left:x_left + TILE_W] += pred01 * win
#         w_sum[..., x_left:x_left + TILE_W] += win

#     out01 = (out_sum / (w_sum + 1e-8)).squeeze().cpu().numpy()
#     return out01[:, :W]

# def save_spectrogram_png(img2d, out_path):
#     plt.figure(figsize=(img2d.shape[1] / DPI, img2d.shape[0] / DPI), dpi=DPI)
#     plt.imshow(img2d, origin='upper', aspect='auto', cmap="magma")
#     plt.axis('off')
#     plt.tight_layout(pad=0)
#     plt.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
#     plt.close()

# def amplify_audio(y: np.ndarray, gain_factor: float):
#     amplified_y = y * gain_factor
#     if np.max(np.abs(amplified_y)) > 1.0:
#         print(f"Warnung: Clipping bei Verstärkung erkannt. Signal wird begrenzt.")
#         amplified_y = np.clip(amplified_y, -1.0, 1.0)
#     return amplified_y

# def save_waveform_comparison(y_noisy, y_denoised_final, sr, out_path):
#     plt.figure(figsize=(15, 5))
#     librosa.display.waveshow(y_noisy, sr=sr, alpha=0.6, label='Original verrauscht', color='blue')
#     librosa.display.waveshow(y_denoised_final, sr=sr, alpha=0.8, label='Finales Ergebnis', color='red')
#     plt.title('Wellenform-Vergleich')
#     plt.xlabel('Zeit (s)')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.grid(True, linestyle='--')
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#     print(f"Wellenform-Vergleich gespeichert: {out_path}")

# # =========================================================================
# #
# #   HAUPTPROZESS
# #
# # =========================================================================

# def denoise_hybrid_process():
#     # --- 0. Vorbereitung ---
#     if not os.path.exists(INPUT_AUDIO_PATH):
#         print(f"FEHLER: Eingabedatei nicht gefunden: {INPUT_AUDIO_PATH}"); return
#     if not os.path.exists(MODEL_PATH):
#         print(f"FEHLER: Modelldatei nicht gefunden: {MODEL_PATH}"); return
#     os.makedirs(RESULTS_DIR, exist_ok=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Verwende Gerät: {device}")

#     # --- 1. Laden & STFT ---
#     print(f"Lade Audio: {INPUT_AUDIO_PATH}")
#     y_noisy, sr = librosa.load(INPUT_AUDIO_PATH, sr=None, mono=True)
#     if sr != TARGET_SR:
#         print(f"Resampling von {sr} Hz auf {TARGET_SR} Hz...")
#         y_noisy = librosa.resample(y=y_noisy, orig_sr=sr, target_sr=TARGET_SR)
#         sr = TARGET_SR

#     print("Führe STFT durch...")
#     S_noisy_complex = librosa.stft(y=y_noisy, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
#     S_noisy_mag = np.abs(S_noisy_complex)
#     noisy_phase = np.angle(S_noisy_complex)

#     # ** KORREKTUR TEIL 1: Speichere den Referenzwert **
#     ref_value_for_db = np.max(S_noisy_mag)
#     S_noisy_db = librosa.amplitude_to_db(S_noisy_mag, ref=ref_value_for_db)

#     # --- 2. Modell laden ---
#     print(f"Lade Modell: {MODEL_PATH}")
#     model = UNetCustom(in_channels=1, out_channels=1, base_channels=BASE_CHANNELS).to(device)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()

#     # ===========================================================
#     #   PFAD A: BILD-BASIERTES DENOISING FÜR VISUALISIERUNG
#     # ===========================================================
#     print("\n--- Prozess A: Erstelle Spektrogramm-Bilder ---")
#     print("Rendere verrauschtes Spektrogramm zu einem Bild...")
#     gray_img_full = render_spec_to_gray_image(S_noisy_db, sr)
#     gray_img_noisy = gray_img_full[1:, :]

#     out_path_noisy_spec = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_spectrogram_noisy.png")
#     save_spectrogram_png(gray_img_noisy, out_path_noisy_spec)
#     print(f"Verrauschtes Spektrogramm-Bild gespeichert: {out_path_noisy_spec}")

#     print("Führe Tiled-Inferenz auf dem Bild durch...")
#     denoised_gray_img = tiled_image_inference(model, gray_img_noisy, device)

#     out_path_denoised_spec = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_spectrogram_denoised.png")
#     save_spectrogram_png(denoised_gray_img, out_path_denoised_spec)
#     print(f"Bereinigtes Spektrogramm-Bild gespeichert: {out_path_denoised_spec}")

#     # ===========================================================
#     #   PFAD B: DATEN-BASIERTES DENOISING FÜR AUDIO-REKONSTRUKTION
#     # ===========================================================
#     print("\n--- Prozess B: Rekonstruiere Audio-Datei ---")
#     S_noisy_db_data = S_noisy_db[1:, :]
#     original_height, original_width = S_noisy_db_data.shape

#     db_min, db_max = S_noisy_db_data.min(), S_noisy_db_data.max()
#     S_norm = (S_noisy_db_data - db_min) / (db_max - db_min + 1e-7)
#     spec_tensor = torch.from_numpy(S_norm).float().unsqueeze(0).unsqueeze(0).to(device)

#     pad_right = (STRIDE - original_width % STRIDE) % STRIDE
#     spec_tensor_padded = F.pad(spec_tensor, (0, pad_right, 0, 0), "constant", 0)

#     print("Führe Inferenz auf Spektrogramm-Daten durch...")
#     with torch.no_grad():
#         denoised_padded = model(spec_tensor_padded)

#     denoised_tensor = denoised_padded[:, :, :, :original_width]
#     denoised_norm = denoised_tensor.squeeze().cpu().numpy()
#     denoised_db = denoised_norm * (db_max - db_min + 1e-7) + db_min

#     denoised_db_full = np.vstack([np.zeros((1, original_width)), denoised_db])

#     print("Kombiniere mit originaler Phase und führe inverse STFT durch...")

#     # ** KORREKTUR TEIL 2: Verwende den gespeicherten Referenzwert erneut **
#     S_denoised_mag = librosa.db_to_amplitude(denoised_db_full, ref=ref_value_for_db)

#     S_denoised_complex = S_denoised_mag * np.exp(1j * noisy_phase)
#     y_denoised = librosa.istft(S_denoised_complex, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, length=len(y_noisy))

#     y_final = amplify_audio(y_denoised, GAIN_FACTOR) if APPLY_AMPLIFICATION else y_denoised

#     sf.write(OUTPUT_AUDIO_PATH, y_final, sr)
#     print(f"Finale Audiodatei gespeichert: {OUTPUT_AUDIO_PATH}")

#     out_path_wave = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_waveforms.png")
#     save_waveform_comparison(y_noisy, y_final, sr, out_path_wave)

#     print("\nAlle Prozesse abgeschlossen.")

# if __name__ == '__main__':
#     denoise_hybrid_process()





























# # ##########################  originale Phase vom noisy audio + waveform + griffin-lim + best sound
# """
# Input : verrauschte M4A/WAV-Datei (einfach unten 'audio_name' / 'audio_extension' anpassen)
# Output:
#   - Noisy-Spektrogramm (PNG, ohne Achsen, training-like)
#   - Denoised-Spektrogramm (PNG, ohne Achsen, training-like)
#   - Noisy-Waveform (Amplitude über Zeit, PNG)
#   - Denoised-Waveform (Amplitude über Zeit, PNG)
#   - Denoised-Audio (WAV, via Griffin-Lim)
#   - Logdatei results_denoising/denoising_output.txt

# Kernideen:
#   * Wir bilden für das Modell genau den gleichen Tensor wie im Training:
#       STFT -> dB (ref = max) -> min/max-Scaling auf [0,1] -> DC-Zeile entfernen -> Normalize(0.5, 0.5)
#   * U-Net-Inferenz in Kacheln (Tiles), Mittelung in Überlappungen.
#   * Für die Audiorekonstruktion:
#       Denormalisieren -> zurück in den ursprünglichen dB-Min/Max-Bereich -> dB->Amplitude -> Griffin-Lim (Phasenschätzung).
#   * Für hübsche Visualisierungen:
#       Speichere die [0,1]-Bilder (ohne Achsen, cmap='magma').
# """

# import os
# import time
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import librosa
# import librosa.display
# import soundfile as sf

# from createModelUnet import UNetCustom
# from trainLogging import setup_logging

# # =========================
# # Pfade
# # =========================
# audios_dir      = os.path.join(os.path.dirname(__file__), "audios")
# audio_name      = "noisy2"
# audio_extension = ".m4a"
# INPUT_AUDIO     = os.path.join(audios_dir, f"{audio_name}{audio_extension}")

# OUT_DIR         = os.path.join(os.path.dirname(__file__), "results_denoising")
# MODEL_PATH      = os.path.join(OUT_DIR, "best_model_baseCh_32_batch_8.pth")
# LOG_FILE        = "denoising_output.txt"

# # =========================
# # STFT / Modell / Darstellung
# # =========================
# STFT_N_FFT = 1024
# STFT_HOP   = 256
# STFT_WIN   = 1024

# DN_TILE_H  = 512
# DN_TILE_W  = 256
# DN_STRIDE  = 256

# MODEL_IN_CHANNELS   = 1
# MODEL_BASE_CHANNELS = 32

# MEAN = 0.5
# STD  = 0.5

# DPI = 100


# # =========================
# # Hilfsfunktionen
# # =========================
# # def save_spectrogram_png(gray01: np.ndarray, out_path: str):
# #     """Speichert ein [0,1]-Spektrogramm als PNG ohne Achsen, cmap='magma'."""
# #     H, W = gray01.shape
# #     fig_w, fig_h = W / DPI, H / DPI
# #     plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
# #     plt.imshow(gray01, origin='lower', aspect='auto', cmap="magma", vmin=0.0, vmax=1.0)
# #     plt.axis('off')
# #     plt.tight_layout(pad=0)
# #     os.makedirs(os.path.dirname(out_path), exist_ok=True)
# #     plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
# #     plt.close()


# def save_spectrogram_db_png(S_db, sr, hop_length, out_path):
#     """Speichert ein dB-Spektrogramm als PNG ohne Achsen."""
#     H, W = S_db.shape
#     fig_w, fig_h = W / DPI, H / DPI

#     plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
#     # librosa.display.specshow ist die professionelle Wahl für Spektrogramme
#     librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, cmap='magma')

#     plt.axis('off')
#     plt.tight_layout(pad=0)
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# # def save_waveform(y: np.ndarray, sr: int, out_path: str, title: str):
# #     plt.figure(figsize=(12, 3))
# #     librosa.display.waveshow(y, sr=sr, color='b')
# #     plt.title(title); plt.xlabel("Zeit (s)"); plt.ylabel("Amplitude")
# #     plt.tight_layout()
# #     os.makedirs(os.path.dirname(out_path), exist_ok=True)
# #     plt.savefig(out_path, dpi=150)
# #     plt.close()

# def save_comparison_waveform(y_noisy, y_denoised, sr, out_path):
#     """Speichert eine überlagerte Darstellung der verrauschten und entrauschten Wellenform."""
#     plt.figure(figsize=(12, 4))

#     # Zeichne die entrauschte Wellenform (prominenter)
#     librosa.display.waveshow(y_denoised, sr=sr, color='blue', alpha=1.0, label='Denoised')
#     # Zeichne die verrauschte Wellenform (im Hintergrund)
#     librosa.display.waveshow(y_noisy, sr=sr, color='grey', alpha=0.5, label='Noisy')

#     plt.title("Waveform Vergleich: Noisy vs. Denoised")
#     plt.xlabel("Zeit (s)")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.savefig(out_path, dpi=150)
#     plt.close()



# def audio_to_model_tensor(y: np.ndarray, sr: int, logger):
#     """
#     Audio -> STFT -> dB (ref=max) -> [0,1]-Scaling pro Datei -> DC-Zeile weg -> Normalize(0.5,0.5).
#     Rückgabe:
#       noisy_tensor  : (C=1, H=512, W) Tensor in Modell-Skala [-1,1]
#       gray01_wo_dc  : (512, W) numpy in [0,1]  (für PNG)
#       S_complex     : komplexes STFT (513, W)
#       db_min, db_max: dB-Min/Max der Datei (für spätere Rückskalierung)
#     """
#     logger.info("1) Audio -> STFT -> dB -> [0,1] (+ DC drop) -> Normalize")
#     S_complex = librosa.stft(y, n_fft=STFT_N_FFT, hop_length=STFT_HOP, win_length=STFT_WIN)
#     S_mag = np.abs(S_complex)                   # (513, W)
#     S_db  = librosa.amplitude_to_db(S_mag, ref=np.max)  # [neg, 0], Peak=0 dB

#     db_min = float(S_db.min())
#     db_max = float(S_db.max())
#     S_db01 = (S_db - db_min) / (db_max - db_min + 1e-8)  # -> [0,1]

#     gray01_wo_dc = S_db01[1:, :].astype(np.float32)      # (512, W)

#     # -> Tensor (C=1,H,W), Normalize(0.5,0.5) => [-1,1]
#     noisy_tensor = torch.from_numpy(gray01_wo_dc).unsqueeze(0)          # (1, 512, W)
#     noisy_tensor = (noisy_tensor - MEAN) / STD                           # normalize
#     return noisy_tensor, gray01_wo_dc, S_complex, db_min, db_max


# @torch.no_grad()
# def denoise_tiles(noisy_tensor: torch.Tensor, model: nn.Module, device: torch.device, logger):
#     """
#     Tiled-Inferenz über die Zeitachse:
#       noisy_tensor: (1, H=512, W)  in Modell-Skala [-1,1]
#     Ergebnis wird über Überlapp gemittelt.
#     Rückgabe:
#       denoised_tensor: (1, 512, W) in Modell-Skala [-1,1]
#     """
#     C, H, W = noisy_tensor.shape
#     assert H == DN_TILE_H, f"Höhe {H} != erwartete {DN_TILE_H}"

#     out = torch.zeros_like(noisy_tensor)
#     cnt = torch.zeros_like(noisy_tensor)

#     x = 0
#     steps = 0
#     while True:
#         x0, x1 = x, x + DN_TILE_W
#         if x1 > W:        # letzte Kachel rechtsbündig
#             x0 = max(0, W - DN_TILE_W)
#             x1 = W

#         tile = noisy_tensor[:, :, x0:x1].unsqueeze(0).to(device)  # (1,1,512,Wt)
#         with torch.no_grad():
#             pred = model(tile).cpu().squeeze(0)                   # (1,512,Wt)

#         out[:, :, x0:x1] += pred
#         cnt[:, :, x0:x1] += 1.0

#         steps += 1
#         if x1 >= W:
#             break
#         x += DN_STRIDE

#     logger.info(f"2) Tiled Denoising: {steps} Kacheln (tile_w={DN_TILE_W}, stride={DN_STRIDE})")
#     denoised = out / (cnt + 1e-8)
#     return denoised


# def tensor_to_audio_griffinlim(denoised_tensor: torch.Tensor, db_min: float, db_max: float, logger):
#     """
#     Denormalisieren -> [0,1] -> zurück in dB -> DC-Zeile auffüllen -> dB->Amplitude -> Griffin-Lim -> y_denoised.
#     Rückgabe:
#       y_denoised        : 1D numpy
#       denoised_db_full  : (513, W) dB-Bild (mit DC-Zeile) für evtl. weitere Visualisierung
#       denoised01_wo_dc  : (512, W) [0,1]-Bild der Modell-Ausgabe (für PNG)
#     """
#     # [-1,1] -> [0,1]
#     denoised01 = denoised_tensor * STD + MEAN
#     denoised01.clamp_(0.0, 1.0)
#     den01_np = denoised01.squeeze(0).numpy()            # (512, W)

#     # [0,1] -> dB (min/max der Datei)
#     den_db = den01_np * (db_max - db_min) + db_min      # (512, W)

#     # DC-Zeile mit db_min füllen
#     H_wo, W = den_db.shape
#     den_db_full = np.full((H_wo + 1, W), db_min, dtype=np.float32)
#     den_db_full[1:, :] = den_db

#     # dB -> Amplitude
#     den_mag = librosa.db_to_amplitude(den_db_full)

#     # Griffin-Lim (Phasenschätzung)
#     logger.info("3) Griffin-Lim Rekonstruktion ...")
#     y_deno = librosa.griffinlim(
#         den_mag,
#         n_iter=64,
#         hop_length=STFT_HOP,
#         win_length=STFT_WIN
#     )
#     return y_deno.astype(np.float32), den_db_full, den01_np

# def tensor_to_audio_with_original_phase(denoised_tensor: torch.Tensor, S_complex_noisy: np.ndarray, db_min: float, db_max: float, logger):
#     """
#     Professionelle Rekonstruktion:
#     Denormalisieren -> [0,1] -> zurück in dB -> DC-Zeile auffüllen -> dB->Amplitude ->
#     Kombinieren der entrauschten Amplitude mit der Phase des Originalsignals.
#     """
#     # [-1,1] -> [0,1]
#     denoised01 = denoised_tensor * STD + MEAN
#     denoised01.clamp_(0.0, 1.0)
#     den01_np = denoised01.squeeze(0).numpy()      # (512, W)

#     # [0,1] -> dB (min/max der Datei)
#     den_db = den01_np * (db_max - db_min) + db_min   # (512, W)

#     # DC-Zeile mit db_min füllen
#     H_wo, W = den_db.shape
#     den_db_full = np.full((H_wo + 1, W), db_min, dtype=np.float32)
#     den_db_full[1:, :] = den_db

#     # dB -> Amplitude
#     den_mag = librosa.db_to_amplitude(den_db_full)

#     # Phase aus dem originalen, verrauschten STFT extrahieren
#     logger.info("3) Rekonstruktion durch Kombination mit Original-Phase...")
#     phase_noisy = np.angle(S_complex_noisy)

#     # Neues komplexes Spektrogramm aus entrauschter Amplitude und verrauschter Phase bauen
#     den_complex = den_mag * np.exp(1j * phase_noisy)

#     # Inverse STFT anwenden, um das Audio zu rekonstruieren
#     y_deno = librosa.istft(
#         den_complex,
#         hop_length=STFT_HOP,
#         win_length=STFT_WIN
#     )
#     return y_deno.astype(np.float32), den_db_full, den01_np


# # =========================
# # Main
# # =========================
# def main():
#     os.makedirs(OUT_DIR, exist_ok=True)
#     logger = setup_logging(OUT_DIR, log_file=LOG_FILE, level="INFO")

#     t0 = time.time()
#     logger.info("========================================")
#     logger.info("=== Start: Externes Audio denoising ===")
#     logger.info("========================================")
#     logger.info(f"Eingabe: {INPUT_AUDIO}")

#     if not os.path.isfile(INPUT_AUDIO):
#         logger.error("Eingabedatei nicht gefunden."); return
#     if not os.path.isfile(MODEL_PATH):
#         logger.error("Modelldatei nicht gefunden."); return

#     # Gerät & Modell
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Gerät: {device}")

#     model = UNetCustom(in_channels=MODEL_IN_CHANNELS, out_channels=MODEL_IN_CHANNELS,
#                        base_channels=MODEL_BASE_CHANNELS)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device).eval()
#     logger.info("Modell geladen.")

#     # Audio laden
#     y, sr = librosa.load(INPUT_AUDIO, sr=None, mono=True)
#     dur = librosa.get_duration(y=y, sr=sr)
#     logger.info(f"Audio geladen: sr={sr} Hz, Dauer={dur:.2f} s, Samples={len(y)}")

#     # Tensor & Visual-Noisy (gray01)
#     noisy_tensor, gray01_no_dc, S_complex, db_min, db_max = audio_to_model_tensor(y, sr, logger)
#     H, W = gray01_no_dc.shape
#     logger.info(f"STFT/Img-Form: H={H}, W={W} (H=512 nach DC-Entfernung)")

#     # Denoising (Tiles)
#     denoised_tensor = denoise_tiles(noisy_tensor, model, device, logger)

#     # Rekonstruktion
#     y_deno, den_db_full, den01_no_dc = tensor_to_audio_griffinlim(denoised_tensor, db_min, db_max, logger)
#     # y_deno, den_db_full, den01_no_dc = tensor_to_audio_with_original_phase(denoised_tensor, S_complex, db_min, db_max, logger)

#     # ==== Ausgaben speichern ====
#     base = os.path.splitext(os.path.basename(INPUT_AUDIO))[0]

#     # Spektrogramme (ohne Achsen) – training-like [0,1]
#     noisy_png   = os.path.join(OUT_DIR, f"extern_noisy_{base}.png")
#     deno_png    = os.path.join(OUT_DIR, f"extern_denoised_{base}.png")

#     # Erzeuge das ursprüngliche dB-Spektrogramm (ohne DC) für die Visualisierung
#     S_db_noisy = librosa.amplitude_to_db(np.abs(S_complex), ref=np.max)

#     # HINWEIS: Wir verwenden S_db_noisy[1:, :] und den_db_full[1:, :], um die DC-Zeile zu ignorieren
#     save_spectrogram_db_png(S_db_noisy[1:, :], sr, STFT_HOP, noisy_png)
#     save_spectrogram_db_png(den_db_full[1:, :], sr, STFT_HOP, deno_png)
#     logger.info(f"dB-Spektrogramme gespeichert: {noisy_png}, {deno_png}")

#     # save_spectrogram_png(gray01_no_dc, noisy_png)
#     # save_spectrogram_png(den01_no_dc, deno_png)
#     # logger.info(f"Spektrogramme: {noisy_png}, {deno_png}")

#     # Waveforms
#     wave_comp_png = os.path.join(OUT_DIR, f"wave_comparison_{base}.png")
#     save_comparison_waveform(y, y_deno, sr, wave_comp_png)
#     logger.info(f"Vergleichs-Waveform gespeichert: {wave_comp_png}")

#     # # Waveforms
#     # wav_noisy_png  = os.path.join(OUT_DIR, f"wave_noisy_{base}.png")
#     # wav_deno_png   = os.path.join(OUT_DIR, f"wave_denoised_{base}.png")
#     # save_waveform(y,      sr, wav_noisy_png, title="Noisy Audio (Time Domain)")
#     # save_waveform(y_deno, sr, wav_deno_png,  title="Denoised Audio (Time Domain)")
#     # logger.info(f"Waveforms: {wav_noisy_png}, {wav_deno_png}")

#     # Denoised WAV
#     out_wav = os.path.join(OUT_DIR, f"{base}_denoised.wav")
#     # # Audio auf maximale Lautstärke normalisieren
#     # if np.max(np.abs(y_deno)) > 0:
#     #     y_deno = y_deno / np.max(np.abs(y_deno))
#     sf.write(out_wav, y_deno, sr)
#     logger.info(f"Denoised WAV: {out_wav}")

#     # Log-Zusatzinfos
#     energy_noisy = float(np.mean(y**2))
#     energy_deno  = float(np.mean(y_deno**2))
#     logger.info(f"Energie noisy: {energy_noisy:.6f}, Energie denoised: {energy_deno:.6f}")

#     logger.info("========================================")
#     logger.info(f"Fertig in {time.time()-t0:.2f} s")
#     logger.info("========================================")


# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()

















# ########################## mit iSTFT + originale Phase vom noisy audio + best waveform
# """
# Externes (Handy-)Audio -> STFT -> specshow (wie convertDataSpec.py) -> Bild -> Graustufe -> DC-Zeile weg -> Tiles
# -> U-Net -> Hann-Overlap-Stitching -> denoised Bild speichern (ohne Achsen)
# UND: Rekonstruktion des bereinigten Audios via iSTFT mit Original-Phase.

# Wichtig:
# - Wir spiegeln dein Trainings-Preprocessing (rendern per specshow ohne vmin/vmax).
# - Für die Audio-Rekonstruktion benutzen wir die aus specshow ermittelten (vmin, vmax), um die denoised-Graustufen
#   wieder auf dB zurückzubringen (annähernd).
# - DC-Zeile (Magnitude in dB) übernehmen wir aus dem Noisy-STFT, Phase ebenfalls aus dem Noisy.
# """

# import os
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import librosa
# import librosa.display
# from PIL import Image
# import soundfile as sf

# from createModelUnet import UNetCustom
# from trainLogging import setup_logging

# # =========================
# #   Pfade
# # =========================
# audios_dir      = os.path.join(os.path.dirname(__file__), "audios")
# audio_name      = "noisy2"
# audio_extension = ".m4a"
# AUDIO_PATH      = os.path.join(audios_dir, f"{audio_name}{audio_extension}")

# RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "results_denoising")
# MODEL_PATH      = os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8.pth")
# LOG_FILE        = "denoising_output.txt"

# # =========================
# #   STFT-Parameter (wie convertDataSpec.py)
# # =========================
# N_FFT      = 1024
# HOP_LENGTH = 256
# WIN_LENGTH = 1024
# SR         = None  # sr=None -> keine Resampling-Änderung

# # Rendering (Pixelgenau wie convertDataSpec.py, default colormap)
# DPI = 100  # Bildgröße = (W/DPI, H/DPI)

# # =========================
# #   Tiling / Modell
# # =========================
# TILE_W  = 256
# STRIDE  = 256
# IN_CH   = 1     # Graustufen
# BASE_CH = 32

# # =========================
# #   Helfer
# # =========================
# def render_spec_like_training_to_gray01(S_dB, sr):
#     """
#     Rendert S_dB mit librosa.display.specshow -> RGB-Canvas -> Graustufe (PIL "L").
#     Keine expliziten vmin/vmax -> identisch zum Konverter (per-Image-Autoskalierung).
#     Rückgabe:
#       gray01: (H, W) in [0,1]
#       clim:   (vmin, vmax) des Matplotlib-Normalizers (für spätere dB-Inversion).
#     """
#     H, W = S_dB.shape
#     fig_w = W / DPI
#     fig_h = H / DPI
#     fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
#     ax.set_position([0, 0, 1, 1])  # vollflächig
#     im = librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax)  # AxesImage
#     ax.axis('off')
#     vmin, vmax = im.get_clim()  # <- wichtig für spätere dB-Rückrechnung
#     fig.canvas.draw()  # rendern

#     # RGB aus Canvas lesen
#     w_px, h_px = fig.canvas.get_width_height()  # (W, H)
#     rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     rgb = rgb.reshape(h_px, w_px, 3)  # (H,W, 3)
#     plt.close(fig)

#     # RGB -> Graustufe (PIL "L")
#     gray = np.array(Image.fromarray(rgb).convert("L"), dtype=np.float32) / 255.0  # (H,W) in [0,1]
#     return gray, (float(vmin), float(vmax))

# def audio_to_gray01_training_like(wav_path):
#     """
#     1) Audio laden (sr=None, mono=True).
#     2) STFT -> |S| -> dB (ref=max pro Datei).
#     3) Rendern zu Gray [0,1] via specshow (wie im Dataset) + clim (vmin, vmax) merken.
#     4) DC-Zeile entfernen (erste Pixelzeile) -> Höhe 512.

#     Rückgabe:
#       gray01_wo_dc: (512, W) in [0,1]
#       clim:         (vmin, vmax) aus specshow
#       S_complex:    komplexes STFT (513, W)
#       amp_ref:      Referenz (Max-Magnitude) für dB<->Amplitude
#       y:            geladene Wellenform (1D)
#       sr_ret:       Samplerate
#     """
#     y, sr_ret = librosa.load(wav_path, sr=SR, mono=True)

#     S = librosa.stft(
#         y=y,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         win_length=WIN_LENGTH,
#         window="hann",
#         center=True,
#         pad_mode="reflect",
#     )
#     S_mag = np.abs(S)                # (513, W)
#     amp_ref = float(np.max(S_mag) + 1e-12)

#     # dB relativ zu amp_ref (Peak -> 0 dB)
#     S_db = librosa.amplitude_to_db(S_mag, ref=amp_ref)  # (513, W), Werte in [negativ, 0]

#     # Bild wie im Training erzeugen + clim merken
#     gray01_full, clim = render_spec_like_training_to_gray01(S_db, sr_ret)  # (513, W)
#     # DC-Zeile weg
#     gray01_wo_dc = gray01_full[1:, :]  # (512, W)
#     return gray01_wo_dc, clim, S, amp_ref, y, sr_ret

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

# @torch.no_grad()
# def tiled_infer_and_stitch(model: nn.Module, img01: np.ndarray, device: torch.device,
#                            tile_w: int = TILE_W, stride: int = STRIDE) -> np.ndarray:
#     """
#     Horizontal Tilen, Inferenz, Hann-Overlap-Stitch.
#     Rückgabe: denoised in [0,1], gleiche Form wie img01.
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
#             tile = img01_pad[:, x_left:x_left + tile_w]      # (H, tile_w)
#             tile_t = torch.from_numpy(tile).unsqueeze(0)      # (1,H,Wt)
#             tile_t = tile_t.unsqueeze(0).to(device)           # (1,1,H,Wt)

#             pred = model(to_model_space(tile_t))              # [-1,1]
#             pred01 = to_img01_space(pred).clamp(0.0, 1.0)

#             pred_w = pred01 * win.to(device)
#             out_sum[:, :, :, x_left:x_left + tile_w] += pred_w
#             w_sum[:, :, :, x_left:x_left + tile_w]   += win.to(device)

#     out01 = (out_sum / (w_sum + 1e-8)).squeeze(0).squeeze(0).cpu().numpy()  # (H,W_pad)
#     return out01[:, :W]

# def save_spectrogram_png(img2d: np.ndarray, out_path: str):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.figure(figsize=(img2d.shape[1] / DPI, img2d.shape[0] / DPI), dpi=DPI)
#     plt.imshow(img2d, origin='upper', aspect='auto', cmap="magma")
#     plt.axis('off'); plt.tight_layout(pad=0)
#     plt.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
#     plt.close()

# def save_waveform(y, sr, out_path, title="Waveform"):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.figure(figsize=(12, 3))
#     librosa.display.waveshow(y, sr=sr, color='b')
#     plt.title(title)
#     plt.xlabel("Zeit (s)")
#     plt.ylabel("Amplitude")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()

# def mean_abs_change01(noisy01: np.ndarray, denoised01: np.ndarray) -> float:
#     return float(np.mean(np.abs(denoised01 - noisy01)))

# def db_to_amplitude(S_db, clim, amp_ref):
#     """
#     Konvertiert dB-Werte zurück in Amplituden.
#     """
#     vmin, vmax = clim
#     S_db_clipped = np.clip(S_db, vmin, vmax)
#     S_db_scaled = (S_db_clipped - vmin) / (vmax - vmin) * (0 - vmin) + vmin
#     return librosa.db_to_amplitude(S_db_scaled, ref=amp_ref)

# def reconstruct_audio_from_denoised(denoised01, clim, S_complex_original, amp_ref, sr):
#     """
#     Rekonstruiert das Audio aus dem denoised01-Bild.
#     Verwendet die Original-Phase und rekonstruierte Magnitude.
#     """
#     # 1. [0,1] -> dB
#     vmin, vmax = clim
#     S_dB_recon = denoised01 * (vmax - vmin) + vmin  # (512, W)

#     # 2. dB -> Amplitude
#     S_mag_recon = db_to_amplitude(S_dB_recon, clim, amp_ref)  # (512, W)

#     # 3. DC-Zeile wiederherstellen (aus Original)
#     S_mag_recon_full = np.zeros((513, S_mag_recon.shape[1]))
#     S_mag_recon_full[0, :] = np.abs(S_complex_original)[0, :]  # DC aus Original
#     S_mag_recon_full[1:, :] = S_mag_recon

#     # 4. Phase aus Original
#     phase = np.angle(S_complex_original)

#     # 5. Komplexes STFT
#     S_recon = S_mag_recon_full * np.exp(1j * phase)

#     # 6. iSTFT
#     y_recon = librosa.istft(
#         S_recon,
#         hop_length=HOP_LENGTH,
#         win_length=WIN_LENGTH,
#         window="hann",
#         center=True
#     )
#     return y_recon


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
#     logger.info("Externes Audio denoisen (training-like rendering + Audio via iSTFT/Original-Phase)")
#     logger.info(f"Device: {device}")
#     logger.info(f"Datei:  {AUDIO_PATH}")

#     # Modell
#     model = UNetCustom(in_channels=IN_CH, out_channels=IN_CH, base_channels=BASE_CH).to(device)
#     state = torch.load(MODEL_PATH, map_location=device)
#     model.load_state_dict(state)

#     # 1) Audio -> S_dB -> Bild -> Gray01 -> DC-Bin entfernen (H=512) + clim + STFT + amp_ref + y
#     gray01, clim, S_complex, amp_ref, y, sr_ret = audio_to_gray01_training_like(AUDIO_PATH)
#     H, W = gray01.shape
#     dur_s = len(y) / float(sr_ret)
#     logger.info(f"Audio-Laenge: {dur_s:.2f} s, SR: {sr_ret} Hz")
#     logger.info(f"Spektrogramm (nach DC-Entfernung): H={H}, W={W}")
#     logger.info(f"STFT: n_fft={N_FFT}, hop={HOP_LENGTH}, win={WIN_LENGTH}")
#     logger.info(f"specshow clim: vmin={clim[0]:.2f}, vmax={clim[1]:.2f} (~0 dB)")

#     # 2) Tiled Inferenz & Stitch (Grauraum [0,1])
#     denoised01 = tiled_infer_and_stitch(model, gray01, device, tile_w=TILE_W, stride=STRIDE)
#     tiles_est = 1 if W <= TILE_W else 1 + (W - TILE_W + STRIDE - 1) // STRIDE
#     logger.info(f"Tiling: tile_w={TILE_W}, stride={STRIDE}, Tiles ~= {tiles_est}")

#     # 3) Bilder speichern (ohne Achsen)
#     base = os.path.splitext(os.path.basename(AUDIO_PATH))[0]
#     out_png_noisy  = os.path.join(RESULTS_DIR, f"extern_noisy_{base}.png")
#     out_png_deno   = os.path.join(RESULTS_DIR, f"extern_denoised_{base}.png")
#     save_spectrogram_png(gray01,     out_png_noisy)
#     save_spectrogram_png(denoised01, out_png_deno)
#     logger.info(f"Spektrogramme gespeichert: {out_png_noisy}, {out_png_deno}")

#     # 4) Heuristik (Bildraum)
#     mac = mean_abs_change01(gray01, denoised01)
#     logger.info(f"Mittlere absolute Aenderung (Grauraum [0,1]): {mac:.6f}")

#     # 5) Wellenform-Darstellung (Zeitbereich)
#     out_wave_noisy = os.path.join(RESULTS_DIR, f"wave_noisy_{base}.png")
#     out_wave_deno  = os.path.join(RESULTS_DIR, f"wave_denoised_{base}.png")
#     save_waveform(y, sr_ret, out_wave_noisy, title="Noisy Audio (Time Domain)")
#     logger.info(f"Wellenform (noisy) gespeichert: {out_wave_noisy}")

#     # 6) Audio-Rekonstruktion
#     y_recon = reconstruct_audio_from_denoised(denoised01, clim, S_complex, amp_ref, sr_ret)
#     save_waveform(y_recon, sr_ret, out_wave_deno, title="Denoised Audio (Time Domain)")
#     logger.info(f"Wellenform (denoised) gespeichert: {out_wave_deno}")

#     # 7) Bereinigtes Audio speichern
#     out_audio_deno = os.path.join(RESULTS_DIR, f"denoised_{base}.wav")
#     sf.write(out_audio_deno, y_recon, sr_ret)
#     logger.info(f"Bereinigtes Audio gespeichert: {out_audio_deno}")

#     # 8) Vergleich der Energien (optional)
#     energy_noisy = np.mean(y ** 2)
#     energy_deno = np.mean(y_recon ** 2)
#     logger.info(f"Energie (noisy): {energy_noisy:.6f}, Energie (denoised): {energy_deno:.6f}")

#     logger.info("=" * 70)
#     logger.info("Fertig.")

# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()
