import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

from createModelUnet import UNetCustom
from trainLogging import setup_logging

# ================================
# KONFIGURATION
# ================================
# --- Eingabe / Ausgabe Pfade ---
INPUT_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audios")
INPUT_AUDIO_NAME = "adhen.mp3"
INPUT_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, INPUT_AUDIO_NAME)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_denoising")
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

OUTPUT_AUDIO_NAME = "adhen_denoised.wav"
OUTPUT_AUDIO_PATH = os.path.join(RESULTS_DIR, OUTPUT_AUDIO_NAME)

NOISY_SPEC_PNG    = os.path.join(RESULTS_DIR, f"{os.path.splitext(INPUT_AUDIO_NAME)[0]}_spec.png")
DENOISED_SPEC_PNG = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_spec.png")
WAVEFORMS_PNG     = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_waveforms.png")
WAVEFORMS_SUB_PNG = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_waveforms_sub.png")

# --- Modelle (Liste) ---
MULTI_MODEL_SELECTION = True   # True = mehrere Modelle testen, False = einzelnes Modell nutzen
ZERO_EPS = 1e-2                # Epsilon für Nullnähe-Anteil

# Falls MULTI_MODEL_SELECTION=False wird nur das erste existierende Modell genutzt.
MODEL_LIST: List[str] = [
    os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8_full_data_aug_60_epochen.pth"),
    os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_16_full_data_aug_60_epochen_step_20.pth"),
    os.path.join(RESULTS_DIR, "best_model_baseCh_64_batch_8_full_data_aug_60_epochen_step_10_aug_high.pth"),
    os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8_full_data_aug_60_epochen.pth"),
    os.path.join(RESULTS_DIR, "best_model_baseCh_64_batch_8_full_data_aug_60_epochen_step_15_aug_low.pth"),
    os.path.join(RESULTS_DIR, "best_model_baseCh_64_batch_8_full_data_aug_60_epochen_step_15_aug_high.pth"),
    os.path.join(RESULTS_DIR, "best_model_baseCh_64_batch_8_full_data_aug_60_epochen.pth"),
    os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_16_full_data_aug_60_epochen.pth"),
    os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8_full_data_aug_60_epochen_val_01.pth"),

]

# --- STFT Parameter ---
SR_TARGET    = None
N_FFT        = 1024
HOP          = 256
WIN_LENGTH   = 1024
WINDOW       = "hann"

# --- Datenvorbereitung ---
DATA_PREP_METHOD = "numeric"   # "numeric" oder "image_tiled"

# für numeric
DOWNSAMPLE_FACTOR = 32

# für image_tiled
TILE_W = 256
STRIDE = 128
DPI = 100

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================
# Utilities (numeric)
# ================================
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

# ================================
# Utilities (image tiled)
# ================================
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

# ================================
# Plots
# ================================
def plot_spectrogram_db(S_db: np.ndarray, sr: int, hop_length: int, out_path: str, title: str = ""):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear")
    plt.colorbar(label="Amplitude [dB]")
    if title: plt.title(title)
    plt.xlabel("Zeit [s]"); plt.ylabel("Frequenz [Hz]")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()

def save_waveform(y_a, y_b, sr, out_path, label_a="Noisy", label_b="Denoised"):
    plt.figure(figsize=(15, 5))
    librosa.display.waveshow(y_a, sr=sr, alpha=0.6, label=label_a, color='blue')
    librosa.display.waveshow(y_b, sr=sr, alpha=0.6, label=label_b, color='red')
    plt.title('Wellenform'); plt.xlabel('Zeit [s]'); plt.ylabel('Amplitude')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()

def save_waveform_subplot(y_a, y_b, sr, out_path, label_a="Noisy", label_b="Denoised"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

    # Obere Grafik: Erste Wellenform
    librosa.display.waveshow(y_a, sr=sr, alpha=0.8, label=label_a, color='blue', ax=ax1)
    ax1.set_title(f'Wellenform - {label_a}')
    ax1.set_xlabel('Zeit [s]')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Untere Grafik: Zweite Wellenform
    librosa.display.waveshow(y_b, sr=sr, alpha=0.8, label=label_b, color='blue', ax=ax2)
    ax2.set_title(f'Wellenform - {label_b}')
    ax2.set_xlabel('Zeit [s]')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# ================================
# Auswahl Metrik
# ================================
def zero_fraction(y: np.ndarray, eps: float = ZERO_EPS) -> float:
    """Anteil der Samples mit |y| < eps (größer ist besser)."""
    y = np.asarray(y, dtype=np.float32)
    return float(np.mean(np.abs(y) < eps))

# ================================
# Model Loading
# ================================
def try_instantiate_unet(base_channels: Optional[int]) -> UNetCustom:
    import inspect
    sig = inspect.signature(UNetCustom)
    if "base_channels" in sig.parameters and base_channels is not None:
        return UNetCustom(base_channels=base_channels)
    return UNetCustom()

def load_model_best_fit(model_path: str, device: str = "cpu") -> torch.nn.Module:
    state = torch.load(model_path, map_location="cpu")
    state_dict = state if isinstance(state, dict) and "state_dict" not in state else state.get("state_dict", state)
    candidates = [8, 16, 24, 32, 48, 64, 96, 128, None]
    best_score, best_model = None, None
    for bc in candidates:
        try:
            model = try_instantiate_unet(bc)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            score = len(missing) + len(unexpected)
            if best_score is None or score < best_score:
                best_score, best_model = score, model
            if score == 0:
                break
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError(f"Keine kompatible UNetCustom-Variante für: {os.path.basename(model_path)}")
    return best_model.to(device).eval()

# ================================
# Inferenz für ein Modell
# ================================
@torch.no_grad()
def run_inference_with_model(
    model: torch.nn.Module,
    *,
    S_db: np.ndarray,
    S_db01: np.ndarray,
    db_min: float,
    db_max: float,
    S_noisy_mag: np.ndarray,
    S_noisy_phase: np.ndarray,
    hop_length: int,
    win_length: int,
    window: str,
    y_noisy: np.ndarray,
    sr: int,
    data_prep_method: str = DATA_PREP_METHOD,
) -> Dict[str, np.ndarray]:
    device = next(model.parameters()).device

    if data_prep_method == "numeric":
        S_db01_pad, orig_hw = pad_to_multiple_reflect(S_db01, DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR)
        x = torch.from_numpy(S_db01_pad[None, None, :, :]).to(device)
        y_hat01_pad = to_img01_space(model(to_model_space(x))).clamp(0, 1)
        S_denoised01 = crop_to_original(y_hat01_pad[0, 0].cpu().numpy(), orig_hw)
        S_denoised_db_full = S_denoised01 * (db_max - db_min) + db_min
    else:
        S_db01_wo_dc = S_db01[1:, :].astype(np.float32)
        den01_wo_dc  = tiled_image_inference(model, S_db01_wo_dc, device)
        den_db_wo_dc = den01_wo_dc * (db_max - db_min) + db_min
        S_denoised_db_full = np.vstack([S_db[0:1, :], den_db_wo_dc])

    # Rücktransformation linear
    ref_val = float(np.max(S_noisy_mag))
    S_denoised_mag = librosa.db_to_amplitude(S_denoised_db_full, ref=ref_val)

    # iSTFT
    S_denoised_complex = S_denoised_mag * np.exp(1j * S_noisy_phase)
    y_denoised = librosa.istft(
        S_denoised_complex, hop_length=hop_length, win_length=win_length, window=window, length=len(y_noisy)
    )

    return {
        "S_denoised_db_full": S_denoised_db_full,
        "y_denoised": y_denoised,
    }

# ================================
# Multi Model Pipeline
# ================================
def denoise_audio(
    noisy_audio_path: str,
    model_paths: List[str],
    output_audio_path: str,
    noisy_spec_png: str,
    denoised_spec_png: str,
    waveforms_png: str,
    waveforms_sub_png: str,
    n_fft: int = N_FFT,
    hop_length: int = HOP,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
    sr_target: Optional[int] = SR_TARGET,
    device: str = DEVICE,
):
    denoise_output_txt = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_output.txt")
    logger = setup_logging(RESULTS_DIR, log_file=denoise_output_txt, level="INFO")
    Path(os.path.dirname(output_audio_path)).mkdir(parents=True, exist_ok=True)

    # --- 1) Audio & STFT einlesen ---
    y_noisy, sr = librosa.load(noisy_audio_path, sr=sr_target)
    duration_s = len(y_noisy) / sr
    logger.info(f"Audio: {os.path.basename(noisy_audio_path)} | sr={sr} Hz | Dauer={duration_s:.3f}s | N={len(y_noisy)}")

    S_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    S_noisy_mag   = np.abs(S_noisy)
    S_noisy_phase = np.angle(S_noisy)
    S_db          = librosa.amplitude_to_db(S_noisy_mag, ref=np.max)

    db_min = float(S_db.min())
    db_max = float(S_db.max())
    S_db01 = ((S_db - db_min) / (db_max - db_min + 1e-12)).astype(np.float32)

    logger.info(f"STFT: n_fft={n_fft}, hop={hop_length}, win={win_length}, window={window} | S_db shape={S_db.shape}")
    logger.info(f"Epsilon für Nullnähe-Anteil: {ZERO_EPS}")

    # --- 2) Modelle testen ---
    results: List[Dict] = []

    # Wenn MULTI_MODEL_SELECTION False ist, wähle das erste vorhandene Modell als einzige Kandidat
    if not MULTI_MODEL_SELECTION:
        logger.info("[MODE] Single-model mode aktiviert. Benutze das erste gefundene Modell aus MODEL_LIST.")
        chosen_model_path = None
        for mp in model_paths:
            if os.path.isfile(mp):
                chosen_model_path = mp
                break
        if chosen_model_path is None:
            raise RuntimeError("Kein Modell gefunden in MODEL_LIST.")
        try:
            model = load_model_best_fit(chosen_model_path, device=device)
            logger.info(f"[LOAD] Modell: {os.path.basename(chosen_model_path)}")
            out = run_inference_with_model(
                model,
                S_db=S_db,
                S_db01=S_db01,
                db_min=db_min,
                db_max=db_max,
                S_noisy_mag=S_noisy_mag,
                S_noisy_phase=S_noisy_phase,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                y_noisy=y_noisy,
                sr=sr,
                data_prep_method=DATA_PREP_METHOD,
            )
            results.append({
                "model_path": chosen_model_path,
                "y_denoised": out["y_denoised"],
                "S_denoised_db_full": out["S_denoised_db_full"],
                "score_frac0": zero_fraction(out["y_denoised"], eps=ZERO_EPS),
            })
        except Exception as e:
            logger.exception(f"[ERR] Fehler bei Single-Model-Inferenz: {e}")
            raise
    else:
        # Multi-Model Modus: alle existierenden Modelle probieren und vergleichen
        logger.info("[MODE] Multi-model selection aktiviert. Probiere alle Modelle in MODEL_LIST.")
        for mp in model_paths:
            if not os.path.isfile(mp):
                logger.warning(f"[SKIP] Modelldatei nicht gefunden: {mp}")
                continue
            try:
                model = load_model_best_fit(mp, device=device)
                logger.info(f"[TEST] Modell: {os.path.basename(mp)}")
                out = run_inference_with_model(
                    model,
                    S_db=S_db,
                    S_db01=S_db01,
                    db_min=db_min,
                    db_max=db_max,
                    S_noisy_mag=S_noisy_mag,
                    S_noisy_phase=S_noisy_phase,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    y_noisy=y_noisy,
                    sr=sr,
                    data_prep_method=DATA_PREP_METHOD,
                )
                y_den = out["y_denoised"]
                frac0 = zero_fraction(y_den, eps=ZERO_EPS)
                logger.info(f"[METRIC] {os.path.basename(mp)} => frac(|y|<{ZERO_EPS:.3f}) = {frac0:.4f}")

                results.append({
                    "model_path": mp,
                    "y_denoised": y_den,
                    "S_denoised_db_full": out["S_denoised_db_full"],
                    "score_frac0": frac0,
                })
            except Exception as e:
                logger.exception(f"[FAIL] Fehler bei Modell {os.path.basename(mp)}: {e}")

    if not results:
        raise RuntimeError("Keine Ergebnisse von Modellen erhalten. Bitte MODEL_LIST prüfen.")

    # --- 3) Auswahl: bestes Modell (max Nullnähe-Anteil) ---
    best = max(results, key=lambda r: r["score_frac0"])
    best_name = os.path.basename(best["model_path"])
    logger.info(f"[BEST] Modell: {best_name} | frac(|y|<{ZERO_EPS:.3f}) = {best['score_frac0']:.4f}")

    # --- 4) Speichern und Plots für bestes Ergebnis ---
    sf.write(output_audio_path, best["y_denoised"].astype(np.float32), sr)
    plot_spectrogram_db(S_db, sr, hop_length, noisy_spec_png,    title=f"Noisy Spektrogramm\n{os.path.basename(noisy_audio_path)}")
    plot_spectrogram_db(best["S_denoised_db_full"], sr, hop_length, denoised_spec_png, title=f"Denoised Spektrogramm\nModell: {best_name}")
    # save_waveform(y_noisy, best["y_denoised"], sr, waveforms_png, label_a="Noisy", label_b="Denoised")
    save_waveform_subplot(y_noisy, best["y_denoised"], sr, waveforms_sub_png, label_a="Noisy", label_b="Denoised")


    logger.info(f"[OK] Denoised WAV:      {output_audio_path}")
    logger.info(f"[OK] Noisy Spec PNG:    {noisy_spec_png}")
    logger.info(f"[OK] Denoised Spec PNG: {denoised_spec_png}")
    logger.info(f"[OK] Waveforms PNG:     {waveforms_png}")

# ================================
# Main
# ================================
if __name__ == "__main__":
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    denoise_audio(
        noisy_audio_path=INPUT_AUDIO_PATH,
        model_paths=MODEL_LIST,
        output_audio_path=OUTPUT_AUDIO_PATH,
        noisy_spec_png=NOISY_SPEC_PNG,
        denoised_spec_png=DENOISED_SPEC_PNG,
        waveforms_png=WAVEFORMS_PNG,
        waveforms_sub_png=WAVEFORMS_SUB_PNG,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN_LENGTH,
        window=WINDOW,
        sr_target=SR_TARGET,
        device=DEVICE,
    )












# # ########################## mit volume normalisierung durch peak
# import os
# from pathlib import Path
# from typing import Optional, Tuple, List, Dict
# import numpy as np
# import torch
# import librosa
# import librosa.display
# import soundfile as sf
# import matplotlib.pyplot as plt

# from createModelUnet import UNetCustom
# from trainLogging import setup_logging

# # ================================
# # KONFIGURATION
# # ================================
# # --- Eingabe / Ausgabe Pfade ---
# INPUT_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audios")
# INPUT_AUDIO_NAME = "schach_noisy.wav"
# INPUT_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, INPUT_AUDIO_NAME)

# RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_denoising")
# Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# OUTPUT_AUDIO_NAME = "schach_denoised.wav"
# OUTPUT_AUDIO_PATH = os.path.join(RESULTS_DIR, OUTPUT_AUDIO_NAME)

# NOISY_SPEC_PNG    = os.path.join(RESULTS_DIR, f"{os.path.splitext(INPUT_AUDIO_NAME)[0]}_spec.png")
# DENOISED_SPEC_PNG = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_spec.png")
# WAVEFORMS_PNG     = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_waveforms.png")
# WAVEFORMS_SUB_PNG = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_waveforms_sub.png")

# # --- Modelle (Liste) ---
# MULTI_MODEL_SELECTION = False   # True = mehrere Modelle testen, False = einzelnes Modell nutzen
# ZERO_EPS = 1e-2                 # Epsilon für Nullnähe-Anteil

# # Falls MULTI_MODEL_SELECTION=False wird nur das erste existierende Modell genutzt.
# MODEL_LIST: List[str] = [

#         os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_16_full_data_aug_60_epochen_step_20.pth"),
#         os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8_full_data_aug_60_epochen.pth"),
#         os.path.join(RESULTS_DIR, "best_model_baseCh_64_batch_8_full_data_aug_60_epochen.pth"),
#         os.path.join(RESULTS_DIR, "best_model_baseCh_64_batch_8_full_data_aug_60_epochen_step_15_aug_low.pth"),
#         os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_16_full_data_aug_60_epochen.pth"),
#         os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8_full_data_aug_60_epochen_val_01.pth"),
#         os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8_60_epochen.pth"),
#         os.path.join(RESULTS_DIR, "best_model_baseCh_32_batch_8_patience_20.pth"),

# ]

# # --- STFT Parameter ---
# SR_TARGET    = None
# N_FFT        = 1024
# HOP          = 256
# WIN_LENGTH   = 1024
# WINDOW       = "hann"

# # --- Datenvorbereitung ---
# DATA_PREP_METHOD = "numeric"   # "numeric" oder "image_tiled"

# # für numeric
# DOWNSAMPLE_FACTOR = 32

# # für image_tiled
# TILE_W = 256
# STRIDE = 128
# DPI = 100

# # --- Device ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # --- Auswahl/Normalisierung Parameter ---
# NORMALIZE_METHOD = "none"      # "peak" | "hard" | "none"
# PEAK_TARGET = 0.25
# HARD_GAIN_FACTOR = 0.5


# # ================================
# # Utilities (numeric)
# # ================================
# def pad_to_multiple_reflect(x: np.ndarray, mult_h: int, mult_w: int) -> Tuple[np.ndarray, Tuple[int,int]]:
#     H, W = x.shape
#     target_H = int(np.ceil(H / mult_h) * mult_h)
#     target_W = int(np.ceil(W / mult_w) * mult_w)
#     pad_h = target_H - H
#     pad_w = target_W - W
#     if pad_h == 0 and pad_w == 0:
#         return x, (H, W)

#     def _reflect_pad_1d(vec, pad):
#         if pad <= 0: return vec
#         if len(vec) < 2: return np.pad(vec, (0, pad), mode="edge")
#         ref = vec[1:-1][::-1]
#         reps = int(np.ceil(pad / len(ref))) if len(ref) > 0 else pad
#         pad_block = np.tile(ref, reps)[:pad] if len(ref) > 0 else np.repeat(vec[-1:], pad)
#         return np.concatenate([vec, pad_block])

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
#     return x_pad.astype(np.float32), (H, W)

# def crop_to_original(x_pad: np.ndarray, orig_hw: Tuple[int,int]) -> np.ndarray:
#     H, W = orig_hw
#     return x_pad[:H, :W]

# def to_model_space(img01: torch.Tensor) -> torch.Tensor:
#     return img01 * 2.0 - 1.0

# def to_img01_space(img_norm: torch.Tensor) -> torch.Tensor:
#     return (img_norm + 1.0) * 0.5

# # ================================
# # Utilities (image tiled)
# # ================================
# @torch.no_grad()
# def tiled_image_inference(model: torch.nn.Module, img01: np.ndarray, device: str) -> np.ndarray:
#     model.eval()
#     H, W = img01.shape
#     if W < TILE_W:
#         img01 = np.pad(img01, ((0, 0), (0, TILE_W - W)), mode="constant")
#     H, W_pad = img01.shape

#     win = torch.from_numpy(np.hanning(TILE_W).astype(np.float32)).to(device).view(1, 1, 1, -1)
#     out_sum = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)
#     w_sum  = torch.zeros((1, 1, H, W_pad), dtype=torch.float32, device=device)

#     for x0 in range(0, W_pad - TILE_W + 1, STRIDE):
#         tile = img01[:, x0:x0+TILE_W]
#         tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
#         pred = model(to_model_space(tile_t))
#         pred01 = to_img01_space(pred).clamp(0.0, 1.0)
#         out_sum[..., x0:x0+TILE_W] += pred01 * win
#         w_sum[...,  x0:x0+TILE_W] += win

#     if (W_pad - TILE_W) % STRIDE != 0:
#         x0 = W_pad - TILE_W
#         tile = img01[:, x0:x0+TILE_W]
#         tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
#         pred = model(to_model_space(tile_t))
#         pred01 = to_img01_space(pred).clamp(0.0, 1.0)
#         out_sum[..., x0:x0+TILE_W] += pred01 * win
#         w_sum[...,  x0:x0+TILE_W] += win

#     out01 = (out_sum / (w_sum + 1e-8)).squeeze().detach().cpu().numpy()
#     return out01[:, :W]

# # ================================
# # Plots
# # ================================
# def plot_spectrogram_db(S_db: np.ndarray, sr: int, hop_length: int, out_path: str, title: str = ""):
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear")
#     plt.colorbar(label="Amplitude [dB]")
#     if title: plt.title(title)
#     plt.xlabel("Zeit [s]"); plt.ylabel("Frequenz [Hz]")
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()

# def save_waveform(y_a, y_b, sr, out_path, label_a="Noisy", label_b="Denoised"):
#     plt.figure(figsize=(15, 5))
#     librosa.display.waveshow(y_a, sr=sr, alpha=0.6, label=label_a, color='blue')
#     librosa.display.waveshow(y_b, sr=sr, alpha=0.6, label=label_b, color='red')
#     plt.title('Wellenform'); plt.xlabel('Zeit [s]'); plt.ylabel('Amplitude')
#     plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()

# def save_waveform_subplot(y_a, y_b, sr, out_path, label_a="Noisy", label_b="Denoised"):
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

#     # Obere Grafik: Erste Wellenform
#     librosa.display.waveshow(y_a, sr=sr, alpha=0.8, label=label_a, color='blue', ax=ax1)
#     ax1.set_title(f'Wellenform - {label_a}')
#     ax1.set_xlabel('Zeit [s]')
#     ax1.set_ylabel('Amplitude')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)

#     # Untere Grafik: Zweite Wellenform
#     librosa.display.waveshow(y_b, sr=sr, alpha=0.8, label=label_b, color='blue', ax=ax2)
#     ax2.set_title(f'Wellenform - {label_b}')
#     ax2.set_xlabel('Zeit [s]')
#     ax2.set_ylabel('Amplitude')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=120)
#     plt.close()

# # ================================
# # Auswahl Metrik
# # ================================
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

# def zero_fraction(y: np.ndarray, eps: float = ZERO_EPS) -> float:
#     """Anteil der Samples mit |y| < eps (größer ist besser)."""
#     y = np.asarray(y, dtype=np.float32)
#     return float(np.mean(np.abs(y) < eps))

# # ================================
# # Model Loading
# # ================================
# def try_instantiate_unet(base_channels: Optional[int]) -> UNetCustom:
#     import inspect
#     sig = inspect.signature(UNetCustom)
#     if "base_channels" in sig.parameters and base_channels is not None:
#         return UNetCustom(base_channels=base_channels)
#     return UNetCustom()

# def load_model_best_fit(model_path: str, device: str = "cpu") -> torch.nn.Module:
#     state = torch.load(model_path, map_location="cpu")
#     state_dict = state if isinstance(state, dict) and "state_dict" not in state else state.get("state_dict", state)
#     candidates = [8, 16, 24, 32, 48, 64, 96, 128, None]
#     best_score, best_model = None, None
#     for bc in candidates:
#         try:
#             model = try_instantiate_unet(bc)
#             missing, unexpected = model.load_state_dict(state_dict, strict=False)
#             score = len(missing) + len(unexpected)
#             if best_score is None or score < best_score:
#                 best_score, best_model = score, model
#             if score == 0:
#                 break
#         except Exception:
#             continue
#     if best_model is None:
#         raise RuntimeError(f"Keine kompatible UNetCustom-Variante für: {os.path.basename(model_path)}")
#     return best_model.to(device).eval()

# # ================================
# # Inferenz für ein Modell
# # ================================
# @torch.no_grad()
# def run_inference_with_model(
#     model: torch.nn.Module,
#     *,
#     S_db: np.ndarray,
#     S_db01: np.ndarray,
#     db_min: float,
#     db_max: float,
#     S_noisy_mag: np.ndarray,
#     S_noisy_phase: np.ndarray,
#     hop_length: int,
#     win_length: int,
#     window: str,
#     y_noisy: np.ndarray,
#     sr: int,
#     data_prep_method: str = DATA_PREP_METHOD,
# ) -> Dict[str, np.ndarray]:
#     device = next(model.parameters()).device

#     if data_prep_method == "numeric":
#         S_db01_pad, orig_hw = pad_to_multiple_reflect(S_db01, DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR)
#         x = torch.from_numpy(S_db01_pad[None, None, :, :]).to(device)
#         y_hat01_pad = to_img01_space(model(to_model_space(x))).clamp(0, 1)
#         S_denoised01 = crop_to_original(y_hat01_pad[0, 0].cpu().numpy(), orig_hw)
#         S_denoised_db_full = S_denoised01 * (db_max - db_min) + db_min
#     else:
#         S_db01_wo_dc = S_db01[1:, :].astype(np.float32)
#         den01_wo_dc  = tiled_image_inference(model, S_db01_wo_dc, device)
#         den_db_wo_dc = den01_wo_dc * (db_max - db_min) + db_min
#         S_denoised_db_full = np.vstack([S_db[0:1, :], den_db_wo_dc])

#     # Rücktransformation linear
#     ref_val = float(np.max(S_noisy_mag))
#     S_denoised_mag = librosa.db_to_amplitude(S_denoised_db_full, ref=ref_val)

#     # iSTFT
#     S_denoised_complex = S_denoised_mag * np.exp(1j * S_noisy_phase)
#     y_denoised_raw = librosa.istft(
#         S_denoised_complex, hop_length=hop_length, win_length=win_length, window=window, length=len(y_noisy)
#     )

#     # finale Pegelanpassung (für Vergleichbarkeit)
#     y_out = apply_output_level(y_denoised_raw, method=NORMALIZE_METHOD, peak_target=PEAK_TARGET, hard_gain=HARD_GAIN_FACTOR)

#     return {
#         "S_denoised_db_full": S_denoised_db_full,
#         "y_denoised": y_out.astype(np.float32),
#     }

# # ================================
# # Multi Model Pipeline
# # ================================
# def denoise_audio(
#     noisy_audio_path: str,
#     model_paths: List[str],
#     output_audio_path: str,
#     noisy_spec_png: str,
#     denoised_spec_png: str,
#     waveforms_png: str,
#     waveforms_sub_png: str,
#     n_fft: int = N_FFT,
#     hop_length: int = HOP,
#     win_length: int = WIN_LENGTH,
#     window: str = WINDOW,
#     sr_target: Optional[int] = SR_TARGET,
#     device: str = DEVICE,
# ):
#     denoise_output_txt = os.path.join(RESULTS_DIR, f"{os.path.splitext(OUTPUT_AUDIO_NAME)[0]}_output.txt")
#     logger = setup_logging(RESULTS_DIR, log_file=denoise_output_txt, level="INFO")
#     Path(os.path.dirname(output_audio_path)).mkdir(parents=True, exist_ok=True)

#     # --- 1) Audio & STFT einlesen ---
#     y_noisy, sr = librosa.load(noisy_audio_path, sr=sr_target)
#     duration_s = len(y_noisy) / sr
#     logger.info(f"Audio: {os.path.basename(noisy_audio_path)} | sr={sr} Hz | Dauer={duration_s:.3f}s | N={len(y_noisy)}")

#     S_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
#     S_noisy_mag   = np.abs(S_noisy)
#     S_noisy_phase = np.angle(S_noisy)
#     S_db          = librosa.amplitude_to_db(S_noisy_mag, ref=np.max)

#     db_min = float(S_db.min())
#     db_max = float(S_db.max())
#     S_db01 = ((S_db - db_min) / (db_max - db_min + 1e-12)).astype(np.float32)

#     logger.info(f"STFT: n_fft={n_fft}, hop={hop_length}, win={win_length}, window={window} | S_db shape={S_db.shape}")
#     logger.info(f"Epsilon für Nullnähe-Anteil: {ZERO_EPS}")

#     # --- 2) Modelle testen ---
#     results: List[Dict] = []

#     # Wenn MULTI_MODEL_SELECTION False ist, wähle das erste vorhandene Modell als einzige Kandidat
#     if not MULTI_MODEL_SELECTION:
#         logger.info("[MODE] Single-model mode aktiviert. Benutze das erste gefundene Modell aus MODEL_LIST.")
#         chosen_model_path = None
#         for mp in model_paths:
#             if os.path.isfile(mp):
#                 chosen_model_path = mp
#                 break
#         if chosen_model_path is None:
#             raise RuntimeError("Kein Modell gefunden in MODEL_LIST.")
#         try:
#             model = load_model_best_fit(chosen_model_path, device=device)
#             logger.info(f"[LOAD] Modell: {os.path.basename(chosen_model_path)}")
#             out = run_inference_with_model(
#                 model,
#                 S_db=S_db,
#                 S_db01=S_db01,
#                 db_min=db_min,
#                 db_max=db_max,
#                 S_noisy_mag=S_noisy_mag,
#                 S_noisy_phase=S_noisy_phase,
#                 hop_length=hop_length,
#                 win_length=win_length,
#                 window=window,
#                 y_noisy=y_noisy,
#                 sr=sr,
#                 data_prep_method=DATA_PREP_METHOD,
#             )
#             results.append({
#                 "model_path": chosen_model_path,
#                 "y_denoised": out["y_denoised"],
#                 "S_denoised_db_full": out["S_denoised_db_full"],
#                 "score_frac0": zero_fraction(out["y_denoised"], eps=ZERO_EPS),
#             })
#         except Exception as e:
#             logger.exception(f"[ERR] Fehler bei Single-Model-Inferenz: {e}")
#             raise
#     else:
#         # Multi-Model Modus: alle existierenden Modelle probieren und vergleichen
#         logger.info("[MODE] Multi-model selection aktiviert. Probiere alle Modelle in MODEL_LIST.")
#         for mp in model_paths:
#             if not os.path.isfile(mp):
#                 logger.warning(f"[SKIP] Modelldatei nicht gefunden: {mp}")
#                 continue
#             try:
#                 model = load_model_best_fit(mp, device=device)
#                 logger.info(f"[TEST] Modell: {os.path.basename(mp)}")
#                 out = run_inference_with_model(
#                     model,
#                     S_db=S_db,
#                     S_db01=S_db01,
#                     db_min=db_min,
#                     db_max=db_max,
#                     S_noisy_mag=S_noisy_mag,
#                     S_noisy_phase=S_noisy_phase,
#                     hop_length=hop_length,
#                     win_length=win_length,
#                     window=window,
#                     y_noisy=y_noisy,
#                     sr=sr,
#                     data_prep_method=DATA_PREP_METHOD,
#                 )
#                 y_den = out["y_denoised"]
#                 frac0 = zero_fraction(y_den, eps=ZERO_EPS)
#                 logger.info(f"[METRIC] {os.path.basename(mp)} => frac(|y|<{ZERO_EPS:.3f}) = {frac0:.4f}")

#                 results.append({
#                     "model_path": mp,
#                     "y_denoised": y_den,
#                     "S_denoised_db_full": out["S_denoised_db_full"],
#                     "score_frac0": frac0,
#                 })
#             except Exception as e:
#                 logger.exception(f"[FAIL] Fehler bei Modell {os.path.basename(mp)}: {e}")

#     if not results:
#         raise RuntimeError("Keine Ergebnisse von Modellen erhalten. Bitte MODEL_LIST prüfen.")

#     # --- 3) Auswahl: bestes Modell (max Nullnähe-Anteil) ---
#     best = max(results, key=lambda r: r["score_frac0"])
#     best_name = os.path.basename(best["model_path"])
#     logger.info(f"[BEST] Modell: {best_name} | frac(|y|<{ZERO_EPS:.3f}) = {best['score_frac0']:.4f}")

#     # --- 4) Speichern und Plots für bestes Ergebnis ---
#     sf.write(output_audio_path, best["y_denoised"].astype(np.float32), sr)
#     plot_spectrogram_db(S_db, sr, hop_length, noisy_spec_png,    title=f"Noisy Spektrogramm\n{os.path.basename(noisy_audio_path)}")
#     plot_spectrogram_db(best["S_denoised_db_full"], sr, hop_length, denoised_spec_png, title=f"Denoised Spektrogramm\nModell: {best_name}")
#     # save_waveform(y_noisy, best["y_denoised"], sr, waveforms_png, label_a="Noisy", label_b="Denoised")
#     save_waveform_subplot(y_noisy, best["y_denoised"], sr, waveforms_sub_png, label_a="Noisy", label_b="Denoised")


#     logger.info(f"[OK] Denoised WAV:      {output_audio_path}")
#     logger.info(f"[OK] Noisy Spec PNG:    {noisy_spec_png}")
#     logger.info(f"[OK] Denoised Spec PNG: {denoised_spec_png}")
#     logger.info(f"[OK] Waveforms PNG:     {waveforms_png}")

# # ================================
# # Main
# # ================================
# if __name__ == "__main__":
#     Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
#     denoise_audio(
#         noisy_audio_path=INPUT_AUDIO_PATH,
#         model_paths=MODEL_LIST,
#         output_audio_path=OUTPUT_AUDIO_PATH,
#         noisy_spec_png=NOISY_SPEC_PNG,
#         denoised_spec_png=DENOISED_SPEC_PNG,
#         waveforms_png=WAVEFORMS_PNG,
#         waveforms_sub_png=WAVEFORMS_SUB_PNG,
#         n_fft=N_FFT,
#         hop_length=HOP,
#         win_length=WIN_LENGTH,
#         window=WINDOW,
#         sr_target=SR_TARGET,
#         device=DEVICE,
#     )
