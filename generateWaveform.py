import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# --- Konfigurationen ---
SR_TARGET = None
FIGSIZE = (15, 5)
DPI = 120

# --- Pfade ---
BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, "results_denoising")
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audios")
NOISY_AUDIO_NAME = "schach_noisy.wav"
NOISY_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, NOISY_AUDIO_NAME)

DENOISED_AUDIO_NAME = "schach_denoised_64.wav"
DENOISED_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, DENOISED_AUDIO_NAME)

DENOISED_AUDIO2_NAME = "schach_denoised_32.wav"
DENOISED_AUDIO2_PATH = os.path.join(INPUT_AUDIO_DIR, DENOISED_AUDIO2_NAME)

CLEAN_AUDIO_NAME = "schach_clean.wav"
CLEAN_AUDIO_PATH = os.path.join(INPUT_AUDIO_DIR, CLEAN_AUDIO_NAME)

WAVEFORMS_PNG     = os.path.join(RESULTS_DIR, f"{os.path.splitext(CLEAN_AUDIO_NAME)[0]}_denoised_waveforms.png")
WAVEFORMS_SUB_PNG = os.path.join(RESULTS_DIR, f"{os.path.splitext(CLEAN_AUDIO_NAME)[0]}_denoised_waveforms_sub.png")
WAVEFORMS_3_SUB_PNG = os.path.join(RESULTS_DIR, f"{os.path.splitext(NOISY_AUDIO_NAME)[0]}_denoised_clean_waveforms_sub.png")
WAVEFORMS_4_SUB_PNG = os.path.join(RESULTS_DIR, f"{os.path.splitext(NOISY_AUDIO_NAME)[0]}_2denoised_clean_waveforms_sub.png")


def load_mono(path: str, sr_target: int | None):
    """Lädt ein Audio."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    return y, sr



def save_waveforms(y_a: np.ndarray,
                   y_b: np.ndarray,
                   sr: int,
                   label_a: str = "Clean",
                   label_b: str = "Denoised",
                   out_path: str = "waveforms.png"):
    """Plottet Wellenformen und speichert sie."""

    plt.figure(figsize=FIGSIZE)
    # Wellenformen gemeinsam
    librosa.display.waveshow(y_a, sr=sr, alpha=0.8, label=label_a, color='blue')
    # librosa.display.waveshow(y_b, sr=sr, alpha=0.6, label=label_b, color='red')

    plt.title('Wellenform')
    plt.xlabel('Zeit [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()

def save_waveform_subplot(y_a, y_b, sr, out_path, label_a="Clean", label_b="Denoised"):
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

def save_three_waveform_subplot(y_a, y_b, y_c, sr, out_path, label_a="Noisy", label_b="Denoised", label_c="Clean"):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))

    # Erste Wellenform
    librosa.display.waveshow(y_a, sr=sr, alpha=0.8, label=label_a, color='blue', ax=ax1)
    ax1.set_title(f'Wellenform - {label_a}')
    ax1.set_xlabel('Zeit [s]')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Zweite Wellenform
    librosa.display.waveshow(y_b, sr=sr, alpha=0.8, label=label_b, color='blue', ax=ax2)
    ax2.set_title(f'Wellenform - {label_b}')
    ax2.set_xlabel('Zeit [s]')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Dritte Wellenform
    librosa.display.waveshow(y_c, sr=sr, alpha=0.8, label=label_c, color='blue', ax=ax3)
    ax3.set_title(f'Wellenform - {label_c}')
    ax3.set_xlabel('Zeit [s]')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def save_four_waveform_subplot(y_a, y_b, y_c, y_d, sr, out_path, label_a="Noisy", label_b="Denoised_1", label_c="Denoised_2", label_d="Clean"):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))

    # Erste Wellenform
    librosa.display.waveshow(y_a, sr=sr, alpha=0.8, label=label_a, color='blue', ax=ax1)
    ax1.set_title(f'Wellenform - {label_a}')
    ax1.set_xlabel('Zeit [s]')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Zweite Wellenform
    librosa.display.waveshow(y_b, sr=sr, alpha=0.8, label=label_b, color='blue', ax=ax2)
    ax2.set_title(f'Wellenform - Denoised_baseCh64_Batch8')
    ax2.set_xlabel('Zeit [s]')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Dritte Wellenform
    librosa.display.waveshow(y_c, sr=sr, alpha=0.8, label=label_c, color='blue', ax=ax3)
    ax3.set_title(f'Wellenform - Denoised_baseCh32_Batch16')
    ax3.set_xlabel('Zeit [s]')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Viertes Wellenform
    librosa.display.waveshow(y_d, sr=sr, alpha=0.8, label=label_d, color='blue', ax=ax4)
    ax4.set_title(f'Wellenform - {label_d}')
    ax4.set_xlabel('Zeit [s]')
    ax4.set_ylabel('Amplitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    # Audios laden
    y_noisy, sr_noisy = load_mono(NOISY_AUDIO_PATH, SR_TARGET)
    y_denoised, sr_denoised = load_mono(DENOISED_AUDIO_PATH, SR_TARGET)
    y_denoised1, sr_denoised = load_mono(DENOISED_AUDIO_PATH, SR_TARGET)
    y_denoised2, sr_denoised = load_mono(DENOISED_AUDIO2_PATH, SR_TARGET)
    y_clean, sr_clean = load_mono(CLEAN_AUDIO_PATH, SR_TARGET)

    if sr_clean != sr_denoised:
        print(f"Warnung: unterschiedliche SR (clean={sr_clean}, denoised={sr_denoised}). "
              f"Zeitachse wird mit {sr_clean} Hz erstellt.")

    # Plot speichern
    # save_waveforms(y_a=y_clean, y_b=y_denoised, sr=sr_clean, label_a="Clean", label_b="Denoised", out_path=WAVEFORMS_PNG)
    # save_waveform_subplot(y_clean, y_denoised, sr_clean, WAVEFORMS_SUB_PNG, label_a="Clean", label_b="Denoised")
    # save_three_waveform_subplot(y_noisy, y_denoised, y_clean, sr_clean, WAVEFORMS_3_SUB_PNG, label_a="Noisy", label_b="Denoised", label_c="Clean")
    save_four_waveform_subplot(y_noisy, y_denoised1, y_denoised2, y_clean, sr_clean, WAVEFORMS_3_SUB_PNG, label_a="Noisy", label_b="Denoised_1", label_c="Denoised_2",label_d="Clean")

    print(f"Wellenformen gespeichert unter: {WAVEFORMS_PNG}")


if __name__ == "__main__":
    main()
