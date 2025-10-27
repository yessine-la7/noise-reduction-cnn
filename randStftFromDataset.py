import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
import torch


SEED = 32

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

set_global_seed(SEED)

# Pfad zur Dataset
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset"))
NOISY_PATH = os.path.join(DATASET_PATH, "noisy_trainset_56spk_wav")
CLEAN_PATH = os.path.join(DATASET_PATH, "clean_trainset_56spk_wav")

# Parameter
DPI = 100
STFT_N_FFT = 1024
STFT_HOP = 256
STFT_WIN = 1024

def get_wav_files(path):
    """Holt alle WAV-Dateien aus einem Ordner."""
    files = [f for f in os.listdir(path) if f.endswith('.wav')]
    return sorted(files)

def plot_spectrogram_subplot(ax, wav_path, title, n_fft=1024, hop_length=256, win_length=1024):
    """Plottet ein Spektrogramm in einem Subplot mit Achsen."""
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_mag = np.abs(S)
    S_dB = librosa.amplitude_to_db(S_mag, ref=np.max)

    # Spektrogramm darstellen
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length,
                                   x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Zeit [s]', fontsize=8)
    ax.set_ylabel('Frequenz [Hz]', fontsize=8)
    ax.label_outer()

    return img

# Hauptfigur erstellen
fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=DPI)
# fig.suptitle(f'Vergleich: Noisy vs Clean Spektrogramme (3 zufällige Beispiele, Seed={SEED})',
#              fontsize=14, fontweight='bold')

# Dateien sortieren
noisy_files = get_wav_files(NOISY_PATH)
clean_files = get_wav_files(CLEAN_PATH)

if len(noisy_files) != len(clean_files):
    print(f"Warnung: Ungleiche Anzahl an Dateien - Noisy: {len(noisy_files)}, Clean: {len(clean_files)}")

# zufällige Beispiele auswählen
num_examples = min(3, len(noisy_files), len(clean_files))
all_indices = list(range(min(len(noisy_files), len(clean_files))))
selected_indices = random.sample(all_indices, num_examples)

print(f"Gewählte Beispiele: {selected_indices}")

images = []

for i, idx in enumerate(selected_indices):
    # Noisy Spektrogramm (obere Reihe)
    noisy_file = os.path.join(NOISY_PATH, noisy_files[idx])
    noisy_filename = os.path.splitext(noisy_files[idx])[0]
    img_noisy = plot_spectrogram_subplot(axes[0, i], noisy_file, f'Noisy: {noisy_filename}')
    images.append(img_noisy)

    # Clean Spektrogramm (untere Reihe)
    clean_file = os.path.join(CLEAN_PATH, clean_files[idx])
    clean_filename = os.path.splitext(clean_files[idx])[0]
    img_clean = plot_spectrogram_subplot(axes[1, i], clean_file, f'Clean: {clean_filename}')
    images.append(img_clean)

# Farbbalken
plt.tight_layout()
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

if images:
    cbar = fig.colorbar(images[0], cax=cbar_ax, label='Amplitude [dB]')
    cbar.set_label('Amplitude [dB]', fontsize=10)
else:
    print("Warnung: Keine Spektrogramme zum Darstellen gefunden")

plt.tight_layout(rect=[0, 0, 0.9, 0.95])

# Figur speichern
output_path = os.path.join(os.path.dirname(__file__), f'spektrogramme_vergleich_seed_{SEED}.png')
plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
plt.close()

print(f"Figur gespeichert als: {output_path}")
print(f"Gefundene Noisy-Dateien: {len(noisy_files)}")
print(f"Gefundene Clean-Dateien: {len(clean_files)}")
print(f"Dargestellte Beispiele: {num_examples}")
for i, idx in enumerate(selected_indices):
    print(f"Beispiel {i+1} (Index {idx}): {noisy_files[idx]} -> {clean_files[idx]}")