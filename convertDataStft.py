import os
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# Pfade & Dateinamen
# ---------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_denoising")
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

INPUT_DIR = os.path.join(os.path.dirname(__file__), "audios")
INPUT_NAME  = "p232_336_clean.wav"
INPUT_PATH = os.path.join(INPUT_DIR, INPUT_NAME)

OUTPUT_NAME = f"{os.path.splitext(INPUT_NAME)[0]}_spec.png"
OUTPUT_PATH = os.path.join(RESULTS_DIR, OUTPUT_NAME)

# ---------------------------
# STFT-Parameter
# ---------------------------
SR_TARGET  = None
N_FFT      = 1024
HOP        = 256
WIN_LENGTH = 1024
WINDOW     = "hann"
CENTER     = True

SHOW_INFO = False

# ---------------------------
# Plot-Funktion
# ---------------------------
def plot_spectrogram_db(S_db, sr, hop_length, out_path, title, filename, duration, min_db, max_db):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear")
    plt.colorbar(label="Amplitude [dB]")
    if title:
        plt.title(title)
    plt.xlabel("Zeit [s]")
    plt.ylabel("Frequenz [Hz]")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Audio-Informationen als Text hinzufügen
    if SHOW_INFO:
        info_text = (f'Datei: {filename}\n'
                    f'Dauer: {duration:.2f}s\n'
                    f'Sample-Rate: {sr} Hz\n'
                    f'Min Amplitude: {min_db:.2f}\n'
                    f'Max Amplitude: {max_db:.2f}')

        # Informationen in der oberen rechten Ecke platzieren
        plt.annotate(info_text, xy=(0.98, 0.98), xycoords='axes fraction',
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10, fontfamily='monospace')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# ---------------------------
# Hauptablauf
# ---------------------------
def main():

    # 1) Audio laden
    y, sr = librosa.load(str(INPUT_PATH), sr=SR_TARGET, mono=True)
    duration = len(y) / sr
    filename = os.path.basename(INPUT_PATH)

    # 2) STFT -> Betrag -> dB
    S_complex = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=CENTER,
    )
    S_mag = np.abs(S_complex)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    min_db = float(np.min(S_db))
    max_db = float(np.max(S_db))

    # 3) Plotten
    title = f"Clean Spektrogramm \n{INPUT_NAME}"
    plot_spectrogram_db(S_db, sr, HOP, OUTPUT_PATH, title=title, filename=filename, duration=duration, min_db=min_db, max_db=max_db)

    print(f"Fertig! Spektrogramm gespeichert unter: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
