import os
import time
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


MEL_MODE = False # True: Mel, False: STFT

DPI = 100  # Pixel = figsize * DPI

# Mel-Parameter
MEL_N_MELS = 128
MEL_N_FFT = 1024
MEL_HOP = 256

# STFT-Parameter
STFT_N_FFT = 1024
STFT_HOP = 256
STFT_WIN = 1024

def save_spectrogram_image(S_dB, sr, output_path, dpi=100):
    """Speichert das Spektrogramm pixelgenau (H x B) ohne Achsen/Rand."""
    H, B = S_dB.shape  # H = Frequenz-Bins, B = Zeit-Frames
    fig_w = B / dpi
    fig_h = H / dpi
    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def wav_to_mel_spectrogram(
    wav_path, output_path, n_mels=128, n_fft=1024, hop_length=256, dpi=100
):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    save_spectrogram_image(S_dB, sr, output_path, dpi=dpi)

def wav_to_stft_spectrogram(
    wav_path, output_path, n_fft=1024, hop_length=256, win_length=1024, dpi=100
):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.stft(
        y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    S_mag = np.abs(S)   # MAGNITUDE
    S_dB = librosa.amplitude_to_db(S_mag, ref=np.max)
    save_spectrogram_image(S_dB, sr, output_path, dpi=dpi)

def process_folder(input_dir, output_dir, use_mel=True, mel_params=None, stft_params=None, dataset_root=None):
    os.makedirs(output_dir, exist_ok=True)
    mel_params = mel_params or {}
    stft_params = stft_params or {}

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".wav"):
            continue

        wav_path = os.path.join(input_dir, filename)
        output_file = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_dir, output_file)

        if dataset_root:
            rel_wav = os.path.relpath(wav_path, dataset_root)
            rel_out = os.path.relpath(output_path, dataset_root)
            print(f"{rel_wav} → {rel_out}")
        else:
            print(f"{wav_path} → {output_path}")

        if use_mel:
            wav_to_mel_spectrogram(wav_path, output_path, **mel_params)
        else:
            wav_to_stft_spectrogram(wav_path, output_path, **stft_params)

def build_datasets(suffix):
    """Erzeugt die Input/Output-Ordnerliste mit dem gegebenen Suffix ('mel' oder 'stft')."""
    return [
        {"input": "clean_trainset_56spk_wav", "output": f"clean_trainset_56spk_{suffix}"},
        {"input": "noisy_trainset_56spk_wav", "output": f"noisy_trainset_56spk_{suffix}"},
        {"input": "clean_testset_wav",        "output": f"clean_testset_{suffix}"},
        {"input": "noisy_testset_wav",        "output": f"noisy_testset_{suffix}"},
    ]

if __name__ == "__main__":
    start_time = time.time()
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset"))

    use_mel = MEL_MODE
    suffix = "mel" if use_mel else "stft"
    datasets = build_datasets(suffix)

    mel_params = {
        "n_mels": MEL_N_MELS,
        "n_fft": MEL_N_FFT,
        "hop_length": MEL_HOP,
        "dpi": DPI,
    }
    stft_params = {
        "n_fft": STFT_N_FFT,
        "hop_length": STFT_HOP,
        "win_length": STFT_WIN,
        "dpi": DPI,
    }

    print(f"\nModus: {'Mel' if use_mel else 'STFT'}  |  Ausgabe-Suffix: _{suffix}")
    if use_mel:
        print(f"Mel-Params: n_mels={MEL_N_MELS}, n_fft={MEL_N_FFT}, hop={MEL_HOP}")
    else:
        print(f"STFT-Params: n_fft={STFT_N_FFT}, hop={STFT_HOP}, win={STFT_WIN}")

    for ds in datasets:
        in_dir = os.path.join(DATASET_PATH, ds["input"])
        out_dir = os.path.join(DATASET_PATH, ds["output"])
        print(f"\nVerarbeite Ordner: {ds['input']} → {ds['output']}")
        process_folder(
            input_dir=in_dir,
            output_dir=out_dir,
            use_mel=use_mel,
            mel_params=mel_params,
            stft_params=stft_params,
            dataset_root=DATASET_PATH
        )
        print(f"Fertig mit: {ds['output']}")

    dur = time.time() - start_time
    print(f"\nScript execution completed in {dur:.2f} seconds")
