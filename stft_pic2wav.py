import numpy as np
from PIL import Image
import scipy.signal
import soundfile as sf
import pyloudnorm as pyln

def spectrogram_png_to_wav_scipy(png_path, output_wav_path,
                                  sr=16000, n_fft=1024, hop_length=256,
                                  n_iter=50, normalize_loudness=True, target_lufs=-23):
    """
    Konvertiert ein STFT-Spektrogramm (PNG-Bild) zurück zu WAV mit scipy.signal.istft und optionaler Lautheitsnormalisierung.
    """
    # 1. PNG-Bild laden (grau, normalisiert [0, 1])
    img = Image.open(png_path).convert('L')
    img_np = np.array(img).astype(np.float32) / 255.0

    # 2. Zurückskalieren von dB → Amplitude
    db = img_np * 80.0 - 80.0  # Annahme: gespeichert zwischen -80 dB und 0 dB
    magnitude = 10 ** (db / 20.0)

    # 3. Initialphase zufällig
    np.random.seed(0)
    phase = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    stft_matrix = magnitude * phase

    # 4. Griffin-Lim Iterationen (mit scipy.signal.istft)
    for i in range(n_iter):
        _, signal = scipy.signal.istft(stft_matrix, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        _, _, new_stft = scipy.signal.stft(signal, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        phase = np.exp(1j * np.angle(new_stft))
        stft_matrix = magnitude * phase

    # 5. Letzte Rekonstruktion
    _, audio = scipy.signal.istft(stft_matrix, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # 6. (Optional) Lautheitsnormalisierung auf -23 LUFS
    if normalize_loudness:
        meter = pyln.Meter(sr)  # EBU R128 Meter
        loudness = meter.integrated_loudness(audio)
        audio = pyln.normalize.loudness(audio, loudness, target_lufs)

    # 7. Speichern als WAV
    sf.write(output_wav_path, audio, sr)
    print(f"✓ WAV gespeichert: {output_wav_path}")

if __name__ == "__main__":
    spectrogram_png_to_wav_scipy(
        png_path="Dataset/noisy_testset_stft/sample001.png",
        output_wav_path="reconstructed.wav",
        sr=16000,
        normalize_loudness=True,
        target_lufs=-23
    )
