import numpy as np
import librosa
import soundfile as sf
from PIL import Image
from scipy.signal import butter, filtfilt
import pyloudnorm as pyln

def png_to_wav_speech(
    png_path,
    wav_path,
    sr,
    target_lufs,
    apply_bandpass=True,
    apply_preemphasis=True,
    apply_compression=True
):
    """
    Konvertiert PNG-Spektrogramm zu WAV mit professioneller Sprachverarbeitung

    Parameter:
    - png_path: Pfad zum Eingabe-PNG
    - wav_path: Ziel-WAV-Pfad
    - sr: Sample Rate
    - target_lufs: Ziel-Lautheit (-16 LUFS für Podcasts, -23 für Broadcast)
    - apply_*: Aktiviert/Deaktiviert Verarbeitungsschritte
    """

    # 1. PNG laden und kalibrieren
    img = Image.open(png_path).convert('L')
    img_array = np.flipud(np.array(img)) / 255.0

    # 2. dB-Skalierung (Sprachbereich -80dB bis 0dB)
    S_dB = (img_array * 80) - 80
    S = librosa.db_to_power(S_dB)

    # 3. Sprachoptimierte Rekonstruktion
    audio = librosa.feature.inverse.mel_to_audio(
        S,
        sr=sr,
        n_fft=1024,
        hop_length=128,
        n_iter=150,
        window='hann',
        center=False
    )

    # 4. Postprocessing Pipeline
    if apply_bandpass:
        audio = butter_bandpass_filter(audio, sr, lowcut=100, highcut=8000)

    if apply_preemphasis:
        audio = librosa.effects.preemphasis(audio, coef=0.97)

    if apply_compression:
        audio = dynamic_range_compression(audio, sr)

    # 5. Lautheitsnormalisierung
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, target_lufs)

    # 6. Abschneiden von Übersteuerungen
    audio = np.clip(audio, -0.99, 0.99)

    sf.write(wav_path, audio, sr)
    print(f"Successfully saved processed speech to {wav_path}")

def butter_bandpass_filter(audio, sr, lowcut, highcut, order=4):
    """Null-Phasen Bandpassfilter"""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, audio)

def dynamic_range_compression(audio, sr, threshold=-20.0, ratio=4.0):
    """Einfacher Dynamikkompressor"""
    rms = np.sqrt(np.mean(audio**2))
    db = 20 * np.log10(rms)

    if db > threshold:
        gain = (threshold / db) ** (1.0 / ratio)
        audio = audio * gain

    return audio

if __name__ == "__main__":
    png_to_wav_speech(
        "p234_001.png",
        "reconstructed_speech_new_101_100_18000.wav",
        sr=44100,
        target_lufs=-16.0,
        apply_bandpass=True,
        apply_preemphasis=False,
        apply_compression=True
    )