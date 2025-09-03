import librosa
import numpy as np

def analyze_wav(file_path):
    # Audiofile laden
    y, sr = librosa.load(file_path, sr=None)

    # Amplitude in dB umrechnen (relative zum Maximum)
    amplitude_db = 20 * np.log10(np.abs(y) + 1e-10)  # +1e-10 um log(0) zu vermeiden

    # Frequenzanalyse mit FFT
    n = len(y)
    fft_result = np.fft.rfft(y)
    frequencies = np.fft.rfftfreq(n, d=1/sr)
    magnitude = np.abs(fft_result)

    # Dominante Frequenzen finden (ignoriere sehr kleine Amplituden)
    threshold = 0.01 * np.max(magnitude)
    significant_freqs = frequencies[magnitude > threshold]

    if len(significant_freqs) > 0:
        min_freq = np.min(significant_freqs)
        max_freq = np.max(significant_freqs)
    else:
        min_freq = max_freq = 0

    # Amplitudenwerte
    min_amp_db = np.min(amplitude_db)
    max_amp_db = np.max(amplitude_db)

    return {
        'min_frequency_hz': min_freq,
        'max_frequency_hz': max_freq,
        'min_amplitude_db': min_amp_db,
        'max_amplitude_db': max_amp_db,
        'sample_rate': sr
    }

# Verwendung
if __name__ == "__main__":
    file_path = "p234_018.wav"
    results = analyze_wav(file_path)

    print(f"Sample Rate: {results['sample_rate']} Hz")
    print(f"Kleinste Frequenz: {results['min_frequency_hz']:.2f} Hz")
    print(f"Größte Frequenz: {results['max_frequency_hz']:.2f} Hz")
    print(f"Kleinste Amplitude: {results['min_amplitude_db']:.2f} dB")
    print(f"Größte Amplitude: {results['max_amplitude_db']:.2f} dB")