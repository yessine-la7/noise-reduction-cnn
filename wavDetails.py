import librosa
import numpy as np
import os

# --- Konfiguration ---

# HIER DEN NAMEN IHRER M4A-DATEI EINTRAGEN
AUDIO_FILENAME = "noisy2.m4a"

# STFT-Parameter (sollten mit Ihrem Projekt übereinstimmen)
STFT_N_FFT = 1024
STFT_HOP   = 256
STFT_WIN   = 1024

# --- Hauptskript ---

def get_db_min_from_file(filepath):
    """
    Lädt eine Audiodatei, berechnet das dB-Spektrogramm und gibt den db_min-Wert zurück.
    """
    if not os.path.isfile(filepath):
        print(f"FEHLER: Datei '{filepath}' nicht gefunden.")
        return None

    try:
        print(f"Lade Audiodatei: {filepath}...")
        y, sr = librosa.load(filepath, sr=None, mono=True)

        if len(y) == 0:
            print("FEHLER: Die Audiodatei ist leer.")
            return None

        print("Berechne STFT und konvertiere zu Dezibel...")
        S_complex = librosa.stft(y, n_fft=STFT_N_FFT, hop_length=STFT_HOP, win_length=STFT_WIN)
        S_mag = np.abs(S_complex)

        if np.max(S_mag) == 0:
            print("Die Audiodatei enthält nur Stille.")
            # Für eine stille Datei ist der dB-Wert negativ unendlich,
            # aber wir können einen Standard-Mindestwert zurückgeben.
            return -80.0

        S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

        db_min = S_db.min()

        return db_min

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None


if __name__ == "__main__":
    db_min_value = get_db_min_from_file(AUDIO_FILENAME)

    if db_min_value is not None:
        print("\n" + "="*40)
        print(f"Ergebnis für '{AUDIO_FILENAME}':")
        print(f"Der db_min-Wert ist: {db_min_value:.4f} dB")
        print("="*40)












# import os
# import librosa
# import numpy as np
# from tqdm import tqdm
# import warnings

# # Ignoriere Warnungen von librosa für nicht standardmäßige Abtastraten
# warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# # --- Konfiguration ---
# # Pfad zum Ordner mit den Original-Trainings-WAV-Dateien
# # Wir analysieren die "clean"-Dateien, da sie die Referenz sind.
# DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset"))
# WAV_DIR = os.path.join(DATASET_ROOT, "clean_trainset_56spk_wav")

# # STFT-Parameter (müssen mit dem Training übereinstimmen)
# STFT_N_FFT = 1024
# STFT_HOP   = 256
# STFT_WIN   = 1024

# def analyze_audio_file(filepath):
#     """Lädt eine Audiodatei, berechnet das dB-Spektrogramm und gibt db_min zurück."""
#     try:
#         y, sr = librosa.load(filepath, sr=None, mono=True)
#         if len(y) == 0:
#             return None  # Leere Datei überspringen

#         S_complex = librosa.stft(y, n_fft=STFT_N_FFT, hop_length=STFT_HOP, win_length=STFT_WIN)
#         S_mag = np.abs(S_complex)

#         # Überprüfen, ob die Magnitude nicht nur aus Nullen besteht
#         if np.max(S_mag) == 0:
#             return None # Stille Datei überspringen

#         S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

#         # Sicherstellen, dass das Ergebnis endlich ist
#         if not np.isfinite(S_db.min()):
#             return None

#         return S_db.min()
#     except Exception as e:
#         print(f"\nFehler bei der Verarbeitung von {os.path.basename(filepath)}: {e}")
#         return None

# def main():
#     print("=====================================================")
#     print("=== Analyse des Dynamikbereichs der WAV-Dateien ===")
#     print(f"Durchsuche Ordner: {WAV_DIR}")
#     print("=====================================================")

#     if not os.path.isdir(WAV_DIR):
#         print(f"FEHLER: Verzeichnis nicht gefunden: {WAV_DIR}")
#         return

#     wav_files = [os.path.join(WAV_DIR, f) for f in os.listdir(WAV_DIR) if f.lower().endswith('.wav')]
#     if not wav_files:
#         print("FEHLER: Keine WAV-Dateien im Verzeichnis gefunden.")
#         return

#     db_min_values = []

#     # tqdm fügt eine Fortschrittsanzeige hinzu
#     for f in tqdm(wav_files, desc="Analysiere WAV-Dateien"):
#         db_min = analyze_audio_file(f)
#         if db_min is not None:
#             db_min_values.append(db_min)

#     if not db_min_values:
#         print("\nFEHLER: Konnte keine gültigen db_min-Werte aus den Dateien extrahieren.")
#         return

#     db_min_values = np.array(db_min_values)

#     print("\n\n--- Ergebnisse der Analyse ---")
#     print(f"Verarbeitete Dateien: {len(db_min_values)} / {len(wav_files)}")
#     print(f"Durchschnitt (Mean) von db_min: {db_min_values.mean():.4f} dB   <-- DIESEN WERT VERWENDEN")
#     print(f"Median von db_min:              {np.median(db_min_values):.4f} dB")
#     print(f"Standardabweichung:             {db_min_values.std():.4f} dB")
#     print(f"Minimaler db_min (leiseste Datei): {db_min_values.min():.4f} dB")
#     print(f"Maximaler db_min (lauteste 'stille' Datei): {db_min_values.max():.4f} dB")
#     print("--------------------------------\n")
#     print("Kopieren Sie den Durchschnittswert (Mean) in Ihr 'denoiseExtern.py'-Skript.")

# if __name__ == "__main__":
#     main()




















# import librosa
# import numpy as np

# def analyze_wav(file_path):
#     # Audiofile laden
#     y, sr = librosa.load(file_path, sr=None)

#     # Amplitude in dB umrechnen (relative zum Maximum)
#     amplitude_db = 20 * np.log10(np.abs(y) + 1e-10)  # +1e-10 um log(0) zu vermeiden

#     # Frequenzanalyse mit FFT
#     n = len(y)
#     fft_result = np.fft.rfft(y)
#     frequencies = np.fft.rfftfreq(n, d=1/sr)
#     magnitude = np.abs(fft_result)

#     # Dominante Frequenzen finden (ignoriere sehr kleine Amplituden)
#     threshold = 0.01 * np.max(magnitude)
#     significant_freqs = frequencies[magnitude > threshold]

#     if len(significant_freqs) > 0:
#         min_freq = np.min(significant_freqs)
#         max_freq = np.max(significant_freqs)
#     else:
#         min_freq = max_freq = 0

#     # Amplitudenwerte
#     min_amp_db = np.min(amplitude_db)
#     max_amp_db = np.max(amplitude_db)

#     return {
#         'min_frequency_hz': min_freq,
#         'max_frequency_hz': max_freq,
#         'min_amplitude_db': min_amp_db,
#         'max_amplitude_db': max_amp_db,
#         'sample_rate': sr
#     }

# # Verwendung
# if __name__ == "__main__":
#     file_path = "noisy2.m4a"
#     results = analyze_wav(file_path)

#     print(f"Sample Rate: {results['sample_rate']} Hz")
#     print(f"Kleinste Frequenz: {results['min_frequency_hz']:.2f} Hz")
#     print(f"Größte Frequenz: {results['max_frequency_hz']:.2f} Hz")
#     print(f"Kleinste Amplitude: {results['min_amplitude_db']:.2f} dB")
#     print(f"Größte Amplitude: {results['max_amplitude_db']:.2f} dB")