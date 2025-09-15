import numpy as np
import librosa
import soundfile as sf
from PIL import Image
import matplotlib.pyplot as plt
import pyloudnorm as pyln

def preprocess_image_to_spectrogram(image_path, db_range=(-80.0, 0.0)):
    """
    Lädt ein Spektrogramm-Bild, verarbeitet es vor und konvertiert es
    in eine lineare Amplituden-Spektrogramm-Matrix.

    HINWEIS: Dieser Teil ist hochgradig heuristisch und muss an das
    spezifische Bild (mit seinen Rändern, Achsen etc.) angepasst werden.
    Die hier gezeigten Werte sind Platzhalter.
    """
    # 1. Bild laden und in Graustufen konvertieren
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # 2. Ränder und Achsen beschneiden (HEURISTISCHE WERTE!)
    # Diese Werte müssen durch Analyse des Bildes ermittelt werden.
    # z.B. top=10, bottom=20, left=30, right=10
    # img_cropped = img_array[top:-bottom, left:-right]
    img_cropped = img_array # Annahme: Bild enthält nur Spektrogramm-Daten

    # 3. Pixelwerte (0-255) auf einen normalisierten Bereich (0.0-1.0) abbilden
    # Annahme: Schwarz (0) ist min_db, Weiß (255) ist max_db
    norm_spec = img_cropped / 255.0

    # 4. Frequenzachse umkehren (oft sind niedrige Frequenzen unten im Bild)
    norm_spec = norm_spec[::-1, :]

    # 5. Logarithmische Skala (dB) in lineare Amplitude umkehren
    min_db, max_db = db_range
    db_spec = min_db + norm_spec * (max_db - min_db)

    # Umrechnung von dB in Amplitude
    # S_linear = 10^(S_db / 20)
    linear_spec = np.power(10.0, db_spec / 20.0)

    return linear_spec

def reconstruct_audio_from_spectrogram(linear_spec, n_fft, hop_length, n_iter):
    """
    Rekonstruiert ein Audiosignal aus einem linearen Amplituden-Spektrogramm
    mit dem Griffin-Lim-Algorithmus.
    """
    print("Starte Griffin-Lim-Rekonstruktion...")
    # Verwendung von librosa.griffinlim
    audio_signal = librosa.griffinlim(
        linear_spec,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=n_fft,
        n_fft=n_fft
    )
    print("Rekonstruktion abgeschlossen.")
    return audio_signal

def normalize_audio(data, sample_rate, target_peak=-30):
    """
    Normalisiert das Audiosignal auf einen Ziel-Spitzenpegel.
    """
    print(f"Normalisiere Audio auf {target_peak} dBFS Peak-Level...")
    # Peak-Normalisierung mit pyloudnorm
    normalized_audio = pyln.normalize.peak(data, target_peak)
    print("Normalisierung abgeschlossen.")
    return normalized_audio


if __name__ == '__main__':
    # --- Konfiguration ---
    IMAGE_PATH = 'extern_denoised_noisy2.png' # Pfad zum Spektrogramm-Bild
    OUTPUT_WAV_PATH = 'reconstructed_speech_clean.wav'
    NORMALIZED_OUTPUT_WAV_PATH = 'reconstructed_speech_max_volume_clean.wav'

    # --- Parameter-Schätzungen (müssen experimentell angepasst werden!) ---
    # n_fft aus der Bildhöhe ableiten: height = 1 + n_fft / 2
    # Lade das Bild nur zur Dimensionsbestimmung
    try:
        temp_img = Image.open(IMAGE_PATH)
    except FileNotFoundError:
        print(f"Fehler: Bilddatei nicht gefunden unter {IMAGE_PATH}")
        exit()

    width, height = temp_img.size
    # Annahme: Höhe des Bildes entspricht den Frequenz-Bins nach dem Beschneiden
    N_FFT = (height - 1) * 2

    HOP_LENGTH = N_FFT // 4 # Gängige Annahme: 75% Überlappung
    N_ITER = 100             # Anzahl der Griffin-Lim-Iterationen
    SAMPLING_RATE = 48000    # Angenommene Abtastrate
    DB_RANGE = (-80.0, 0.0)  # Angenommener dB-Bereich der Visualisierung

    print(f"Abgeleitete/Angenommene Parameter:")
    print(f"  n_fft: {N_FFT}")
    print(f"  hop_length: {HOP_LENGTH}")
    print(f"  sampling_rate: {SAMPLING_RATE}")

    # Schritt 1: Bild vorverarbeiten, um lineares Spektrogramm zu erhalten
    linear_spectrogram = preprocess_image_to_spectrogram(IMAGE_PATH, DB_RANGE)

    # Visuelle Überprüfung des wiederhergestellten Spektrogramms
    plt.figure(figsize=(10, 4))
    # KORREKTUR: np.max wird als Referenz übergeben, nicht mit () aufgerufen.
    S_db = librosa.amplitude_to_db(linear_spectrogram, ref=np.max)
    librosa.display.specshow(S_db,
                             sr=SAMPLING_RATE,
                             hop_length=HOP_LENGTH,
                             x_axis='time',
                             y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Wiederhergestelltes lineares Spektrogramm (in dB zur Anzeige)')
    plt.tight_layout()
    plt.close()

    # Schritt 2: Audio aus dem Spektrogramm rekonstruieren
    reconstructed_audio = reconstruct_audio_from_spectrogram(
        linear_spectrogram,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_iter=N_ITER
    )

    # Schritt 3: Rekonstruiertes Audio als WAV-Datei speichern (unnormalisiert)
    sf.write(OUTPUT_WAV_PATH, reconstructed_audio, SAMPLING_RATE)
    print(f"Unnormalisierte Audiodatei gespeichert unter: {OUTPUT_WAV_PATH}")

    # Schritt 4: Audio normalisieren
    normalized_audio = normalize_audio(reconstructed_audio, SAMPLING_RATE)

    # Schritt 5: Normalisiertes Audio als neue WAV-Datei speichern
    sf.write(NORMALIZED_OUTPUT_WAV_PATH, normalized_audio, SAMPLING_RATE)
    print(f"Normalisierte Audiodatei erfolgreich gespeichert unter: {NORMALIZED_OUTPUT_WAV_PATH}")















########################################### gleich oben no max volume
# import numpy as np
# import librosa
# import soundfile as sf
# from PIL import Image
# import matplotlib.pyplot as plt

# def preprocess_image_to_spectrogram(image_path, db_range=(-80.0, 0.0)):
#     """
#     Lädt ein Spektrogramm-Bild, verarbeitet es vor und konvertiert es
#     in eine lineare Amplituden-Spektrogramm-Matrix.

#     HINWEIS: Dieser Teil ist hochgradig heuristisch und muss an das
#     spezifische Bild (mit seinen Rändern, Achsen etc.) angepasst werden.
#     Die hier gezeigten Werte sind Platzhalter.
#     """
#     # 1. Bild laden und in Graustufen konvertieren
#     img = Image.open(image_path).convert('L')
#     img_array = np.array(img)

#     # 2. Ränder und Achsen beschneiden (HEURISTISCHE WERTE!)
#     # Diese Werte müssen durch Analyse des Bildes ermittelt werden.
#     # z.B. top=10, bottom=20, left=30, right=10
#     # img_cropped = img_array[top:-bottom, left:-right]
#     img_cropped = img_array # Annahme: Bild enthält nur Spektrogramm-Daten

#     # 3. Pixelwerte (0-255) auf einen normalisierten Bereich (0.0-1.0) abbilden
#     # Annahme: Schwarz (0) ist min_db, Weiß (255) ist max_db
#     norm_spec = img_cropped / 255.0

#     # 4. Frequenzachse umkehren (oft sind niedrige Frequenzen unten im Bild)
#     norm_spec = norm_spec[::-1, :]

#     # 5. Logarithmische Skala (dB) in lineare Amplitude umkehren
#     min_db, max_db = db_range
#     db_spec = min_db + norm_spec * (max_db - min_db)

#     # Umrechnung von dB in Amplitude
#     # S_linear = 10^(S_db / 20)
#     linear_spec = np.power(10.0, db_spec / 20.0)

#     return linear_spec

# def reconstruct_audio_from_spectrogram(linear_spec, n_fft, hop_length, n_iter, sr):
#     """
#     Rekonstruiert ein Audiosignal aus einem linearen Amplituden-Spektrogramm
#     mit dem Griffin-Lim-Algorithmus.
#     """
#     print("Starte Griffin-Lim-Rekonstruktion...")
#     # Verwendung von librosa.griffinlim
#     audio_signal = librosa.griffinlim(
#         linear_spec,
#         n_iter=n_iter,
#         hop_length=hop_length,
#         win_length=n_fft,
#         n_fft=n_fft
#     )
#     print("Rekonstruktion abgeschlossen.")
#     return audio_signal

# if __name__ == '__main__':
#     # --- Konfiguration ---
#     IMAGE_PATH = 'extern_denoised_noisy2.png' # Pfad zum Spektrogramm-Bild
#     OUTPUT_WAV_PATH = 'extern_denoised_noisy2.wav'

#     # --- Parameter-Schätzungen (müssen experimentell angepasst werden!) ---
#     # n_fft aus der Bildhöhe ableiten: height = 1 + n_fft / 2
#     # Lade das Bild nur zur Dimensionsbestimmung
#     temp_img = Image.open(IMAGE_PATH)
#     width, height = temp_img.size
#     # Annahme: Höhe des Bildes entspricht den Frequenz-Bins nach dem Beschneiden
#     N_FFT = (height - 1) * 2

#     HOP_LENGTH = N_FFT // 4 # Gängige Annahme: 75% Überlappung
#     N_ITER = 100             # Anzahl der Griffin-Lim-Iterationen
#     SAMPLING_RATE = 48000    # Angenommene Abtastrate
#     DB_RANGE = (-80.0, 0.0)  # Angenommener dB-Bereich der Visualisierung

#     print(f"Abgeleitete/Angenommene Parameter:")
#     print(f"  n_fft: {N_FFT}")
#     print(f"  hop_length: {HOP_LENGTH}")
#     print(f"  sampling_rate: {SAMPLING_RATE}")

#     # Schritt 1: Bild vorverarbeiten, um lineares Spektrogramm zu erhalten
#     try:
#         linear_spectrogram = preprocess_image_to_spectrogram(IMAGE_PATH, DB_RANGE)
#     except FileNotFoundError:
#         print(f"Fehler: Bilddatei nicht gefunden unter {IMAGE_PATH}")
#         exit()

#     # Visuelle Überprüfung des wiederhergestellten Spektrogramms
#     plt.figure(figsize=(10, 4))

#     S_db = librosa.amplitude_to_db(linear_spectrogram, ref=np.max)
#     librosa.display.specshow(S_db,
#                              sr=SAMPLING_RATE,
#                              hop_length=HOP_LENGTH,
#                              x_axis='time',
#                              y_axis='linear')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Wiederhergestelltes lineares Spektrogramm (in dB zur Anzeige)')
#     plt.tight_layout()
#     plt.savefig('reconstructed_spectrogram_visualization.png')
#     plt.close()

#     # Schritt 2: Audio aus dem Spektrogramm rekonstruieren
#     reconstructed_audio = reconstruct_audio_from_spectrogram(
#         linear_spectrogram,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         n_iter=N_ITER,
#         sr=SAMPLING_RATE
#     )

#     # Schritt 3: Rekonstruiertes Audio als WAV-Datei speichern
#     sf.write(OUTPUT_WAV_PATH, reconstructed_audio, SAMPLING_RATE)
#     print(f"Audiodatei erfolgreich gespeichert unter: {OUTPUT_WAV_PATH}")