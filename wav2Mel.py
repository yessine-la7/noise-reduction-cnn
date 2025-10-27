import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def wav_to_mel_spectrogram_with_info(wav_path, output_path, n_mels=128, dpi=100,
                                    show_info=True, figsize=(10, 4)):
    """
    Konvertiert WAV zu Mel-Spektrogramm mit Audio-Informationen.
    """

    # Audio laden
    y, sr = librosa.load(wav_path, sr=None)
    duration = len(y) / sr
    filename = os.path.basename(wav_path)

    # Mel-Spektrogramm berechnen
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot erstellen
    plt.figure(figsize=figsize, dpi=dpi)

    # Spektrogramm anzeigen
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')

    # Farbbalken hinzufügen
    plt.colorbar(img, format='%+2.0f dB', label='Amplitude [dB]')

    # Titel mit Dateinamen
    # plt.title(f'Mel-Spektrogramm: {filename}', fontsize=14, fontweight='bold', pad=20)

    plt.xlabel('Zeit [s]', fontsize=12)
    plt.ylabel('Frequenz [Hz]', fontsize=12)

    # Audio-Informationen als Text hinzufügen
    if show_info:
        info_text = (f'Datei: {filename}\n'
                    f'Dauer: {duration:.2f}s\n'
                    f'Sample-Rate: {sr} Hz\n'
                    f'Mel-Bänder: {n_mels}\n'
                    f'Max Amplitude: {np.max(np.abs(y)):.3f}')

        # Informationen in der oberen rechten Ecke platzieren
        plt.annotate(info_text, xy=(0.98, 0.98), xycoords='axes fraction',
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10, fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close()

    print(f"Spektrogramm gespeichert: {output_path}")


if __name__ == "__main__":
    wav_to_mel_spectrogram_with_info(
        "p234_018.wav",
        "p234_018_spec_with_axis.png",
        n_mels=128,
        figsize=(10, 4)
    )
