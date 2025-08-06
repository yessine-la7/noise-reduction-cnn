import os
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def wav_to_stft_spectrogram(wav_path, output_path, n_fft=2048, hop_length=512):
    y, sr = librosa.load(wav_path, sr=None)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D)
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis=None, y_axis='linear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.endswith('.wav'):
            continue
        wav_path = os.path.join(input_dir, filename)
        output_file = filename.replace('.wav', '.png')
        output_path = os.path.join(output_dir, output_file)

        rel_wav = os.path.relpath(wav_path, DATASET_PATH)
        rel_out = os.path.relpath(output_path, DATASET_PATH)
        print(f"{rel_wav} → {rel_out}")

        # print(f"{filename} → {output_file}")
        wav_to_stft_spectrogram(wav_path, output_path)

if __name__ == '__main__':
    start_time = time.time()
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))

    datasets = [
        {
            "input": "clean_trainset_56spk_wav",
            "output": "clean_trainset_stft"
        },
        {
            "input": "noisy_trainset_56spk_wav",
            "output": "noisy_trainset_stft"
        },
        {
            "input": "clean_testset_wav",
            "output": "clean_testset_stft"
        },
        {
            "input": "noisy_testset_wav",
            "output": "noisy_testset_stft"
        },
    ]

    for ds in datasets:
        input_dir = os.path.join(DATASET_PATH, ds["input"])
        output_dir = os.path.join(DATASET_PATH, ds["output"])
        print(f"\nVerarbeite Ordner: {ds['input']} → {ds['output']}")
        process_folder(input_dir, output_dir)
        print(f"Fertig mit: {ds['output']}\n")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Script execution completed in {duration:.2f} seconds")
