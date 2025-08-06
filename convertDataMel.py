import os
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def wav_to_mel_spectrogram(wav_path, output_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.56, 2.56))
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            wav_path = os.path.join(input_dir, filename)
            output_file = filename.replace('.wav', '.png')
            output_path = os.path.join(output_dir, output_file)

            rel_wav_path = os.path.relpath(wav_path, DATASET_PATH)
            rel_output_path = os.path.relpath(output_path, DATASET_PATH)
            print(f"{rel_wav_path} → {rel_output_path}")

            # print(f"{filename} → {output_file}")
            wav_to_mel_spectrogram(wav_path, output_path)

if __name__ == '__main__':
    start_time = time.time()
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))

    datasets = [
        {
            "input": "clean_trainset_56spk_wav",
            "output": "clean_trainset_56spk_mel_256"
        },
        {
            "input": "noisy_trainset_56spk_wav",
            "output": "noisy_trainset_56spk_mel_256"
        },
        {
            "input": "clean_testset_wav",
            "output": "clean_testset_mel_256"
        },
        {
            "input": "noisy_testset_wav",
            "output": "noisy_testset_mel_256"
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





# import os
# import time
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np

# def wav_to_mel_spectrogram(wav_path, output_path):
#     y, sr = librosa.load(wav_path, sr=None)
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#     S_dB = librosa.power_to_db(S, ref=np.max)

#     plt.figure(figsize=(2.56, 2.56))
#     librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
#     plt.axis('off')
#     plt.tight_layout(pad=0)
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# def process_folder(input_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.wav'):
#             wav_path = os.path.join(input_dir, filename)
#             output_file = filename.replace('.wav', '.png')
#             output_path = os.path.join(output_dir, output_file)

#             rel_wav_path = os.path.relpath(wav_path, DATASET_PATH)
#             rel_output_path = os.path.relpath(output_path, DATASET_PATH)
#             print(f"{rel_wav_path} → {rel_output_path}")

#             print(f"{filename} → {output_file}")
#             wav_to_mel_spectrogram(wav_path, output_path)

# if __name__ == '__main__':
#     start_time = time.time()
#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))

#     datasets = [
#         {
#             "input": "1noisy_trainset_56spk_wav",
#             "output": "1noisy_trainset_56spk_mel"
#         }
#     ]

#     for ds in datasets:
#         input_dir = os.path.join(DATASET_PATH, ds["input"])
#         output_dir = os.path.join(DATASET_PATH, ds["output"])
#         print(f"\nVerarbeite Ordner: {ds['input']} → {ds['output']}")
#         process_folder(input_dir, output_dir)
#         print(f"Fertig mit: {ds['output']}\n")

#     end_time = time.time()
#     duration = end_time - start_time
#     print(f"Script execution completed in {duration:.2f} seconds")
