# ###################### snr with pixel value
# import os
# import random
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from loadData import ClassificationSpectrogramDataset
# from createModel import ResNet18Custom
# import logging
# import numpy as np
# from PIL import Image

# def compute_snr_from_color(img_path):
#     """
#     Berechne SNR aus RGB-Spektrogramm, basierend auf Farbinformation:
#     Gelb wird als Signal interpretiert, andere Farben als Rauschen.
#     """
#     img = Image.open(img_path).convert("RGB")
#     img_np = np.array(img, dtype=np.float32)

#     # Gelb: (R≈255, G≈255, B≈0)
#     yellow = np.array([255, 255, 0], dtype=np.float32)
#     distance = np.linalg.norm(img_np - yellow, axis=2)

#     # Signal: Pixel nahe an Gelb
#     signal_mask = distance < 50  # Schwelle kann angepasst werden
#     noise_mask = ~signal_mask

#     # Leistung berechnen
#     signal_power = np.mean(img_np[signal_mask] ** 2) if np.any(signal_mask) else 1e-8
#     noise_power = np.mean(img_np[noise_mask] ** 2) if np.any(noise_mask) else 1e-8
#     snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
#     return snr

# def predict_and_plot(model, dataset, device, results_dir, num_samples=10):
#     model.eval()
#     indices = random.sample(range(len(dataset)), num_samples)

#     fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(3*num_samples, 6))

#     for i, idx in enumerate(indices):
#         img, true = dataset[idx]
#         inp = img.unsqueeze(0).to(device)

#         with torch.no_grad():
#             out = model(inp)
#             probs = F.softmax(out, dim=1).cpu().numpy()[0]
#             pred = probs.argmax()
#             conf = probs[pred]

#         # Pfad zum Originalbild (für SNR)
#         path, _ = dataset.files[idx]
#         snr = compute_snr_from_color(path)

#         # Obere Reihe: True Label + SNR
#         axes[0, i].imshow(Image.open(path))  # RGB für visuelle Klarheit
#         axes[0, i].axis('off')
#         axes[0, i].set_title(f"T: {'Noisy' if true==1 else 'Clean'}\nSNR: {snr:.1f} dB")

#         # Untere Reihe: Prediction + Confidence
#         axes[1, i].imshow(Image.open(path))
#         axes[1, i].axis('off')
#         axes[1, i].set_title(f"P: {'Noisy' if pred==1 else 'Clean'}\n{conf*100:.1f}%")

#     plt.tight_layout()
#     outpath = os.path.join(results_dir, 'predictions.png')
#     plt.savefig(outpath)
#     plt.close()

# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), 'results')
#     os.makedirs(results_dir, exist_ok=True)
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     # Datensatz laden in Graustufen für das trainierte Modell
#     dataset = ClassificationSpectrogramDataset(
#         clean_dir=os.path.join('..', 'Dataset', 'clean_testset_mel'),
#         noisy_dir=os.path.join('..', 'Dataset', 'noisy_testset_mel'),
#         image_size=256,
#         in_channels=1  # wichtig: Modell wurde mit Graustufen trainiert
#     )

#     model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=False).to(device)
#     model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model_2.pth'), map_location=device))
#     logger.info("Loaded model weights")

#     predict_and_plot(model, dataset, device, results_dir, num_samples=10)

# if __name__ == "__main__":
#     main()






# ####################### snr with percentile
# import os
# import random
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from loadData import ClassificationSpectrogramDataset
# from createModel import ResNet18Custom
# import logging
# import numpy as np
# from PIL import Image


# def compute_snr_pegelbasiert(img_path):
#     img = Image.open(img_path)
#     img_np = np.array(img, dtype=np.float32) / 255.0

#     threshold = np.percentile(img_np, 70)

#     signal_pixels = img_np[img_np >= threshold]
#     noise_pixels = img_np[img_np < threshold]

#     signal_power = np.mean(signal_pixels ** 2) if len(signal_pixels) else 1e-8
#     noise_power = np.mean(noise_pixels ** 2) if len(noise_pixels) else 1e-8

#     snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
#     return snr



# def predict_and_plot(model, dataset, device, results_dir, num_samples=10):
#     model.eval()
#     indices = random.sample(range(len(dataset)), num_samples)

#     fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(3*num_samples, 6))

#     for i, idx in enumerate(indices):
#         img, true = dataset[idx]
#         inp = img.unsqueeze(0).to(device)

#         with torch.no_grad():
#             out = model(inp)
#             probs = F.softmax(out, dim=1).cpu().numpy()[0]
#             pred = probs.argmax()
#             conf = probs[pred]

#         # Pfad zum Originalbild (für SNR)
#         path, _ = dataset.files[idx]
#         snr = compute_snr_pegelbasiert(path)

#         # Für Visualisierung: Originalbild in Farbe laden (nicht das transformierte)
#         img_orig = Image.open(path)
#         img_vis = np.array(img_orig.resize((256, 256)))  # Zum Plotten in gleicher Größe

#         # Obere Reihe: True Label + SNR
#         axes[0, i].imshow(img_vis)
#         axes[0, i].axis('off')
#         axes[0, i].set_title(f"T: {'Noisy' if true==1 else 'Clean'}\nSNR: {snr:.1f} dB")

#         # Untere Reihe: Prediction + Confidence
#         axes[1, i].imshow(img_vis)
#         axes[1, i].axis('off')
#         axes[1, i].set_title(f"P: {'Noisy' if pred==1 else 'Clean'}\n{conf*100:.1f}%")

#     plt.tight_layout()
#     outpath = os.path.join(results_dir, 'predictions.png')
#     plt.savefig(outpath)
#     plt.close()


# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), 'results')
#     os.makedirs(results_dir, exist_ok=True)
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     # Dataset: Grau für Modell, Originalbild für SNR
#     dataset = ClassificationSpectrogramDataset(
#         clean_dir=os.path.join('..','Dataset','clean_testset_mel'),
#         noisy_dir=os.path.join('..','Dataset','noisy_testset_mel'),
#         image_size=256,
#         in_channels=1
#     )

#     model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=False).to(device)
#     model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model_2.pth'), map_location=device))
#     logger.info("Loaded model weights")

#     predict_and_plot(model, dataset, device, results_dir, num_samples=10)


# if __name__ == "__main__":
#     main()









############### snr aus bilder 256*256, np.mean
# import os
# import random
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from loadData import ClassificationSpectrogramDataset
# from createModel import ResNet18Custom
# import logging
# import numpy as np

# def compute_snr(img_tensor):
#     """
#     Berechne das SNR (Signal-to-Noise Ratio) für ein gegebenes Spektrogramm-Bild (Tensor).
#     """
#     img_np = img_tensor.squeeze().numpy()
#     signal_power = np.mean(img_np ** 2)
#     noise_power = np.mean((img_np - np.mean(img_np)) ** 2)
#     snr = 10 * np.log10(signal_power / (noise_power + 1e-8))  # +epsilon für numerische Stabilität
#     return snr

# def predict_and_plot(model, dataset, device, results_dir, num_samples=10):
#     model.eval()
#     indices = random.sample(range(len(dataset)), num_samples)

#     fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(3*num_samples, 6))

#     for i, idx in enumerate(indices):
#         img, true = dataset[idx]
#         inp = img.unsqueeze(0).to(device)

#         with torch.no_grad():
#             out = model(inp)
#             probs = F.softmax(out, dim=1).cpu().numpy()[0]
#             pred = probs.argmax()
#             conf = probs[pred]

#         snr = compute_snr(img)

#         # Obere Reihe: True Label + SNR
#         axes[0, i].imshow(img.squeeze())
#         axes[0, i].axis('off')
#         axes[0, i].set_title(f"T: {'Noisy' if true==1 else 'Clean'}\nSNR: {snr:.1f} dB")

#         # Untere Reihe: Prediction + Confidence
#         axes[1, i].imshow(img.squeeze())
#         axes[1, i].axis('off')
#         axes[1, i].set_title(f"P: {'Noisy' if pred==1 else 'Clean'}\n{conf*100:.1f}%")

#     plt.tight_layout()
#     outpath = os.path.join(results_dir, 'predictions.png')
#     plt.savefig(outpath)
#     plt.close()

# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), 'results')
#     os.makedirs(results_dir, exist_ok=True)
#     logger = logging.getLogger()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     # Testdaten laden
#     dataset = ClassificationSpectrogramDataset(
#         clean_dir=os.path.join('..','Dataset','clean_testset_mel'),
#         noisy_dir=os.path.join('..','Dataset','noisy_testset_mel'),
#         image_size=256
#     )

#     # Modell laden
#     model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=False).to(device)
#     model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model_2.pth'), map_location=device))

#     # Vorhersagen und Visualisierung
#     predict_and_plot(model, dataset, device, results_dir, num_samples=10)

# if __name__ == "__main__":
#     main()







# ########################## no snr
# import os
# import random
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from loadData import ClassificationSpectrogramDataset
# from createModel import ResNet18Custom
# import logging

# def predict_and_plot(model, dataset, device, results_dir, num_samples=10):
#     model.eval()
#     indices = random.sample(range(len(dataset)), num_samples)

#     fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(3*num_samples, 6))

#     for i, idx in enumerate(indices):
#         img, true = dataset[idx]
#         inp = img.unsqueeze(0).to(device)
#         with torch.no_grad():
#             out = model(inp)
#             probs = F.softmax(out, dim=1).cpu().numpy()[0]
#             pred = probs.argmax()
#             conf = probs[pred]

#         # Obere Reihe: noisy oder clean (True)
#         axes[0, i].imshow(img.squeeze())
#         axes[0, i].axis('off')
#         axes[0, i].set_title(f"T: {'Noisy' if true==1 else 'Clean'}")

#         # Untere Reihe: Vorhersage + Confidence
#         axes[1, i].imshow(img.squeeze())
#         axes[1, i].axis('off')
#         axes[1, i].set_title(f"P: {'Noisy' if pred==1 else 'Clean'}\n{conf*100:.1f}%")

#     plt.tight_layout()
#     outpath = os.path.join(results_dir, 'predictions.png')
#     plt.savefig(outpath)
#     plt.close()

# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), 'results')
#     os.makedirs(results_dir, exist_ok=True)
#     logger = logging.getLogger()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     # Test-Dataset vorbereiten
#     dataset = ClassificationSpectrogramDataset(
#         clean_dir=os.path.join('..','Dataset','clean_testset_mel'),
#         noisy_dir=os.path.join('..','Dataset','noisy_testset_mel'),
#         image_size=256
#     )

#     # Modell laden
#     model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=False).to(device)
#     model.load_state_dict(torch.load(os.path.join(results_dir,'best_model_MIL_batch_16_seed_regul.pth'), map_location=device))
#     logger.info("Loaded best_model.pth")

#     # Vorhersage
#     predict_and_plot(model, dataset, device, results_dir, num_samples=10)

# if __name__ == "__main__":
#     main()




######################### MIL
import os
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loadData import get_data_loaders
from createModelResNet import ResNet18Custom
import logging

def predict_and_plot(model, dataset, device, results_dir, num_samples=10):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(3*num_samples, 6))

    for i, idx in enumerate(indices):
        img, label, _ = dataset[idx]   # (Bild, Label, file_id)
        inp = img.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
            pred = probs.argmax()
            conf = probs[pred]

        # Obere Reihe: True Label
        axes[0, i].imshow(img.squeeze())
        axes[0, i].axis('off')
        axes[0, i].set_title(f"T: {'Noisy' if label==1 else 'Clean'}")

        # Untere Reihe: Prediction + Confidence
        axes[1, i].imshow(img.squeeze())
        axes[1, i].axis('off')
        axes[1, i].set_title(f"P: {'Noisy' if pred==1 else 'Clean'}\n{conf*100:.1f}%")

    plt.tight_layout()
    outpath = os.path.join(results_dir, "predictions.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Vorhersagen gespeichert unter: {outpath}")

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset"))

    # Lade nur das Test-Set
    loaders = get_data_loaders(
        DATASET_PATH,
        batch_size=1,
        num_workers=0,
        cls_in_channels=1,
        cls_tile_h=128,
        cls_tile_w=256,
        cls_stride_w=128,
        cls_resize_height=True,
        cls_val_ratio=0.2,
        cls_seed=42,
        enable_classification=True,
        enable_denoising=False,
    )
    _, _, test_loader = loaders["classification"]
    test_dataset = test_loader.dataset

    # Modell laden
    model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=False).to(device)
    best_model_path = os.path.join(results_dir, "best_model_MIL_batch_16_seed_regul.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    logger.info(f"Loaded model weights from {best_model_path}")

    # Vorhersage für 10 zufällige Bilder
    predict_and_plot(model, test_dataset, device, results_dir, num_samples=10)

if __name__ == "__main__":
    main()
