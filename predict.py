import os
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loadData import ClassificationSpectrogramDataset
from createModel import ResNet18Custom
import logging

def predict_and_plot(model, dataset, device, results_dir, num_samples=10):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(3*num_samples, 6))

    for i, idx in enumerate(indices):
        img, true = dataset[idx]
        inp = img.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
            pred = probs.argmax()
            conf = probs[pred]

        # Obere Reihe: noisy oder clean (True)
        axes[0, i].imshow(img.squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f"T: {'Noisy' if true==1 else 'Clean'}")

        # Untere Reihe: Vorhersage + Confidence
        axes[1, i].imshow(img.squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f"P: {'Noisy' if pred==1 else 'Clean'}\n{conf*100:.1f}%")

    plt.tight_layout()
    outpath = os.path.join(results_dir, 'predictions.png')
    plt.savefig(outpath)
    plt.close()

def main():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    logger = logging.getLogger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Test-Dataset vorbereiten
    dataset = ClassificationSpectrogramDataset(
        clean_dir=os.path.join('..','Dataset','clean_testset_mel'),
        noisy_dir=os.path.join('..','Dataset','noisy_testset_mel'),
        image_size=224
    )

    # Modell laden (Graustufen)
    model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(results_dir,'best_model_1.pth'), map_location=device))
    logger.info("Loaded best_model.pth")

    # Vorhersage & Plot
    predict_and_plot(model, dataset, device, results_dir, num_samples=10)

if __name__ == "__main__":
    main()
