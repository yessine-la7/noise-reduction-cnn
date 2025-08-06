# # ################ custom split tensorboard MIL
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix
# from torch.utils.tensorboard import SummaryWriter
# from PIL import Image
# import numpy as np
# from io import BytesIO

# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging

# def fig_to_tensorboard_image(fig):
#     buf = BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     image = Image.open(buf).convert('RGB')
#     image = np.array(image).astype(np.float32) / 255.0
#     image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
#     return image

# def aggregate_predictions(outputs, mode="mean"):
#     if mode == "mean":
#         return outputs.mean(dim=1)
#     elif mode == "max":
#         return outputs.max(dim=1).values
#     else:
#         raise ValueError(f"Unbekannter Aggregationstyp: {mode}")

# def train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4):
#     logger = logging.getLogger()
#     writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#     early_stopper = EarlyStopping(patience=5, min_delta=None, verbose=True)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info("Starting training")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

#         model.train()
#         running_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             B, M, C, H, W = inputs.shape
#             inputs = inputs.view(B * M, C, H, W)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             outputs = outputs.view(B, M, -1)
#             outputs_agg = aggregate_predictions(outputs, mode="mean")

#             loss = criterion(outputs_agg, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * B

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)
#         writer.add_scalar("Loss/Train", train_loss, epoch)

#         model.eval()
#         running_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 B, M, C, H, W = inputs.shape
#                 inputs = inputs.view(B * M, C, H, W)

#                 outputs = model(inputs)
#                 outputs = outputs.view(B, M, -1)
#                 outputs_agg = aggregate_predictions(outputs, mode="mean")

#                 loss = criterion(outputs_agg, targets)
#                 running_loss += loss.item() * B

#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)
#         writer.add_scalar("Loss/Val", val_loss, epoch)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.warning(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training completed in {total_minutes:.2f} minutes")
#     writer.close()

#     plt.figure(figsize=(8,5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Loss Curves with EarlyStopping')
#     plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
#     plt.close()

# def evaluate(model, loader, device, name, results_dir, writer=None):
#     logger = logging.getLogger()
#     model.eval()
#     all_preds, all_targets = [], []

#     with torch.no_grad():
#         for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             B, M, C, H, W = inputs.shape
#             inputs = inputs.view(B * M, C, H, W)
#             outputs = model(inputs)
#             outputs = outputs.view(B, M, -1)
#             outputs_agg = aggregate_predictions(outputs, mode="mean")
#             preds = torch.argmax(outputs_agg, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())

#     report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
#     df = pd.DataFrame(report_dict).transpose()
#     logger.info(f"\n{name} Classification Report:\n{df.to_string()}\n")

#     if writer is not None:
#         for cls in ['0', '1']:
#             writer.add_scalar(f"{name}/Precision_class_{cls}", report_dict[cls]['precision'], 0)
#             writer.add_scalar(f"{name}/Recall_class_{cls}", report_dict[cls]['recall'], 0)
#             writer.add_scalar(f"{name}/F1_class_{cls}", report_dict[cls]['f1-score'], 0)

#     cm = confusion_matrix(all_targets, all_preds)
#     fig, ax = plt.subplots(figsize=(5,4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean','Noisy'], yticklabels=['Clean','Noisy'], ax=ax)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_title(f'{name} Confusion Matrix')

#     cm_path = os.path.join(results_dir, f'{name.lower()}_confusion_matrix.png')
#     fig.savefig(cm_path)
#     plt.close(fig)

#     if writer is not None:
#         img_tensor = fig_to_tensorboard_image(fig)
#         writer.add_images(f"{name}/Confusion_Matrix", img_tensor, 0)

# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     os.makedirs(results_dir, exist_ok=True)
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, batch_size=8, image_size=256)
#     train_loader, val_loader, test_loader = loaders["classification"]

#     model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=True).to(device)
#     train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4)

#     model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
#     evaluate(model, train_loader, device, name="Train", results_dir=results_dir, writer=writer)
#     evaluate(model, val_loader, device, name="Validation", results_dir=results_dir, writer=writer)
#     evaluate(model, test_loader, device, name="Test", results_dir=results_dir, writer=writer)

#     writer.close()

# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()












################ custom split tensorboard
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from io import BytesIO

from loadData import get_data_loaders
from createModel import ResNet18Custom
from earlyStopping import EarlyStopping
from trainLogging import setup_logging


def fig_to_tensorboard_image(fig):
    """Konvertiere matplotlib-Figure zu TensorBoard-kompatiblem Tensor."""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    return image


def log_hyperparams(logger, writer, *, lr, batch_size, image_size, in_channels, patience, adapt_start_epoch):
    """Logge Hyperparameter in Terminal/Datei und TensorBoard."""
    logger.info("======== Hyperparameter ========")
    logger.info(f"Learning Rate:           {lr}")
    logger.info(f"Batch Size:                   {batch_size}")
    logger.info(f"Image Size:                   {image_size}x{image_size}")
    logger.info(f"in_channels (1=Gray,3=RGB):   {in_channels}")
    logger.info(f"EarlyStopping patience:       {patience}")
    logger.info(f"EarlyStopping adapt_start_epoch: {adapt_start_epoch}")
    logger.info("================================")

    # Im TensorBoard als Text ablegen (gut sichtbar im SCALARS/TEXT)
    writer.add_text("config/learning_rate", str(lr))
    writer.add_text("config/batch_size", str(batch_size))
    writer.add_text("config/image_size", f"{image_size}x{image_size}")
    writer.add_text("config/in_channels", str(in_channels))
    writer.add_text("config/early_stopping_patience", str(patience))
    writer.add_text("config/early_stopping_adapt_start_epoch", str(adapt_start_epoch))

    # Optional: HParams-Tab in TensorBoard
    try:
        writer.add_hparams(
            {
                "lr": lr,
                "batch_size": batch_size,
                "image_size": image_size,
                "in_channels": in_channels,
                "patience": patience,
                "adapt_start_epoch": adapt_start_epoch,
            },
            {}
        )
    except Exception as e:
        logger.warning(f"add_hparams konnte nicht geschrieben werden: {e}")


def train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4,
          patience=10, adapt_start_epoch=10):
    logger = logging.getLogger()
    writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopper = EarlyStopping(patience=patience, min_delta=None, verbose=True, adapt_start_epoch=adapt_start_epoch)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    start_time = time.time()

    logger.info("Starting training")
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

        elapsed = time.time() - epoch_start
        logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

        if early_stopper(val_loss):
            logger.warning(f"Early stopping at epoch {epoch}")
            break

        scheduler.step()

    total_minutes = (time.time() - start_time) / 60
    logger.info(f"Training completed in {total_minutes:.2f} minutes")
    writer.close()

    # Plot Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves with EarlyStopping')
    plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
    plt.close()


def evaluate(model, loader, device, name, results_dir, writer=None):
    logger = logging.getLogger()
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    logger.info(f"\n{name} Classification Report:\n{df.to_string()}\n")

    if writer is not None:
        # Log Klassenmetriken als Scalars
        for cls in ['0', '1']:
            writer.add_scalar(f"{name}/Precision_class_{cls}", report_dict[cls]['precision'], 0)
            writer.add_scalar(f"{name}/Recall_class_{cls}", report_dict[cls]['recall'], 0)
            writer.add_scalar(f"{name}/F1_class_{cls}", report_dict[cls]['f1-score'], 0)
        # Macro/Weighted
        writer.add_scalar(f"{name}/F1_macro", report_dict["macro avg"]["f1-score"], 0)
        writer.add_scalar(f"{name}/F1_weighted", report_dict["weighted avg"]["f1-score"], 0)
        writer.add_scalar(f"{name}/Accuracy", report_dict["accuracy"], 0)

    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Noisy'], yticklabels=['Clean', 'Noisy'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{name} Confusion Matrix')

    cm_path = os.path.join(results_dir, f'{name.lower()}_confusion_matrix.png')
    fig.savefig(cm_path)
    plt.close(fig)

    if writer is not None:
        img_tensor = fig_to_tensorboard_image(fig)
        writer.add_images(f"{name}/Confusion_Matrix", img_tensor, 0)


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

    # ==== Hyperparameter ====
    lr = 1e-4
    batch_size = 8
    image_size = 256
    in_channels = 1           # 1 = Graustufen, 3 = RGB
    patience = 10
    adapt_start_epoch = 10
    num_epochs = 50
    # ==========================================

    writer = SummaryWriter(log_dir=os.path.join(results_dir, "runs"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Hyperparameter in Terminal/Datei + TensorBoard loggen
    log_hyperparams(
        logger, writer,
        lr=lr, batch_size=batch_size, image_size=image_size, in_channels=in_channels,
        patience=patience, adapt_start_epoch=adapt_start_epoch
    )

    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
    loaders = get_data_loaders(DATASET_PATH, batch_size=batch_size, image_size=image_size)
    train_loader, val_loader, test_loader = loaders["classification"]

    model = ResNet18Custom(num_classes=2, in_channels=in_channels, pretrained=True).to(device)
    train(model, train_loader, val_loader, device, results_dir,
          num_epochs=num_epochs, lr=lr, patience=patience, adapt_start_epoch=adapt_start_epoch)

    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
    evaluate(model, train_loader, device, name="Train", results_dir=results_dir, writer=writer)
    evaluate(model, val_loader, device, name="Validation", results_dir=results_dir, writer=writer)
    evaluate(model, test_loader, device, name="Test", results_dir=results_dir, writer=writer)

    writer.close()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()









# ################## custom split ohne early stopping
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix

# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from trainLogging import setup_logging

# def train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4):
#     logger = logging.getLogger()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info("Starting training")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

#         model.train()
#         running_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)

#         model.eval()
#         running_loss = 0.0
#         for inputs, targets in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             running_loss += loss.item() * inputs.size(0)

#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training completed in {total_minutes:.2f} minutes")

#     plt.figure(figsize=(8,5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Loss Curves (No EarlyStopping)')
#     plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
#     plt.close()

# def evaluate(model, loader, device, name, results_dir):
#     logger = logging.getLogger()
#     model.eval()
#     all_preds, all_targets = [], []
#     for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_targets.extend(targets.cpu().numpy())

#     report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
#     df = pd.DataFrame(report_dict).transpose()
#     logger.info(f"\n{name} Classification Report:\n{df.to_string()}\n")

#     cm = confusion_matrix(all_targets, all_preds)
#     plt.figure(figsize=(5,4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean','Noisy'], yticklabels=['Clean','Noisy'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(f'{name} Confusion Matrix')
#     plt.savefig(os.path.join(results_dir, f'{name.lower()}_confusion_matrix.png'))
#     plt.close()

# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, batch_size=16, image_size=224)
#     train_loader, val_loader, test_loader = loaders["classification"]

#     model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=True).to(device)
#     train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4)

#     model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
#     evaluate(model, train_loader, device, name="Train", results_dir=results_dir)
#     evaluate(model, val_loader, device, name="Validation", results_dir=results_dir)
#     evaluate(model, test_loader, device, name="Test", results_dir=results_dir)

# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()






# ############### custom cross validation mit average
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np

# # Deine Module
# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging

# def train_fold(model, train_loader, val_loader, device, results_dir, fold_idx, num_epochs=50, lr=1e-4):
#     logger = logging.getLogger()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     early_stopper = EarlyStopping(patience=5, min_delta=None, verbose=True)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info(f"Starting training Fold {fold_idx}")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Fold {fold_idx} Epoch {epoch}/{num_epochs} ---")

#         model.train()
#         running_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Train Fold {fold_idx}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)

#         model.eval()
#         running_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets in tqdm(val_loader, desc=f"Val Fold {fold_idx}", unit="batch", leave=False):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 running_loss += loss.item() * inputs.size(0)

#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             model_path = os.path.join(results_dir, f"best_model_fold{fold_idx}.pth")
#             torch.save(model.state_dict(), model_path)
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.error(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training Fold {fold_idx} completed in {total_minutes:.2f} minutes")

#     # Save loss curves
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title(f'Loss Curves Fold {fold_idx}')
#     plt.savefig(os.path.join(results_dir, f'loss_curves_fold{fold_idx}.png'))
#     plt.close()

#     return best_val_loss


# def evaluate(model, loader, device, name, results_dir, fold_idx):
#     logger = logging.getLogger()
#     model.eval()
#     all_preds, all_targets = [], []
#     with torch.no_grad():
#         for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())

#     # Classification Report
#     report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
#     df = pd.DataFrame(report_dict).transpose()
#     logger.info(f"\n{name} Classification Report (Fold {fold_idx}):\n{df.to_string()}\n")

#     # Confusion Matrix
#     cm = confusion_matrix(all_targets, all_preds)
#     plt.figure(figsize=(5, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean','Noisy'], yticklabels=['Clean','Noisy'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(f'{name} Confusion Matrix Fold {fold_idx}')
#     plt.savefig(os.path.join(results_dir, f'{name.lower()}_confusion_matrix_fold{fold_idx}.png'))
#     plt.close()

#     return df


# def aggregate_fold_results(fold_metrics, results_dir):
#     """
#     Aggregiert die Ergebnisse über alle Folds.
#     """
#     logger = logging.getLogger()
#     logger.info("\n" + "="*50)
#     logger.info("           CROSS-VALIDATION RESULTS")
#     logger.info("="*50)

#     # Sammle Metriken
#     f1_clean = [df.loc['0', 'f1-score'] for df in fold_metrics]
#     f1_noisy = [df.loc['1', 'f1-score'] for df in fold_metrics]
#     acc = [df.loc['accuracy', 'precision'] for df in fold_metrics]

#     # Berechne Mittelwert und Standardabweichung
#     def log_metric(name, values):
#         mean = np.mean(values)
#         std = np.std(values)
#         logger.info(f"{name} - Mean: {mean:.4f} ± {std:.4f}")

#     log_metric("F1-Score (Clean)", f1_clean)
#     log_metric("F1-Score (Noisy)", f1_noisy)
#     log_metric("Accuracy", acc)

#     # Bestes Modell identifizieren (nach Val-F1)
#     val_f1_scores = [df.loc['macro avg', 'f1-score'] for df in fold_metrics]
#     best_fold_idx = np.argmax(val_f1_scores) + 1
#     logger.info(f"\nBest Validation F1-Score: Fold {best_fold_idx}")
#     return best_fold_idx


# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results_cv")
#     os.makedirs(results_dir, exist_ok=True)
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, batch_size=16, image_size=224)

#     folds = loaders["classification"]["folds"]
#     test_loader = loaders["classification"]["test"]

#     # Speichern der Metriken und Modelle
#     fold_metrics = []

#     for fold_idx, (train_loader, val_loader) in enumerate(folds, start=1):
#         logger.info(f"\n{'='*20} Fold {fold_idx}/5 {'='*20}")

#         model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=True).to(device)

#         #  KORREKT: 'device' wird jetzt übergeben!
#         train_fold(model, train_loader, val_loader, device, results_dir, fold_idx, num_epochs=50, lr=1e-4)

#         # Lade bestes Modell des Folds
#         best_model_path = os.path.join(results_dir, f"best_model_fold{fold_idx}.pth")
#         model.load_state_dict(torch.load(best_model_path, map_location=device))

#         # Evaluate auf allen Sets
#         train_metrics = evaluate(model, train_loader, device, "Train", results_dir, fold_idx)
#         val_metrics = evaluate(model, val_loader, device, "Validation", results_dir, fold_idx)
#         test_metrics = evaluate(model, test_loader, device, "Test", results_dir, fold_idx)

#         fold_metrics.append(val_metrics)

#     # --- Aggregation der Ergebnisse ---
#     best_fold_idx = aggregate_fold_results(fold_metrics, results_dir)

#     # --- Finale Evaluation mit bestem Modell ---
#     logger.info(f"\n\n{'='*20} FINAL EVALUATION {'='*20}")
#     best_model_path = os.path.join(results_dir, f"best_model_fold{best_fold_idx}.pth")
#     final_model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=True).to(device)
#     final_model.load_state_dict(torch.load(best_model_path, map_location=device))

#     final_test_metrics = evaluate(final_model, test_loader, device, "Final Test", results_dir, fold_idx="BEST")
#     logger.info(f"Final Test Accuracy (Best Model): {final_test_metrics.loc['accuracy', 'precision']:.4f}")


# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()







# ############### custom cross validation
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix

# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging

# def train(model, train_loader, val_loader, device, results_dir, fold_idx, num_epochs=50, lr=1e-4):
#     logger = logging.getLogger()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     early_stopper = EarlyStopping(patience=5, min_delta=None, verbose=True)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info(f"Starting training Fold {fold_idx}")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Fold {fold_idx} Epoch {epoch}/{num_epochs} ---")

#         model.train()
#         running_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Train Fold {fold_idx}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)

#         model.eval()
#         running_loss = 0.0
#         for inputs, targets in tqdm(val_loader, desc=f"Val Fold {fold_idx}", unit="batch", leave=False):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             running_loss += loss.item() * inputs.size(0)

#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             model_path = os.path.join(results_dir, f"best_model_fold{fold_idx}.pth")
#             torch.save(model.state_dict(), model_path)
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.error(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training Fold {fold_idx} completed in {total_minutes:.2f} minutes")

#     # Save loss curves
#     plt.figure(figsize=(8,5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title(f'Loss Curves Fold {fold_idx}')
#     plt.savefig(os.path.join(results_dir, f'loss_curves_fold{fold_idx}.png'))
#     plt.close()

# def evaluate(model, loader, device, name, results_dir, fold_idx):
#     logger = logging.getLogger()
#     model.eval()
#     all_preds, all_targets = [], []
#     for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_targets.extend(targets.cpu().numpy())

#     report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
#     df = pd.DataFrame(report_dict).transpose()
#     logger.info(f"\n{name} Classification Report (Fold {fold_idx}):\n{df.to_string()}\n")

#     cm = confusion_matrix(all_targets, all_preds)
#     plt.figure(figsize=(5,4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean','Noisy'], yticklabels=['Clean','Noisy'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(f'{name} Confusion Matrix Fold {fold_idx}')
#     plt.savefig(os.path.join(results_dir, f'{name.lower()}_confusion_matrix_fold{fold_idx}.png'))
#     plt.close()

# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     os.makedirs(results_dir, exist_ok=True)
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, batch_size=16, image_size=224)

#     folds = loaders["classification"]["folds"]
#     test_loader = loaders["classification"]["test"]

#     for fold_idx, (train_loader, val_loader) in enumerate(folds, start=1):
#         logger.info(f"==== Fold {fold_idx} ====")
#         model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=True).to(device)
#         train(model, train_loader, val_loader, device, results_dir, fold_idx=fold_idx, num_epochs=50, lr=1e-4)

#         best_model_path = os.path.join(results_dir, f"best_model_fold{fold_idx}.pth")
#         model.load_state_dict(torch.load(best_model_path, map_location=device))

#         evaluate(model, train_loader, device, name="Train", results_dir=results_dir, fold_idx=fold_idx)
#         evaluate(model, val_loader, device, name="Validation", results_dir=results_dir, fold_idx=fold_idx)
#         evaluate(model, test_loader, device, name="Test", results_dir=results_dir, fold_idx=fold_idx)

# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()
















# ################## custom split
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix

# from loadData import get_data_loaders
# from createModel import ResNet18Custom
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging

# def train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4):
#     logger = logging.getLogger()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     early_stopper = EarlyStopping(patience=5, min_delta=None, verbose=True)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info("Starting training")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

#         model.train()
#         running_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)

#         model.eval()
#         running_loss = 0.0
#         for inputs, targets in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             running_loss += loss.item() * inputs.size(0)

#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.error(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training completed in {total_minutes:.2f} minutes")

#     plt.figure(figsize=(8,5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Loss Curves with EarlyStopping')
#     plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
#     plt.close()

# def evaluate(model, loader, device, name, results_dir):
#     logger = logging.getLogger()
#     model.eval()
#     all_preds, all_targets = [], []
#     for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_targets.extend(targets.cpu().numpy())

#     report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
#     df = pd.DataFrame(report_dict).transpose()
#     logger.info(f"\n{name} Classification Report:\n{df.to_string()}\n")

#     cm = confusion_matrix(all_targets, all_preds)
#     plt.figure(figsize=(5,4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean','Noisy'], yticklabels=['Clean','Noisy'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(f'{name} Confusion Matrix')
#     plt.savefig(os.path.join(results_dir, f'{name.lower()}_confusion_matrix.png'))
#     plt.close()

# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, batch_size=16, image_size=224)
#     train_loader, val_loader, test_loader = loaders["classification"]

#     model = ResNet18Custom(num_classes=2, in_channels=1, pretrained=True).to(device)
#     train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4)

#     model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
#     evaluate(model, train_loader, device, name="Train", results_dir=results_dir)
#     evaluate(model, val_loader, device, name="Validation", results_dir=results_dir)
#     evaluate(model, test_loader, device, name="Test", results_dir=results_dir)

# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()


















################################## Graustufen
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
# from loadData import get_data_loaders
# from createModel import ResNet18Gray
# from earlyStopping import EarlyStopping
# from trainLogging import setup_logging
# import logging
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns


# def train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4):
#     logger = logging.getLogger()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=lr)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     early_stopper = EarlyStopping(patience=5, min_delta=None, verbose=True)

#     train_losses, val_losses = [], []
#     best_val_loss = float('inf')
#     start_time = time.time()

#     logger.info("Starting training")
#     for epoch in range(1, num_epochs + 1):
#         epoch_start = time.time()
#         logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

#         model.train()
#         running_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Train {epoch}", unit="batch"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)

#         train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)

#         model.eval()
#         running_loss = 0.0
#         for inputs, targets in tqdm(val_loader, desc=f"Val {epoch}", unit="batch", leave=False):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             running_loss += loss.item() * inputs.size(0)

#         val_loss = running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
#             logger.info(f"Best model saved: val_loss={best_val_loss:.4f}")

#         elapsed = time.time() - epoch_start
#         logger.info(f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.1f}s")

#         if early_stopper(val_loss):
#             logger.error(f"Early stopping at epoch {epoch}")
#             break

#         scheduler.step()

#     total_minutes = (time.time() - start_time) / 60
#     logger.info(f"Training completed in {total_minutes:.2f} minutes")

#     plt.figure(figsize=(8,5))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Loss Curves with EarlyStopping')
#     plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
#     plt.close()

# def evaluate(model, loader, device, name, results_dir):
#     logger = logging.getLogger()
#     model.eval()
#     all_preds, all_targets = [], []
#     for inputs, targets in tqdm(loader, desc=f"Eval {name}", unit="batch"):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_targets.extend(targets.cpu().numpy())

#     report = classification_report(all_targets, all_preds, digits=4)
#     logger.info(f"\n{name} Metrics:\n{report}")
#     cm = confusion_matrix(all_targets, all_preds)
#     plt.figure(figsize=(5,4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Clean','Noisy'], yticklabels=['Clean','Noisy'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title(f'{name} Confusion Matrix')
#     plt.savefig(os.path.join(results_dir, f'{name.lower()}_confusion_matrix.png'))
#     plt.close()

# def main():
#     results_dir = os.path.join(os.path.dirname(__file__), "results")
#     logger = setup_logging(results_dir, log_file="terminal_output.txt", level="INFO")
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, batch_size=16, image_size=224)
#     train_loader, val_loader, test_loader = loaders["classification"]

#     model = ResNet18Gray(num_classes=2, pretrained=True).to(device)
#     train(model, train_loader, val_loader, device, results_dir, num_epochs=50, lr=1e-4)

#     model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
#     evaluate(model, train_loader, device, name="Train", results_dir=results_dir)
#     evaluate(model, val_loader, device, name="Validation", results_dir=results_dir)
#     evaluate(model, test_loader, device, name="Test", results_dir=results_dir)

# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()
