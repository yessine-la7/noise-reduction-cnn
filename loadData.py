# # ################################## Graustufen split MIL
# import os
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# import torchvision.transforms as transforms
# import logging

# logger = logging.getLogger()

# # --- Hilfsfunktion: Extrahiere überlappende Patches ---
# def extract_patches(image, patch_size, stride):
#     _, h, w = image.size()
#     patches = []
#     for i in range(0, h - patch_size + 1, stride):
#         for j in range(0, w - patch_size + 1, stride):
#             patch = image[:, i:i + patch_size, j:j + patch_size]
#             patches.append(patch)
#     return torch.stack(patches)

# # --- 1. Classification Dataset mit MIL ---
# class MILClassificationSpectrogramDataset(Dataset):
#     def __init__(self, clean_dir, noisy_dir, image_size=256, patch_size=64, stride=32):
#         self.files = []
#         for f in os.listdir(clean_dir):
#             if f.endswith('.png'):
#                 self.files.append((os.path.join(clean_dir, f), 0))
#         for f in os.listdir(noisy_dir):
#             if f.endswith('.png'):
#                 self.files.append((os.path.join(noisy_dir, f), 1))
#         self.files.sort(key=lambda x: x[0])

#         self.transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#         self.patch_size = patch_size
#         self.stride = stride

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path, label = self.files[idx]
#         img = Image.open(path).convert('L')
#         img = self.transform(img)
#         patches = extract_patches(img, self.patch_size, self.stride)
#         return patches, torch.tensor(label, dtype=torch.long)

# # --- 2. Denoising Dataset mit MIL ---
# class MILPairedSpectrogramDataset(Dataset):
#     def __init__(self, noisy_dir, clean_dir, image_size=256, patch_size=64, stride=32):
#         self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
#         self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.png')])
#         assert len(self.noisy_files) == len(self.clean_files), "Anzahl noisy/clean stimmt nicht!"
#         for nf, cf in zip(self.noisy_files, self.clean_files):
#             assert nf == cf, f"Dateien stimmen nicht: {nf} ≠ {cf}"

#         self.noisy_dir, self.clean_dir = noisy_dir, clean_dir
#         self.transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#         self.patch_size = patch_size
#         self.stride = stride

#     def __len__(self):
#         return len(self.noisy_files)

#     def __getitem__(self, idx):
#         noisy = Image.open(os.path.join(self.noisy_dir, self.noisy_files[idx])).convert('L')
#         clean = Image.open(os.path.join(self.clean_dir, self.clean_files[idx])).convert('L')
#         noisy = self.transform(noisy)
#         clean = self.transform(clean)
#         noisy_patches = extract_patches(noisy, self.patch_size, self.stride)
#         clean_patches = extract_patches(clean, self.patch_size, self.stride)
#         return noisy_patches, clean_patches

# # --- 3. Loader-Funktion für Classification und Denoising mit MIL ---
# def get_data_loaders(dataset_path, batch_size=1, image_size=256, patch_size=64, stride=32,
#                      num_workers=4, val_split=0.2):
#     clean_train = os.path.join(dataset_path, "clean_trainset_56spk_mel")
#     noisy_train = os.path.join(dataset_path, "noisy_trainset_56spk_mel")
#     clean_test = os.path.join(dataset_path, "clean_testset_mel")
#     noisy_test = os.path.join(dataset_path, "noisy_testset_mel")

#     # Classification
#     class_ds = MILClassificationSpectrogramDataset(clean_train, noisy_train, image_size, patch_size, stride)
#     test_cls = MILClassificationSpectrogramDataset(clean_test, noisy_test, image_size, patch_size, stride)
#     n = len(class_ds)
#     n_val = int(val_split * n)
#     n_train = n - n_val
#     train_cls, val_cls = random_split(class_ds, [n_train, n_val])
#     train_loader_cls = DataLoader(train_cls, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader_cls   = DataLoader(val_cls, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     test_loader_cls  = DataLoader(test_cls, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     # Denoising
#     paired_ds = MILPairedSpectrogramDataset(noisy_train, clean_train, image_size, patch_size, stride)
#     test_dn = MILPairedSpectrogramDataset(noisy_test, clean_test, image_size, patch_size, stride)
#     n = len(paired_ds)
#     n_val = int(val_split * n)
#     n_train = n - n_val
#     train_dn, val_dn = random_split(paired_ds, [n_train, n_val])
#     train_loader_dn = DataLoader(train_dn, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader_dn   = DataLoader(val_dn, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     test_loader_dn  = DataLoader(test_dn, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     logger.info(f"Classification: Train {len(train_cls)}, Val {len(val_cls)}, Test {len(test_cls)}")
#     logger.info(f"Denoising: Train {len(train_dn)}, Val {len(val_dn)}, Test {len(test_dn)}")

#     return {
#         "classification": (train_loader_cls, val_loader_cls, test_loader_cls),
#         "denoising": (train_loader_dn, val_loader_dn, test_loader_dn)
#     }


# if __name__ == "__main__":
#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH)
#     train_cls, val_cls, test_cls = loaders["classification"]
#     train_dn, val_dn, test_dn = loaders["denoising"]






################################## Graustufen split
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import logging


logger = logging.getLogger()

# --- 1. Classificationsdataset: clean + noisy ---
class ClassificationSpectrogramDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, image_size=256, transform=None):
        self.files = []
        for f in os.listdir(clean_dir):
            if f.endswith('.png'):
                self.files.append((os.path.join(clean_dir, f), 0))
        for f in os.listdir(noisy_dir):
            if f.endswith('.png'):
                self.files.append((os.path.join(noisy_dir, f), 1))
        self.files.sort(key=lambda x: x[0])

        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        img = Image.open(path).convert('L')
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# --- 2. Paired dataset for Denoising ---
class PairedSpectrogramDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, image_size=256, transform=None):
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.png')])
        assert len(self.noisy_files) == len(self.clean_files), "Anzahl noisy/clean stimmt nicht!"
        for nf, cf in zip(self.noisy_files, self.clean_files):
            assert nf == cf, f"Dateien stimmen nicht: {nf} ≠ {cf}"

        self.noisy_dir, self.clean_dir = noisy_dir, clean_dir
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        n = Image.open(os.path.join(self.noisy_dir, self.noisy_files[idx])).convert('L')
        c = Image.open(os.path.join(self.clean_dir, self.clean_files[idx])).convert('L')
        return self.transform(n), self.transform(c)


# --- 3. DataLoader-Funktion für beide Szenarien ---
def get_data_loaders(dataset_path, batch_size=8, image_size=256,
                     num_workers=4, val_split=0.2):
    clean_train = os.path.join(dataset_path, "clean_trainset_56spk_mel")
    noisy_train = os.path.join(dataset_path, "noisy_trainset_56spk_mel")
    clean_test = os.path.join(dataset_path, "clean_testset_mel")
    noisy_test = os.path.join(dataset_path, "noisy_testset_mel")

    # Klassifikations-Dataset
    class_ds = ClassificationSpectrogramDataset(clean_train, noisy_train, image_size)
    test_cls = ClassificationSpectrogramDataset(clean_test, noisy_test, image_size)
    n = len(class_ds)
    n_val = int(val_split * n)
    n_train = n - n_val
    train_cls, val_cls = random_split(class_ds, [n_train, n_val])
    train_loader_cls = DataLoader(train_cls, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_cls   = DataLoader(val_cls, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_cls = DataLoader(test_cls, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Denoising-Dataset
    paired_train = PairedSpectrogramDataset(noisy_train, clean_train, image_size=image_size)
    paired_test = PairedSpectrogramDataset(noisy_test, clean_test, image_size=image_size)
    n = len(paired_train)
    n_val = int(val_split * n)
    n_train = n - n_val
    train_dn, val_dn = random_split(paired_train, [n_train, n_val])
    train_loader_dn = DataLoader(train_dn, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_dn   = DataLoader(val_dn, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_dn  = DataLoader(paired_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info(f"Classification: Train {len(train_cls)}, Val {len(val_cls)}, Test {len(test_cls)}")
    logger.info(f"Denoising: Train {len(train_dn)}, Val {len(val_dn)}, Test {len(paired_test)}")


    return {
        "classification": (train_loader_cls, val_loader_cls, test_loader_cls),
        "denoising": (train_loader_dn, val_loader_dn, test_loader_dn)
    }


if __name__ == "__main__":
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
    loaders = get_data_loaders(DATASET_PATH)
    train_cls, val_cls, test_cls = loaders["classification"]
    train_dn, val_dn, test_dn = loaders["denoising"]





# ################################ Graustufen cross validation
# import os
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split, Subset
# import torchvision.transforms as transforms
# import logging
# from sklearn.model_selection import KFold

# logger = logging.getLogger()

# class ClassificationSpectrogramDataset(Dataset):
#     def __init__(self, clean_dir, noisy_dir, image_size=256, transform=None):
#         self.files = []
#         for f in os.listdir(clean_dir):
#             if f.endswith('.png'):
#                 self.files.append((os.path.join(clean_dir, f), 0))
#         for f in os.listdir(noisy_dir):
#             if f.endswith('.png'):
#                 self.files.append((os.path.join(noisy_dir, f), 1))
#         self.files.sort(key=lambda x: x[0])
#         self.transform = transform or transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#     def __len__(self):
#         return len(self.files)
#     def __getitem__(self, idx):
#         path, label = self.files[idx]
#         img = Image.open(path).convert('L')
#         img = self.transform(img)
#         return img, torch.tensor(label, dtype=torch.long)

# class PairedSpectrogramDataset(Dataset):
#     def __init__(self, noisy_dir, clean_dir, image_size=256, transform=None):
#         self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
#         self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.png')])
#         assert len(self.noisy_files) == len(self.clean_files), "Anzahl noisy/clean stimmt nicht!"
#         for nf, cf in zip(self.noisy_files, self.clean_files):
#             assert nf == cf, f"{nf} ≠ {cf}"
#         self.noisy_dir, self.clean_dir = noisy_dir, clean_dir
#         self.transform = transform or transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#     def __len__(self):
#         return len(self.noisy_files)
#     def __getitem__(self, idx):
#         n = Image.open(os.path.join(self.noisy_dir, self.noisy_files[idx])).convert('L')
#         c = Image.open(os.path.join(self.clean_dir, self.clean_files[idx])).convert('L')
#         return self.transform(n), self.transform(c)

# def get_data_loaders(dataset_path, batch_size=16, image_size=256,
#                      num_workers=4, val_split=0.2, folds=5):
#     clean_train = os.path.join(dataset_path, "clean_trainset_56spk_mel")
#     noisy_train = os.path.join(dataset_path, "noisy_trainset_56spk_mel")
#     clean_test = os.path.join(dataset_path, "clean_testset_mel")
#     noisy_test = os.path.join(dataset_path, "noisy_testset_mel")

#     # Klassifikations-Dataset
#     class_ds = ClassificationSpectrogramDataset(clean_train, noisy_train, image_size)

#     # Setup 5-Fold Cross Validation für Klassifikation
#     kf = KFold(n_splits=folds, shuffle=True, random_state=42)
#     class_folds = []
#     for fold_idx, (train_idx, val_idx) in enumerate(kf.split(class_ds), start=1):
#         train_sub = Subset(class_ds, train_idx)
#         val_sub = Subset(class_ds, val_idx)
#         train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#         val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#         class_folds.append((train_loader, val_loader))
#         logger.info(f"Classification Fold {fold_idx}: Train {len(train_sub)}, Val {len(val_sub)}")
#     # kompletter Test für Klassifikation
#     test_cls = ClassificationSpectrogramDataset(clean_test, noisy_test, image_size)
#     test_loader_cls = DataLoader(test_cls, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     # Denoising-Dataset mit normalem Split
#     paired_ds = PairedSpectrogramDataset(noisy_train, clean_train, image_size=image_size)
#     n = len(paired_ds)
#     n_val = int(val_split * n)
#     train_dn, val_dn = random_split(paired_ds, [n - n_val, n_val])
#     train_dn_loader = DataLoader(train_dn, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_dn_loader = DataLoader(val_dn, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     paired_test = PairedSpectrogramDataset(noisy_test, clean_test, image_size=image_size)
#     test_dn_loader = DataLoader(paired_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     logger.info(f"Denoising: Train {len(train_dn)}, Val {len(val_dn)}, Test {len(paired_test)}")

#     return {
#         "classification": {
#             "folds": class_folds,
#             "test": test_loader_cls
#         },
#         "denoising": (train_dn_loader, val_dn_loader, test_dn_loader)
#     }

# if __name__ == "__main__":
#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH)
#     cls = loaders["classification"]
#     logger.info(f"Classification Test-size: {len(cls['test'].dataset)}")
#     for i, (tr, va) in enumerate(cls['folds']):
#         logger.info(f"Fold {i+1}: train {len(tr.dataset)}, val {len(va.dataset)}")
#     dn = loaders["denoising"]
#     logger.info(f"Denoising train/val/test sizes: {[len(x.dataset) for x in dn]}")




# ########################################################## RGB split
# import os
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# import torchvision.transforms as transforms
# from trainLogging import setup_logging
# import logging

# logger = logging.getLogger()

# # --- 1. ClassificationDataset: clean + noisy in RGB ---
# class ClassificationSpectrogramDataset(Dataset):
#     def __init__(self, clean_dir, noisy_dir, image_size=256, transform=None):
#         self.files = []
#         for f in os.listdir(clean_dir):
#             if f.endswith('.png'):
#                 self.files.append((os.path.join(clean_dir, f), 0))
#         for f in os.listdir(noisy_dir):
#             if f.endswith('.png'):
#                 self.files.append((os.path.join(noisy_dir, f), 1))
#         self.files.sort(key=lambda x: x[0])

#         self.transform = transform or transforms.Compose([
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path, label = self.files[idx]
#         img = Image.open(path).convert('RGB')  # RGB conversion
#         img = self.transform(img)
#         return img, torch.tensor(label, dtype=torch.long)

# # --- 2. Paired dataset for denoising in RGB ---
# class PairedSpectrogramDataset(Dataset):
#     def __init__(self, noisy_dir, clean_dir, image_size=256, transform=None):
#         self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
#         self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.png')])
#         assert len(self.noisy_files) == len(self.clean_files), "Anzahl noisy/clean stimmt nicht!"
#         for nf, cf in zip(self.noisy_files, self.clean_files):
#             assert nf == cf, f"Dateien stimmen nicht: {nf} ≠ {cf}"

#         self.noisy_dir, self.clean_dir = noisy_dir, clean_dir
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])

#     def __len__(self):
#         return len(self.noisy_files)

#     def __getitem__(self, idx):
#         n = Image.open(os.path.join(self.noisy_dir, self.noisy_files[idx])).convert('RGB')
#         c = Image.open(os.path.join(self.clean_dir, self.clean_files[idx])).convert('RGB')
#         return self.transform(n), self.transform(c)

# # --- 3. Loader-Funktion für beide Szenarien ---
# def get_data_loaders(dataset_path, batch_size=16, image_size=256,
#                      num_workers=4, val_split=0.2):
#     clean_train = os.path.join(dataset_path, "clean_trainset_56spk_mel")
#     noisy_train = os.path.join(dataset_path, "noisy_trainset_56spk_mel")
#     clean_test = os.path.join(dataset_path, "clean_testset_mel")
#     noisy_test = os.path.join(dataset_path, "noisy_testset_mel")

#     class_ds = ClassificationSpectrogramDataset(clean_train, noisy_train, image_size)
#     test_cls = ClassificationSpectrogramDataset(clean_test, noisy_test, image_size)
#     n = len(class_ds)
#     n_val = int(val_split * n)
#     train_cls, val_cls = random_split(class_ds, [n - n_val, n_val])
#     train_loader_cls = DataLoader(train_cls, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader_cls = DataLoader(val_cls, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     test_loader_cls = DataLoader(test_cls, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     paired_train = PairedSpectrogramDataset(noisy_train, clean_train, image_size=image_size)
#     paired_test = PairedSpectrogramDataset(noisy_test, clean_test, image_size=image_size)
#     n_p = len(paired_train)
#     n_val_p = int(val_split * n_p)
#     train_dn, val_dn = random_split(paired_train, [n_p - n_val_p, n_val_p])
#     train_loader_dn = DataLoader(train_dn, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader_dn = DataLoader(val_dn, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     test_loader_dn = DataLoader(paired_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     logger.info(f"Classification: Train={len(train_cls)}, Val={len(val_cls)}, Test={len(test_cls)}")
#     logger.info(f"Denoising: Train={len(train_dn)}, Val={len(val_dn)}, Test={len(paired_test)}")

#     return {
#         "classification": (train_loader_cls, val_loader_cls, test_loader_cls),
#         "denoising": (train_loader_dn, val_loader_dn, test_loader_dn)
#     }

# if __name__ == "__main__":
#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH)
#     train_c, val_c, test_c = loaders["classification"]
#     train_d, val_d, test_d = loaders["denoising"]
