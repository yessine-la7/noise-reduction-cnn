################################## mil
import os
import random
import logging
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -------------------------------
# Hilfsfunktionen
# -------------------------------
def _list_png_files(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Folder not found: {dir_path}")
    return sorted(
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(".png")
    )


def _basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def deterministic_classwise_split(
    clean_files: List[str],
    noisy_files: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Datei-weise, deterministische Aufteilung je Klasse.
    """
    rng = random.Random(seed)
    clean = sorted(clean_files)
    noisy = sorted(noisy_files)
    rng.shuffle(clean)
    rng.shuffle(noisy)

    n_val_clean = int(round(val_ratio * len(clean)))
    n_val_noisy = int(round(val_ratio * len(noisy)))

    val_clean = clean[:n_val_clean]
    train_clean = clean[n_val_clean:]
    val_noisy = noisy[:n_val_noisy]
    train_noisy = noisy[n_val_noisy:]

    train_list = [(p, 0) for p in train_clean] + [(p, 1) for p in train_noisy]
    val_list = [(p, 0) for p in val_clean] + [(p, 1) for p in val_noisy]

    train_list.sort(key=lambda x: x[0])
    val_list.sort(key=lambda x: x[0])
    return train_list, val_list


def pair_noisy_clean_files(
    noisy_dir: str, clean_dir: str
) -> List[Tuple[str, str]]:
    """
    Bildet Paare (noisy_path, clean_path) über identische Basenamen.
    """
    noisy_files = _list_png_files(noisy_dir)
    clean_files = _list_png_files(clean_dir)
    map_clean = {_basename_no_ext(p): p for p in clean_files}

    pairs: List[Tuple[str, str]] = []
    missing_clean: List[str] = []

    for npath in noisy_files:
        base = _basename_no_ext(npath)
        cpath = map_clean.get(base)
        if cpath is not None:
            pairs.append((npath, cpath))
        else:
            missing_clean.append(base)

    if not pairs:
        raise RuntimeError(
            f"Keine Paare gefunden zwischen\n  noisy={noisy_dir}\n  clean={clean_dir}.\n"
            "Bitte Dateinamen angleichen."
        )
    if missing_clean:
        logger.warning(f"[Denoising] {len(missing_clean)} noisy-Dateien ohne Clean-Version.")

    pairs.sort(key=lambda t: t[0])
    return pairs


def deterministic_paired_split(
    pairs: List[Tuple[str, str]],
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    rng = random.Random(seed)
    idxs = list(range(len(pairs)))
    rng.shuffle(idxs)
    n_val = int(round(val_ratio * len(pairs)))
    val_idx = set(idxs[:n_val])
    train_pairs, val_pairs = [], []
    for i, p in enumerate(pairs):
        (val_pairs if i in val_idx else train_pairs).append(p)
    train_pairs.sort(key=lambda t: t[0])
    val_pairs.sort(key=lambda t: t[0])
    return train_pairs, val_pairs


def _pad_width_to_at_least(img: Image.Image, min_w: int, pad_value: int = 0) -> Image.Image:
    """
    Falls die Bildbreite < min_w ist, rechtsseitiges Padding bis genau min_w.
    Höhe bleibt unverändert.
    """
    W, H = img.size
    if W >= min_w:
        return img
    pad_right = min_w - W
    return ImageOps.expand(img, border=(0, 0, pad_right, 0), fill=pad_value)


# -------------------------------
# Klassifikation
# -------------------------------
class ClassificationMILDataset(Dataset):
    """
    Kachelt Bilder entlang der Zeitachse in überlappende Tiles.
    """
    def __init__(
        self,
        files_with_labels: List[Tuple[str, int]],
        in_channels: int = 1,
        tile_h: int = 128,
        tile_w: int = 256,
        stride_w: int = 128,
        normalize_mean_std: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
    ):
        super().__init__()
        assert in_channels in (1, 3)
        self.files: List[Tuple[str, int]] = files_with_labels
        self.in_channels = in_channels
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.stride_w = stride_w

        if normalize_mean_std is None:
            normalize_mean_std = ((0.5,), (0.5,)) if in_channels == 1 else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        to_tensor_tfms = [transforms.ToTensor(), transforms.Normalize(*normalize_mean_std)]
        self.pre = transforms.Compose(([transforms.Grayscale(num_output_channels=1)] if in_channels == 1 else []) + to_tensor_tfms)

        self.index: List[Tuple[int, int]] = []
        for fid, (path, _) in enumerate(self.files):
            with Image.open(path) as im:
                im = im.convert("L") if self.in_channels == 1 else im.convert("RGB")
                W, H = im.size
                x = 0
                if W <= self.tile_w:
                    self.index.append((fid, 0))
                else:
                    while True:
                        self.index.append((fid, x))
                        x += self.stride_w
                        if x + self.tile_w >= W:
                            break
        logger.info(f"[ClassificationMILDataset] Dateien: {len(self.files)}, Tiles: {len(self.index)}")

    def __len__(self) -> int:
        return len(self.index)

    def _load_and_preprocess(self, path: str) -> Image.Image:
        im = Image.open(path)
        im = im.convert("L") if self.in_channels == 1 else im.convert("RGB")
        return im

    def __getitem__(self, idx: int):
        fid, x_left = self.index[idx]
        path, label = self.files[fid]
        im = self._load_and_preprocess(path)
        W, H = im.size

        # Sicherheit: Höhe sollte 128 sein
        if H != self.tile_h:
            raise ValueError(f"Erwartete Mel-Höhe {self.tile_h}, bekam {H} für Datei {path}")

        if x_left + self.tile_w > W:
            x_left = max(0, W - self.tile_w)
        box = (x_left, 0, x_left + self.tile_w, self.tile_h)
        tile = im.crop(box)  # füllt rechts mit 0
        tile_t = self.pre(tile)
        return tile_t, torch.tensor(label, dtype=torch.long), torch.tensor(fid, dtype=torch.long)


# -------------------------------
# Paired Denoising
# -------------------------------
class DenoisingPairedDataset(Dataset):
    """
    Liefert (noisy_tensor, clean_tensor) als Paar in Kacheln (Tiles).
    Zeile 0 wird entfernt -> feste Höhe 512.
    """
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        in_channels: int = 1,
        tile_h: int = 512,          # fest 512
        tile_w: int = 256,
        stride_w: int = 128,
        normalize_mean_std: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
    ):
        super().__init__()
        assert in_channels in (1, 3)
        self.pairs = pairs
        self.in_channels = in_channels
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.stride_w = stride_w

        if normalize_mean_std is None:
            normalize_mean_std = ((0.5,), (0.5,)) if in_channels == 1 else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        to_tensor_tfms = [transforms.ToTensor(), transforms.Normalize(*normalize_mean_std)]
        self.pre = transforms.Compose(([transforms.Grayscale(num_output_channels=1)] if in_channels == 1 else []) + to_tensor_tfms)

        self.index: List[Tuple[int, int]] = []
        for pid, (npath, _cpath) in enumerate(self.pairs):
            with Image.open(npath) as nim:
                nim = nim.convert("L") if self.in_channels == 1 else nim.convert("RGB")
                W, _ = nim.size
                if W <= self.tile_w:
                    self.index.append((pid, 0))
                else:
                    x = 0
                    while True:
                        self.index.append((pid, x))
                        x += self.stride_w
                        if x + self.tile_w >= W:
                            break
        logger.info(f"[DenoisingPairedDataset/Tiles] Paare: {len(self.pairs)}, Tiles: {len(self.index)}")

    def __len__(self) -> int:
        return len(self.index)

    def _load_pair(self, pid: int) -> Tuple[Image.Image, Image.Image]:
        npath, cpath = self.pairs[pid]
        noisy = Image.open(npath)
        clean = Image.open(cpath)
        noisy = noisy.convert("L") if self.in_channels == 1 else noisy.convert("RGB")
        clean = clean.convert("L") if self.in_channels == 1 else clean.convert("RGB")
        return noisy, clean

    @staticmethod
    def _remove_dc_row(img: Image.Image) -> Image.Image:
        """Entfernt die erste Pixelzeile -> Höhe 512."""
        W, H = img.size
        return img.crop((0, 1, W, H))

    def _crop_tile(self, img: Image.Image, x_left: int) -> Image.Image:
        """
        Schneidet einen Tile von Breite tile_w und Höhe tile_h aus.
        Wenn Breite < tile_w, vorher rechts auf tile_w auffüllen.
        """
        W, H = img.size
        if H != self.tile_h:
            raise ValueError(f"Erwartete Höhe {self.tile_h}, bekam {H}")
        if W < self.tile_w:
            img = _pad_width_to_at_least(img, self.tile_w, pad_value=0)
            W = self.tile_w
        if x_left + self.tile_w > W:
            x_left = max(0, W - self.tile_w)
        return img.crop((x_left, 0, x_left + self.tile_w, self.tile_h))


    def __getitem__(self, idx):
        pid, x_left = self.index[idx]
        noisy, clean = self._load_pair(pid)

        noisy = self._remove_dc_row(noisy)
        clean = self._remove_dc_row(clean)

        # Sicherheit: identische Breite der Paare
        assert noisy.size[0] == clean.size[0], \
            f"Width mismatch: noisy={noisy.size[0]} vs clean={clean.size[0]} (pair_id={pid})"

        noisy_t = self._crop_tile(noisy, x_left)
        clean_t = self._crop_tile(clean, x_left)

        x = self.pre(noisy_t)   # (C, 512, tile_w)
        y = self.pre(clean_t)   # (C, 512, tile_w)
        return x, y


# -------------------------------
# Loader-Erzeugung
# -------------------------------
def get_data_loaders(
    dataset_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    # Klassifikation
    enable_classification: bool = False,
    cls_in_channels: int = 1,
    cls_tile_h: int = 128,
    cls_tile_w: int = 256,
    cls_stride_w: int = 128,
    cls_val_ratio: float = 0.2,
    cls_seed: int = 42,
    # Denoising
    enable_denoising: bool = False,
    dn_in_channels: int = 1,
    dn_tile_h: int = 512,
    dn_tile_w: int = 256,
    dn_stride_w: int = 128,
    dn_val_ratio: float = 0.2,
    dn_seed: int = 42,
) -> Dict[str, Optional[Tuple[DataLoader, DataLoader, DataLoader]]]:

    out: Dict[str, Optional[Tuple[DataLoader, DataLoader, DataLoader]]] = {
        "classification": None,
        "denoising": None,
    }

    g_cls = torch.Generator().manual_seed(cls_seed)
    g_dn  = torch.Generator().manual_seed(dn_seed)

    # --- Klassifikation (Mel) ---
    if enable_classification:
        clean_train_mel = os.path.join(dataset_path, "clean_trainset_56spk_mel")
        noisy_train_mel = os.path.join(dataset_path, "noisy_trainset_56spk_mel")
        clean_test_mel = os.path.join(dataset_path, "clean_testset_mel")
        noisy_test_mel = os.path.join(dataset_path, "noisy_testset_mel")

        clean_train_files = _list_png_files(clean_train_mel)
        noisy_train_files = _list_png_files(noisy_train_mel)
        clean_test_files = _list_png_files(clean_test_mel)
        noisy_test_files = _list_png_files(noisy_test_mel)

        cls_train_files, cls_val_files = deterministic_classwise_split(
            clean_train_files, noisy_train_files, val_ratio=cls_val_ratio, seed=cls_seed
        )

        cls_test_files = ([(p, 0) for p in clean_test_files] +
                          [(p, 1) for p in noisy_test_files])
        cls_test_files.sort(key=lambda x: x[0])

        cls_train_ds = ClassificationMILDataset(
            files_with_labels=cls_train_files,
            in_channels=cls_in_channels,
            tile_h=cls_tile_h,
            tile_w=cls_tile_w,
            stride_w=cls_stride_w,
        )
        cls_val_ds = ClassificationMILDataset(
            files_with_labels=cls_val_files,
            in_channels=cls_in_channels,
            tile_h=cls_tile_h,
            tile_w=cls_tile_w,
            stride_w=cls_stride_w,
        )
        cls_test_ds = ClassificationMILDataset(
            files_with_labels=cls_test_files,
            in_channels=cls_in_channels,
            tile_h=cls_tile_h,
            tile_w=cls_tile_w,
            stride_w=cls_stride_w,
        )

        train_loader_cls = DataLoader(
            cls_train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            worker_init_fn=_seed_worker, generator=g_cls,
        )
        val_loader_cls = DataLoader(
            cls_val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            worker_init_fn=_seed_worker, generator=g_cls,
        )
        test_loader_cls = DataLoader(
            cls_test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            worker_init_fn=_seed_worker, generator=g_cls,
        )

        logger.info(
            f"[Classification/MIL] Dateien Train/Val/Test: "
            f"{len(cls_train_files)}/{len(cls_val_files)}/{len(cls_test_files)} | "
            f"Tiles Train/Val/Test: {len(cls_train_ds)}/{len(cls_val_ds)}/{len(cls_test_ds)}"
        )

        out["classification"] = (train_loader_cls, val_loader_cls, test_loader_cls)

    # --- Denoising (STFT) ---
    if enable_denoising:
        clean_train_stft = os.path.join(dataset_path, "clean_trainset_56spk_stft")
        noisy_train_stft = os.path.join(dataset_path, "noisy_trainset_56spk_stft")
        clean_test_stft  = os.path.join(dataset_path, "clean_testset_stft")
        noisy_test_stft  = os.path.join(dataset_path, "noisy_testset_stft")

        pairs_train_all = pair_noisy_clean_files(noisy_train_stft, clean_train_stft)
        pairs_test_all  = pair_noisy_clean_files(noisy_test_stft,  clean_test_stft)

        dn_train_pairs, dn_val_pairs = deterministic_paired_split(
            pairs_train_all, val_ratio=dn_val_ratio, seed=dn_seed
        )

        dn_train_ds = DenoisingPairedDataset(
            pairs=dn_train_pairs,
            in_channels=dn_in_channels,
            tile_h=dn_tile_h,
            tile_w=dn_tile_w,
            stride_w=dn_stride_w,
        )
        dn_val_ds = DenoisingPairedDataset(
            pairs=dn_val_pairs,
            in_channels=dn_in_channels,
            tile_h=dn_tile_h,
            tile_w=dn_tile_w,
            stride_w=dn_stride_w,
        )
        dn_test_ds = DenoisingPairedDataset(
            pairs=pairs_test_all,
            in_channels=dn_in_channels,
            tile_h=dn_tile_h,
            tile_w=dn_tile_w,
            stride_w=dn_stride_w,
        )

        train_loader_dn = DataLoader(
            dn_train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            worker_init_fn=_seed_worker, generator=g_dn,
        )
        val_loader_dn = DataLoader(
            dn_val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            worker_init_fn=_seed_worker, generator=g_dn,
        )
        test_loader_dn = DataLoader(
            dn_test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            worker_init_fn=_seed_worker, generator=g_dn,
        )

        logger.info(
            f"[Denoising/Paired Tiles] "
            f"Paare Train/Val/Test: {len(dn_train_pairs)}/{len(dn_val_pairs)}/{len(pairs_test_all)} | "
            f"Samples Train/Val/Test: {len(dn_train_ds)}/{len(dn_val_ds)}/{len(dn_test_ds)}"
        )

        out["denoising"] = (train_loader_dn, val_loader_dn, test_loader_dn)

    return out


if __name__ == "__main__":
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
    loaders = get_data_loaders(
        DATASET_PATH,
        enable_classification=True,
        enable_denoising=True,
        dn_in_channels=1,
        dn_tile_h=512,
        dn_tile_w=256,
        dn_stride_w=128,
    )
    summary = {
        k: (
            tuple(len(v[i].dataset) for i in range(3)) if v else None
        )
        for k, v in loaders.items()
    }
    print("OK:", summary)
















# ################################## resize no mil
# import os
# import random
# import logging
# from typing import List, Tuple, Dict, Optional
# from PIL import Image
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms

# logger = logging.getLogger(__name__)


# def _seed_worker(worker_id: int):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


# # -------------------------------
# # Hilfsfunktionen
# # -------------------------------
# def _list_png_files(dir_path: str) -> List[str]:
#     if not os.path.isdir(dir_path):
#         raise FileNotFoundError(f"Folder not found: {dir_path}")
#     return sorted(
#         os.path.join(dir_path, f)
#         for f in os.listdir(dir_path)
#         if f.lower().endswith(".png")
#     )


# def _basename_no_ext(path: str) -> str:
#     return os.path.splitext(os.path.basename(path))[0]


# def deterministic_classwise_split(
#     clean_files: List[str],
#     noisy_files: List[str],
#     val_ratio: float = 0.2,
#     seed: int = 42,
# ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
#     """
#     Datei-weise, deterministische Aufteilung je Klasse.
#     """
#     rng = random.Random(seed)
#     clean = sorted(clean_files)
#     noisy = sorted(noisy_files)
#     rng.shuffle(clean)
#     rng.shuffle(noisy)

#     n_val_clean = int(round(val_ratio * len(clean)))
#     n_val_noisy = int(round(val_ratio * len(noisy)))

#     val_clean = clean[:n_val_clean]
#     train_clean = clean[n_val_clean:]
#     val_noisy = noisy[:n_val_noisy]
#     train_noisy = noisy[n_val_noisy:]

#     train_list = [(p, 0) for p in train_clean] + [(p, 1) for p in train_noisy]
#     val_list = [(p, 0) for p in val_clean] + [(p, 1) for p in val_noisy]

#     train_list.sort(key=lambda x: x[0])
#     val_list.sort(key=lambda x: x[0])
#     return train_list, val_list


# def pair_noisy_clean_files(
#     noisy_dir: str, clean_dir: str
# ) -> List[Tuple[str, str]]:
#     """
#     Bildet Paare (noisy_path, clean_path) über identische Basenamen.
#     """
#     noisy_files = _list_png_files(noisy_dir)
#     clean_files = _list_png_files(clean_dir)
#     map_clean = {_basename_no_ext(p): p for p in clean_files}

#     pairs: List[Tuple[str, str]] = []
#     missing_clean: List[str] = []

#     for npath in noisy_files:
#         base = _basename_no_ext(npath)
#         cpath = map_clean.get(base)
#         if cpath is not None:
#             pairs.append((npath, cpath))
#         else:
#             missing_clean.append(base)

#     if not pairs:
#         raise RuntimeError(
#             f"Keine Paare gefunden zwischen\n  noisy={noisy_dir}\n  clean={clean_dir}.\n"
#             "Bitte Dateinamen angleichen."
#         )
#     if missing_clean:
#         logger.warning(f"[Denoising] {len(missing_clean)} noisy-Dateien ohne Clean-Version.")

#     pairs.sort(key=lambda t: t[0])
#     return pairs


# def deterministic_paired_split(
#     pairs: List[Tuple[str, str]],
#     val_ratio: float = 0.2,
#     seed: int = 42
# ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
#     rng = random.Random(seed)
#     idxs = list(range(len(pairs)))
#     rng.shuffle(idxs)
#     n_val = int(round(val_ratio * len(pairs)))
#     val_idx = set(idxs[:n_val])
#     train_pairs, val_pairs = [], []
#     for i, p in enumerate(pairs):
#         (val_pairs if i in val_idx else train_pairs).append(p)
#     train_pairs.sort(key=lambda t: t[0])
#     val_pairs.sort(key=lambda t: t[0])
#     return train_pairs, val_pairs


# # --- 1. Classificationsdataset: clean + noisy ---
# class ClassificationSpectrogramDataset(Dataset):
#     def __init__(self, files_with_labels: List[Tuple[str, int]], image_size=128, in_channels=1, transform=None):

#         super().__init__()
#         assert in_channels in (1, 3)
#         self.files: List[Tuple[str, int]] = files_with_labels
#         self.in_channels = in_channels

#         if transform:
#             self.transform = transform
#         else:
#             if in_channels == 1:
#                 self.transform = transforms.Compose([
#                     transforms.Grayscale(num_output_channels=1),
#                     transforms.Resize((image_size, image_size)),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5,), (0.5,))
#                 ])
#             elif in_channels == 3:
#                 self.transform = transforms.Compose([
#                     transforms.Resize((image_size, image_size)),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                 ])
#             else:
#                 raise ValueError("in_channels must be 1 (grayscale) or 3 (RGB)")

#         self.in_channels = in_channels

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path, label = self.files[idx]
#         img = Image.open(path)
#         if self.in_channels == 1:
#             img = img.convert('L')
#         elif self.in_channels == 3:
#             img = img.convert('RGB')
#         img = self.transform(img)
#         return img, torch.tensor(label, dtype=torch.long)


# # -------------------------------
# # Loader-Erzeugung
# # -------------------------------
# def get_data_loaders(
#     dataset_path: str,
#     batch_size: int = 8,
#     num_workers: int = 4,
#     # Klassifikation
#     enable_classification: bool = False,
#     cls_in_channels: int = 1,
#     cls_val_ratio: float = 0.2,
#     cls_seed: int = 42,
#     image_size: int = 128,
#     # Denoising
#     enable_denoising: bool = False,
# ) -> Dict[str, Optional[Tuple[DataLoader, DataLoader, DataLoader]]]:

#     out: Dict[str, Optional[Tuple[DataLoader, DataLoader, DataLoader]]] = {
#         "classification": None,
#         "denoising": None,
#     }

#     g_cls = torch.Generator().manual_seed(cls_seed)
#     # g_dn  = torch.Generator().manual_seed(dn_seed)

#     # --- Klassifikation (Mel) ---
#     if enable_classification:
#         clean_train_mel = os.path.join(dataset_path, "clean_trainset_56spk_mel")
#         noisy_train_mel = os.path.join(dataset_path, "noisy_trainset_56spk_mel")
#         clean_test_mel = os.path.join(dataset_path, "clean_testset_mel")
#         noisy_test_mel = os.path.join(dataset_path, "noisy_testset_mel")

#         clean_train_files = _list_png_files(clean_train_mel)
#         noisy_train_files = _list_png_files(noisy_train_mel)
#         clean_test_files = _list_png_files(clean_test_mel)
#         noisy_test_files = _list_png_files(noisy_test_mel)

#         cls_train_files, cls_val_files = deterministic_classwise_split(
#             clean_train_files, noisy_train_files, val_ratio=cls_val_ratio, seed=cls_seed
#         )

#         cls_test_files = ([(p, 0) for p in clean_test_files] +
#                           [(p, 1) for p in noisy_test_files])
#         cls_test_files.sort(key=lambda x: x[0])

#         cls_train_ds = ClassificationSpectrogramDataset(
#             files_with_labels=cls_train_files,
#             in_channels=cls_in_channels,
#             image_size=image_size,
#         )
#         cls_val_ds = ClassificationSpectrogramDataset(
#             files_with_labels=cls_val_files,
#             in_channels=cls_in_channels,
#             image_size=image_size,
#         )
#         cls_test_ds = ClassificationSpectrogramDataset(
#             files_with_labels=cls_test_files,
#             in_channels=cls_in_channels,
#             image_size=image_size,
#         )

#         train_loader_cls = DataLoader(
#             cls_train_ds, batch_size=batch_size, shuffle=True,
#             num_workers=num_workers, pin_memory=True,
#             worker_init_fn=_seed_worker, generator=g_cls,
#         )
#         val_loader_cls = DataLoader(
#             cls_val_ds, batch_size=batch_size, shuffle=False,
#             num_workers=num_workers, pin_memory=True,
#             worker_init_fn=_seed_worker, generator=g_cls,
#         )
#         test_loader_cls = DataLoader(
#             cls_test_ds, batch_size=batch_size, shuffle=False,
#             num_workers=num_workers, pin_memory=True,
#             worker_init_fn=_seed_worker, generator=g_cls,
#         )

#         logger.info(
#             f"[Classification] Dateien Train/Val/Test: "
#             f"{len(cls_train_files)}/{len(cls_val_files)}/{len(cls_test_files)} | "
#             f"Train/Val/Test: {len(cls_train_ds)}/{len(cls_val_ds)}/{len(cls_test_ds)}"
#         )

#         out["classification"] = (train_loader_cls, val_loader_cls, test_loader_cls)

#     # --- Denoising (STFT) ---
#     if enable_denoising:
#         pass

#     return out


# if __name__ == "__main__":
#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(
#         DATASET_PATH,
#         enable_classification=True,
#         enable_denoising=True,
#         dn_in_channels=1,
#         image_size=128
#     )
