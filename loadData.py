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


# # -------------------------------
# # Repro: Worker-Seeding für DataLoader
# # -------------------------------
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
#     Gibt (train_list, val_list) mit (pfad, label) zurück.
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
#     Bildet Paare (noisy_path, clean_path) über identische Basenamen (ohne Extension).
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
#         logger.warning(f"[Denoising] {len(missing_clean)} noisy-Dateien ohne Clean-Gegenstück (ignoriert).")

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
# # Loader-Erzeugung (deterministisch)
# # -------------------------------
# def get_data_loaders(
#     dataset_path: str,
#     batch_size: int = 8,
#     num_workers: int = 4,
#     # Klassifikation (Mel)
#     enable_classification: bool = False,
#     cls_in_channels: int = 1,
#     cls_val_ratio: float = 0.2,
#     cls_seed: int = 42,
#     image_size: int = 128,
#     # Denoising (STFT)
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















################################## Graustufen split MIL mit seed + Paired Denoising (STFT)
"""
- Klassifikation (Mel) mit MIL (Tiles), feste Höhe 128.
- Denoising (STFT) als Paardataset (Noisy↔Clean), DC-Zeile entfernt -> Höhe 512.
"""
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


# -------------------------------
# Repro: Worker-Seeding für DataLoader
# -------------------------------
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
    Gibt (train_list, val_list) mit (pfad, label) zurück.
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
    Bildet Paare (noisy_path, clean_path) über identische Basenamen (ohne Extension).
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
        logger.warning(f"[Denoising] {len(missing_clean)} noisy-Dateien ohne Clean-Gegenstück (ignoriert).")

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
# MIL Dataset (Klassifikation/Mel)
# -------------------------------
class ClassificationMILDataset(Dataset):
    """
    Kachelt PNGs entlang der Zeitachse in überlappende Tiles.
    Für jedes Tile wird (tensor, label, file_id) zurückgegeben.
    Erwartete Höhe der Mel-Spektrogramme: 128.
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

        # baue Index: (file_id, x_left)
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

        # Sicherheit: Mel-Höhe sollte fix 128 sein
        if H != self.tile_h:
            raise ValueError(f"Erwartete Mel-Höhe {self.tile_h}, bekam {H} für Datei {path}")

        if x_left + self.tile_w > W:
            x_left = max(0, W - self.tile_w)
        box = (x_left, 0, x_left + self.tile_w, self.tile_h)
        tile = im.crop(box)  # PIL füllt ggf. rechts mit 0, falls Box überragt
        tile_t = self.pre(tile)
        return tile_t, torch.tensor(label, dtype=torch.long), torch.tensor(fid, dtype=torch.long)


# -------------------------------
# Paired Denoising Dataset (STFT)
# -------------------------------
class DenoisingPairedDataset(Dataset):
    """
    Liefert (noisy_tensor, clean_tensor) als Paar in Kacheln (Tiles).
    - DC-Zeile (Zeile 0) wird entfernt -> feste Höhe 512.
    - Kein vertikales Padding.
    - Horizontales Padding NUR, wenn Breite < tile_w (auf genau tile_w).
    - Paare (noisy, clean) haben identische Breite.
    """
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        in_channels: int = 1,
        tile_h: int = 512,          # fest 512 (513-1)
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

        # Index über Tiles: Liste von (pair_id, x_left)
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
        """Entfernt die erste Pixelzeile (DC-Bin) -> Höhe 512 (ausgehend von 513)."""
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

        # DC-Zeile entfernen (H=513 -> 512)
        noisy = self._remove_dc_row(noisy)
        clean = self._remove_dc_row(clean)

        # Sicherheitsnetz: identische Breite der Paare
        assert noisy.size[0] == clean.size[0], \
            f"Width mismatch: noisy={noisy.size[0]} vs clean={clean.size[0]} (pair_id={pid})"

        noisy_t = self._crop_tile(noisy, x_left)
        clean_t = self._crop_tile(clean, x_left)

        x = self.pre(noisy_t)   # (C, 512, tile_w)
        y = self.pre(clean_t)   # (C, 512, tile_w)
        return x, y


# -------------------------------
# Loader-Erzeugung (deterministisch)
# -------------------------------
def get_data_loaders(
    dataset_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    # Klassifikation (Mel)
    enable_classification: bool = False,
    cls_in_channels: int = 1,
    cls_tile_h: int = 128,
    cls_tile_w: int = 256,
    cls_stride_w: int = 128,
    cls_val_ratio: float = 0.2,
    cls_seed: int = 42,
    # Denoising (STFT)
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



















# ################################## Graustufen split MIL mit seed
# import os
# import random
# from typing import List, Tuple, Dict, Optional

# from PIL import Image
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# import logging

# logger = logging.getLogger(__name__)


# # -------------------------------
# # Repro: Worker-Seeding für DataLoader
# # -------------------------------
# def _seed_worker(worker_id: int):
#     """
#     Sorgt dafür, dass jeder DataLoader-Worker einen reproduzierbaren Seed bekommt.
#     Nutzt den von PyTorch initialisierten Seed als Quelle und verteilt ihn auf
#     numpy/python.random.
#     """
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
#         [
#             os.path.join(dir_path, f)
#             for f in os.listdir(dir_path)
#             if f.lower().endswith(".png")
#         ]
#     )


# def deterministic_classwise_split(
#     clean_files: List[str],
#     noisy_files: List[str],
#     val_ratio: float = 0.2,
#     seed: int = 42,
# ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
#     """
#     Datei-weise, deterministische Aufteilung je Klasse: (1 - val_ratio)/val_ratio. 80/20 (default).
#     Gibt (train_list, val_list) mit (pfad, label) zurück.
#     """
#     rng = random.Random(seed)
#     clean = sorted(clean_files)
#     noisy = sorted(noisy_files)

#     rng.shuffle(clean)
#     rng.shuffle(noisy)

#     n_clean = len(clean)
#     n_noisy = len(noisy)
#     n_val_clean = int(round(val_ratio * n_clean))
#     n_val_noisy = int(round(val_ratio * n_noisy))

#     val_clean = clean[:n_val_clean]
#     train_clean = clean[n_val_clean:]
#     val_noisy = noisy[:n_val_noisy]
#     train_noisy = noisy[n_val_noisy:]

#     train_list = [(p, 0) for p in train_clean] + [(p, 1) for p in train_noisy]
#     val_list = [(p, 0) for p in val_clean] + [(p, 1) for p in val_noisy]

#     train_list.sort(key=lambda x: x[0])
#     val_list.sort(key=lambda x: x[0])
#     return train_list, val_list


# # -------------------------------
# # MIL Dataset (Klassifikation/Mel)
# # -------------------------------
# class ClassificationMILDataset(Dataset):
#     """
#     Kachelt PNGs entlang der Zeitachse in überlappende Tiles.
#     Für jedes Tile wird (tensor, label, file_id) zurückgegeben.
#     Erwartete Höhe der Mel-Spektrogramme: 128.
#     """
#     def __init__(
#         self,
#         files_with_labels: List[Tuple[str, int]],
#         in_channels: int = 1,
#         tile_h: int = 128,
#         tile_w: int = 256,
#         stride_w: int = 128,
#         normalize_mean_std: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
#     ):
#         super().__init__()
#         assert in_channels in (1, 3)
#         self.files: List[Tuple[str, int]] = files_with_labels
#         self.in_channels = in_channels
#         self.tile_h = tile_h
#         self.tile_w = tile_w
#         self.stride_w = stride_w

#         if normalize_mean_std is None:
#             normalize_mean_std = ((0.5,), (0.5,)) if in_channels == 1 else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         to_tensor_tfms = [transforms.ToTensor(), transforms.Normalize(*normalize_mean_std)]
#         self.pre = transforms.Compose(([transforms.Grayscale(num_output_channels=1)] if in_channels == 1 else []) + to_tensor_tfms)

#         # baue Index: (file_id, x_left)
#         self.index: List[Tuple[int, int]] = []
#         for fid, (path, _) in enumerate(self.files):
#             with Image.open(path) as im:
#                 im = im.convert("L") if self.in_channels == 1 else im.convert("RGB")
#                 W, H = im.size
#                 # kein Resize: H sollte 128 sein
#                 x = 0
#                 if W <= self.tile_w:
#                     self.index.append((fid, 0))
#                 else:
#                     while True:
#                         self.index.append((fid, x))
#                         x += self.stride_w
#                         if x + self.tile_w >= W:
#                             break
#         logger.info(f"[ClassificationMILDataset] Dateien: {len(self.files)}, Tiles: {len(self.index)}")

#     def __len__(self) -> int:
#         return len(self.index)

#     def _load_and_preprocess(self, path: str) -> Image.Image:
#         im = Image.open(path)
#         im = im.convert("L") if self.in_channels == 1 else im.convert("RGB")
#         return im

#     def __getitem__(self, idx: int):
#         fid, x_left = self.index[idx]
#         path, label = self.files[fid]
#         im = self._load_and_preprocess(path)
#         W, H = im.size

#         # Sicherheit: Mel-Höhe sollte fix 128 sein
#         if H != self.tile_h:
#             raise ValueError(f"Erwartete Mel-Höhe {self.tile_h}, bekam {H} für Datei {path}")

#         if x_left + self.tile_w > W:
#             x_left = max(0, W - self.tile_w)
#         box = (x_left, 0, x_left + self.tile_w, self.tile_h)
#         tile = im.crop(box)  # PIL füllt ggf. rechts mit 0, falls Box überragt
#         tile_t = self.pre(tile)
#         return tile_t, torch.tensor(label, dtype=torch.long), torch.tensor(fid, dtype=torch.long)


# # -------------------------------
# # Loader-Erzeugung (deterministisch)
# # -------------------------------
# def get_data_loaders(
#     dataset_path: str,
#     batch_size: int = 8,
#     num_workers: int = 4,
#     # Klassifikation (Mel)
#     cls_in_channels: int = 1,
#     cls_tile_h: int = 128,
#     cls_tile_w: int = 256,
#     cls_stride_w: int = 128,
#     cls_val_ratio: float = 0.2,
#     cls_seed: int = 42,
#     enable_classification: bool = True,
#     enable_denoising: bool = False,
# ) -> Dict[str, Optional[Tuple[DataLoader, DataLoader, DataLoader]]]:
#     """
#     Gibt ein Dict zurück mit Schlüsseln:
#         - "classification": (train_loader, val_loader, test_loader) oder None
#         - "denoising": (train_loader, val_loader, test_loader) oder None

#     Reproduzierbarkeit:
#       * deterministische Datei-Splits via cls_seed
#       * deterministisches Shuffling/Worker-Seeding via torch.Generator + _seed_worker
#     """
#     out: Dict[str, Optional[Tuple[DataLoader, DataLoader, DataLoader]]] = {
#         "classification": None,
#         "denoising": None,
#     }

#     # Gemeinsamer Generator für alle Loader → deterministische Reihenfolge
#     g_cls = torch.Generator()
#     g_cls.manual_seed(cls_seed)

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

#         cls_train_ds = ClassificationMILDataset(
#             files_with_labels=cls_train_files,
#             in_channels=cls_in_channels,
#             tile_h=cls_tile_h,
#             tile_w=cls_tile_w,
#             stride_w=cls_stride_w,
#         )
#         cls_val_ds = ClassificationMILDataset(
#             files_with_labels=cls_val_files,
#             in_channels=cls_in_channels,
#             tile_h=cls_tile_h,
#             tile_w=cls_tile_w,
#             stride_w=cls_stride_w,
#         )
#         cls_test_ds = ClassificationMILDataset(
#             files_with_labels=cls_test_files,
#             in_channels=cls_in_channels,
#             tile_h=cls_tile_h,
#             tile_w=cls_tile_w,
#             stride_w=cls_stride_w,
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
#             f"[Classification/MIL] Dateien Train/Val/Test: "
#             f"{len(cls_train_files)}/{len(cls_val_files)}/{len(cls_test_files)} | "
#             f"Tiles Train/Val/Test: {len(cls_train_ds)}/{len(cls_val_ds)}/{len(cls_test_ds)}"
#         )

#         out["classification"] = (train_loader_cls, val_loader_cls, test_loader_cls)

#     # --- Denoising (STFT) – aktuell abgeschaltet ---
#     if enable_denoising:
#         pass

#     return out


# if __name__ == "__main__":
#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, enable_classification=True, enable_denoising=False)
#     print(
#         "OK:",
#         {
#             k: tuple(len(v[i].dataset) for i in range(3)) if v else None
#             for k, v in loaders.items()
#         },
#     )

















# ################################## Graustufen split MIL
# import os
# import random
# from typing import List, Tuple, Dict, Optional

# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# import logging

# logger = logging.getLogger(__name__)


# # -------------------------------
# # Hilfsfunktionen
# # -------------------------------
# def _list_png_files(dir_path: str) -> List[str]:
#     if not os.path.isdir(dir_path):
#         raise FileNotFoundError(f"Folder not found: {dir_path}")
#     return sorted([os.path.join(dir_path, f)
#                    for f in os.listdir(dir_path)
#                    if f.lower().endswith(".png")])

# def deterministic_classwise_split(
#     clean_files: List[str],
#     noisy_files: List[str],
#     val_ratio: float = 0.2,
#     seed: int = 42,
# ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
#     """
#     Datei-weise, deterministische Aufteilung je Klasse: 80/20 (default).
#     Gibt (train_list, val_list) mit (pfad, label) zurück.
#     """
#     rng = random.Random(seed)
#     clean = sorted(clean_files)
#     noisy = sorted(noisy_files)

#     rng.shuffle(clean)
#     rng.shuffle(noisy)

#     n_clean = len(clean)
#     n_noisy = len(noisy)
#     n_val_clean = int(round(val_ratio * n_clean))
#     n_val_noisy = int(round(val_ratio * n_noisy))

#     val_clean = clean[:n_val_clean]
#     train_clean = clean[n_val_clean:]
#     val_noisy = noisy[:n_val_noisy]
#     train_noisy = noisy[n_val_noisy:]

#     train_list = [(p, 0) for p in train_clean] + [(p, 1) for p in train_noisy]
#     val_list = [(p, 0) for p in val_clean] + [(p, 1) for p in val_noisy]

#     train_list.sort(key=lambda x: x[0])
#     val_list.sort(key=lambda x: x[0])
#     return train_list, val_list


# # -------------------------------
# # MIL Dataset (Klassifikation/Mel)
# # -------------------------------
# class ClassificationMILDataset(Dataset):
#     """
#     Kachelt jede PNG-Datei entlang der Zeitachse (Breite) in überlappende Tiles.
#     Für jedes Tile wird (tensor, label, file_id) zurückgegeben.
#     file_id verweist auf self.files (Liste von (pfad,label)).
#     """
#     def __init__(
#         self,
#         files_with_labels: List[Tuple[str, int]],
#         in_channels: int = 1,
#         tile_h: int = 128,
#         tile_w: int = 256,
#         stride_w: int = 128,
#         resize_height: bool = True,
#         normalize_mean_std: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
#     ):
#         super().__init__()
#         assert in_channels in (1, 3)
#         self.files: List[Tuple[str, int]] = files_with_labels
#         self.in_channels = in_channels
#         self.tile_h = tile_h
#         self.tile_w = tile_w
#         self.stride_w = stride_w
#         self.resize_height = resize_height

#         if normalize_mean_std is None:
#             if in_channels == 1:
#                 normalize_mean_std = ((0.5,), (0.5,))
#             else:
#                 normalize_mean_std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

#         # Bild-zu-Tensor
#         to_tensor_tfms = [transforms.ToTensor(), transforms.Normalize(*normalize_mean_std)]
#         if in_channels == 1:
#             self.pre = transforms.Compose([transforms.Grayscale(num_output_channels=1)] + to_tensor_tfms)
#         else:
#             self.pre = transforms.Compose([transforms.ConvertImageDtype(torch.float32)] + to_tensor_tfms)

#         # baue Index: (file_id, x_left) je Tile
#         self.index: List[Tuple[int, int]] = []
#         for fid, (path, _) in enumerate(self.files):
#             with Image.open(path) as im:
#                 if self.in_channels == 1:
#                     im = im.convert("L")
#                 else:
#                     im = im.convert("RGB")
#                 W, H = im.size  # (Breite, Höhe) in Pixeln

#                 # Höhe ggf. auf tile_h skalieren (Mel hat H=128 → verlustfreie Anpassung)
#                 if self.resize_height and H != self.tile_h:
#                     new_w = int(round(W * (self.tile_h / H)))
#                     W = new_w  # nur für Index, tatsächliches Cropping passiert in __getitem__

#                 # entlang Breite sliden
#                 x = 0
#                 if W <= self.tile_w:
#                     self.index.append((fid, 0))
#                 else:
#                     while True:
#                         self.index.append((fid, x))
#                         x += self.stride_w
#                         if x + self.tile_w >= W:
#                             # sicherstellen, dass das letzte Tile am Ende anliegt
#                             if x < W - self.tile_w:
#                                 self.index.append((fid, W - self.tile_w))
#                             break

#         logger.info(f"[ClassificationMILDataset] Dateien: {len(self.files)}, Tiles: {len(self.index)}")

#     def __len__(self) -> int:
#         return len(self.index)

#     def _load_and_preprocess(self, path: str) -> Image.Image:
#         im = Image.open(path)
#         if self.in_channels == 1:
#             im = im.convert("L")
#         else:
#             im = im.convert("RGB")
#         return im

#     def __getitem__(self, idx: int):
#         fid, x_left = self.index[idx]
#         path, label = self.files[fid]

#         im = self._load_and_preprocess(path)
#         W, H = im.size

#         # Höhe ggf. auf tile_h skalieren (proportional in Breite)
#         if self.resize_height and H != self.tile_h:
#             new_w = int(round(W * (self.tile_h / H)))
#             im = im.resize((new_w, self.tile_h), Image.BILINEAR)
#             W, H = im.size  # update

#         # sichere Cropping-Koords
#         if x_left + self.tile_w > W:
#             x_left = max(0, W - self.tile_w)
#         box = (x_left, 0, x_left + self.tile_w, self.tile_h)
#         tile = im.crop(box)

#         tile_t = self.pre(tile)  # (C,H,W) + Normalize
#         return tile_t, torch.tensor(label, dtype=torch.long), torch.tensor(fid, dtype=torch.long)


# # -------------------------------
# # Loader-Erzeugung
# # -------------------------------
# def get_data_loaders(
#     dataset_path: str,
#     batch_size: int = 8,
#     num_workers: int = 4,
#     # Klassifikation (Mel)
#     cls_in_channels: int = 1,
#     cls_tile_h: int = 128,
#     cls_tile_w: int = 256,
#     cls_stride_w: int = 128,
#     cls_resize_height: bool = True,
#     cls_val_ratio: float = 0.2,
#     cls_seed: int = 42,
#     enable_classification: bool = True,
#     # (Denoising könntest du später wieder aktivieren – hier default aus)
#     enable_denoising: bool = False,
# ):
#     """
#     Gibt ein Dict zurück mit Schlüsseln:
#         - "classification": (train_loader, val_loader, test_loader) oder None
#         - "denoising": (train_loader, val_loader, test_loader) oder None
#     """
#     out: Dict[str, Optional[Tuple[DataLoader, DataLoader, DataLoader]]] = {
#         "classification": None,
#         "denoising": None,
#     }

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

#         # deterministischer 80/20 Split je Klasse
#         cls_train_files, cls_val_files = deterministic_classwise_split(
#             clean_train_files, noisy_train_files, val_ratio=cls_val_ratio, seed=cls_seed
#         )

#         # Test-Liste
#         cls_test_files = ([(p, 0) for p in clean_test_files] +
#                           [(p, 1) for p in noisy_test_files])
#         cls_test_files.sort(key=lambda x: x[0])

#         # Datasets
#         cls_train_ds = ClassificationMILDataset(
#             files_with_labels=cls_train_files,
#             in_channels=cls_in_channels,
#             tile_h=cls_tile_h,
#             tile_w=cls_tile_w,
#             stride_w=cls_stride_w,
#             resize_height=cls_resize_height,
#         )
#         cls_val_ds = ClassificationMILDataset(
#             files_with_labels=cls_val_files,
#             in_channels=cls_in_channels,
#             tile_h=cls_tile_h,
#             tile_w=cls_tile_w,
#             stride_w=cls_stride_w,
#             resize_height=cls_resize_height,
#         )
#         cls_test_ds = ClassificationMILDataset(
#             files_with_labels=cls_test_files,
#             in_channels=cls_in_channels,
#             tile_h=cls_tile_h,
#             tile_w=cls_tile_w,
#             stride_w=cls_stride_w,
#             resize_height=cls_resize_height,
#         )

#         # Loader
#         train_loader_cls = DataLoader(cls_train_ds, batch_size=batch_size, shuffle=True,
#                                       num_workers=num_workers, pin_memory=True)
#         val_loader_cls = DataLoader(cls_val_ds, batch_size=batch_size, shuffle=False,
#                                     num_workers=num_workers, pin_memory=True)
#         test_loader_cls = DataLoader(cls_test_ds, batch_size=batch_size, shuffle=False,
#                                      num_workers=num_workers, pin_memory=True)

#         logger.info(
#             f"[Classification/MIL] Dateien Train/Val/Test: "
#             f"{len(cls_train_files)}/{len(cls_val_files)}/{len(cls_test_files)} | "
#             f"Tiles Train/Val/Test: {len(cls_train_ds)}/{len(cls_val_ds)}/{len(cls_test_ds)}"
#         )

#         out["classification"] = (train_loader_cls, val_loader_cls, test_loader_cls)

#     # --- Denoising (STFT) – hier aktuell abgeschaltet ---
#     if enable_denoising:
#         # Platzhalter – du kannst später analog ein MIL-Denoising-Dataset reaktivieren.
#         pass

#     return out


# if __name__ == "__main__":
#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, enable_classification=True, enable_denoising=False)
#     print("OK:", {k: tuple(len(v[i].dataset) for i in range(3)) if v else None for k, v in loaders.items()})










# ################################ custom split
# import os
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# import torchvision.transforms as transforms
# import logging

# logger = logging.getLogger()

# # --- 1. Classificationsdataset: clean + noisy ---
# class ClassificationSpectrogramDataset(Dataset):
#     def __init__(self, clean_dir, noisy_dir, image_size=256, in_channels=1, transform=None):
#         self.files = []
#         for f in os.listdir(clean_dir):
#             if f.endswith('.png'):
#                 self.files.append((os.path.join(clean_dir, f), 0))
#         for f in os.listdir(noisy_dir):
#             if f.endswith('.png'):
#                 self.files.append((os.path.join(noisy_dir, f), 1))
#         self.files.sort(key=lambda x: x[0])

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


# # --- 2. Paired dataset for Denoising ---
# class PairedSpectrogramDataset(Dataset):
#     def __init__(self, noisy_dir, clean_dir, image_size=256, in_channels=1, transform=None):
#         self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
#         self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.png')])
#         assert len(self.noisy_files) == len(self.clean_files), "Anzahl noisy/clean stimmt nicht!"
#         for nf, cf in zip(self.noisy_files, self.clean_files):
#             assert nf == cf, f"Dateien stimmen nicht: {nf} ≠ {cf}"

#         self.noisy_dir = noisy_dir
#         self.clean_dir = clean_dir

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
#                 raise ValueError("in_channels must be 1 or 3")

#         self.in_channels = in_channels

#     def __len__(self):
#         return len(self.noisy_files)

#     def __getitem__(self, idx):
#         n = Image.open(os.path.join(self.noisy_dir, self.noisy_files[idx]))
#         c = Image.open(os.path.join(self.clean_dir, self.clean_files[idx]))
#         if self.in_channels == 1:
#             n = n.convert('L')
#             c = c.convert('L')
#         elif self.in_channels == 3:
#             n = n.convert('RGB')
#             c = c.convert('RGB')
#         return self.transform(n), self.transform(c)


# # --- 3. DataLoader-Funktion für beide Szenarien ---
# def get_data_loaders(dataset_path, batch_size=8, image_size=256,
#                      num_workers=4, val_split=0.2, in_channels=1):
#     clean_train = os.path.join(dataset_path, "clean_trainset_56spk_mel")
#     noisy_train = os.path.join(dataset_path, "noisy_trainset_56spk_mel")
#     clean_test = os.path.join(dataset_path, "clean_testset_mel")
#     noisy_test = os.path.join(dataset_path, "noisy_testset_mel")

#     # Klassifikations-Dataset
#     class_ds = ClassificationSpectrogramDataset(clean_train, noisy_train, image_size=image_size, in_channels=in_channels)
#     test_cls = ClassificationSpectrogramDataset(clean_test, noisy_test, image_size=image_size, in_channels=in_channels)
#     n = len(class_ds)
#     n_val = int(val_split * n)
#     n_train = n - n_val
#     train_cls, val_cls = random_split(class_ds, [n_train, n_val])
#     train_loader_cls = DataLoader(train_cls, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader_cls   = DataLoader(val_cls, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     test_loader_cls  = DataLoader(test_cls, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     # Denoising-Dataset
#     paired_train = PairedSpectrogramDataset(noisy_train, clean_train, image_size=image_size, in_channels=in_channels)
#     paired_test  = PairedSpectrogramDataset(noisy_test, clean_test, image_size=image_size, in_channels=in_channels)
#     n = len(paired_train)
#     n_val = int(val_split * n)
#     n_train = n - n_val
#     train_dn, val_dn = random_split(paired_train, [n_train, n_val])
#     train_loader_dn = DataLoader(train_dn, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader_dn   = DataLoader(val_dn, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     test_loader_dn  = DataLoader(paired_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     logger.info(f"Classification: Train {len(train_cls)}, Val {len(val_cls)}, Test {len(test_cls)}")
#     logger.info(f"Denoising: Train {len(train_dn)}, Val {len(val_dn)}, Test {len(paired_test)}")

#     return {
#         "classification": (train_loader_cls, val_loader_cls, test_loader_cls),
#         "denoising": (train_loader_dn, val_loader_dn, test_loader_dn)
#     }


# if __name__ == "__main__":
#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(DATASET_PATH, in_channels=1)  # Oder in_channels=3 für RGB
#     train_cls, val_cls, test_cls = loaders["classification"]
#     train_dn, val_dn, test_dn = loaders["denoising"]
