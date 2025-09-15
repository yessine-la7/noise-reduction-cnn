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






























# ################################ Graustufen split MIL mit seed + Paired Denoising (PNG/NPY/WAV)
# """
# Datasets & Loader für:
# - Klassifikation (Mel, PNG) mit MIL (Tiles), feste Höhe 128.
# - Denoising (STFT, PNG) als Paardataset (Noisy↔Clean), DC-Zeile entfernt -> Höhe 512.
# - Denoising (WAV) als Paardataset (Noisy↔Clean) mit On-the-fly-STFT und Zeitkacheln; optional DC entfernen.

# Wichtig: Für WAV-"cRM"-Training liefert der Loader Tensoren (2, F', T) mit Re/Im.
# """

# import os
# import random
# import logging
# from typing import List, Tuple, Dict, Optional
# from collections import OrderedDict

# from PIL import Image, ImageOps
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms

# # WAV/STFT
# import soundfile as sf
# import librosa

# logger = logging.getLogger(__name__)

# # -------------------------------
# # Repro: Worker-Seeding für DataLoader
# # -------------------------------
# def _seed_worker(worker_id: int):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# # -------------------------------
# # Allgemeine Hilfsfunktionen (Dateiliste, Namen, etc.)
# # -------------------------------
# def _list_png_files(dir_path: str) -> List[str]:
#     if not os.path.isdir(dir_path):
#         raise FileNotFoundError(f"Folder not found: {dir_path}")
#     return sorted(os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".png"))

# def _list_wav_files(dir_path: str) -> List[str]:
#     if not os.path.isdir(dir_path):
#         raise FileNotFoundError(f"Folder not found: {dir_path}")
#     return sorted(os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".wav"))

# def _basename_no_ext(path: str) -> str:
#     return os.path.splitext(os.path.basename(path))[0]

# # -------------------------------
# # Pairing & Splits (PNG/WAV)
# # -------------------------------
# def deterministic_classwise_split(
#     clean_files: List[str],
#     noisy_files: List[str],
#     val_ratio: float = 0.2,
#     seed: int = 42,
# ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
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

# def pair_noisy_clean_files(noisy_dir: str, clean_dir: str) -> List[Tuple[str, str]]:
#     noisy_files = _list_png_files(noisy_dir)
#     clean_files = _list_png_files(clean_dir)
#     map_clean = {_basename_no_ext(p): p for p in clean_files}
#     pairs, missing = [], []
#     for npath in noisy_files:
#         base = _basename_no_ext(npath)
#         cpath = map_clean.get(base)
#         if cpath is None:
#             missing.append(base)
#         else:
#             pairs.append((npath, cpath))
#     logger.info(f"[PNG Pairing] matched={len(pairs)}, missing_noisy={len(missing)}")
#     if missing:
#         logger.warning(f"[PNG Pairing] Beispiele ohne Clean-Gegenstück (erste 5): {missing[:5]}")
#     if not pairs:
#         raise RuntimeError(f"Keine PNG-Paare gefunden zwischen\n  noisy={noisy_dir}\n  clean={clean_dir}.")
#     pairs.sort(key=lambda t: t[0])
#     return pairs

# def pair_noisy_clean_wav_files(noisy_dir: str, clean_dir: str) -> List[Tuple[str, str]]:
#     noisy_files = _list_wav_files(noisy_dir)
#     clean_files = _list_wav_files(clean_dir)
#     map_clean = {_basename_no_ext(p): p for p in clean_files}
#     pairs, missing = [], []
#     for npath in noisy_files:
#         base = _basename_no_ext(npath)
#         cpath = map_clean.get(base)
#         if cpath is None:
#             missing.append(base)
#         else:
#             pairs.append((npath, cpath))
#     logger.info(f"[WAV Pairing] matched={len(pairs)}, missing_noisy={len(missing)}")
#     if missing:
#         logger.warning(f"[WAV Pairing] Beispiele ohne Clean-Gegenstück (erste 5): {missing[:5]}")
#     if not pairs:
#         raise RuntimeError(f"Keine WAV-Paare gefunden zwischen\n  noisy={noisy_dir}\n  clean={clean_dir}.")
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

# def _pad_width_to_at_least(img: Image.Image, min_w: int, pad_value: int = 0) -> Image.Image:
#     W, H = img.size
#     if W >= min_w:
#         return img
#     pad_right = min_w - W
#     return ImageOps.expand(img, border=(0, 0, pad_right, 0), fill=pad_value)

# # -------------------------------
# # PNG-basierte Datasets (unverändert)
# # -------------------------------
# class ClassificationMILDataset(Dataset):
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
#         self.files = files_with_labels
#         self.in_channels = in_channels
#         self.tile_h = tile_h
#         self.tile_w = tile_w
#         self.stride_w = stride_w

#         if normalize_mean_std is None:
#             normalize_mean_std = ((0.5,), (0.5,)) if in_channels == 1 else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         to_tensor = [transforms.ToTensor(), transforms.Normalize(*normalize_mean_std)]
#         self.pre = transforms.Compose(([transforms.Grayscale(num_output_channels=1)] if in_channels == 1 else []) + to_tensor)

#         self.index: List[Tuple[int, int]] = []
#         for fid, (path, _) in enumerate(self.files):
#             with Image.open(path) as im:
#                 im = im.convert("L") if self.in_channels == 1 else im.convert("RGB")
#                 W, H = im.size
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
#         if H != self.tile_h:
#             raise ValueError(f"Erwartete Mel-Höhe {self.tile_h}, bekam {H} für Datei {path}")
#         if x_left + self.tile_w > W:
#             x_left = max(0, W - self.tile_w)
#         box = (x_left, 0, x_left + self.tile_w, self.tile_h)
#         tile = im.crop(box)
#         tile_t = self.pre(tile)
#         return tile_t, torch.tensor(label, dtype=torch.long), torch.tensor(fid, dtype=torch.long)

# class DenoisingPairedDataset(Dataset):
#     def __init__(
#         self,
#         pairs: List[Tuple[str, str]],
#         in_channels: int = 1,
#         tile_h: int = 512,   # 513-1
#         tile_w: int = 256,
#         stride_w: int = 128,
#         normalize_mean_std: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
#     ):
#         super().__init__()
#         assert in_channels in (1, 3)
#         self.pairs = pairs
#         self.in_channels = in_channels
#         self.tile_h = tile_h
#         self.tile_w = tile_w
#         self.stride_w = stride_w

#         if normalize_mean_std is None:
#             normalize_mean_std = ((0.5,), (0.5,)) if in_channels == 1 else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         to_tensor = [transforms.ToTensor(), transforms.Normalize(*normalize_mean_std)]
#         self.pre = transforms.Compose(([transforms.Grayscale(num_output_channels=1)] if in_channels == 1 else []) + to_tensor)

#         self.index: List[Tuple[int, int]] = []
#         for pid, (npath, _cpath) in enumerate(self.pairs):
#             with Image.open(npath) as nim:
#                 nim = nim.convert("L") if self.in_channels == 1 else nim.convert("RGB")
#                 W, _ = nim.size
#                 if W <= self.tile_w:
#                     self.index.append((pid, 0))
#                 else:
#                     x = 0
#                     while True:
#                         self.index.append((pid, x))
#                         x += self.stride_w
#                         if x + self.tile_w >= W:
#                             break
#         logger.info(f"[DenoisingPairedDataset/Tiles] Paare: {len(self.pairs)}, Tiles: {len(self.index)}")

#     def __len__(self) -> int:
#         return len(self.index)

#     def _load_pair(self, pid: int) -> Tuple[Image.Image, Image.Image]:
#         npath, cpath = self.pairs[pid]
#         noisy = Image.open(npath)
#         clean = Image.open(cpath)
#         noisy = noisy.convert("L") if self.in_channels == 1 else noisy.convert("RGB")
#         clean = clean.convert("L") if self.in_channels == 1 else clean.convert("RGB")
#         return noisy, clean

#     @staticmethod
#     def _remove_dc_row(img: Image.Image) -> Image.Image:
#         W, H = img.size
#         return img.crop((0, 1, W, H))

#     def _crop_tile(self, img: Image.Image, x_left: int) -> Image.Image:
#         W, H = img.size
#         if H != self.tile_h:
#             raise ValueError(f"Erwartete Höhe {self.tile_h}, bekam {H}")
#         if W < self.tile_w:
#             img = _pad_width_to_at_least(img, self.tile_w, pad_value=0)
#             W = self.tile_w
#         if x_left + self.tile_w > W:
#             x_left = max(0, W - self.tile_w)
#         return img.crop((x_left, 0, x_left + self.tile_w, self.tile_h))

#     def __getitem__(self, idx):
#         pid, x_left = self.index[idx]
#         noisy, clean = self._load_pair(pid)
#         noisy = self._remove_dc_row(noisy)
#         clean = self._remove_dc_row(clean)
#         assert noisy.size[0] == clean.size[0], "Width mismatch im Paar"

#         noisy_t = self._crop_tile(noisy, x_left)
#         clean_t = self._crop_tile(clean, x_left)

#         x = self.pre(noisy_t)   # (C, 512, tile_w)
#         y = self.pre(clean_t)   # (C, 512, tile_w)
#         return x, y

# # -------------------------------
# # WAV-Paired Dataset (On-the-fly STFT, mit paarweiser Normierung)
# # -------------------------------
# class DenoisingWavPairedDataset(Dataset):
#     """
#     Lädt WAV-Paare (noisy, clean), berechnet on-the-fly STFT (Re/Im),
#     kachelt entlang der Zeit (Frames) und entfernt optional den DC-Bin.
#     Rückgabe: (2, F', tile_T) als float32.

#     get_full_stft(pid)  -> vollständige (X,S) STFTs
#     get_full_wav_pair(pid) -> (noisy, clean) im Zeitbereich (mit identischer Normierung)
#     """

#     def __init__(
#         self,
#         pairs: List[Tuple[str, str]],
#         sr: int = 48000,
#         n_fft: int = 1024,
#         hop_length: int = 256,
#         win_length: int = 1024,
#         window: str = "hann",
#         center: bool = True,
#         tile_T: int = 256,
#         stride_T: int = 128,
#         remove_dc_bin: bool = True,
#         normalize_mode: str = "pair_peak",   # "pair_peak" | "per_file_peak" | "none"
#         cache_size: int = 8,
#     ):
#         super().__init__()
#         assert normalize_mode in ("pair_peak", "per_file_peak", "none")
#         self.pairs = pairs
#         self.sr = sr
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.window = window
#         self.center = center
#         self.tile_T = tile_T
#         self.stride_T = stride_T
#         self.remove_dc_bin = remove_dc_bin
#         self.normalize_mode = normalize_mode

#         self.params = dict(sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
#                            window=window, center=center, remove_dc_bin=remove_dc_bin, normalize_mode=normalize_mode)

#         # Index über (pair_id, t_left)
#         self.index: List[Tuple[int, int]] = []
#         for pid, (npath, _cpath) in enumerate(self.pairs):
#             y, sr = self._try_read_wav(npath)
#             y = self._to_mono(y)
#             if sr != self.sr:
#                 y = librosa.resample(y, orig_sr=sr, target_sr=self.sr, res_type="kaiser_best")
#             X = librosa.stft(y.astype(np.float32), n_fft=self.n_fft, hop_length=self.hop_length,
#                              win_length=self.win_length, window=self.window, center=self.center, pad_mode="reflect")
#             T = X.shape[1]
#             if T <= self.tile_T:
#                 self.index.append((pid, 0))
#             else:
#                 t = 0
#                 while True:
#                     self.index.append((pid, t))
#                     t += self.stride_T
#                     if t + self.tile_T >= T:
#                         break

#         # LRU-Cache für STFT-Paare
#         self._cache: OrderedDict[int, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
#         self._cache_size = max(1, cache_size)

#         logger.info(f"[DenoisingWavPairedDataset] Paare: {len(self.pairs)} | Tiles: {len(self.index)} | "
#                     f"sr={self.sr}, n_fft={self.n_fft}, hop={self.hop_length}, win={self.win_length}, "
#                     f"remove_dc={self.remove_dc_bin}, normalize={self.normalize_mode}")

#     @staticmethod
#     def _to_mono(y: np.ndarray) -> np.ndarray:
#         y = np.asarray(y)
#         if y.ndim == 1:
#             return y
#         return np.mean(y, axis=0)

#     @staticmethod
#     def _try_read_wav(path: str):
#         try:
#             y, sr = sf.read(path, always_2d=False)
#             y = np.asarray(y)
#             if y.ndim == 2:
#                 y = y.T
#         except Exception:
#             y, sr = librosa.load(path, sr=None, mono=False)
#         return y, sr

#     def _post_resample_pair_norm(self, ny: np.ndarray, cy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """ wendet die gewünschte Normalisierung auf das Paar an """
#         if self.normalize_mode == "none":
#             return ny, cy
#         if self.normalize_mode == "per_file_peak":
#             def peak_norm(y):
#                 m = np.max(np.abs(y)) + 1e-9
#                 return (y / m).astype(np.float32, copy=False)
#             return peak_norm(ny), peak_norm(cy)
#         # pair_peak
#         m = max(np.max(np.abs(ny)), np.max(np.abs(cy))) + 1e-9
#         return (ny / m).astype(np.float32, copy=False), (cy / m).astype(np.float32, copy=False)

#     def _load_wav_pair(self, pid: int) -> Tuple[np.ndarray, np.ndarray]:
#         npath, cpath = self.pairs[pid]
#         ny, nsr = self._try_read_wav(npath)
#         cy, csr = self._try_read_wav(cpath)
#         ny = self._to_mono(ny); cy = self._to_mono(cy)
#         if nsr != self.sr:
#             ny = librosa.resample(ny, orig_sr=nsr, target_sr=self.sr, res_type="kaiser_best")
#         if csr != self.sr:
#             cy = librosa.resample(cy, orig_sr=csr, target_sr=self.sr, res_type="kaiser_best")
#         ny, cy = self._post_resample_pair_norm(ny.astype(np.float32), cy.astype(np.float32))
#         return ny, cy

#     def _stft_pair(self, pid: int) -> Tuple[np.ndarray, np.ndarray]:
#         if pid in self._cache:
#             x = self._cache.pop(pid)
#             self._cache[pid] = x
#             return x

#         ny, cy = self._load_wav_pair(pid)

#         X = librosa.stft(ny, n_fft=self.n_fft, hop_length=self.hop_length,
#                          win_length=self.win_length, window=self.window, center=self.center, pad_mode="reflect")
#         S = librosa.stft(cy, n_fft=self.n_fft, hop_length=self.hop_length,
#                          win_length=self.win_length, window=self.window, center=self.center, pad_mode="reflect")

#         X = np.stack((X.real, X.imag), axis=0).astype(np.float32)  # (2,F,T)
#         S = np.stack((S.real, S.imag), axis=0).astype(np.float32)

#         if self.remove_dc_bin:
#             X = X[:, 1:, :]
#             S = S[:, 1:, :]

#         self._cache[pid] = (X, S)
#         while len(self._cache) > self._cache_size:
#             self._cache.popitem(last=False)
#         return X, S

#     # ---- Public helpers ----
#     def get_full_stft(self, pid: int) -> Tuple[np.ndarray, np.ndarray]:
#         """ vollständige STFTs (Noisy=X, Clean=S) eines Paares. (2,F',T) float32 """
#         return self._stft_pair(pid)

#     def get_full_wav_pair(self, pid: int) -> Tuple[np.ndarray, np.ndarray]:
#         """ Zeitbereich (noisy, clean) nach Resampling & gewählter Normierung """
#         return self._load_wav_pair(pid)

#     @property
#     def num_pairs(self) -> int:
#         return len(self.pairs)

#     def __len__(self) -> int:
#         return len(self.index)

#     def __getitem__(self, idx: int):
#         pid, t_left = self.index[idx]
#         X, S = self._stft_pair(pid)   # (2, F', T)
#         T = X.shape[2]
#         end = t_left + self.tile_T
#         if end > T:
#             pad = end - T
#             Xp = np.pad(X, ((0,0),(0,0),(0,pad)), mode="constant")
#             Sp = np.pad(S, ((0,0),(0,0),(0,pad)), mode="constant")
#             x = Xp[:, :, t_left:end]
#             y = Sp[:, :, t_left:end]
#         else:
#             x = X[:, :, t_left:end]
#             y = S[:, :, t_left:end]
#         return torch.from_numpy(x), torch.from_numpy(y)

# # -------------------------------
# # Loader-Erzeugung (deterministisch)
# # -------------------------------
# def get_data_loaders(
#     dataset_path: str,
#     batch_size: int = 8,
#     num_workers: int = 4,

#     # Klassifikation (Mel, PNG)
#     enable_classification: bool = False,
#     cls_in_channels: int = 1,
#     cls_tile_h: int = 128,
#     cls_tile_w: int = 256,
#     cls_stride_w: int = 128,
#     cls_val_ratio: float = 0.2,
#     cls_seed: int = 42,

#     # Denoising (STFT, PNG)
#     enable_denoising: bool = False,
#     dn_in_channels: int = 1,
#     dn_tile_h: int = 512,
#     dn_tile_w: int = 256,
#     dn_stride_w: int = 128,
#     dn_val_ratio: float = 0.2,
#     dn_seed: int = 42,

#     # Denoising (WAV, on-the-fly STFT)
#     enable_denoising_wav: bool = False,
#     wav_sr: int = 48000,
#     wav_n_fft: int = 1024,
#     wav_hop_length: int = 256,
#     wav_win_length: int = 1024,
#     wav_window: str = "hann",
#     wav_center: bool = True,
#     wav_tile_T: int = 256,
#     wav_stride_T: int = 128,
#     wav_remove_dc_bin: bool = True,
#     wav_val_ratio: float = 0.1,
#     wav_seed: int = 42,
#     wav_normalize_mode: str = "pair_peak",  # <— wichtig
# ) -> Dict[str, Optional[Tuple[DataLoader, DataLoader, DataLoader]]]:

#     out: Dict[str, Optional[Tuple[DataLoader, DataLoader, DataLoader]]] = {
#         "classification": None,
#         "denoising": None,
#         "denoising_wav": None,
#     }

#     g_cls = torch.Generator().manual_seed(cls_seed)
#     g_dn  = torch.Generator().manual_seed(dn_seed)
#     g_wav = torch.Generator().manual_seed(wav_seed)

#     # Klassifikation (PNG/Mel)
#     if enable_classification:
#         clean_train_mel = os.path.join(dataset_path, "clean_trainset_56spk_mel")
#         noisy_train_mel = os.path.join(dataset_path, "noisy_trainset_56spk_mel")
#         clean_test_mel  = os.path.join(dataset_path, "clean_testset_mel")
#         noisy_test_mel  = os.path.join(dataset_path, "noisy_testset_mel")

#         clean_train_files = _list_png_files(clean_train_mel)
#         noisy_train_files = _list_png_files(noisy_train_mel)
#         clean_test_files  = _list_png_files(clean_test_mel)
#         noisy_test_files  = _list_png_files(noisy_test_mel)

#         cls_train_files, cls_val_files = deterministic_classwise_split(
#             clean_train_files, noisy_train_files, val_ratio=cls_val_ratio, seed=cls_seed
#         )
#         cls_test_files = ([(p, 0) for p in clean_test_files] +
#                           [(p, 1) for p in noisy_test_files])
#         cls_test_files.sort(key=lambda x: x[0])

#         cls_train_ds = ClassificationMILDataset(cls_train_files, cls_in_channels, cls_tile_h, cls_tile_w, cls_stride_w)
#         cls_val_ds   = ClassificationMILDataset(cls_val_files,   cls_in_channels, cls_tile_h, cls_tile_w, cls_stride_w)
#         cls_test_ds  = ClassificationMILDataset(cls_test_files,  cls_in_channels, cls_tile_h, cls_tile_w, cls_stride_w)

#         train_loader_cls = DataLoader(cls_train_ds, batch_size=batch_size, shuffle=True,
#                                       num_workers=num_workers, pin_memory=True,
#                                       worker_init_fn=_seed_worker, generator=g_cls)
#         val_loader_cls = DataLoader(cls_val_ds, batch_size=batch_size, shuffle=False,
#                                     num_workers=num_workers, pin_memory=True,
#                                     worker_init_fn=_seed_worker, generator=g_cls)
#         test_loader_cls = DataLoader(cls_test_ds, batch_size=batch_size, shuffle=False,
#                                      num_workers=num_workers, pin_memory=True,
#                                      worker_init_fn=_seed_worker, generator=g_cls)

#         logger.info(f"[Classification/MIL] Tiles Train/Val/Test: {len(cls_train_ds)}/{len(cls_val_ds)}/{len(cls_test_ds)}")
#         out["classification"] = (train_loader_cls, val_loader_cls, test_loader_cls)

#     # Denoising (PNG)
#     if enable_denoising:
#         clean_train_stft = os.path.join(dataset_path, "clean_trainset_56spk_stft")
#         noisy_train_stft = os.path.join(dataset_path, "noisy_trainset_56spk_stft")
#         clean_test_stft  = os.path.join(dataset_path, "clean_testset_stft")
#         noisy_test_stft  = os.path.join(dataset_path, "noisy_testset_stft")

#         pairs_train_all = pair_noisy_clean_files(noisy_train_stft, clean_train_stft)
#         pairs_test_all  = pair_noisy_clean_files(noisy_test_stft,  clean_test_stft)

#         dn_train_pairs, dn_val_pairs = deterministic_paired_split(pairs_train_all, val_ratio=dn_val_ratio, seed=dn_seed)

#         dn_train_ds = DenoisingPairedDataset(dn_train_pairs, dn_in_channels, dn_tile_h, dn_tile_w, dn_stride_w)
#         dn_val_ds   = DenoisingPairedDataset(dn_val_pairs,   dn_in_channels, dn_tile_h, dn_tile_w, dn_stride_w)
#         dn_test_ds  = DenoisingPairedDataset(pairs_test_all, dn_in_channels, dn_tile_h, dn_tile_w, dn_stride_w)

#         train_loader_dn = DataLoader(dn_train_ds, batch_size=batch_size, shuffle=True,
#                                      num_workers=num_workers, pin_memory=True,
#                                      worker_init_fn=_seed_worker, generator=g_dn)
#         val_loader_dn = DataLoader(dn_val_ds, batch_size=batch_size, shuffle=False,
#                                    num_workers=num_workers, pin_memory=True,
#                                    worker_init_fn=_seed_worker, generator=g_dn)
#         test_loader_dn = DataLoader(dn_test_ds, batch_size=batch_size, shuffle=False,
#                                     num_workers=num_workers, pin_memory=True,
#                                     worker_init_fn=_seed_worker, generator=g_dn)

#         logger.info(f"[Denoising/PNG] Tiles Train/Val/Test: {len(dn_train_ds)}/{len(dn_val_ds)}/{len(dn_test_ds)}")
#         out["denoising"] = (train_loader_dn, val_loader_dn, test_loader_dn)

#     # Denoising (WAV / on-the-fly STFT)
#     if enable_denoising_wav:
#         clean_train_wav = os.path.join(dataset_path, "clean_trainset_56spk_wav")
#         noisy_train_wav = os.path.join(dataset_path, "noisy_trainset_56spk_wav")
#         clean_test_wav  = os.path.join(dataset_path, "clean_testset_wav")
#         noisy_test_wav  = os.path.join(dataset_path, "noisy_testset_wav")

#         wav_pairs_train_all = pair_noisy_clean_wav_files(noisy_train_wav, clean_train_wav)
#         wav_pairs_test_all  = pair_noisy_clean_wav_files(noisy_test_wav,  clean_test_wav)
#         wav_train_pairs, wav_val_pairs = deterministic_paired_split(wav_pairs_train_all, val_ratio=wav_val_ratio, seed=wav_seed)

#         wav_train_ds = DenoisingWavPairedDataset(
#             wav_train_pairs, sr=wav_sr, n_fft=wav_n_fft, hop_length=wav_hop_length, win_length=wav_win_length,
#             window=wav_window, center=wav_center, tile_T=wav_tile_T, stride_T=wav_stride_T,
#             remove_dc_bin=wav_remove_dc_bin, normalize_mode=wav_normalize_mode, cache_size=8
#         )
#         wav_val_ds = DenoisingWavPairedDataset(
#             wav_val_pairs, sr=wav_sr, n_fft=wav_n_fft, hop_length=wav_hop_length, win_length=wav_win_length,
#             window=wav_window, center=wav_center, tile_T=wav_tile_T, stride_T=wav_stride_T,
#             remove_dc_bin=wav_remove_dc_bin, normalize_mode=wav_normalize_mode, cache_size=8
#         )
#         wav_test_ds = DenoisingWavPairedDataset(
#             wav_pairs_test_all, sr=wav_sr, n_fft=wav_n_fft, hop_length=wav_hop_length, win_length=wav_win_length,
#             window=wav_window, center=wav_center, tile_T=wav_tile_T, stride_T=wav_stride_T,
#             remove_dc_bin=wav_remove_dc_bin, normalize_mode=wav_normalize_mode, cache_size=8
#         )

#         train_loader_wav = DataLoader(wav_train_ds, batch_size=batch_size, shuffle=True,
#                                       num_workers=num_workers, pin_memory=True,
#                                       worker_init_fn=_seed_worker, generator=g_wav)
#         val_loader_wav = DataLoader(wav_val_ds, batch_size=batch_size, shuffle=False,
#                                     num_workers=num_workers, pin_memory=True,
#                                     worker_init_fn=_seed_worker, generator=g_wav)
#         test_loader_wav = DataLoader(wav_test_ds, batch_size=batch_size, shuffle=False,
#                                      num_workers=num_workers, pin_memory=True,
#                                      worker_init_fn=_seed_worker, generator=g_wav)

#         logger.info(f"[Denoising/WAV] Tiles Train/Val/Test: {len(wav_train_ds)}/{len(wav_val_ds)}/{len(wav_test_ds)}")
#         out["denoising_wav"] = (train_loader_wav, val_loader_wav, test_loader_wav)

#     return out

# if __name__ == "__main__":
#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(
#         DATASET_PATH,
#         enable_classification=False,
#         enable_denoising=False,
#         enable_denoising_wav=True,
#         wav_sr=48000, wav_n_fft=1024, wav_hop_length=256, wav_win_length=1024,
#         wav_tile_T=256, wav_stride_T=128, wav_normalize_mode="pair_peak",
#     )
#     print("OK")



















# ################################## Graustufen split MIL mit seed + Paired Denoising (PNG/NPY)
# """
# - Klassifikation (Mel, PNG) mit MIL (Tiles), feste Höhe 128.
# - Denoising (STFT, PNG) als Paardataset (Noisy↔Clean), DC-Zeile entfernt -> Höhe 512.
# - NEU: Denoising (STFT, NPY/NPZ-Punkte) als Paardataset (Noisy↔Clean), optional DC-Entfernung,
#        Zeitachse wird gekachelt (Frames).

# Hinweis zu NPY/NPZ:
# - Erwartete Shape pro Datei: (C, FreqBins, TimeFrames), ideal: C=2 (Re/Im).
# - Optional wird die DC-Zeile (FreqBin 0) entfernt. Danach sollte FreqBins konstant sein.
# - Die Paare werden über identische Basenamen (ohne Extension) gebildet.
# """
# import os
# import random
# import logging
# from typing import List, Tuple, Dict, Optional

# from PIL import Image, ImageOps
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
# # Allgemeine Hilfsfunktionen
# # -------------------------------
# def _list_png_files(dir_path: str) -> List[str]:
#     if not os.path.isdir(dir_path):
#         raise FileNotFoundError(f"Folder not found: {dir_path}")
#     return sorted(
#         os.path.join(dir_path, f)
#         for f in os.listdir(dir_path)
#         if f.lower().endswith(".png")
#     )


# def _list_npy_like_files(dir_path: str) -> List[str]:
#     if not os.path.isdir(dir_path):
#         raise FileNotFoundError(f"Folder not found: {dir_path}")
#     return sorted(
#         os.path.join(dir_path, f)
#         for f in os.listdir(dir_path)
#         if f.lower().endswith(".npy") or f.lower().endswith(".npz")
#     )


# def _basename_no_ext(path: str) -> str:
#     return os.path.splitext(os.path.basename(path))[0]


# def _first_existing_dir(candidates: List[str]) -> str:
#     for p in candidates:
#         if os.path.isdir(p):
#             return p
#     raise FileNotFoundError(
#         "Kein passendes Verzeichnis gefunden. Getestet:\n  - " + "\n  - ".join(candidates)
#     )


# def _load_npy_like(path: str) -> np.ndarray:
#     """
#     Lädt .npy oder .npz.
#     Erwartet (C, F, T). Bei .npz wird zuerst nach Key 'X' gesucht,
#     sonst das erste Array im File genommen.
#     """
#     if path.lower().endswith(".npy"):
#         arr = np.load(path)
#     else:
#         data = np.load(path)
#         if isinstance(data, np.lib.npyio.NpzFile):
#             if "X" in data:
#                 arr = data["X"]
#             else:
#                 # erstes Array nehmen
#                 key = data.files[0]
#                 arr = data[key]
#         else:
#             arr = data  # Fallback
#     if not isinstance(arr, np.ndarray):
#         raise TypeError(f"{path}: geladenes Objekt ist kein np.ndarray")
#     return arr


# # -------------------------------
# # Splits / Pairing (PNG)
# # -------------------------------
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
#     Bildet PNG-Paare (noisy_path, clean_path) über identische Basenamen.
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
#             f"Keine PNG-Paare gefunden zwischen\n  noisy={noisy_dir}\n  clean={clean_dir}.\n"
#             "Bitte Dateinamen angleichen."
#         )
#     if missing_clean:
#         logger.warning(f"[Denoising(PNG)] {len(missing_clean)} noisy-Dateien ohne Clean-Gegenstück (ignoriert).")

#     pairs.sort(key=lambda t: t[0])
#     return pairs


# def pair_noisy_clean_np_files(
#     noisy_dir: str, clean_dir: str
# ) -> List[Tuple[str, str]]:
#     """
#     Bildet NPY/NPZ-Paare (noisy_path, clean_path) über identische Basenamen.
#     """
#     noisy_files = _list_npy_like_files(noisy_dir)
#     clean_files = _list_npy_like_files(clean_dir)
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
#             f"Keine NPY/NPZ-Paare gefunden zwischen\n  noisy={noisy_dir}\n  clean={clean_dir}.\n"
#             "Bitte Dateinamen angleichen."
#         )
#     if missing_clean:
#         logger.warning(f"[Denoising(NPY)] {len(missing_clean)} noisy-Dateien ohne Clean-Gegenstück (ignoriert).")

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


# def _pad_width_to_at_least(img: Image.Image, min_w: int, pad_value: int = 0) -> Image.Image:
#     """
#     Falls die Bildbreite < min_w ist, rechtsseitiges Padding bis genau min_w.
#     Höhe bleibt unverändert.
#     """
#     W, H = img.size
#     if W >= min_w:
#         return img
#     pad_right = min_w - W
#     return ImageOps.expand(img, border=(0, 0, pad_right, 0), fill=pad_value)


# # -------------------------------
# # MIL Dataset (Klassifikation/Mel, PNG)
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
# # Paired Denoising Dataset (STFT, PNG)
# # -------------------------------
# class DenoisingPairedDataset(Dataset):
#     """
#     Liefert (noisy_tensor, clean_tensor) als Paar in Kacheln (Tiles).
#     - DC-Zeile (Zeile 0) wird entfernt -> feste Höhe 512.
#     - Kein vertikales Padding.
#     - Horizontales Padding NUR, wenn Breite < tile_w (auf genau tile_w).
#     - Paare (noisy, clean) haben identische Breite.
#     """
#     def __init__(
#         self,
#         pairs: List[Tuple[str, str]],
#         in_channels: int = 1,
#         tile_h: int = 512,          # fest 512 (513-1)
#         tile_w: int = 256,
#         stride_w: int = 128,
#         normalize_mean_std: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
#     ):
#         super().__init__()
#         assert in_channels in (1, 3)
#         self.pairs = pairs
#         self.in_channels = in_channels
#         self.tile_h = tile_h
#         self.tile_w = tile_w
#         self.stride_w = stride_w

#         if normalize_mean_std is None:
#             normalize_mean_std = ((0.5,), (0.5,)) if in_channels == 1 else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         to_tensor_tfms = [transforms.ToTensor(), transforms.Normalize(*normalize_mean_std)]
#         self.pre = transforms.Compose(([transforms.Grayscale(num_output_channels=1)] if in_channels == 1 else []) + to_tensor_tfms)

#         # Index über Tiles: Liste von (pair_id, x_left)
#         self.index: List[Tuple[int, int]] = []
#         for pid, (npath, _cpath) in enumerate(self.pairs):
#             with Image.open(npath) as nim:
#                 nim = nim.convert("L") if self.in_channels == 1 else nim.convert("RGB")
#                 W, _ = nim.size
#                 if W <= self.tile_w:
#                     self.index.append((pid, 0))
#                 else:
#                     x = 0
#                     while True:
#                         self.index.append((pid, x))
#                         x += self.stride_w
#                         if x + self.tile_w >= W:
#                             break
#         logger.info(f"[DenoisingPairedDataset/Tiles] Paare: {len(self.pairs)}, Tiles: {len(self.index)}")

#     def __len__(self) -> int:
#         return len(self.index)

#     def _load_pair(self, pid: int) -> Tuple[Image.Image, Image.Image]:
#         npath, cpath = self.pairs[pid]
#         noisy = Image.open(npath)
#         clean = Image.open(cpath)
#         noisy = noisy.convert("L") if self.in_channels == 1 else noisy.convert("RGB")
#         clean = clean.convert("L") if self.in_channels == 1 else clean.convert("RGB")
#         return noisy, clean

#     @staticmethod
#     def _remove_dc_row(img: Image.Image) -> Image.Image:
#         """Entfernt die erste Pixelzeile (DC-Bin) -> Höhe 512 (ausgehend von 513)."""
#         W, H = img.size
#         return img.crop((0, 1, W, H))

#     def _crop_tile(self, img: Image.Image, x_left: int) -> Image.Image:
#         """
#         Schneidet einen Tile von Breite tile_w und Höhe tile_h aus.
#         Wenn Breite < tile_w, vorher rechts auf tile_w auffüllen.
#         """
#         W, H = img.size
#         if H != self.tile_h:
#             raise ValueError(f"Erwartete Höhe {self.tile_h}, bekam {H}")
#         if W < self.tile_w:
#             img = _pad_width_to_at_least(img, self.tile_w, pad_value=0)
#             W = self.tile_w
#         if x_left + self.tile_w > W:
#             x_left = max(0, W - self.tile_w)
#         return img.crop((x_left, 0, x_left + self.tile_w, self.tile_h))

#     def __getitem__(self, idx):
#         pid, x_left = self.index[idx]
#         noisy, clean = self._load_pair(pid)

#         # DC-Zeile entfernen (H=513 -> 512)
#         noisy = self._remove_dc_row(noisy)
#         clean = self._remove_dc_row(clean)

#         # Sicherheitsnetz: identische Breite der Paare
#         assert noisy.size[0] == clean.size[0], \
#             f"Width mismatch: noisy={noisy.size[0]} vs clean={clean.size[0]} (pair_id={pid})"

#         noisy_t = self._crop_tile(noisy, x_left)
#         clean_t = self._crop_tile(clean, x_left)

#         x = self.pre(noisy_t)   # (C, 512, tile_w)
#         y = self.pre(clean_t)   # (C, 512, tile_w)
#         return x, y


# # -------------------------------
# # Paired Denoising Dataset (STFT, NPY/NPZ-Punkte)
# # -------------------------------
# class DenoisingPointsPairedDataset(Dataset):
#     """
#     Lädt Paare aus NPY/NPZ (Noisy, Clean) mit komplexen STFT-Punkten.

#     Erwartung an jede Datei:
#       - Array-Shape: (C, F, T), üblicherweise C=2 (Re/Im).
#       - Bei .npz: bevorzugt Key 'X'; sonst erstes Array.
#     Kachelung:
#       - Entlang der Zeit (Frames) in Fenster der Breite tile_T mit stride_T.
#       - Falls T < tile_T: Zero-Padding bis tile_T.
#     Optional:
#       - remove_dc_bin=True entfernt das 0.-Freq-Bin -> F wird um 1 reduziert.

#     Rückgabe: (x, y) als torch.FloatTensor mit Shape (C, F', tile_T).
#     """
#     def __init__(
#         self,
#         pairs: List[Tuple[str, str]],
#         in_channels: int = 2,          # Re/Im
#         tile_T: int = 256,             # Anzahl Zeit-Frames pro Tile
#         stride_T: int = 128,
#         remove_dc_bin: bool = True,
#         enforce_freq_bins: Optional[int] = None,  # wenn gesetzt, wird F' validiert
#         dtype: np.dtype = np.float32,
#     ):
#         super().__init__()
#         self.pairs = pairs
#         self.in_channels = in_channels
#         self.tile_T = tile_T
#         self.stride_T = stride_T
#         self.remove_dc_bin = remove_dc_bin
#         self.dtype = dtype

#         # Index über Zeitfenster: Liste (pair_id, t_left)
#         self.index: List[Tuple[int, int]] = []
#         self.freq_bins_after = None

#         for pid, (npath, _cpath) in enumerate(self.pairs):
#             arr = _load_npy_like(npath)
#             if arr.ndim == 2:
#                 # (F, T) -> C=1
#                 arr = np.expand_dims(arr, 0)
#             if arr.shape[0] < self.in_channels:
#                 raise ValueError(f"{npath}: erwartete mind. {self.in_channels} Kanäle, bekam {arr.shape[0]}")

#             # Freq-Bins mit/ohne DC
#             F = arr.shape[1] - (1 if self.remove_dc_bin else 0)
#             T = arr.shape[2]

#             if self.freq_bins_after is None:
#                 self.freq_bins_after = F
#             elif self.freq_bins_after != F:
#                 raise ValueError(f"Uneinheitliche Freq-Bins nach DC-Entfernung: {self.freq_bins_after} vs {F} ({npath})")

#             if enforce_freq_bins is not None and enforce_freq_bins != F:
#                 raise ValueError(f"Frequenzhöhe {F} != erwarteten {enforce_freq_bins}")

#             if T <= self.tile_T:
#                 self.index.append((pid, 0))
#             else:
#                 t = 0
#                 while True:
#                     self.index.append((pid, t))
#                     t += self.stride_T
#                     if t + self.tile_T >= T:
#                         break

#         logger.info(f"[DenoisingPointsPairedDataset] Paare: {len(self.pairs)}, Tiles: {len(self.index)}, F'={self.freq_bins_after}")

#     def __len__(self) -> int:
#         return len(self.index)

#     def _load_pair(self, pid: int) -> Tuple[np.ndarray, np.ndarray]:
#         npath, cpath = self.pairs[pid]
#         noisy = _load_npy_like(npath).astype(self.dtype, copy=False)
#         clean = _load_npy_like(cpath).astype(self.dtype, copy=False)
#         if noisy.ndim == 2:
#             noisy = noisy[None, ...]
#         if clean.ndim == 2:
#             clean = clean[None, ...]
#         return noisy, clean

#     def _cut_freq(self, arr: np.ndarray) -> np.ndarray:
#         # Entferne DC-Bin falls gewünscht
#         if self.remove_dc_bin:
#             return arr[:, 1:, :]
#         return arr

#     def __getitem__(self, idx: int):
#         pid, t_left = self.index[idx]
#         noisy, clean = self._load_pair(pid)    # (C, F, T)

#         noisy = self._cut_freq(noisy)
#         clean = self._cut_freq(clean)

#         # Sicherheitsprüfungen
#         if noisy.shape != clean.shape:
#             raise ValueError(f"Shape-Mismatch im Paar {pid}: noisy {noisy.shape} vs clean {clean.shape}")
#         C, F, T = noisy.shape
#         if C < self.in_channels:
#             raise ValueError(f"Erwartete mind. {self.in_channels} Kanäle, bekam {C}")

#         # Zeitfenster schneiden/padden
#         if t_left + self.tile_T > T:
#             # Pad rechts mit Nullen
#             pad = t_left + self.tile_T - T
#             pad_noisy = np.pad(noisy, ((0, 0), (0, 0), (0, pad)), mode="constant")
#             pad_clean = np.pad(clean, ((0, 0), (0, 0), (0, pad)), mode="constant")
#             x = pad_noisy[:self.in_channels, :, t_left:t_left + self.tile_T]
#             y = pad_clean[:self.in_channels, :, t_left:t_left + self.tile_T]
#         else:
#             x = noisy[:self.in_channels, :, t_left:t_left + self.tile_T]
#             y = clean[:self.in_channels, :, t_left:t_left + self.tile_T]

#         # -> Torch
#         x_t = torch.from_numpy(x)           # (C, F', tile_T)
#         y_t = torch.from_numpy(y)
#         return x_t, y_t


# # -------------------------------
# # Loader-Erzeugung (deterministisch)
# # -------------------------------
# def get_data_loaders(
#     dataset_path: str,
#     batch_size: int = 8,
#     num_workers: int = 4,

#     # Klassifikation (Mel, PNG)
#     enable_classification: bool = False,
#     cls_in_channels: int = 1,
#     cls_tile_h: int = 128,
#     cls_tile_w: int = 256,
#     cls_stride_w: int = 128,
#     cls_val_ratio: float = 0.2,
#     cls_seed: int = 42,

#     # Denoising (STFT, PNG)
#     enable_denoising: bool = False,
#     dn_in_channels: int = 1,
#     dn_tile_h: int = 512,
#     dn_tile_w: int = 256,
#     dn_stride_w: int = 128,
#     dn_val_ratio: float = 0.2,
#     dn_seed: int = 42,

#     # NEU: Denoising (STFT, NPY/NPZ-Punkte)
#     enable_denoising_points: bool = False,
#     dp_in_channels: int = 2,          # Re/Im
#     dp_tile_T: int = 256,
#     dp_stride_T: int = 128,
#     dp_remove_dc_bin: bool = True,
#     dp_val_ratio: float = 0.2,
#     dp_seed: int = 42,
# ) -> Dict[str, Optional[Tuple[DataLoader, DataLoader, DataLoader]]]:

#     out: Dict[str, Optional[Tuple[DataLoader, DataLoader, DataLoader]]] = {
#         "classification": None,
#         "denoising": None,
#         "denoising_points": None,
#     }

#     g_cls = torch.Generator().manual_seed(cls_seed)
#     g_dn  = torch.Generator().manual_seed(dn_seed)
#     g_dp  = torch.Generator().manual_seed(dp_seed)

#     # --- Klassifikation (Mel, PNG) ---
#     if enable_classification:
#         clean_train_mel = os.path.join(dataset_path, "clean_trainset_56spk_mel")
#         noisy_train_mel = os.path.join(dataset_path, "noisy_trainset_56spk_mel")
#         clean_test_mel  = os.path.join(dataset_path, "clean_testset_mel")
#         noisy_test_mel  = os.path.join(dataset_path, "noisy_testset_mel")

#         clean_train_files = _list_png_files(clean_train_mel)
#         noisy_train_files = _list_png_files(noisy_train_mel)
#         clean_test_files  = _list_png_files(clean_test_mel)
#         noisy_test_files  = _list_png_files(noisy_test_mel)

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

#     # --- Denoising (STFT, PNG) ---
#     if enable_denoising:
#         clean_train_stft = os.path.join(dataset_path, "clean_trainset_56spk_stft")
#         noisy_train_stft = os.path.join(dataset_path, "noisy_trainset_56spk_stft")
#         clean_test_stft  = os.path.join(dataset_path, "clean_testset_stft")
#         noisy_test_stft  = os.path.join(dataset_path, "noisy_testset_stft")

#         pairs_train_all = pair_noisy_clean_files(noisy_train_stft, clean_train_stft)
#         pairs_test_all  = pair_noisy_clean_files(noisy_test_stft,  clean_test_stft)

#         dn_train_pairs, dn_val_pairs = deterministic_paired_split(
#             pairs_train_all, val_ratio=dn_val_ratio, seed=dn_seed
#         )

#         dn_train_ds = DenoisingPairedDataset(
#             pairs=dn_train_pairs,
#             in_channels=dn_in_channels,
#             tile_h=dn_tile_h,
#             tile_w=dn_tile_w,
#             stride_w=dn_stride_w,
#         )
#         dn_val_ds = DenoisingPairedDataset(
#             pairs=dn_val_pairs,
#             in_channels=dn_in_channels,
#             tile_h=dn_tile_h,
#             tile_w=dn_tile_w,
#             stride_w=dn_stride_w,
#         )
#         dn_test_ds = DenoisingPairedDataset(
#             pairs=pairs_train_all if pairs_test_all is None else pairs_test_all,
#             in_channels=dn_in_channels,
#             tile_h=dn_tile_h,
#             tile_w=dn_tile_w,
#             stride_w=dn_stride_w,
#         )

#         train_loader_dn = DataLoader(
#             dn_train_ds, batch_size=batch_size, shuffle=True,
#             num_workers=num_workers, pin_memory=True,
#             worker_init_fn=_seed_worker, generator=g_dn,
#         )
#         val_loader_dn = DataLoader(
#             dn_val_ds, batch_size=batch_size, shuffle=False,
#             num_workers=num_workers, pin_memory=True,
#             worker_init_fn=_seed_worker, generator=g_dn,
#         )
#         test_loader_dn = DataLoader(
#             dn_test_ds, batch_size=batch_size, shuffle=False,
#             num_workers=num_workers, pin_memory=True,
#             worker_init_fn=_seed_worker, generator=g_dn,
#         )

#         logger.info(
#             f"[Denoising/Paired PNG-Tiles] "
#             f"Paare Train/Val/Test: {len(dn_train_pairs)}/{len(dn_val_pairs)}/{len(pairs_test_all)} | "
#             f"Samples Train/Val/Test: {len(dn_train_ds)}/{len(dn_val_ds)}/{len(dn_test_ds)}"
#         )

#         out["denoising"] = (train_loader_dn, val_loader_dn, test_loader_dn)

#     # --- Denoising (STFT, NPY/NPZ-Punkte) ---
#     if enable_denoising_points:
#         clean_train_pts = os.path.join(dataset_path, "clean_trainset_56spk_points")
#         noisy_train_pts = os.path.join(dataset_path, "noisy_trainset_56spk_points")
#         clean_test_pts  = os.path.join(dataset_path, "clean_testset_points")
#         noisy_test_pts  = os.path.join(dataset_path, "noisy_testset_points")

#         dp_pairs_train_all = pair_noisy_clean_np_files(noisy_train_pts, clean_train_pts)
#         dp_pairs_test_all  = pair_noisy_clean_np_files(noisy_test_pts,  clean_test_pts)

#         dp_train_pairs, dp_val_pairs = deterministic_paired_split(
#             dp_pairs_train_all, val_ratio=dp_val_ratio, seed=dp_seed
#         )

#         dp_train_ds = DenoisingPointsPairedDataset(
#             pairs=dp_train_pairs,
#             in_channels=dp_in_channels,
#             tile_T=dp_tile_T,
#             stride_T=dp_stride_T,
#             remove_dc_bin=dp_remove_dc_bin,
#         )
#         dp_val_ds = DenoisingPointsPairedDataset(
#             pairs=dp_val_pairs,
#             in_channels=dp_in_channels,
#             tile_T=dp_tile_T,
#             stride_T=dp_stride_T,
#             remove_dc_bin=dp_remove_dc_bin,
#         )
#         dp_test_ds = DenoisingPointsPairedDataset(
#             pairs=dp_pairs_test_all,
#             in_channels=dp_in_channels,
#             tile_T=dp_tile_T,
#             stride_T=dp_stride_T,
#             remove_dc_bin=dp_remove_dc_bin,
#         )

#         train_loader_dp = DataLoader(
#             dp_train_ds, batch_size=batch_size, shuffle=True,
#             num_workers=num_workers, pin_memory=True,
#             worker_init_fn=_seed_worker, generator=g_dp,
#         )
#         val_loader_dp = DataLoader(
#             dp_val_ds, batch_size=batch_size, shuffle=False,
#             num_workers=num_workers, pin_memory=True,
#             worker_init_fn=_seed_worker, generator=g_dp,
#         )
#         test_loader_dp = DataLoader(
#             dp_test_ds, batch_size=batch_size, shuffle=False,
#             num_workers=num_workers, pin_memory=True,
#             worker_init_fn=_seed_worker, generator=g_dp,
#         )

#         logger.info(
#             f"[Denoising/NPY-Punkte] "
#             f"Paare Train/Val/Test: {len(dp_train_pairs)}/{len(dp_val_pairs)}/{len(dp_pairs_test_all)} | "
#             f"Tiles Train/Val/Test: {len(dp_train_ds)}/{len(dp_val_ds)}/{len(dp_test_ds)} | "
#             f"FreqBins(after DC) ~ {dp_train_ds.freq_bins_after}"
#         )

#         out["denoising_points"] = (train_loader_dp, val_loader_dp, test_loader_dp)

#     return out


# if __name__ == "__main__":
#     DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
#     loaders = get_data_loaders(
#         DATASET_PATH,
#         enable_classification=True,
#         enable_denoising=True,
#         enable_denoising_points=True,
#         # PNG-Params
#         dn_in_channels=1,
#         dn_tile_h=512,
#         dn_tile_w=256,
#         dn_stride_w=128,
#         # NPY-Params
#         dp_in_channels=2,
#         dp_tile_T=256,
#         dp_stride_T=128,
#         dp_remove_dc_bin=True,
#     )
#     summary = {
#         k: (
#             tuple(len(v[i].dataset) for i in range(3)) if v else None
#         )
#         for k, v in loaders.items()
#     }
#     print("OK:", summary)

























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
