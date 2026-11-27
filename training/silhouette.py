"""Extraction de silhouettes : respect de la vie privée (« sans intrusion visuelle directe »).

Principe : on dispose d'une image de la pièce *vide* (le « fond »). Pour chaque image
reçue, on soustrait ce fond et on ne conserve que la silhouette binaire de la personne.
L'image RGB n'est ni affichée, ni stockée, ni transmise au modèle.

Ce module est utilisé à l'identique par le notebook d'entraînement
(training/train_cnn_silhouette.ipynb) et par l'application (agents/vision_agent.py),
afin que l'entraînement et l'inférence voient exactement la même représentation.
Dépendances : numpy, Pillow, scipy (déjà installé avec scikit-learn).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

WORK_SIZE = (160, 120)   # (largeur, hauteur) de calcul
OUT_SIZE = 128           # côté de la silhouette fournie au CNN
DIFF_THRESHOLD = 30.0    # écart de niveau de gris (0-255) à partir duquel un pixel est « avant-plan »
MIN_AREA_FRAC = 0.004    # une composante < 0,4 % de l'image est du bruit
KEEP_RATIO = 0.25        # on garde les composantes >= 25 % de la plus grande (membres isolés)


def to_gray(image) -> np.ndarray:
    """Image (PIL, ndarray ou chemin) -> niveaux de gris float32 de taille WORK_SIZE."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return to_gray(handle.convert("RGB"))
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    gray = image.convert("L").resize(WORK_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.float32)


def compute_background(grays: Iterable[np.ndarray]) -> np.ndarray:
    """Fond = médiane pixel à pixel d'images de la scène où la personne occupe des positions variées."""
    stack = np.stack(list(grays), axis=0)
    if stack.shape[0] < 3:
        raise ValueError("Au moins 3 images sont nécessaires pour estimer un fond")
    return np.median(stack, axis=0).astype(np.float32)


def background_from_paths(paths: Sequence, n_images: int = 300) -> np.ndarray:
    """Échantillonne n_images chemins répartis uniformément et calcule le fond."""
    paths = list(paths)
    idx = np.unique(np.linspace(0, len(paths) - 1, min(n_images, len(paths))).astype(int))
    return compute_background(to_gray(paths[i]) for i in idx)


def foreground_mask(image, background: np.ndarray, threshold: float = DIFF_THRESHOLD) -> np.ndarray:
    """Masque booléen (hauteur x largeur de WORK_SIZE) de la personne."""
    gray = to_gray(image)
    if gray.shape != background.shape:
        raise ValueError(f"Fond {background.shape} incompatible avec l'image {gray.shape}")
    delta = gray - background
    delta = delta - np.median(delta)               # compense un changement global d'éclairage
    diff = ndi.gaussian_filter(np.abs(delta), 1.0)
    mask = diff > threshold
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)), iterations=2)
    mask = ndi.binary_fill_holes(mask)
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.asarray(ndi.sum(mask, labels, index=np.arange(1, count + 1)))
    biggest = sizes.max()
    if biggest < MIN_AREA_FRAC * mask.size:
        return np.zeros_like(mask, dtype=bool)
    keep = [i + 1 for i, s in enumerate(sizes) if s >= KEEP_RATIO * biggest]
    return np.isin(labels, keep)


def to_silhouette(image, background: np.ndarray, threshold: float = DIFF_THRESHOLD) -> np.ndarray:
    """Silhouette uint8 (OUT_SIZE x OUT_SIZE), valeurs 0..255."""
    mask = foreground_mask(image, background, threshold)
    pic = Image.fromarray((mask * 255).astype(np.uint8))
    pic = pic.resize((OUT_SIZE, OUT_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(pic, dtype=np.uint8)


def to_model_input(silhouette: np.ndarray) -> np.ndarray:
    """(OUT_SIZE, OUT_SIZE) uint8 -> (1, OUT_SIZE, OUT_SIZE, 1) float32 valeurs 0..255.

    Le CNN contient sa propre couche Rescaling(1/255), le même tableau est donc
    utilisé à l'entraînement (uint8 converti en float) et à l'inférence.
    """
    return silhouette.astype("float32")[None, ..., None]


def foreground_ratio(silhouette: np.ndarray) -> float:
    """Part de pixels d'avant-plan (outil de diagnostic)."""
    return float((silhouette > 127).mean())
