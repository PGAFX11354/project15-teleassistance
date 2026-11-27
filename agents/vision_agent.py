"""Agent de vision : probabilité de chute à partir d'une image.

Deux modes, choisis automatiquement par ``VisionAgent.from_models_dir`` :

* ``silhouette`` (privilégié) : le modèle ``vision_cnn_silhouette.keras`` ne voit qu'une
  silhouette binaire obtenue par soustraction du fond ``background.npy``. L'image RGB
  n'est ni conservée ni transmise au modèle (« sans intrusion visuelle directe »).
* ``rgb`` (modèle de base ``vision_cnn_best.keras``) : l'image RGB est transmise au CNN.
  Ce mode ne respecte pas à lui seul l'objectif de confidentialité.

TensorFlow n'est importé qu'au chargement du modèle (le reste du module est testable sans lui).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from training import silhouette as sil

IMG_SIZE = 224          # entrée du CNN RGB
MODES = ("rgb", "silhouette")


class VisionAgent:
    def __init__(self, model_path=None, threshold=0.5, mode="rgb", background=None, model=None):
        if mode not in MODES:
            raise ValueError(f"mode doit être dans {MODES}")
        if mode == "silhouette" and background is None:
            raise ValueError("Le mode silhouette nécessite l'image de fond (background.npy)")
        self.model_path = model_path
        self.threshold = threshold
        self.mode = mode
        self.background = None if background is None else np.asarray(background, dtype=np.float32)
        if model is None:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
        self.model = model

    @classmethod
    def from_models_dir(cls, models_dir, threshold=0.5):
        """Silhouette si ses deux fichiers existent, sinon modèle RGB de base."""
        folder = Path(models_dir)
        sil_model, background = folder / "vision_cnn_silhouette.keras", folder / "background.npy"
        if sil_model.is_file() and background.is_file():
            return cls(sil_model, threshold, "silhouette", np.load(background))
        return cls(folder / "vision_cnn_best.keras", threshold, "rgb")

    def _prepare_rgb(self, frame):
        """Prétraitement identique à celui du notebook train_cnn.ipynb."""
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = np.array(frame, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = tf.image.resize(arr[..., :3], [IMG_SIZE, IMG_SIZE]).numpy()
        return np.expand_dims(preprocess_input(arr), axis=0)

    def silhouette(self, frame) -> np.ndarray:
        """Silhouette uint8 (mode silhouette uniquement)."""
        if self.mode != "silhouette":
            raise RuntimeError("silhouette() n'est disponible qu'en mode silhouette")
        return sil.to_silhouette(frame, self.background)

    def analyze(self, frame_features, return_silhouette=False):
        silhouette = None
        if self.mode == "silhouette":
            silhouette = self.silhouette(frame_features)
            x = sil.to_model_input(silhouette)
        else:
            x = self._prepare_rgb(frame_features)
        prob = float(np.asarray(self.model.predict(x, verbose=0)).reshape(-1)[0])
        out = {
            "agent": "VisionAgent",
            "mode": self.mode,
            "posture": "Fallen" if prob > self.threshold else "Normal",
            "fall_prob": round(prob, 4),
        }
        if return_silhouette:
            out["silhouette"] = silhouette
        return out
