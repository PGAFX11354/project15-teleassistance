import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224


class VisionAgent:
    def __init__(self, model_path="models/vision_cnn_best.keras", threshold=0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.model = tf.keras.models.load_model(model_path)

    def _prepare(self, frame):
        """Accepts a file path, a PIL image, or a numpy array."""
        if isinstance(frame, str):
            img = tf.keras.utils.load_img(frame, target_size=(IMG_SIZE, IMG_SIZE))
            arr = tf.keras.utils.img_to_array(img)
        else:
            arr = np.array(frame, dtype=np.float32)
            if arr.ndim == 2:  # grayscale -> 3 channels
                arr = np.stack([arr] * 3, axis=-1)
            arr = tf.image.resize(arr[..., :3], [IMG_SIZE, IMG_SIZE]).numpy()
        arr = preprocess_input(arr)  # same scaling as training: [-1, 1]
        return np.expand_dims(arr, axis=0)

    def analyze(self, frame_features):
        prob = float(self.model.predict(self._prepare(frame_features), verbose=0)[0][0])
        return {
            "agent": "VisionAgent",
            "posture": "Fallen" if prob > self.threshold else "Normal",
            "fall_prob": round(prob, 4),
        }
