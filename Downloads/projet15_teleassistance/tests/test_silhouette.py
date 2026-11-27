import unittest

import numpy as np
from PIL import Image

from training import silhouette as S


def _scene(rng, box=None, lum=200, shift=0.0, noise=3.0):
    h, w = 480, 640
    xx = np.arange(w)[None, :].repeat(h, 0)
    img = (120 + 30 * np.sin(xx / 80.0)).astype(np.float32)
    if box:
        y0, y1, x0, x1 = box
        img[y0:y1, x0:x1] = lum
    img += rng.normal(0, noise, img.shape) + shift
    return Image.fromarray(np.clip(np.stack([img] * 3, -1), 0, 255).astype(np.uint8))


class SilhouetteTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.bg = S.compute_background(S.to_gray(_scene(self.rng)) for _ in range(5))

    def test_shapes_and_dtype(self):
        sil = S.to_silhouette(_scene(self.rng, (100, 420, 280, 360)), self.bg)
        self.assertEqual(sil.shape, (S.OUT_SIZE, S.OUT_SIZE))
        self.assertEqual(sil.dtype, np.uint8)
        self.assertEqual(S.to_model_input(sil).shape, (1, S.OUT_SIZE, S.OUT_SIZE, 1))

    def test_person_detected_with_good_overlap(self):
        sil = S.to_silhouette(_scene(self.rng, (100, 420, 280, 360)), self.bg) > 127
        truth = np.zeros((480, 640), np.uint8)
        truth[100:420, 280:360] = 255
        truth = np.asarray(Image.fromarray(truth).resize((S.OUT_SIZE, S.OUT_SIZE), Image.Resampling.BILINEAR)) > 127
        iou = (truth & sil).sum() / (truth | sil).sum()
        self.assertGreater(iou, 0.8)

    def test_standing_and_lying_have_different_shapes(self):
        standing = S.to_silhouette(_scene(self.rng, (100, 420, 280, 360)), self.bg) > 127
        lying = S.to_silhouette(_scene(self.rng, (330, 420, 150, 500)), self.bg) > 127
        def aspect(m):
            ys, xs = np.where(m)
            return (ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1)
        self.assertGreater(aspect(standing), 1.5)
        self.assertLess(aspect(lying), 0.7)

    def test_empty_room_gives_empty_silhouette(self):
        self.assertEqual(S.foreground_ratio(S.to_silhouette(_scene(self.rng), self.bg)), 0.0)

    def test_global_lighting_change_is_ignored(self):
        self.assertEqual(S.foreground_ratio(S.to_silhouette(_scene(self.rng, shift=25), self.bg)), 0.0)

    def test_background_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            S.to_silhouette(_scene(self.rng), np.zeros((10, 10), np.float32))

    def test_background_needs_enough_images(self):
        with self.assertRaises(ValueError):
            S.compute_background([self.bg, self.bg])


if __name__ == "__main__":
    unittest.main()
