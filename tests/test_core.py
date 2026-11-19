import unittest

import numpy as np

from ctdemo.data import assert_no_subject_leakage, make_case, normalize_hu
from ctdemo.metrics import segmentation_metrics
from ctdemo.model import TinyUNet3D


class CoreTests(unittest.TestCase):
    def test_case_is_reproducible(self):
        a, ma, _ = make_case(7); b, mb, _ = make_case(7)
        self.assertTrue(np.array_equal(a, b)); self.assertTrue(np.array_equal(ma, mb))

    def test_normalization_range(self):
        x = normalize_hu(np.array([-1000, -200, 300, 1000], dtype=np.float32))
        self.assertTrue(np.all((0 <= x) & (x <= 1)))

    def test_metrics_perfect(self):
        x = np.array([0, 1, 1]); self.assertAlmostEqual(segmentation_metrics(x, x)["dice"], 1.0)

    def test_leakage_guard(self):
        manifest = {"cases": [{"subject_id": "a", "split": "train"}, {"subject_id": "a", "split": "test"}]}
        with self.assertRaises(ValueError): assert_no_subject_leakage(manifest)

    def test_model_shape(self):
        import torch
        x = torch.zeros(1, 1, 16, 24, 24)
        self.assertEqual(TinyUNet3D(base=2)(x).shape, x.shape)


if __name__ == "__main__": unittest.main()

