# coding: ascii
"""Test del dataset de Exp F con npz sinteticos chicos. Identico al de Fase
3 (dataset_e3.py) -- Exp F no cambia el dataset, solo el modelo y la loss."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from dataset_f import NMRTwoSetsDataset

NORM = {"c13_ppm_min": 0.0, "c13_ppm_max": 220.0,
        "h1_ppm_min": -1.0, "h1_ppm_max": 15.0, "amp_ch0_scale": 3.0}


def _make_tmp(tmp):
    # 2 moleculas. crosspeaks: (2, 2, 4); 13c: (2, 3, 1).
    peaks_ch = np.zeros((2, 2, 4), dtype=np.float32)
    peaks_ch[0, 0] = [220.0, 15.0, 3.0, 1.0]   # extremos de calibracion
    peaks_ch[0, 1] = [0.0, -1.0, -3.0, 0.333]
    mask_ch = np.array([[True, True], [True, False]])
    np.savez(tmp / "ch.npz", peaks=peaks_ch, peaks_mask=mask_ch)

    peaks_13c = np.zeros((2, 3, 1), dtype=np.float32)
    peaks_13c[0, 0, 0] = 220.0
    peaks_13c[0, 1, 0] = 110.0
    mask_13c = np.array([[True, True, False], [True, False, False]])
    np.savez(tmp / "c13.npz", peaks_13c=peaks_13c, mask_13c=mask_13c)

    labels = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                      dtype=np.int64)
    np.save(tmp / "labels.npy", labels)
    np.save(tmp / "smiles.npy", np.array(["CCO", "CC(=O)C"], dtype=object))


def test_shapes_and_normalization(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    _make_tmp(tmp)
    ds = NMRTwoSetsDataset(str(tmp / "ch.npz"), str(tmp / "c13.npz"),
                           str(tmp / "labels.npy"), str(tmp / "smiles.npy"), NORM)
    (peaks_ch, mask_ch, peaks_13c, mask_13c, cond), target = ds[0]
    assert peaks_ch.shape == (2, 4), peaks_ch.shape
    assert peaks_13c.shape == (3, 1), peaks_13c.shape
    assert cond.shape == (8,), cond.shape
    assert target.shape == (19,), target.shape
    assert abs(peaks_ch[0, 0].item() - 1.0) < 1e-5
    assert abs(peaks_ch[0, 1].item() - 1.0) < 1e-5
    assert abs(peaks_ch[0, 2].item() - 1.0) < 1e-5
    assert abs(peaks_13c[0, 0].item() - 1.0) < 1e-5
    assert abs(peaks_13c[1, 0].item() - 0.5) < 1e-5
    print("[OK] shapes y normalizacion correctas")


if __name__ == "__main__":
    test_shapes_and_normalization()
    print("\n>>> TEST DATASET F OK <<<")
