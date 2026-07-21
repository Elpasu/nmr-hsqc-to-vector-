# coding: ascii
"""Tests de blob_detect.py -- requiere scipy, correr en el cluster (login node)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from blob_detect import detect_peaks


def _gaussian_blob(matrix, row, col, sigma=0.5, intensity=1.0):
    size = matrix.shape[0]
    radius = int(6 * sigma) + 1
    r0, r1 = max(0, row - radius), min(size, row + radius + 1)
    c0, c1 = max(0, col - radius), min(size, col + radius + 1)
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    g = intensity * np.exp(-0.5 * (((rr - row) / sigma) ** 2 + ((cc - col) / sigma) ** 2))
    matrix[r0:r1, c0:c1] += g
    return matrix


def test_two_separated_peaks_detected_individually():
    ch0 = np.zeros((256, 256), dtype=np.float32)
    ch1 = np.zeros((256, 256), dtype=np.float32)
    _gaussian_blob(ch0, row=50, col=60, intensity=1.0)
    _gaussian_blob(ch1, row=50, col=60, intensity=0.33)
    _gaussian_blob(ch0, row=200, col=210, intensity=-2.0)  # CH2, fase negativa
    _gaussian_blob(ch1, row=200, col=210, intensity=0.67)

    peaks = detect_peaks(ch0, ch1)
    assert len(peaks) == 2, f"esperados 2 picos, salieron {len(peaks)}"

    peaks_sorted = sorted(peaks, key=lambda p: p[0])
    r0, c0, a0_ch0, a0_ch1 = peaks_sorted[0]
    assert abs(r0 - 50) < 1.0 and abs(c0 - 60) < 1.0
    assert a0_ch0 > 0  # CH/CH3, fase positiva

    r1, c1, a1_ch0, a1_ch1 = peaks_sorted[1]
    assert abs(r1 - 200) < 1.0 and abs(c1 - 210) < 1.0
    assert a1_ch0 < 0  # CH2, fase negativa
    print(f"[OK] test_two_separated_peaks_detected_individually -> {peaks_sorted}")


def test_two_adjacent_peaks_merge_into_one_blob():
    # Dos carbonos distintos a 1 pixel de distancia: sus gaussianas (radio
    # ~4px) se solapan y scipy.ndimage.label los ve como UN componente
    # conexo -- esto es la "colision real" que Fase 1 tiene que poder medir.
    ch0 = np.zeros((256, 256), dtype=np.float32)
    ch1 = np.zeros((256, 256), dtype=np.float32)
    _gaussian_blob(ch0, row=100, col=100, intensity=1.0)
    _gaussian_blob(ch0, row=101, col=101, intensity=1.0)

    peaks = detect_peaks(ch0, ch1)
    assert len(peaks) == 1, f"esperado 1 blob fusionado, salieron {len(peaks)}"
    print(f"[OK] test_two_adjacent_peaks_merge_into_one_blob -> {peaks}")


def test_empty_image_returns_no_peaks():
    ch0 = np.zeros((256, 256), dtype=np.float32)
    ch1 = np.zeros((256, 256), dtype=np.float32)
    peaks = detect_peaks(ch0, ch1)
    assert peaks == []
    print("[OK] test_empty_image_returns_no_peaks")


if __name__ == "__main__":
    test_two_separated_peaks_detected_individually()
    test_two_adjacent_peaks_merge_into_one_blob()
    test_empty_image_returns_no_peaks()
    print("\n>>> test_blob_detect.py OK <<<")
