# coding: ascii
"""Test de build_padded_arrays -- corre localmente, solo numpy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from extract_peaks import build_padded_arrays


def test_build_padded_arrays_shapes_and_mask():
    peaks_per_molecule = [
        [(55.0, 1.2, 1.0, 0.33)],
        [(55.0, 1.2, 1.0, 0.33), (110.0, 4.5, -2.0, 0.67)],
        [],  # molecula sin picos detectados (caso limite, ej. imagen vacia)
    ]
    peaks_array, mask_array = build_padded_arrays(peaks_per_molecule)

    assert peaks_array.shape == (3, 2, 4), peaks_array.shape
    assert mask_array.shape == (3, 2), mask_array.shape
    assert mask_array.dtype == bool

    assert mask_array[0].tolist() == [True, False]
    assert mask_array[1].tolist() == [True, True]
    assert mask_array[2].tolist() == [False, False]

    assert np.allclose(peaks_array[0, 0], [55.0, 1.2, 1.0, 0.33])
    assert np.allclose(peaks_array[1, 1], [110.0, 4.5, -2.0, 0.67])
    # Filas con mask=False deben quedar en cero (padding)
    assert np.allclose(peaks_array[0, 1], [0.0, 0.0, 0.0, 0.0])
    print("[OK] test_build_padded_arrays_shapes_and_mask")


def test_build_padded_arrays_empty_input():
    peaks_array, mask_array = build_padded_arrays([])
    assert peaks_array.shape == (0, 0, 4)
    assert mask_array.shape == (0, 0)
    print("[OK] test_build_padded_arrays_empty_input")


if __name__ == "__main__":
    test_build_padded_arrays_shapes_and_mask()
    test_build_padded_arrays_empty_input()
    print("\n>>> test_padding.py OK <<<")
