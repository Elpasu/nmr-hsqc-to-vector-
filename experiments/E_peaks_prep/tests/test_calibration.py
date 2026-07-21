# coding: ascii
"""Tests de calibration.py -- corre localmente, solo depende de numpy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration import bin_to_ppm


def test_bin_to_ppm_extremes():
    # C13: [0, 220], resolution=256 -> bin 0 = 0 ppm, bin 255 = 220 ppm
    assert abs(bin_to_ppm(0, 0, 220, 256) - 0.0) < 1e-6
    assert abs(bin_to_ppm(255, 0, 220, 256) - 220.0) < 1e-6
    print("[OK] test_bin_to_ppm_extremes")


def test_bin_to_ppm_midpoint():
    # H1: [-1, 15], bin 127.5 (centroide fraccionario) -> punto medio del rango
    mid = bin_to_ppm(127.5, -1, 15, 256)
    expected = -1 + (127.5 / 255) * 16
    assert abs(mid - expected) < 1e-6
    print(f"[OK] test_bin_to_ppm_midpoint ({mid:.4f})")


def test_bin_to_ppm_roundtrip_against_generator_forward():
    # ppm_to_bin_uniform tal como esta en Genera_mapas_de_pkl_v2.py (copiado
    # aca SOLO para este test, no es parte del script de produccion).
    import numpy as np

    def ppm_to_bin_uniform(ppm, ppm_min, ppm_max, resolution):
        ppm_clamped = max(ppm_min, min(ppm_max, ppm))
        normalized = (ppm_clamped - ppm_min) / (ppm_max - ppm_min)
        return int(np.clip(normalized * (resolution - 1), 0, resolution - 1))

    for ppm in (0.0, 55.3, 110.0, 219.9):
        b = ppm_to_bin_uniform(ppm, 0, 220, 256)
        back = bin_to_ppm(b, 0, 220, 256)
        # El roundtrip no es exacto (binning pierde resolucion sub-bin), pero
        # debe caer dentro de un bin de distancia (~0.86 ppm para C13/256).
        assert abs(back - ppm) < (220 / 255) + 1e-6, (ppm, b, back)
    print("[OK] test_bin_to_ppm_roundtrip_against_generator_forward")


if __name__ == "__main__":
    test_bin_to_ppm_extremes()
    test_bin_to_ppm_midpoint()
    test_bin_to_ppm_roundtrip_against_generator_forward()
    print("\n>>> test_calibration.py OK <<<")
