# coding: ascii
"""dataset_j.py -- normaliza la 5a feature (degeneracion), recorta a
peak_features, y arma el condicionante con la semantica nueva:
cond[0] = carbonos totales (== C de la formula), cond[1] = carbonos CH2."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_j import NMRTwoSetsDatasetJ

NORM = {"c13_ppm_min": 0, "c13_ppm_max": 220, "h1_ppm_min": -1,
        "h1_ppm_max": 15, "amp_ch0_scale": 3.0, "degeneracion_scale": 4.0}

# Benceno: 1 crosspeak (delta_c 128.5, delta_h 7.26, CH -> amp0 +1, amp1 1/3),
# degeneracion 6. Label de carbonos totales: 6 en =CH/Ar (indice 13).
PEAKS_CH = np.array([[[128.5, 7.26, 1.0, 1.0 / 3.0, 6.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
MASK_CH = np.array([[True, False]])
PEAKS_13C = np.array([[[128.5], [0.0]]], dtype=np.float32)
MASK_13C = np.array([[True, False]])
LABEL = np.zeros((1, 19), dtype=np.float32)
LABEL[0, 13] = 6.0          # =CH/Ar : 6 carbonos


def _fixture(tmp):
    tmp = Path(tmp)
    np.savez(tmp / "ch.npz", peaks=PEAKS_CH, peaks_mask=MASK_CH)
    np.savez(tmp / "c13.npz", peaks_13c=PEAKS_13C, mask_13c=MASK_13C)
    np.save(tmp / "labels.npy", LABEL)
    np.save(tmp / "smiles.npy", np.array(["c1ccccc1"], dtype=object))
    return (str(tmp / "ch.npz"), str(tmp / "c13.npz"),
            str(tmp / "labels.npy"), str(tmp / "smiles.npy"))


def test_cinco_features_normaliza_la_degeneracion():
    with tempfile.TemporaryDirectory() as tmp:
        ds = NMRTwoSetsDatasetJ(*_fixture(tmp), NORM, peak_features=5)
        (pch, _, _, _, _), _ = ds[0]
    assert pch.shape == (2, 5), pch.shape
    assert abs(float(pch[0, 0]) - 128.5 / 220.0) < 1e-4      # delta_c / 220
    assert abs(float(pch[0, 1]) - (7.26 + 1.0) / 16.0) < 1e-4  # (delta_h+1)/16
    assert abs(float(pch[0, 2]) - 1.0 / 3.0) < 1e-4          # amp_ch0 / 3
    assert abs(float(pch[0, 4]) - 6.0 / 4.0) < 1e-4          # degeneracion / 4
    print("[OK] 5 features, degeneracion normalizada por degeneracion_scale")


def test_cuatro_features_recorta_la_columna():
    """El control lee el MISMO .npz y descarta la 5a columna. Que sea el mismo
    archivo es lo que garantiza que J-A y J-0 difieran solo en la feature."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = NMRTwoSetsDatasetJ(*_fixture(tmp), NORM, peak_features=4)
        (pch, _, _, _, _), _ = ds[0]
    assert pch.shape == (2, 4), pch.shape
    assert abs(float(pch[0, 2]) - 1.0 / 3.0) < 1e-4
    print("[OK] 4 features: la 5a columna se recorta, las otras no cambian")


def test_cond_usa_carbonos_totales():
    """cond[0] == 6 (carbonos, no senales) y cond[2] == 6 (C de la formula del
    benceno). Que coincidan es el punto del experimento."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = NMRTwoSetsDatasetJ(*_fixture(tmp), NORM, peak_features=5)
        (_, _, _, _, cond), target = ds[0]
    assert cond.shape == (8,), cond.shape
    assert float(cond[0]) == 6.0, float(cond[0])   # suma del vector = carbonos
    assert float(cond[1]) == 0.0, float(cond[1])   # el benceno no tiene CH2
    assert float(cond[2]) == 6.0, float(cond[2])   # C de la formula
    assert float(target.sum()) == 6.0
    print("[OK] cond[0] == cond[2] == 6: la suma del vector es C de la formula")


def test_cond_cuenta_carbonos_ch2():
    """cond[1] suma las 4 clases del cupo CH2 (indices 1, 5, 9, 12)."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _fixture(tmp)
        lab = np.zeros((1, 19), dtype=np.float32)
        lab[0, 1] = 2.0    # CH2
        lab[0, 5] = 1.0    # CH2-O
        lab[0, 13] = 3.0   # =CH/Ar
        np.save(paths[2], lab)
        ds = NMRTwoSetsDatasetJ(*paths, NORM, peak_features=5)
        (_, _, _, _, cond), _ = ds[0]
    assert float(cond[0]) == 6.0
    assert float(cond[1]) == 3.0, float(cond[1])
    print("[OK] cond[1] == 3 carbonos CH2 (2 CH2 + 1 CH2-O)")


def test_columnas_0_a_3_identicas_entre_peak_features_4_y_5():
    """Toda la comparacion J-A (5 features) vs J-0 (4 features) depende de que
    las columnas 0-3 (delta_c, delta_h, amp_ch0, amp_ch1) sean IDENTICAS entre
    ambas variantes -- solo la 5a columna (degeneracion) debe diferir. Esto
    vale por estructura del codigo (las columnas 0-3 se normalizan sin
    depender de peak_features), pero nada lo aseveraba directamente."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _fixture(tmp)
        ds5 = NMRTwoSetsDatasetJ(*paths, NORM, peak_features=5)
        ds4 = NMRTwoSetsDatasetJ(*paths, NORM, peak_features=4)
        (pch5, _, _, _, _), _ = ds5[0]
        (pch4, _, _, _, _), _ = ds4[0]
    assert pch5.shape == (2, 5), pch5.shape
    assert pch4.shape == (2, 4), pch4.shape
    import torch
    assert torch.allclose(pch5[:, :4], pch4), (pch5[:, :4], pch4)
    print("[OK] columnas 0-3 identicas entre peak_features=5 y peak_features=4")


def test_peak_features_cinco_con_npz_de_cuatro_falla():
    """Pedir 5 features sobre un .npz que solo tiene 4 columnas tiene que
    romper fuerte: si no, entrenaria con una columna de ceros haciendose pasar
    por la degeneracion."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = list(_fixture(tmp))
        np.savez(paths[0], peaks=PEAKS_CH[:, :, :4], peaks_mask=MASK_CH)
        try:
            NMRTwoSetsDatasetJ(*paths, NORM, peak_features=5)
        except ValueError:
            print("[OK] peak_features=5 con un .npz de 4 columnas -> ValueError")
            return
    raise AssertionError("se esperaba ValueError")


def test_falta_degeneracion_scale_falla():
    """Si el config no trae degeneracion_scale y se piden 5 features, hay que
    fallar en vez de inventar un default silencioso (regla dura 3)."""
    norm_sin = {k: v for k, v in NORM.items() if k != "degeneracion_scale"}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            NMRTwoSetsDatasetJ(*_fixture(tmp), norm_sin, peak_features=5)
        except KeyError:
            print("[OK] sin degeneracion_scale en el config -> KeyError")
            return
    raise AssertionError("se esperaba KeyError")


if __name__ == "__main__":
    test_cinco_features_normaliza_la_degeneracion()
    test_cuatro_features_recorta_la_columna()
    test_cond_usa_carbonos_totales()
    test_cond_cuenta_carbonos_ch2()
    test_columnas_0_a_3_identicas_entre_peak_features_4_y_5()
    test_peak_features_cinco_con_npz_de_cuatro_falla()
    test_falta_degeneracion_scale_falla()
    print("\n>>> DATASET J OK <<<")
