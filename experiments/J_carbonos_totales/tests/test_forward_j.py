# coding: ascii
"""Smoke test del Set Transformer del Exp J (regla dura 5). La diferencia con
el E3 es peak_features configurable: 5 para la corrida experimental (con
degeneracion) y 4 para el control. Un solo archivo de modelo sirve a las dos."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_j_settransformer import NMR_SetTransformerJ

N_CLASSES, MAX_CH, MAX_13C = 19, 32, 40


def _batch(n_feat, B=3):
    """Batch sintetico con mascaras MIXTAS: una molecula llena, una parcial y
    una totalmente enmascarada (el caso que produce NaN si el softmax no esta
    protegido)."""
    mask_ch = torch.ones(B, MAX_CH)
    mask_13c = torch.ones(B, MAX_13C)
    if B > 1:
        mask_ch[1, 10:] = 0.0
        mask_13c[1, 12:] = 0.0
    if B > 2:
        mask_ch[2] = 0.0
        mask_13c[2] = 0.0
    return (torch.randn(B, MAX_CH, n_feat), mask_ch,
            torch.randn(B, MAX_13C, 1), mask_13c,
            torch.randn(B, 8))


def test_forward_cinco_features():
    m = NMR_SetTransformerJ(num_classes=N_CLASSES, peak_features=5).eval()
    with torch.no_grad():
        out = m(*_batch(5))
    assert out.shape == (3, N_CLASSES), out.shape
    assert torch.isfinite(out).all(), "NaN/Inf en la salida"
    print(f"[OK] forward con 5 features (J-A) -> {tuple(out.shape)}")


def test_forward_cuatro_features():
    m = NMR_SetTransformerJ(num_classes=N_CLASSES, peak_features=4).eval()
    with torch.no_grad():
        out = m(*_batch(4))
    assert out.shape == (3, N_CLASSES), out.shape
    assert torch.isfinite(out).all(), "NaN/Inf en la salida"
    print(f"[OK] forward con 4 features (J-0 control) -> {tuple(out.shape)}")


def test_proj_ch_respeta_peak_features():
    assert NMR_SetTransformerJ(peak_features=5).proj_ch.in_features == 5
    assert NMR_SetTransformerJ(peak_features=4).proj_ch.in_features == 4
    print("[OK] proj_ch.in_features sigue a peak_features")


def test_default_es_cinco():
    """El default es la corrida experimental: el control tiene que pedir 4
    explicitamente en su config."""
    assert NMR_SetTransformerJ().proj_ch.in_features == 5
    print("[OK] peak_features default = 5")


def test_shape_equivocada_falla_fuerte():
    """Pasarle 4 features a un modelo de 5 tiene que romper, no producir
    numeros silenciosamente equivocados."""
    m = NMR_SetTransformerJ(peak_features=5).eval()
    try:
        with torch.no_grad():
            m(*_batch(4))
    except RuntimeError:
        print("[OK] entrada de 4 features a un modelo de 5 -> RuntimeError")
        return
    raise AssertionError("se esperaba RuntimeError por mismatch de dimensiones")


def test_peak_features_invalido_rechazado():
    for malo in (0, 3, -1):
        try:
            NMR_SetTransformerJ(peak_features=malo)
        except ValueError:
            continue
        raise AssertionError(f"se esperaba ValueError con peak_features={malo}")
    print("[OK] peak_features fuera de {4, 5} -> ValueError")


def test_invariante_a_permutacion_de_picos():
    """Propiedad central de un Set Transformer: el orden de los picos no
    puede cambiar la prediccion."""
    m = NMR_SetTransformerJ(peak_features=5).eval()
    pch, mch, p13, m13, cond = _batch(5, B=1)
    perm = torch.randperm(MAX_CH)
    with torch.no_grad():
        o1 = m(pch, mch, p13, m13, cond)
        o2 = m(pch[:, perm], mch[:, perm], p13, m13, cond)
    assert torch.allclose(o1, o2, atol=1e-4), (o1 - o2).abs().max()
    print("[OK] invariante a permutacion de los crosspeaks")


if __name__ == "__main__":
    test_forward_cinco_features()
    test_forward_cuatro_features()
    test_proj_ch_respeta_peak_features()
    test_default_es_cinco()
    test_shape_equivocada_falla_fuerte()
    test_peak_features_invalido_rechazado()
    test_invariante_a_permutacion_de_picos()
    print("\n>>> FORWARD J OK <<<")
