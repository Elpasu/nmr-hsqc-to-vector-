# coding: ascii
"""Smoke test offline del Set Transformer (rule 5). Verifica shape, ausencia
de NaN con molecula vacia, invariancia a permutacion, y tamano chico."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model_e3_settransformer import NMR_SetTransformer

N_CLASSES, MAX_CH, MAX_13C = 19, 32, 40


def test_forward_shape():
    model = NMR_SetTransformer(num_classes=N_CLASSES).eval()
    B = 4
    with torch.no_grad():
        out = model(torch.randn(B, MAX_CH, 4), torch.ones(B, MAX_CH),
                    torch.randn(B, MAX_13C, 1), torch.ones(B, MAX_13C),
                    torch.randn(B, 8))
    assert out.shape == (B, N_CLASSES), out.shape
    print(f"[OK] forward -> {tuple(out.shape)}")


def test_empty_molecule_no_nan():
    model = NMR_SetTransformer(num_classes=N_CLASSES).eval()
    B = 2
    mask_ch = torch.ones(B, MAX_CH); mask_13c = torch.ones(B, MAX_13C)
    mask_ch[0] = 0.0; mask_13c[0] = 0.0
    with torch.no_grad():
        out = model(torch.randn(B, MAX_CH, 4), mask_ch,
                    torch.randn(B, MAX_13C, 1), mask_13c, torch.randn(B, 8))
    assert torch.isfinite(out).all(), out
    print("[OK] molecula sin picos -> sin NaN/Inf")


def test_permutation_invariance():
    model = NMR_SetTransformer(num_classes=N_CLASSES).eval()
    peaks_ch = torch.randn(1, MAX_CH, 4); mask_ch = torch.ones(1, MAX_CH)
    peaks_13c = torch.randn(1, MAX_13C, 1); mask_13c = torch.ones(1, MAX_13C)
    cond = torch.randn(1, 8)
    perm = torch.randperm(MAX_CH)
    with torch.no_grad():
        o1 = model(peaks_ch, mask_ch, peaks_13c, mask_13c, cond)
        o2 = model(peaks_ch[:, perm], mask_ch[:, perm], peaks_13c, mask_13c, cond)
    assert torch.allclose(o1, o2, atol=1e-4), (o1 - o2).abs().max()
    print("[OK] invariante a permutacion de los picos")


def test_param_count_small():
    model = NMR_SetTransformer(num_classes=N_CLASSES)
    n = sum(p.numel() for p in model.parameters())
    assert n < 200_000, n   # chico por diseno (V10 ~8.6M)
    print(f"[OK] parametros = {n:,} (chico por diseno)")


if __name__ == "__main__":
    test_forward_shape(); test_empty_molecule_no_nan()
    test_permutation_invariance(); test_param_count_small()
    print("\n>>> SMOKE SET TRANSFORMER E3 OK <<<")
