# coding: ascii
"""Test puro (sin torch) del subsampleo deterministico de train_idx, para
el estudio de escalado de datos (Parte 2 del spec de Exp F). Corre local
ahora mismo -- no necesita torch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from split_utils import subsample_train_idx


def test_reproducible_same_seed():
    idx = np.arange(1000)
    a = subsample_train_idx(idx, 0.25, seed=42)
    b = subsample_train_idx(idx, 0.25, seed=42)
    assert np.array_equal(a, b)
    print("[OK] mismo seed -> mismos indices")


def test_fraction_size():
    idx = np.arange(1000)
    out = subsample_train_idx(idx, 0.25, seed=42)
    assert len(out) == 250, len(out)
    print(f"[OK] fraccion 0.25 de 1000 -> {len(out)} indices")


def test_fractions_are_nested():
    idx = np.arange(1000)
    f10 = set(subsample_train_idx(idx, 0.10, seed=42).tolist())
    f25 = set(subsample_train_idx(idx, 0.25, seed=42).tolist())
    f50 = set(subsample_train_idx(idx, 0.50, seed=42).tolist())
    assert f10.issubset(f25), "10% deberia ser subconjunto de 25%"
    assert f25.issubset(f50), "25% deberia ser subconjunto de 50%"
    print("[OK] fracciones anidadas (10% subset 25% subset 50%)")


def test_fraction_one_returns_all_unchanged():
    idx = np.arange(1000)
    out = subsample_train_idx(idx, 1.0, seed=42)
    assert np.array_equal(out, idx), "fraccion 1.0 no debe alterar train_idx"
    print("[OK] fraccion 1.0 devuelve train_idx intacto")


if __name__ == "__main__":
    test_reproducible_same_seed()
    test_fraction_size()
    test_fractions_are_nested()
    test_fraction_one_returns_all_unchanged()
    print("\n>>> TEST TRAIN_FRACTION OK <<<")
