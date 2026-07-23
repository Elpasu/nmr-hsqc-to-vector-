# coding: ascii
"""
Test del oraculo v2 (ajustar_conteo_hetero) -- numpy PURO, corre local sin torch.

Verifica las 4 reglas airtight de zeroing por ausencia de heteroatomo y que,
tras el zeroing, se siguen cumpliendo las igualdades del ajuste de doble
restriccion (total de carbonos + cupo CH2). Ademas: con N>0, O>0 y N+O>=3 el
resultado es IDENTICO al oraculo v1 (ajustar_conteo_doble_exacto).

Correr:  python tests/test_oraculo_hetero.py   (desde experiments/E3_dos_conjuntos)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oraculo import (
    N_CLASSES, IDX_CH2, IDX_N, IDX_O, IDX_2X, IDX_3X,
    ajustar_conteo_doble_exacto, ajustar_conteo_hetero,
)


def _z(kv):
    """Vector pred_raw de 19 floats, 0 salvo los indices dados (dict idx->val)."""
    v = np.zeros(N_CLASSES, dtype=np.float64)
    for i, val in kv.items():
        v[int(i)] = val
    return v


def test_n0_zeroes_n_classes_and_moves_ch2n_to_ch2():
    # Molecula SIN N; el modelo confunde CH2 (idx1) con CH2-N (idx9).
    # Verdad: 2 CH2. total=2, ch2=2, n=0, o=0.
    pred = _z({1: 1.1, 9: 0.9})   # CH2=1.1, CH2-N=0.9
    out = ajustar_conteo_hetero(pred, total_real=2, ch2_real=2, n_atoms=0, o_atoms=0)
    for i in IDX_N:
        assert out[i] == 0, f"clase N idx {i} deberia ser 0 con n_atoms=0"
    assert out[1] == 2, f"CH2 deberia absorber el cupo: {out[1]}"
    assert out.sum() == 2
    assert sum(out[i] for i in IDX_CH2) == 2


def test_o0_zeroes_o_classes():
    # Molecula SIN O; modelo predice CH-O (idx6). Verdad: 2 CH (idx2).
    pred = _z({2: 1.2, 6: 0.8})
    out = ajustar_conteo_hetero(pred, total_real=2, ch2_real=0, n_atoms=1, o_atoms=0)
    for i in IDX_O:
        assert out[i] == 0, f"clase O idx {i} deberia ser 0 con o_atoms=0"
    assert out.sum() == 2
    assert sum(out[i] for i in IDX_CH2) == 0


def test_c2x_zeroed_when_no_second_heteroatom():
    # N+O = 1 (<2): C-2X imposible; (<3): C-3X imposible.
    pred = _z({3: 1.2, 17: 0.9, 18: 0.9})  # Cq=1.2, C-2X=0.9, C-3X=0.9
    out = ajustar_conteo_hetero(pred, total_real=1, ch2_real=0, n_atoms=1, o_atoms=0)
    assert out[IDX_2X] == 0, "C-2X debe ser 0 con N+O<2"
    assert out[IDX_3X] == 0, "C-3X debe ser 0 con N+O<3"
    assert out.sum() == 1


def test_c3x_zeroed_but_c2x_allowed_when_no_exactly_two():
    # N+O = 2: C-2X permitido (idx17 NO se anula), pero C-3X (<3) si.
    pred = _z({17: 1.4, 18: 0.6})  # C-2X=1.4, C-3X=0.6
    out = ajustar_conteo_hetero(pred, total_real=1, ch2_real=0, n_atoms=1, o_atoms=1)
    assert out[IDX_3X] == 0, "C-3X debe ser 0 con N+O<3"
    assert out[IDX_2X] == 1, "C-2X permitido con N+O>=2 y debe absorber el total"
    assert out.sum() == 1


def test_identical_to_v1_when_heteroatoms_present():
    # N>0, O>0, N+O>=3: no dispara ninguna regla -> identico a v1.
    rng = np.random.RandomState(0)
    for _ in range(50):
        pred = rng.rand(N_CLASSES) * 3.0
        total = int(round(pred.sum()))
        ch2 = int(round(sum(pred[i] for i in IDX_CH2)))
        v1 = ajustar_conteo_doble_exacto(pred.copy(), total, ch2)
        v2 = ajustar_conteo_hetero(pred.copy(), total, ch2, n_atoms=2, o_atoms=2)
        assert np.array_equal(v1, v2), "v2 debe ser identico a v1 sin ausencias"


def test_does_not_mutate_input():
    pred = _z({1: 1.1, 9: 0.9})
    before = pred.copy()
    _ = ajustar_conteo_hetero(pred, total_real=2, ch2_real=2, n_atoms=0, o_atoms=0)
    assert np.array_equal(pred, before), "no debe mutar el pred_raw de entrada"


def test_equalities_hold_general():
    # En un barrido con ausencias variadas, siempre se cumplen las 2 igualdades.
    rng = np.random.RandomState(1)
    for _ in range(200):
        pred = rng.rand(N_CLASSES) * 2.5
        n_atoms = rng.randint(0, 3)
        o_atoms = rng.randint(0, 3)
        # total/ch2 coherentes con el propio pred redondeado (como en eval real)
        total = int(round(pred.sum()))
        ch2 = int(round(sum(pred[i] for i in IDX_CH2)))
        if total < ch2:
            total = ch2
        out = ajustar_conteo_hetero(pred, total, ch2, n_atoms, o_atoms)
        assert out.sum() == total, f"sum {out.sum()} != total {total}"
        assert sum(out[i] for i in IDX_CH2) == ch2
        assert (out >= 0).all()
        if n_atoms == 0:
            assert all(out[i] == 0 for i in IDX_N)
        if o_atoms == 0:
            assert all(out[i] == 0 for i in IDX_O)
        if n_atoms + o_atoms < 2:
            assert out[IDX_2X] == 0
        if n_atoms + o_atoms < 3:
            assert out[IDX_3X] == 0


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n>>> {len(fns)} TESTS OK <<<")


if __name__ == "__main__":
    _run()
