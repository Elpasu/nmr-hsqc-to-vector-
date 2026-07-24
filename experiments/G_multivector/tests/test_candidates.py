# coding: ascii
"""Test del generador de candidatos intra-nH -- numpy PURO, corre local sin torch.
Correr:  python tests/test_candidates.py   (desde experiments/G_multivector)"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oraculo import IDX_CH2, ajustar_conteo_hetero
from candidates import generate_candidates, NH_GROUPS

N = 19


def _raw(indices_dict):
    v = np.zeros(N, dtype=np.float64)
    for i, val in indices_dict.items():
        v[int(i)] = val
    return v


def _fm_ok(c, total, ch2):
    return c.sum() == total and sum(c[i] for i in IDX_CH2) == ch2


def test_top1_is_oraculo_v2():
    raw = _raw({1: 1.4, 9: 0.6, 0: 1.9})
    cands = generate_candidates(raw, total=4, ch2=2, n_atoms=1, o_atoms=1, K=3)
    v2 = ajustar_conteo_hetero(raw, 4, 2, 1, 1)
    assert np.array_equal(cands[0], v2), "el top-1 debe ser el oraculo v2"


def test_2h_split_genera_ambos_candidatos():
    # CH2=1.4 vs CH2-N=0.6, cupo 2 -> deben aparecer (CH2=2,CH2-N=0) y (CH2=1,CH2-N=1)
    raw = _raw({1: 1.4, 9: 0.6})
    cands = generate_candidates(raw, total=2, ch2=2, n_atoms=1, o_atoms=1, K=4)
    got = {(c[1], c[9]) for c in cands}
    assert (2, 0) in got and (1, 1) in got, f"faltan candidatos 2H: {got}"


def test_todos_fm_consistentes():
    rng = np.random.RandomState(0)
    for _ in range(100):
        raw = rng.rand(N) * 2.5
        total = int(round(raw.sum()))
        ch2 = int(round(sum(raw[i] for i in IDX_CH2)))
        if total < ch2:
            total = ch2
        cands = generate_candidates(raw, total, ch2, n_atoms=2, o_atoms=2, K=5)
        for c in cands:
            assert _fm_ok(c, total, ch2), f"candidato no FM-consistente: {c}"
            assert (c >= 0).all()


def test_no_puebla_clases_prohibidas():
    # n_atoms=0 -> ninguna clase -N poblada en ningun candidato
    raw = _raw({1: 1.1, 9: 0.9})
    cands = generate_candidates(raw, total=2, ch2=2, n_atoms=0, o_atoms=0, K=5)
    for c in cands:
        assert all(c[i] == 0 for i in [8, 9, 10, 11, 16]), f"poblo clase -N: {c}"


def test_no_puebla_clases_prohibidas_o():
    # o_atoms=0 -> ninguna clase -O poblada en ningun candidato
    raw = _raw({0: 1.0, 4: 0.5, 6: 0.5})
    cands = generate_candidates(raw, total=2, ch2=0, n_atoms=1, o_atoms=0, K=5)
    for c in cands:
        assert all(c[i] == 0 for i in [4, 5, 6, 7, 15]), f"poblo clase -O: {c}"


def test_no_puebla_c2x_cuando_n_o_menos_2():
    # n_atoms+o_atoms < 2 -> C-2X (idx 17) en 0
    raw = _raw({0: 0.9, 17: 1.1})
    cands = generate_candidates(raw, total=2, ch2=0, n_atoms=1, o_atoms=0, K=5)
    for c in cands:
        assert c[17] == 0, f"poblo C-2X con n_o < 2: {c}"


def test_no_puebla_c3x_cuando_n_o_menos_3():
    # n_atoms+o_atoms < 3 -> C-3X (idx 18) en 0
    raw = _raw({0: 0.9, 18: 1.1})
    cands = generate_candidates(raw, total=2, ch2=0, n_atoms=2, o_atoms=0, K=5)
    for c in cands:
        assert c[18] == 0, f"poblo C-3X con n_o < 3: {c}"


def test_sin_duplicados_y_len_max_K():
    raw = _raw({1: 1.4, 9: 0.6, 2: 1.5, 6: 0.5})
    cands = generate_candidates(raw, total=4, ch2=2, n_atoms=1, o_atoms=1, K=3)
    assert len(cands) <= 3
    keys = {c.tobytes() for c in cands}
    assert len(keys) == len(cands), "hay candidatos duplicados"


def test_ranking_por_L1_al_crudo():
    # el 2do candidato debe estar mas cerca en L1 del crudo que el 3ro
    raw = _raw({1: 1.4, 9: 0.6, 2: 1.4, 6: 0.6})
    cands = generate_candidates(raw, total=4, ch2=2, n_atoms=1, o_atoms=1, K=5)
    d = [np.abs(c - raw).sum() for c in cands[1:]]
    assert d == sorted(d), f"el resto no esta rankeado por L1: {d}"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n>>> {len(fns)} TESTS OK <<<")


if __name__ == "__main__":
    _run()
