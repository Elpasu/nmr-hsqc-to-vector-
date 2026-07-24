# coding: ascii
"""Test del generador guiado por incertidumbre (Fase 1b) -- numpy puro, corre local.
Correr:  python tests/test_candidates_uncertainty.py   (desde experiments/G_multivector)"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oraculo import IDX_CH2, ajustar_conteo_hetero
from candidates import generate_candidates_uncertainty

N = 19


def _raw(kv):
    v = np.zeros(N, dtype=np.float64)
    for i, val in kv.items():
        v[int(i)] = val
    return v


def _fm_ok(c, total, ch2):
    return c.sum() == total and sum(c[i] for i in IDX_CH2) == ch2


def test_top1_es_oraculo_v2():
    raw = _raw({1: 1.55, 9: 0.55})
    cands = generate_candidates_uncertainty(raw, 2, 2, 1, 0, tau=0.5, K_max=6)
    v2 = ajustar_conteo_hetero(raw, 2, 2, 1, 0)
    assert np.array_equal(cands[0], v2), "el top-1 debe ser el oraculo v2"


def test_molecula_segura_emite_uno():
    # CH3=1.02, CH2=1.98 (hidrocarburo, sin duda). total=3, ch2=2, n=o=0.
    raw = _raw({0: 1.02, 1: 1.98})
    cands = generate_candidates_uncertainty(raw, 3, 2, 0, 0, tau=0.5, K_max=6)
    assert len(cands) == 1, f"molecula segura deberia emitir 1 solo candidato, emitio {len(cands)}"


def test_molecula_con_duda_incluye_alternativa():
    # CH2=1.55 vs CH2-N=0.55 (empate) -> debe incluir (CH2=1, CH2-N=1)
    raw = _raw({1: 1.55, 9: 0.55})
    cands = generate_candidates_uncertainty(raw, 2, 2, 1, 0, tau=0.5, K_max=6)
    got = {(c[1], c[9]) for c in cands}
    assert (1, 1) in got, f"falta la alternativa de la duda CH2/CH2-N: {got}"
    assert len(cands) >= 2


def test_tau0_solo_empates_exactos():
    # CH2=1.6 vs CH2-N=0.4: la alternativa cuesta extra-L1=0.4 > 0 -> con tau=0 NO se emite.
    raw = _raw({1: 1.6, 9: 0.4})
    cands = generate_candidates_uncertainty(raw, 2, 2, 1, 0, tau=0.0, K_max=6)
    assert len(cands) == 1, f"con tau=0 y sin empate exacto deberia emitir 1, emitio {len(cands)}"


def test_todos_fm_consistentes():
    rng = np.random.RandomState(0)
    for _ in range(100):
        raw = rng.rand(N) * 2.5
        total = int(round(raw.sum()))
        ch2 = int(round(sum(raw[i] for i in IDX_CH2)))
        if total < ch2:
            total = ch2
        cands = generate_candidates_uncertainty(raw, total, ch2, 2, 2, tau=1.0, K_max=6)
        for c in cands:
            assert _fm_ok(c, total, ch2), f"candidato no FM-consistente: {c}"
            assert (c >= 0).all()


def test_no_puebla_clases_prohibidas():
    raw = _raw({1: 1.1, 9: 0.9})
    cands = generate_candidates_uncertainty(raw, 2, 2, 0, 0, tau=2.0, K_max=6)
    for c in cands:
        assert all(c[i] == 0 for i in [8, 9, 10, 11, 16]), f"poblo clase -N con n=0: {c}"


def test_monotonia_en_tau():
    # el set de candidatos con tau grande contiene al de tau chico (antes del corte K_max)
    raw = _raw({1: 1.55, 9: 0.55, 2: 1.4, 6: 0.6})
    chico = generate_candidates_uncertainty(raw, 4, 2, 1, 1, tau=0.2, K_max=20)
    grande = generate_candidates_uncertainty(raw, 4, 2, 1, 1, tau=1.5, K_max=20)
    s_chico = {c.tobytes() for c in chico}
    s_grande = {c.tobytes() for c in grande}
    assert s_chico <= s_grande, "tau grande deberia contener a tau chico"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n>>> {len(fns)} TESTS OK <<<")


if __name__ == "__main__":
    _run()
