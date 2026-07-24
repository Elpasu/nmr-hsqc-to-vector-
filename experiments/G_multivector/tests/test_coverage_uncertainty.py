# coding: ascii
"""Test de la metrica de barrido de tau (Fase 1b) -- numpy puro, corre local.
Correr:  python tests/test_coverage_uncertainty.py   (desde experiments/G_multivector)"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coverage import coverage_uncertainty

N = 19


def _v(kv):
    v = np.zeros(N, dtype=int)
    for i, val in kv.items():
        v[int(i)] = val
    return v


def test_molecula_segura_kmean_1():
    # verdadero = crudo redondeado exacto -> 1 candidato, cubierto.
    yt = _v({0: 1, 1: 2})
    raw = np.zeros((1, N)); raw[0, 0] = 1.02; raw[0, 1] = 1.98
    res = coverage_uncertainty(yt[None, :], raw, np.array([0]), np.array([0]),
                               tau=0.5, K_max=6)
    assert res["coverage"] == 100.0
    assert res["k_mean"] == 1.0, f"molecula segura deberia dar k_mean=1, dio {res['k_mean']}"


def test_duda_cubierta_con_tau():
    # verdadero CH2=1, CH2-N=1; crudo 1.55/0.55; con tau razonable se cubre.
    yt = _v({1: 1, 9: 1})
    raw = np.zeros((1, N)); raw[0, 1] = 1.55; raw[0, 9] = 0.55
    res = coverage_uncertainty(yt[None, :], raw, np.array([1]), np.array([0]),
                               tau=0.5, K_max=6)
    assert res["coverage"] == 100.0
    assert res["k_mean"] >= 2.0


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n>>> {len(fns)} TESTS OK <<<")


if __name__ == "__main__":
    _run()
