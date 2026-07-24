# coding: ascii
"""Test de la metrica cobertura@K -- numpy puro, corre local sin torch.
Correr:  python tests/test_coverage.py   (desde experiments/G_multivector)"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coverage import coverage_curve

N = 19


def _v(kv):
    v = np.zeros(N, dtype=int)
    for i, val in kv.items():
        v[int(i)] = val
    return v


def test_cobertura_monotona_y_top1_reproduce_v2():
    # Molecula 1: el crudo redondea exacto al verdadero -> cubierta en K=1.
    yt1 = _v({0: 2, 1: 2}); raw1 = np.array(yt1, float) + 0.05
    # Molecula 2: confusion CH2 vs CH2-N; verdadero CH2=2, crudo reparte 1.4/0.6.
    yt2 = _v({1: 2}); raw2 = _v({1: 1}).astype(float); raw2[1] = 1.4; raw2[9] = 0.6
    yt = np.vstack([yt1, yt2]); raw = np.vstack([raw1, raw2])
    n = np.array([0, 1]); o = np.array([0, 0])
    res = coverage_curve(yt, raw, n, o, Ks=[1, 2, 3])
    assert res[1]["coverage"] <= res[2]["coverage"] <= res[3]["coverage"]
    assert res[3]["coverage"] == 100.0, "con K=3 deben estar las 2"
    assert res[1]["k_max"] == 1


def test_k_mean_y_k_max_reportados():
    yt = _v({1: 2})[None, :]
    raw = np.zeros((1, N)); raw[0, 1] = 1.4; raw[0, 9] = 0.6
    res = coverage_curve(yt, raw, np.array([1]), np.array([1]), Ks=[3])
    # Con esa entrada se generan 3 candidatos (anchor + movimientos hasta K=3),
    # ambos emitidos -> k_mean y k_max son 3.0 y 3 respectivamente.
    assert res[3]["k_mean"] == 3.0 and res[3]["k_max"] == 3


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n>>> {len(fns)} TESTS OK <<<")


if __name__ == "__main__":
    _run()
