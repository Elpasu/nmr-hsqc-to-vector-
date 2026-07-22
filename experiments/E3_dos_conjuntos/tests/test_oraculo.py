# coding: ascii
"""El oraculo de doble restriccion es identico al de E2/Exp C. Este test fija
su comportamiento: con la restriccion de suma total y de CH2, la prediccion
asistida debe respetar ambos totales."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from evaluate import ajustar_conteo_doble_exacto, crude_predict, IDX_CH2


def test_asistida_respeta_totales():
    pred_raw = np.array([0.6, 1.4, 0.2, 0.1, 0, 0, 0, 0, 0, 0.9, 0, 0,
                         0.3, 2.1, 1.2, 0, 0, 0, 0], dtype=float)
    total_real, ch2_real = 8, 3
    out = ajustar_conteo_doble_exacto(pred_raw, total_real, ch2_real)
    assert out.sum() == total_real, out.sum()
    assert sum(out[i] for i in IDX_CH2) == ch2_real, out
    print(f"[OK] asistida respeta total={total_real} y ch2={ch2_real}: {out}")


def test_crude_es_floor_no_negativo():
    pred_raw = np.array([-0.3, 1.9, 0.4] + [0.0] * 16, dtype=float)
    out = crude_predict(pred_raw)
    assert out[0] == 0 and out[1] == 1 and out[2] == 0, out
    print("[OK] crude = floor con clip a >=0")


if __name__ == "__main__":
    test_asistida_respeta_totales(); test_crude_es_floor_no_negativo()
    print("\n>>> TEST ORACULO E3 OK <<<")
