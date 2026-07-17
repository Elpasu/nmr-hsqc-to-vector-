# coding: ascii
"""
Smoke test OFFLINE del evaluador Exp D - rule 5 de CLAUDE.md.

NO depende del checkpoint ni del h5 real: usa tensores random y arrays
sinteticos. Valida:
  (1) el forward de model_v10 con HSQC de 2 canales -> (B, 19),
  (2) la prediccion CRUDA (floor + clip a >=0),
  (3) el ORACULO de doble restriccion,
  (4) el calculo de EMA / MAE,
  (5) que Subset(dataset, indices) respeta el orden de
      val_indices_frozen.npy -- el mecanismo que usa evaluate.py.

Correr en CPU (login node) antes de cualquier sbatch:
    python tests/test_forward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from model_v10 import NMR_Net
from evaluate import (
    crude_predict,
    ajustar_conteo_doble_exacto,
    compute_ema,
    compute_mae,
    ema_entorno,
)

N_CLASSES = 19
IDX_CH2 = [1, 5, 9, 12]   # CH2, CH2-O, CH2-N, =CH2


def test_forward_2ch():
    model = NMR_Net(num_classes=N_CLASSES)
    model.eval()
    B = 4
    hsqc = torch.randn(B, 2, 256, 256)
    proj = torch.randn(B, 512)
    cond = torch.randn(B, 8)
    with torch.no_grad():
        out = model(hsqc, proj, cond)
    assert out.shape == (B, N_CLASSES), out.shape
    print(f"[OK] forward 2ch -> {tuple(out.shape)} (esperado ({B}, {N_CLASSES}))")


def test_crude():
    pred_raw = np.array([0.9, -0.3, 2.7, 1.1] + [0.0] * 15, dtype=np.float32)
    p = crude_predict(pred_raw)
    assert p.dtype.kind == "i"
    assert (p >= 0).all(), "clip a >=0 fallo"
    assert p[0] == 0 and p[1] == 0 and p[2] == 2 and p[3] == 1, p[:4]
    print("[OK] crudo: floor + clip a >=0")


def test_oraculo_doble_restriccion():
    rng = np.random.default_rng(0)
    for _ in range(500):
        target = rng.integers(0, 3, size=N_CLASSES)
        total = int(target.sum())
        ch2 = int(sum(target[i] for i in IDX_CH2))
        pred_raw = np.clip(target + rng.uniform(-0.4, 0.4, size=N_CLASSES), 0, None)
        pred_raw = pred_raw.astype(np.float32)
        p = ajustar_conteo_doble_exacto(pred_raw, total, ch2)
        assert (p >= 0).all()
        assert int(p.sum()) == total, (int(p.sum()), total)
        assert int(sum(p[i] for i in IDX_CH2)) == ch2, (sum(p[i] for i in IDX_CH2), ch2)
        assert np.array_equal(p, target), (p, target)
    print("[OK] oraculo: sum==total_senales, sum(CH2)==total_CH2 y recupera el "
          "vector (500 casos)")


def test_metrics():
    t = np.array([[1, 0, 2], [3, 1, 0]])
    assert compute_ema(t.copy(), t) == 100.0
    p = t.copy()
    p[0, 0] += 1
    assert compute_ema(p, t) == 50.0
    mae = compute_mae(p, t)
    assert abs(mae[0] - 0.5) < 1e-9, mae
    assert ema_entorno(p, t, [1, 2]) == 100.0
    assert ema_entorno(p, t, [0]) == 50.0
    print("[OK] EMA / MAE / EMA por entorno")


def test_subset_from_frozen_indices():
    """Valida que Subset(dataset, indices) selecciona exactamente esos
    indices, en el mismo orden -- el mecanismo que usa evaluate.py para
    leer val_indices_frozen.npy."""

    class DummyDataset(Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return idx

    ds = DummyDataset()
    frozen = np.array([7, 2, 5])
    subset = Subset(ds, frozen.tolist())
    assert len(subset) == 3
    assert [subset[i] for i in range(3)] == [7, 2, 5]
    print("[OK] Subset respeta el orden de val_indices_frozen.npy")


if __name__ == "__main__":
    test_forward_2ch()
    test_crude()
    test_oraculo_doble_restriccion()
    test_metrics()
    test_subset_from_frozen_indices()
    print("\n>>> SMOKE EXP D OK - listo para sbatch run_eval.sh <<<")
