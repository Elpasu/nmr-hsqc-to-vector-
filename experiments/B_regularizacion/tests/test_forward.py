# coding: ascii
"""
Smoke test OFFLINE de Exp B (V10 + dropout + weight_decay) - rule 5 de
CLAUDE.md.

NO depende de checkpoint ni h5 real. Valida:
  (1) el forward de model_v11b con HSQC 2 canales -> (B, 19), mismas
      dimensiones que V10 (dropout no cambia shapes),
  (2) dropout esta activo en train() (dos forwards con el mismo input dan
      resultados distintos) e inactivo en eval() (dos forwards dan el
      mismo resultado) -- catch para el bug clasico de olvidarse
      model.eval() antes de evaluar/predecir.

Correr en CPU (login node) antes de cualquier sbatch:
    python tests/test_forward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model_v11b import NMR_Net

N_CLASSES = 19


def test_forward_shape():
    model = NMR_Net(num_classes=N_CLASSES, dropout=0.25)
    model.eval()
    B = 4
    hsqc = torch.randn(B, 2, 256, 256)
    proj = torch.randn(B, 512)
    cond = torch.randn(B, 8)
    with torch.no_grad():
        out = model(hsqc, proj, cond)
    assert out.shape == (B, N_CLASSES), out.shape
    print(f"[OK] forward -> {tuple(out.shape)} (esperado ({B}, {N_CLASSES}), igual que V10)")


def test_dropout_active_in_train_mode():
    torch.manual_seed(0)
    model = NMR_Net(num_classes=N_CLASSES, dropout=0.5)
    model.train()
    hsqc = torch.randn(2, 2, 256, 256)
    proj = torch.randn(2, 512)
    cond = torch.randn(2, 8)
    out1 = model(hsqc, proj, cond)
    out2 = model(hsqc, proj, cond)
    assert not torch.allclose(out1, out2), "dropout deberia dar resultados distintos en train()"
    print("[OK] dropout activo en train(): dos forwards con el mismo input difieren")


def test_dropout_inactive_in_eval_mode():
    torch.manual_seed(0)
    model = NMR_Net(num_classes=N_CLASSES, dropout=0.5)
    model.eval()
    hsqc = torch.randn(2, 2, 256, 256)
    proj = torch.randn(2, 512)
    cond = torch.randn(2, 8)
    with torch.no_grad():
        out1 = model(hsqc, proj, cond)
        out2 = model(hsqc, proj, cond)
    assert torch.allclose(out1, out2), "dropout NO deberia afectar en eval()"
    print("[OK] dropout inactivo en eval(): dos forwards con el mismo input coinciden")


if __name__ == "__main__":
    test_forward_shape()
    test_dropout_active_in_train_mode()
    test_dropout_inactive_in_eval_mode()
    print("\n>>> SMOKE EXP B OK - listo para sbatch run_train.sh <<<")
