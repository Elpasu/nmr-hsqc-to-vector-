# coding: ascii
"""
Smoke test OFFLINE de Exp C (V10 + GAP, sin regularizacion) - rule 5 de
CLAUDE.md.

NO depende de checkpoint ni h5 real. Valida:
  (1) el forward de model_c con HSQC 2 canales -> (B, 19), misma salida
      que V10 (el cambio es interno: GAP en vez de flatten),
  (2) que el conteo de parametros bajo drasticamente respecto a V10
      (~8.6M -> se espera <500k; calculado a mano: ~223k) -- la prueba
      concreta de que GAP esta realmente conectado y no es un no-op.

Correr en CPU (login node) antes de cualquier sbatch:
    python tests/test_forward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model_c import NMR_Net

N_CLASSES = 19
V10_PARAM_COUNT = 8_603_299


def test_forward_shape():
    model = NMR_Net(num_classes=N_CLASSES)
    model.eval()
    B = 4
    hsqc = torch.randn(B, 2, 256, 256)
    proj = torch.randn(B, 512)
    cond = torch.randn(B, 8)
    with torch.no_grad():
        out = model(hsqc, proj, cond)
    assert out.shape == (B, N_CLASSES), out.shape
    print(f"[OK] forward -> {tuple(out.shape)} (esperado ({B}, {N_CLASSES}), igual que V10)")


def test_param_count_dropped():
    model = NMR_Net(num_classes=N_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 500_000, (
        f"Parametros = {n_params}, esperado <500000 (GAP no parece estar conectado, "
        f"revisar que forward() use self.gap y no self.flat_dim)"
    )
    print(f"[OK] parametros = {n_params:,} (<500,000; V10 original ~{V10_PARAM_COUNT:,}, "
          f"reduccion de ~{V10_PARAM_COUNT / n_params:.0f}x)")


if __name__ == "__main__":
    test_forward_shape()
    test_param_count_dropped()
    print("\n>>> SMOKE EXP C OK - listo para sbatch run_train.sh <<<")
