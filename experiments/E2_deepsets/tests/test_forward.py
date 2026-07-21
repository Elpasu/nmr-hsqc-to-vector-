# coding: ascii
"""
Smoke test OFFLINE de Exp E Fase 2 (DeepSets sobre picos) - rule 5 de
CLAUDE.md.

NO depende de checkpoint ni datos reales. Valida:
  (1) el forward de model_e2 con picos + mascara + condicionante ->
      (B, 19), mismo contrato de salida que V10/Exp C,
  (2) que una molecula SIN picos validos (mascara toda en cero) no rompe
      el forward (division por cero en la agregacion) y da salida finita,
  (3) el conteo de parametros: se espera ~23,315 (mucho menos que los
      ~223k de Exp C o los ~8.6M de V10 -- confirma que el modelo es
      chico a proposito, no por error).

Correr en CPU (login node) antes de cualquier sbatch:
    python tests/test_forward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model_e2 import NMR_Net

N_CLASSES = 19
MAX_PEAKS = 32
EXPECTED_PARAMS = 23_315


def test_forward_shape():
    model = NMR_Net(num_classes=N_CLASSES)
    model.eval()
    B = 4
    peaks = torch.randn(B, MAX_PEAKS, 4)
    mask = torch.ones(B, MAX_PEAKS)
    cond = torch.randn(B, 8)
    with torch.no_grad():
        out = model(peaks, mask, cond)
    assert out.shape == (B, N_CLASSES), out.shape
    print(f"[OK] forward -> {tuple(out.shape)} (esperado ({B}, {N_CLASSES}))")


def test_forward_with_empty_molecule_no_nan():
    # Una molecula sin picos validos (mascara toda en cero) no debe romper
    # la agregacion (division por cero) ni dar NaN/Inf en la salida.
    model = NMR_Net(num_classes=N_CLASSES)
    model.eval()
    B = 3
    peaks = torch.randn(B, MAX_PEAKS, 4)
    mask = torch.ones(B, MAX_PEAKS)
    mask[0] = 0.0   # molecula 0: sin picos validos
    cond = torch.randn(B, 8)
    with torch.no_grad():
        out = model(peaks, mask, cond)
    assert torch.isfinite(out).all(), out
    print(f"[OK] forward con molecula sin picos -> sin NaN/Inf: {out[0]}")


def test_param_count_is_small_by_design():
    model = NMR_Net(num_classes=N_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAMS, (
        f"Parametros = {n_params}, esperado {EXPECTED_PARAMS}. Si cambio, "
        f"revisar que las dimensiones de model_e2.py no se hayan tocado "
        f"por error (el tamano chico es una decision deliberada, ver RATIONALE.md)."
    )
    print(f"[OK] parametros = {n_params:,} (esperado {EXPECTED_PARAMS:,}; "
          f"Exp C ~223,000, V10 original ~8,603,299)")


if __name__ == "__main__":
    test_forward_shape()
    test_forward_with_empty_molecule_no_nan()
    test_param_count_is_small_by_design()
    print("\n>>> SMOKE EXP E FASE 2 OK - listo para sbatch run_train.sh <<<")
