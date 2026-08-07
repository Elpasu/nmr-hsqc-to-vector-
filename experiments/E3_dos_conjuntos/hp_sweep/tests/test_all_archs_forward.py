# coding: ascii
"""Regla dura 5, aplicada a las 23 corridas de una sola vez: construye el modelo
de CADA config del sweep y le hace un forward en CPU con shapes reales.

Un mismatch de dimensiones se descubre en segundos en la PC local, no despues de
40 minutos de cola de GPU. Corre sin datos: solo necesita torch."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_E3 = _HERE.parent.parent                      # experiments/E3_dos_conjuntos
sys.path.insert(0, str(_E3))
sys.path.insert(0, str(_HERE.parent))          # hp_sweep (para make_configs)

import torch
import yaml

from train import build_model

CONFIG_DIR = _HERE.parent / "configs"
N_CLASSES, MAX_CH, MAX_13C, B = 19, 32, 40, 3
N_EXPECTED = 23


def _batch():
    """Batch sintetico con mascaras MIXTAS: una molecula llena, una parcial y
    una totalmente enmascarada (el caso que hace NaN si el softmax no esta
    protegido)."""
    mask_ch = torch.ones(B, MAX_CH)
    mask_13c = torch.ones(B, MAX_13C)
    mask_ch[1, 10:] = 0.0
    mask_13c[1, 12:] = 0.0
    mask_ch[2] = 0.0
    mask_13c[2] = 0.0
    return (torch.randn(B, MAX_CH, 4), mask_ch,
            torch.randn(B, MAX_13C, 1), mask_13c,
            torch.randn(B, 8))


def test_all_configs_present():
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(files) == N_EXPECTED, f"{len(files)} configs en {CONFIG_DIR}"
    print(f"[OK] {N_EXPECTED} configs encontrados")


def test_forward_every_config():
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    peaks_ch, mask_ch, peaks_13c, mask_13c, cond = _batch()
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg, num_classes=N_CLASSES).eval()
        with torch.no_grad():
            out = model(peaks_ch, mask_ch, peaks_13c, mask_13c, cond)
        assert out.shape == (B, N_CLASSES), f"{path.name}: shape {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{path.name}: NaN/Inf en la salida"
        n = sum(p.numel() for p in model.parameters())
        print(f"  [OK] {path.name:<24} params={n:>9,}  out={tuple(out.shape)}")
    print(f"[OK] forward correcto en los {len(files)} configs")


def test_param_counts_span_a_range():
    """Sanity del diseno: si todas las variantes tuvieran el mismo tamano, el
    estudio no estaria variando la capacidad realmente."""
    counts = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        counts.append(sum(p.numel() for p in build_model(cfg, N_CLASSES).parameters()))
    assert max(counts) > 3 * min(counts), (min(counts), max(counts))
    print(f"[OK] rango de parametros: {min(counts):,} - {max(counts):,}")


if __name__ == "__main__":
    test_all_configs_present()
    test_forward_every_config()
    test_param_counts_span_a_range()
    print("\n>>> SMOKE 23 ARQUITECTURAS OK <<<")
