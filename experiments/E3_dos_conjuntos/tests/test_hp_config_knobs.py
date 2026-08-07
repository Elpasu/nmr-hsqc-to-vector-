# coding: ascii
"""Verifica los dos knobs que el estudio de hiperparametros (Exp I) necesita:
fusion_hidden parametrizable en el modelo y seed configurable en train.py.
Ambos son ADITIVOS: el default debe reproducir exactamente el comportamiento
historico, o el checkpoint congelado deja de cargar."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model_e3_settransformer import NMR_SetTransformer
from train import build_model

N_CLASSES, MAX_CH, MAX_13C = 19, 32, 40


def _forward(model, B=2):
    with torch.no_grad():
        return model(torch.randn(B, MAX_CH, 4), torch.ones(B, MAX_CH),
                     torch.randn(B, MAX_13C, 1), torch.ones(B, MAX_13C),
                     torch.randn(B, 8))


def test_default_fusion_shapes_unchanged():
    """El default (128, 64) tiene que dar EXACTAMENTE las dimensiones
    historicas, o el state_dict del checkpoint congelado no carga."""
    m = NMR_SetTransformer(num_classes=N_CLASSES)
    assert m.fc_fusion1.out_features == 128, m.fc_fusion1.out_features
    assert m.fc_fusion2.in_features == 128, m.fc_fusion2.in_features
    assert m.fc_fusion2.out_features == 64, m.fc_fusion2.out_features
    assert m.fc_out.in_features == 64, m.fc_out.in_features
    print("[OK] default fusion_hidden = (128, 64), dimensiones historicas")


def test_default_param_count_unchanged():
    """70,163 parametros: el numero exacto que RESULTS.md atribuye al
    checkpoint congelado. Si esto cambia, el cambio NO fue aditivo."""
    n = sum(p.numel() for p in NMR_SetTransformer(num_classes=N_CLASSES).parameters())
    assert n == 70163, n
    print(f"[OK] parametros con defaults = {n:,} (identico al historico)")


def test_custom_fusion_hidden_applies():
    m = NMR_SetTransformer(num_classes=N_CLASSES, fusion_hidden=(256, 128))
    assert m.fc_fusion1.out_features == 256
    assert m.fc_fusion2.in_features == 256 and m.fc_fusion2.out_features == 128
    assert m.fc_out.in_features == 128
    assert _forward(m).shape == (2, N_CLASSES)
    print("[OK] fusion_hidden=(256, 128) aplicado y forward OK")


def test_small_fusion_hidden_applies():
    m = NMR_SetTransformer(num_classes=N_CLASSES, fusion_hidden=(64, 32))
    assert m.fc_fusion1.out_features == 64 and m.fc_fusion2.out_features == 32
    assert _forward(m).shape == (2, N_CLASSES)
    print("[OK] fusion_hidden=(64, 32) aplicado y forward OK")


def _cfg(model_extra=None):
    m = {"arch": "settransformer", "d_model": 64, "n_heads": 4,
         "n_layers": 2, "n_seeds": 1}
    if model_extra:
        m.update(model_extra)
    return {"model": m}


def test_build_model_default_fusion():
    m = build_model(_cfg(), num_classes=N_CLASSES)
    assert m.fc_fusion1.out_features == 128 and m.fc_fusion2.out_features == 64
    print("[OK] build_model sin fusion_hidden -> default historico")


def test_build_model_reads_fusion_hidden_from_config():
    """Una lista de YAML (no una tupla) tiene que funcionar: yaml.safe_load
    devuelve listas, nunca tuplas."""
    m = build_model(_cfg({"fusion_hidden": [256, 128]}), num_classes=N_CLASSES)
    assert m.fc_fusion1.out_features == 256 and m.fc_fusion2.out_features == 128
    print("[OK] build_model lee fusion_hidden (lista de YAML) del config")


def test_train_reads_seed_from_config():
    """train.py no debe tener el 42 hardcodeado: el estudio de ruido (Exp I)
    necesita correr la MISMA config con seeds distintas."""
    src = (Path(__file__).resolve().parent.parent / "train.py").read_text(encoding="utf-8")
    assert "set_seed(42)" not in src, "train.py todavia hardcodea set_seed(42)"
    assert "'seed'" in src or '"seed"' in src, "train.py no lee 'seed' del config"
    print("[OK] train.py toma la seed del config")


if __name__ == "__main__":
    test_default_fusion_shapes_unchanged()
    test_default_param_count_unchanged()
    test_custom_fusion_hidden_applies()
    test_small_fusion_hidden_applies()
    test_build_model_default_fusion()
    test_build_model_reads_fusion_hidden_from_config()
    test_train_reads_seed_from_config()
    print("\n>>> HP CONFIG KNOBS OK <<<")
