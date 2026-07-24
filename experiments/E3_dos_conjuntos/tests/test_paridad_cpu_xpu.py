# coding: ascii
"""Paridad numerica CPU <-> acelerador (Fase 2 de la migracion a Intel XPU,
ver docs/MIGRACION_XPU_Clementina_XXI.md).

Corre el MISMO modelo con los MISMOS pesos y las MISMAS entradas en CPU y en el
acelerador (XPU o CUDA), y compara. Foco en el patron de riesgo identificado en
el analisis: la atencion enmascarada de MAB.forward()
--- masked_fill(-inf) + softmax + nan_to_num --- cuando una molecula no tiene
picos y una fila entera del softmax queda en -inf.

Se corre en un nodo con acelerador:
    python tests/test_paridad_cpu_xpu.py

En una maquina sin acelerador reporta SKIP y sale con 0: sirve igual como
smoke test de que el script no esta roto.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from device_utils import pick_device, seed_everything
from model_e3_settransformer import NMR_SetTransformer

N_CLASSES, MAX_CH, MAX_13C = 19, 32, 40

# FP32 en hardware distinto no da bit-exactitud: el orden de reduccion de las
# matmuls cambia. Estas tolerancias detectan un operador mal implementado sin
# saltar por ruido de punto flotante.
ATOL, RTOL = 2e-5, 1e-4


def _make_batch(B=8, vacias=(0,)):
    """Batch determinista. `vacias` = indices de moleculas sin ningun pico
    (el caso critico: fila entera enmascarada -> softmax de puros -inf)."""
    g = torch.Generator().manual_seed(1234)
    peaks_ch = torch.randn(B, MAX_CH, 4, generator=g)
    peaks_13c = torch.randn(B, MAX_13C, 1, generator=g)
    cond = torch.randn(B, 8, generator=g)
    mask_ch = torch.ones(B, MAX_CH)
    mask_13c = torch.ones(B, MAX_13C)
    # padding parcial realista: la molecula i usa i+3 crosspeaks
    for i in range(B):
        mask_ch[i, min(i + 3, MAX_CH):] = 0.0
        mask_13c[i, min(i + 5, MAX_13C):] = 0.0
    for i in vacias:
        mask_ch[i] = 0.0
        mask_13c[i] = 0.0
    return peaks_ch, mask_ch, peaks_13c, mask_13c, cond


def _forward_en(device, batch, model):
    m = model.to(device).eval()
    args = [t.to(device) for t in batch]
    with torch.no_grad():
        out = m(*args)
    return out.cpu()


def _reporte(nombre, a, b):
    diff = (a - b).abs()
    print(f"    {nombre}: max|dif|={diff.max().item():.3e}  "
          f"media={diff.mean().item():.3e}")
    return diff.max().item()


def test_autoverificacion_el_test_detecta_diferencias():
    """El test de paridad tiene que ser capaz de FALLAR. Si comparamos contra
    un modelo con otros pesos, allclose debe dar False; si no, la comparacion
    no estaria midiendo nada."""
    seed_everything(42)
    m1 = NMR_SetTransformer(num_classes=N_CLASSES).eval()
    seed_everything(7)
    m2 = NMR_SetTransformer(num_classes=N_CLASSES).eval()
    batch = _make_batch()
    cpu = torch.device("cpu")
    o1 = _forward_en(cpu, batch, m1)
    o2 = _forward_en(cpu, batch, m2)
    assert not torch.allclose(o1, o2, atol=ATOL, rtol=RTOL), (
        "la comparacion no detecta pesos distintos: el test no mide nada")
    print("[OK] autoverificacion: el test detecta diferencias reales")


def test_paridad_forward(device):
    seed_everything(42)
    model = NMR_SetTransformer(num_classes=N_CLASSES)
    batch = _make_batch(vacias=())          # sin moleculas vacias
    out_cpu = _forward_en(torch.device("cpu"), batch, model)
    out_dev = _forward_en(device, batch, model)
    _reporte("forward", out_cpu, out_dev)
    assert torch.isfinite(out_dev).all(), "salida no finita en el acelerador"
    assert torch.allclose(out_cpu, out_dev, atol=ATOL, rtol=RTOL), \
        (out_cpu - out_dev).abs().max()
    print(f"[OK] paridad forward CPU <-> {device.type.upper()}")


def test_paridad_molecula_vacia(device):
    """El caso de riesgo: masked_fill(-inf) + softmax + nan_to_num."""
    seed_everything(42)
    model = NMR_SetTransformer(num_classes=N_CLASSES)
    batch = _make_batch(vacias=(0, 3))
    out_cpu = _forward_en(torch.device("cpu"), batch, model)
    out_dev = _forward_en(device, batch, model)
    _reporte("molecula vacia", out_cpu, out_dev)
    assert torch.isfinite(out_cpu).all(), "NaN/Inf ya en CPU"
    assert torch.isfinite(out_dev).all(), \
        f"NaN/Inf en {device.type}: nan_to_num no se comporta igual"
    assert torch.allclose(out_cpu, out_dev, atol=ATOL, rtol=RTOL), \
        (out_cpu - out_dev).abs().max()
    print(f"[OK] paridad con moleculas sin picos (atencion 100% enmascarada)")


def test_paridad_gradientes(device):
    """El entrenamiento depende del backward, no solo del forward."""
    batch = _make_batch(vacias=(0,))
    grads = {}
    for dev in (torch.device("cpu"), device):
        seed_everything(42)
        model = NMR_SetTransformer(num_classes=N_CLASSES).to(dev).train()
        args = [t.to(dev) for t in batch]
        out = model(*args)
        loss = out.pow(2).mean()
        loss.backward()
        grads[dev.type] = torch.cat(
            [p.grad.detach().reshape(-1).cpu() for p in model.parameters()
             if p.grad is not None])
    g_cpu, g_dev = grads["cpu"], grads[device.type]
    _reporte("gradientes", g_cpu, g_dev)
    assert torch.isfinite(g_dev).all(), "gradientes no finitos en el acelerador"
    assert torch.allclose(g_cpu, g_dev, atol=ATOL, rtol=RTOL), \
        (g_cpu - g_dev).abs().max()
    print(f"[OK] paridad de gradientes CPU <-> {device.type.upper()}")


def test_determinismo_en_el_acelerador(device):
    seed_everything(42)
    model = NMR_SetTransformer(num_classes=N_CLASSES)
    batch = _make_batch(vacias=(0,))
    a = _forward_en(device, batch, model)
    b = _forward_en(device, batch, model)
    assert torch.equal(a, b), "dos forwards identicos dan distinto en el acelerador"
    print(f"[OK] {device.type.upper()} es determinista entre corridas")


if __name__ == "__main__":
    test_autoverificacion_el_test_detecta_diferencias()

    device = pick_device("auto")
    if device.type == "cpu":
        print("\n[SKIP] No hay acelerador visible: la paridad necesita XPU o CUDA.")
        print("       Corre esto en un nodo gpunode:")
        print("         srun -p gpunode --gres=gpu:intel_xt1550:1 --pty bash")
        sys.exit(0)

    print(f"\n--- Comparando CPU contra {device.type.upper()} "
          f"(atol={ATOL}, rtol={RTOL}) ---")
    test_paridad_forward(device)
    test_paridad_molecula_vacia(device)
    test_paridad_gradientes(device)
    test_determinismo_en_el_acelerador(device)
    print(f"\n>>> PARIDAD CPU <-> {device.type.upper()} OK <<<")
