# coding: ascii
"""Tests de la seleccion de dispositivo (Fase 1 de la migracion a Intel XPU,
ver docs/MIGRACION_XPU_Clementina_XXI.md).

La logica de pick_device() depende de que backends hay en la maquina, y ninguna
maquina tiene CUDA y XPU a la vez: por eso los probes de disponibilidad se
inyectan (has_cuda/has_xpu) y estos tests cubren las 3 plataformas sin
necesitar el hardware. Los probes reales se testean aparte (ver
test_probes_reales_no_explotan).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from device_utils import pick_device, wants_pin_memory, seed_everything, synchronize


# --- pick_device: modo "auto" (D5: cuda -> xpu -> cpu) ---------------------

def test_auto_prefiere_cuda():
    dev = pick_device("auto", has_cuda=True, has_xpu=True)
    assert dev.type == "cuda", dev
    print("[OK] auto + cuda disponible -> cuda (preserva comportamiento actual)")


def test_auto_cae_a_xpu_sin_cuda():
    dev = pick_device("auto", has_cuda=False, has_xpu=True)
    assert dev.type == "xpu", dev
    print("[OK] auto + sin cuda + xpu -> xpu")


def test_auto_cae_a_cpu_sin_nada():
    dev = pick_device("auto", has_cuda=False, has_xpu=False)
    assert dev.type == "cpu", dev
    print("[OK] auto sin aceleradores -> cpu")


# --- pick_device: eleccion explicita desde el config ----------------------

def test_explicito_cpu_ignora_aceleradores():
    # Necesario para el test de paridad CPU<->XPU de la Fase 2.
    dev = pick_device("cpu", has_cuda=True, has_xpu=True)
    assert dev.type == "cpu", dev
    print("[OK] device='cpu' fuerza cpu aunque haya acelerador")


def test_explicito_xpu_disponible():
    dev = pick_device("xpu", has_cuda=False, has_xpu=True)
    assert dev.type == "xpu", dev
    print("[OK] device='xpu' con xpu disponible -> xpu")


def test_explicito_no_disponible_es_error():
    # Regla dura del proyecto: nada de fallos silenciosos. Pedir 'xpu' y
    # entrenar 3 dias en CPU sin enterarse es exactamente el bug caro.
    for pedido, hc, hx in [("xpu", False, False), ("cuda", False, True)]:
        try:
            pick_device(pedido, has_cuda=hc, has_xpu=hx)
        except RuntimeError as e:
            assert pedido in str(e), str(e)
        else:
            raise AssertionError(f"pick_device('{pedido}') debio fallar, no caer a cpu")
    print("[OK] pedir un backend ausente levanta RuntimeError (no cae en silencio)")


def test_valor_desconocido_es_error():
    try:
        pick_device("gpu", has_cuda=False, has_xpu=False)
    except ValueError:
        print("[OK] device desconocido -> ValueError")
    else:
        raise AssertionError("un device invalido debio dar ValueError")


# --- guardas derivadas ----------------------------------------------------

def test_pin_memory_solo_en_acelerador():
    assert wants_pin_memory(torch.device("cuda")) is True
    assert wants_pin_memory(torch.device("xpu")) is True
    assert wants_pin_memory(torch.device("cpu")) is False
    print("[OK] pin_memory: True en cuda/xpu, False en cpu")


def test_synchronize_en_cpu_es_noop():
    synchronize(torch.device("cpu"))   # no debe explotar
    print("[OK] synchronize(cpu) es no-op")


def test_seed_everything_reproducible():
    seed_everything(42)
    a = torch.randn(8)
    seed_everything(42)
    b = torch.randn(8)
    assert torch.equal(a, b), (a, b)
    print("[OK] seed_everything(42) reproduce la misma secuencia")


# --- probes reales --------------------------------------------------------

def test_probes_reales_no_explotan():
    # En torch sin build XPU, torch.xpu puede no existir: el probe debe
    # devolver False, no romper.
    dev = pick_device("auto")
    assert dev.type in ("cuda", "xpu", "cpu"), dev
    print(f"[OK] pick_device('auto') real -> {dev.type}")


if __name__ == "__main__":
    test_auto_prefiere_cuda()
    test_auto_cae_a_xpu_sin_cuda()
    test_auto_cae_a_cpu_sin_nada()
    test_explicito_cpu_ignora_aceleradores()
    test_explicito_xpu_disponible()
    test_explicito_no_disponible_es_error()
    test_valor_desconocido_es_error()
    test_pin_memory_solo_en_acelerador()
    test_synchronize_en_cpu_es_noop()
    test_seed_everything_reproducible()
    test_probes_reales_no_explotan()
    print("\n>>> DEVICE UTILS OK <<<")
