# coding: ascii
"""device_utils.py -- seleccion de dispositivo agnostica de plataforma.

Fase 1 de la migracion a Intel XPU (docs/MIGRACION_XPU_Clementina_XXI.md).
El proyecto entreno historicamente en CUDA (A10, login-1) y ahora tambien debe
correr en XPU (Intel GPU Max 1550, Clementina XXI). Los cambios son ADITIVOS
(decision D8): CUDA y CPU siguen funcionando exactamente igual que antes.

Orden de preferencia en modo 'auto' (decision D5): cuda -> xpu -> cpu.
"""
import random

import numpy as np
import torch

VALID_DEVICES = ("auto", "cuda", "xpu", "cpu")


def cuda_available():
    """True si hay un backend CUDA usable."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def xpu_available():
    """True si hay un backend XPU usable.

    Ojo: `torch.xpu` existe como atributo incluso en builds CPU/CUDA sin
    soporte XPU, asi que hasattr() NO alcanza como probe -- hay que llamar
    is_available() y atajar cualquier error del runtime Level-Zero.
    """
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


def pick_device(prefer="auto", has_cuda=None, has_xpu=None):
    """Devuelve el torch.device a usar.

    prefer: valor de `system.device` del config ("auto"|"cuda"|"xpu"|"cpu").
    has_cuda / has_xpu: overrides de los probes, para tests (ninguna maquina
        tiene CUDA y XPU a la vez).

    Pedir explicitamente un backend que no esta disponible levanta RuntimeError
    en vez de caer a CPU en silencio: entrenar dias en CPU creyendo que estas
    en GPU es el tipo de fallo silencioso que este proyecto ya pago caro.
    """
    if prefer is None:
        prefer = "auto"
    prefer = str(prefer).strip().lower()
    if prefer not in VALID_DEVICES:
        raise ValueError(
            f"system.device invalido: {prefer!r}. Validos: {VALID_DEVICES}")

    has_cuda = cuda_available() if has_cuda is None else bool(has_cuda)
    has_xpu = xpu_available() if has_xpu is None else bool(has_xpu)

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "cuda":
        if not has_cuda:
            raise RuntimeError(
                "El config pide device='cuda' pero torch.cuda.is_available() es "
                "False. Usa device='auto' si queres fallback a CPU.")
        return torch.device("cuda")

    if prefer == "xpu":
        if not has_xpu:
            raise RuntimeError(
                "El config pide device='xpu' pero torch.xpu.is_available() es "
                "False. Revisa que el job corra en un nodo gpunode, que hayas "
                "hecho `unset ZE_AFFINITY_MASK` y que el env sea el de XPU. "
                "Usa device='auto' si queres fallback a CPU.")
        return torch.device("xpu")

    # auto (D5): cuda -> xpu -> cpu
    if has_cuda:
        return torch.device("cuda")
    if has_xpu:
        return torch.device("xpu")
    return torch.device("cpu")


def wants_pin_memory(device):
    """pin_memory solo tiene sentido copiando a un acelerador."""
    return device.type in ("cuda", "xpu")


def synchronize(device):
    """Barrera del dispositivo. No-op en CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def seed_everything(seed=42):
    """Fija las semillas de python/numpy/torch y del acelerador presente.

    Reemplaza a set_seed() de train.py; identico en CUDA, mas la rama XPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if xpu_available():
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)
    # Inocuo (no-op) fuera de CUDA; se mantiene para no cambiar el
    # comportamiento historico en A10.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
