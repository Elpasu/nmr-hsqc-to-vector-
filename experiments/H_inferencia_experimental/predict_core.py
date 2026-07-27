# coding: utf-8
"""Exp H -- forward local (torch CPU) del E3 congelado + post-proceso.

Carga el checkpoint Clementina XPU con map_location='cpu'. El post-proceso
(oraculo v2 + Fase 1b) se IMPORTA de E3/G para no duplicar el oraculo (regla 7).
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_E3 = os.path.abspath(os.path.join(_HERE, "..", "E3_dos_conjuntos"))
_G = os.path.abspath(os.path.join(_HERE, "..", "G_multivector"))
for _p in (_E3, _G):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_e3_settransformer import NMR_SetTransformer  # noqa: E402
from oraculo import ajustar_conteo_hetero  # noqa: E402
from candidates import generate_candidates_uncertainty  # noqa: E402


def load_model(checkpoint_path, model_cfg):
    """Instancia NMR_SetTransformer con los hiperparametros del config y carga
    el state_dict en CPU. Devuelve el modelo en eval()."""
    model = NMR_SetTransformer(
        num_classes=19,
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        n_seeds=int(model_cfg["n_seeds"]),
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_raw(model, inputs):
    """inputs = (peaks_ch, mask_ch, peaks_13c, mask_13c, cond) de build_inputs
    (numpy). Agrega dimension de batch, corre el forward y devuelve el crudo (19,)."""
    peaks_ch, mask_ch, peaks_13c, mask_13c, cond = inputs

    def _b(a):
        return torch.tensor(a, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out = model(_b(peaks_ch), _b(mask_ch), _b(peaks_13c), _b(mask_13c), _b(cond))
    return out.squeeze(0).cpu().numpy().astype(np.float64)


def candidatos(raw, formula, total, ch2, tau, k_max):
    """Ancla v2 + alternativas Fase 1b. [0] = oraculo v2. FM-consistentes."""
    return generate_candidates_uncertainty(
        raw, int(total), int(ch2), int(formula["N"]), int(formula["O"]),
        float(tau), int(k_max),
    )
