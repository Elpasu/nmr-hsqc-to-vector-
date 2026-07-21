# coding: ascii
import torch
import torch.nn as nn
import torch.nn.functional as F


class NMR_Net(nn.Module):
    """
    Modelo Exp E Fase 2 (DeepSets): reemplaza la imagen HSQC + proyecciones
    1D (V10/Exp B/Exp C) por un conjunto de picos (delta_c, delta_h,
    amp_ch0, amp_ch1) extraidos directamente del pkl original (Exp E Fase
    1b, ver docs/Runs/RESULTS.md), hasta 32 por molecula con mascara de
    validos.
      - Por pico: MLP compartido 4 -> 64 -> 64 (mismos pesos para todos los
        picos de todas las moleculas -> invariante a permutacion).
      - Agregacion: promedio enmascarado sobre los picos validos (si una
        molecula no tiene picos validos, el agregado queda en cero).
      - Fusion: agregado (64) + condicionante FM (8) = 72 -> 128 -> 64 -> 19.
    Capacidad deliberadamente chica (~23k parametros, ver RATIONALE.md) --
    no es un modelo grande, es una decision tomada con evidencia previa
    (V10 8.6M parametros peor que Exp C 223k).
    """
    def __init__(self, num_classes=19):
        super(NMR_Net, self).__init__()

        self.peak_mlp1 = nn.Linear(4, 64)
        self.peak_mlp2 = nn.Linear(64, 64)
        self.agg_dim = 64

        fusion_dim = self.agg_dim + 8  # + condicionante FM

        self.fc_fusion1 = nn.Linear(fusion_dim, 128)
        self.fc_fusion2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x_peaks, x_mask, x_cond):
        # x_peaks: (batch, max_peaks, 4)
        # x_mask:  (batch, max_peaks) -- 1.0 en picos validos, 0.0 en padding
        x = F.relu(self.peak_mlp1(x_peaks))   # (batch, max_peaks, 64)
        x = F.relu(self.peak_mlp2(x))         # (batch, max_peaks, 64)

        mask = x_mask.unsqueeze(-1)                     # (batch, max_peaks, 1)
        x_masked = x * mask
        counts = mask.sum(dim=1).clamp(min=1.0)         # (batch, 1)
        agg = x_masked.sum(dim=1) / counts               # (batch, 64)

        x = torch.cat((agg, x_cond), dim=1)
        x = F.relu(self.fc_fusion1(x))
        x = F.relu(self.fc_fusion2(x))
        return self.fc_out(x)
