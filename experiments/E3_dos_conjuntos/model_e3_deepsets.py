# coding: ascii
import torch
import torch.nn as nn
import torch.nn.functional as F


class NMR_DeepSets(nn.Module):
    """Exp E Fase 3 -- DeepSets de dos ramas:
      - Rama crosspeaks: MLP 4->64->64 por pico, promedio enmascarado -> aggA.
      - Rama 13C:        MLP 1->64->64 por pico, promedio enmascarado -> aggB.
      - Fusion: [aggA(64), aggB(64), cond(8)] -> 128 -> 64 -> num_classes.
    """
    def __init__(self, num_classes=19):
        super().__init__()
        self.ch_mlp1 = nn.Linear(4, 64)
        self.ch_mlp2 = nn.Linear(64, 64)
        self.c13_mlp1 = nn.Linear(1, 64)
        self.c13_mlp2 = nn.Linear(64, 64)

        fusion_dim = 64 + 64 + 8
        self.fc_fusion1 = nn.Linear(fusion_dim, 128)
        self.fc_fusion2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)

    @staticmethod
    def _masked_mean(x, mask):
        # x: (B, T, 64), mask: (B, T)
        m = mask.unsqueeze(-1)                    # (B, T, 1)
        counts = m.sum(dim=1).clamp(min=1.0)      # (B, 1)
        return (x * m).sum(dim=1) / counts        # (B, 64)

    def forward(self, peaks_ch, mask_ch, peaks_13c, mask_13c, cond):
        a = F.relu(self.ch_mlp1(peaks_ch))
        a = F.relu(self.ch_mlp2(a))
        aggA = self._masked_mean(a, mask_ch)

        b = F.relu(self.c13_mlp1(peaks_13c))
        b = F.relu(self.c13_mlp2(b))
        aggB = self._masked_mean(b, mask_13c)

        x = torch.cat((aggA, aggB, cond), dim=1)
        x = F.relu(self.fc_fusion1(x))
        x = F.relu(self.fc_fusion2(x))
        return self.fc_out(x)
