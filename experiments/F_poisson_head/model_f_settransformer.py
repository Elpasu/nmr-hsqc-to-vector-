# coding: ascii
"""Set Transformer (Lee et al. 2019) sobre la union de los dos conjuntos de
picos, identico a experiments/E3_dos_conjuntos/model_e3_settransformer.py
salvo un cambio: la salida final pasa por softplus para garantizar
lambda >= 0 (lo exige PoissonNLLLoss con log_input=False en train.py). No
agrega parametros nuevos."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, valid_mask=None):
        Qp = self.fc_q(Q); Kp = self.fc_k(K); Vp = self.fc_v(K)
        H = self.num_heads
        d = self.dim_V // H
        Qh = torch.cat(Qp.split(d, 2), 0)
        Kh = torch.cat(Kp.split(d, 2), 0)
        Vh = torch.cat(Vp.split(d, 2), 0)
        logits = Qh.bmm(Kh.transpose(1, 2)) / math.sqrt(d)
        if valid_mask is not None:
            vm = (valid_mask > 0.5)
            vm = vm.repeat(H, 1).unsqueeze(1)
            logits = logits.masked_fill(~vm, float("-inf"))
            A = torch.softmax(logits, dim=2)
            A = torch.nan_to_num(A, nan=0.0)
        else:
            A = torch.softmax(logits, dim=2)
        O = Qh + A.bmm(Vh)
        O = torch.cat(O.split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, X, valid_mask=None):
        return self.mab(X, X, valid_mask)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, X, valid_mask=None):
        S = self.S.repeat(X.size(0), 1, 1)
        return self.mab(S, X, valid_mask)


class NMR_SetTransformer(nn.Module):
    def __init__(self, num_classes=19, d_model=64, n_heads=4, n_layers=2, n_seeds=1):
        super().__init__()
        self.proj_ch = nn.Linear(4, d_model)
        self.proj_13c = nn.Linear(1, d_model)
        self.type_emb = nn.Embedding(2, d_model)   # 0=crosspeak, 1=13C
        self.encoder = nn.ModuleList([SAB(d_model, n_heads) for _ in range(n_layers)])
        self.pma = PMA(d_model, n_heads, n_seeds)

        fusion_dim = d_model * n_seeds + 8
        self.fc_fusion1 = nn.Linear(fusion_dim, 128)
        self.fc_fusion2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, peaks_ch, mask_ch, peaks_13c, mask_13c, cond):
        B = peaks_ch.size(0)
        tok_ch = self.proj_ch(peaks_ch) + self.type_emb.weight[0].view(1, 1, -1)
        tok_13c = self.proj_13c(peaks_13c) + self.type_emb.weight[1].view(1, 1, -1)
        tokens = torch.cat([tok_ch, tok_13c], dim=1)
        valid = torch.cat([mask_ch, mask_13c], dim=1)

        x = tokens
        for sab in self.encoder:
            x = sab(x, valid)
        pooled = self.pma(x, valid).reshape(B, -1)

        h = torch.cat([pooled, cond], dim=1)
        h = F.relu(self.fc_fusion1(h))
        h = F.relu(self.fc_fusion2(h))
        return F.softplus(self.fc_out(h))
