# coding: ascii
import torch
import torch.nn as nn
import torch.nn.functional as F

class NMR_Net(nn.Module):
    """
    Modelo Exp C: igual a V10 (HSQC 2 canales + Formula Molecular + 19 clases)
    pero reemplaza el flatten+Linear(65664,128) por Global Average Pooling
    (Exp C: rebalanceo de la fusion). Ataca el desbalance de ramas: antes la
    conv aportaba 65536 features contra 128 del 1D y 8 de la FM (512:1);
    con GAP la conv aporta solo 64.
      - Conv2d(2->16): 2 canales (de V8)
      - GAP tras el ultimo bloque conv: (batch,64,32,32) -> (batch,64) (Exp C)
      - fusion_dim: 64 (GAP) + 128 (1D) + 8 (FM) = 200 (Exp C, antes 65672)
    """
    def __init__(self, num_classes=19):
        super(NMR_Net, self).__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)  # 2 canales
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)  # (batch,64,32,32) -> (batch,64,1,1)
        self.gap_dim = 64

        self.fc_proj1 = nn.Linear(512, 256)
        self.fc_proj2 = nn.Linear(256, 128)

        # Condicionante 8 valores (con FM)
        fusion_dim = self.gap_dim + 128 + 8

        self.fc_fusion1 = nn.Linear(fusion_dim, 128)
        self.fc_fusion2 = nn.Linear(128, 64)
        self.fc_out     = nn.Linear(64, num_classes)

    def forward(self, x_img, x_proj, x_cond):
        # x_img: (batch, 2, 256, 256)
        x1 = self.pool1(F.relu(self.conv1(x_img)))
        x1 = self.pool2(F.relu(self.conv2(x1)))
        x1 = self.pool3(F.relu(self.conv3(x1)))
        x1 = self.gap(x1)
        x1 = x1.view(-1, self.gap_dim)

        x2 = F.relu(self.fc_proj1(x_proj))
        x2 = F.relu(self.fc_proj2(x2))

        x = torch.cat((x1, x2, x_cond), dim=1)
        x = F.relu(self.fc_fusion1(x))
        x = F.relu(self.fc_fusion2(x))
        return self.fc_out(x)
