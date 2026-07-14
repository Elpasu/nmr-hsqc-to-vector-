
# coding: ascii
import torch
import torch.nn as nn
import torch.nn.functional as F

class NMR_Net(nn.Module):
    """
    Modelo V8.
    Cambios respecto a V7:
      - Conv2d(1->2): HSQC de 2 canales (DEPT escalado + tipo CH)
      - fusion_dim: flat_dim + 128 + 2  (condicionante reducido a 2 valores)
    """
    def __init__(self, num_classes=17):
        super(NMR_Net, self).__init__()

        # --- RAMA 1: HSQC (2 canales) ---
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)  # antes: 1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flat_dim = 64 * 32 * 32   # 65536

        # --- RAMA 2: Proyecciones 1D ---
        self.fc_proj1 = nn.Linear(512, 256)
        self.fc_proj2 = nn.Linear(256, 128)

        # --- FUSION ---
        # Condicionante: [total_senales, total_CH2] = 2 valores (sin formula molecular)
        fusion_dim = self.flat_dim + 128 + 2

        self.fc_fusion1 = nn.Linear(fusion_dim, 128)
        self.fc_fusion2 = nn.Linear(128, 64)
        self.fc_out     = nn.Linear(64, num_classes)

    def forward(self, x_img, x_proj, x_cond):
        # x_img: (batch, 2, 256, 256)
        x1 = self.pool1(F.relu(self.conv1(x_img)))
        x1 = self.pool2(F.relu(self.conv2(x1)))
        x1 = self.pool3(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, self.flat_dim)

        x2 = F.relu(self.fc_proj1(x_proj))
        x2 = F.relu(self.fc_proj2(x2))

        x = torch.cat((x1, x2, x_cond), dim=1)
        x = F.relu(self.fc_fusion1(x))
        x = F.relu(self.fc_fusion2(x))
        return self.fc_out(x)