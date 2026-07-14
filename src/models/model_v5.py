# -*- coding: utf-8 -*-
"""
Modelo V5 - Arquitectura Original de 1 Canal + Doble Condición (Pura)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class NMR_Net(nn.Module):
    def __init__(self, num_classes=13):
        super(NMR_Net, self).__init__()
        
        # --- RAMA 1: HSQC (1 Canal Crudo) ---
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) 
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) 
        
        self.flat_dim = 64 * 32 * 32 
        
        # --- RAMA 2: Proyecciones 1D ---
        self.fc_proj1 = nn.Linear(512, 256)
        self.fc_proj2 = nn.Linear(256, 128)
        
        # --- FUSION DE DOBLE CONDICION ---
        # Sumamos 2 neuronas extra para el cond_tensor (Total + Total CH2)
        fusion_dim = self.flat_dim + 128 + 2
        
        self.fc_fusion1 = nn.Linear(fusion_dim, 128) 
        self.fc_fusion2 = nn.Linear(128, 64)         
        self.fc_out = nn.Linear(64, num_classes)     

    def forward(self, x_img, x_proj, x_cond):
        # Rama 1
        x1 = self.pool1(F.relu(self.conv1(x_img)))
        x1 = self.pool2(F.relu(self.conv2(x1)))
        x1 = self.pool3(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, self.flat_dim) 
        
        # Rama 2
        x2 = F.relu(self.fc_proj1(x_proj))
        x2 = F.relu(self.fc_proj2(x2))
        
        # Fusión
        x_cat = torch.cat((x1, x2, x_cond), dim=1)
        
        x = F.relu(self.fc_fusion1(x_cat))
        x = F.relu(self.fc_fusion2(x))
        out = self.fc_out(x)
        
        return out