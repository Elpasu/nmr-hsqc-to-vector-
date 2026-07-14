# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:40:46 2026

@author: Pasu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NMR_Net(nn.Module):
    def __init__(self, num_classes=13):
        super(NMR_Net, self).__init__()
        
        # --- RAMA 1: HSQC (Imagen 2D) ---
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
        
        # --- FUSION ---
        fusion_dim = self.flat_dim + 128
        
        self.fc_fusion1 = nn.Linear(fusion_dim, 128)
        self.fc_fusion2 = nn.Linear(128, 64)
        
        # ==========================================
        # MULTI-TASK LEARNING: DOS CABEZAS DE SALIDA
        # ==========================================
        # CABEZA 1: Clasificacion de los 13 grupos
        self.fc_out_grupos = nn.Linear(64, num_classes)
        
        # CABEZA 2: Conteo total de senales (1 solo numero)
        self.fc_out_total = nn.Linear(64, 1)

    def forward(self, x_img, x_proj):
        # --- Paso Rama 1 (CNN) ---
        x1 = self.pool1(F.relu(self.conv1(x_img)))
        x1 = self.pool2(F.relu(self.conv2(x1)))
        x1 = self.pool3(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, self.flat_dim)
        
        # --- Paso Rama 2 (Dense) ---
        x2 = F.relu(self.fc_proj1(x_proj))
        x2 = F.relu(self.fc_proj2(x2))
        
        # --- Fusion ---
        x_cat = torch.cat((x1, x2), dim=1)
        
        # --- Clasificacion compartida ---
        x = F.relu(self.fc_fusion1(x_cat))
        x = F.relu(self.fc_fusion2(x))
        
        # --- SALIDAS INDEPENDIENTES ---
        out_grupos = F.softplus(self.fc_out_grupos(x))
        
        # Usamos softplus tambien aca para garantizar que el total sea un numero positivo
        out_total = F.softplus(self.fc_out_total(x))
        
        # Retornamos ambas predicciones
        return out_grupos, out_total