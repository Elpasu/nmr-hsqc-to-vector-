# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 17:00:54 2026

@author: UCA Team
"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class NMRDataset(Dataset):
    def __init__(self, h5_path, labels_path, transform=None):
        self.h5_path = h5_path
        self.labels = np.load(labels_path).astype(np.float32)
        self.transform = transform
        self.h5_file = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)

        # 1. Cargar HSQC crudo
        hsqc_raw = self.h5_file['hsqc'][idx]
        hsqc_raw = torch.tensor(hsqc_raw, dtype=torch.float32).unsqueeze(0) 
        
        # --- VISION DEPT (2 CANALES) ---
        # Canal 1: Solo positivos (CH, CH3)
        hsqc_pos = torch.clamp(hsqc_raw, min=0.0)
        # Canal 2: Solo negativos, pasados a positivo absoluto (CH2)
        hsqc_neg = torch.abs(torch.clamp(hsqc_raw, max=0.0))
        
        # Juntamos los canales. Queda (2, 256, 256)
        hsqc_2ch = torch.cat((hsqc_pos, hsqc_neg), dim=0)
        
        # 2. Vectores 1D
        vec_c = self.h5_file['vec_c'][idx]
        vec_h = self.h5_file['vec_h'][idx]
        vec_cat = np.concatenate((vec_c, vec_h))
        vec_cat = torch.tensor(vec_cat, dtype=torch.float32)
        
        # 3. Input Condicional (Total de señales)
        target_vec = self.labels[idx]
        total_signals = np.sum(target_vec).astype(np.float32)
        total_tensor = torch.tensor([total_signals], dtype=torch.float32)

        return (hsqc_2ch, vec_cat, total_tensor), torch.tensor(target_vec)