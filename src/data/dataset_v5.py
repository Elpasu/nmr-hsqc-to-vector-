# -*- coding: utf-8 -*-
"""
Dataset V5 - Doble Condición (Total de Señales + Total de CH2)
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

        # 1. HSQC de 1 Canal (Arquitectura Original Pura)
        hsqc_raw = self.h5_file['hsqc'][idx]
        hsqc_raw = torch.tensor(hsqc_raw, dtype=torch.float32).unsqueeze(0) 
        
        # 2. Vectores 1D
        vec_c = self.h5_file['vec_c'][idx]
        vec_h = self.h5_file['vec_h'][idx]
        vec_cat = np.concatenate((vec_c, vec_h))
        vec_cat = torch.tensor(vec_cat, dtype=torch.float32)
        
        # 3. DOBLE CONDICIÓN (Total Señales + Total CH2 genéricos)
        target_vec = self.labels[idx]
        total_signals = np.sum(target_vec).astype(np.float32)
        
        # Sumamos las posiciones de los metilenos: 1 (CH2), 5 (CH2-X) y 8 (=CH2)
        total_ch2 = (target_vec[1] + target_vec[5] + target_vec[8]).astype(np.float32)
        
        # Tensor condicional con 2 valores [Total, Total_CH2]
        cond_tensor = torch.tensor([total_signals, total_ch2], dtype=torch.float32)

        return (hsqc_raw, vec_cat, cond_tensor), torch.tensor(target_vec)