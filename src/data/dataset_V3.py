# -*- coding: utf-8 -*-
"""
Dataset V2 - Modificado para red condicional (Pasa el total de señales como input)
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

        # 1. Cargar Inputs Base
        hsqc = self.h5_file['hsqc'][idx]
        hsqc = torch.tensor(hsqc, dtype=torch.float32).unsqueeze(0) 
        
        vec_c = self.h5_file['vec_c'][idx]
        vec_h = self.h5_file['vec_h'][idx]
        vec_cat = np.concatenate((vec_c, vec_h))
        vec_cat = torch.tensor(vec_cat, dtype=torch.float32)
        
        # 2. EL NUEVO INPUT: Extraemos el total de señales de la molecula
        target_vec = self.labels[idx]
        total_signals = np.sum(target_vec).astype(np.float32)
        total_tensor = torch.tensor([total_signals], dtype=torch.float32)

        # Retornamos (hsqc, proyecciones, total_señales) como entradas, y el vector target
        return (hsqc, vec_cat, total_tensor), torch.tensor(target_vec)