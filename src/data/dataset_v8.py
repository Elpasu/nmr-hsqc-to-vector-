# coding: ascii
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class NMRDataset(Dataset):
    """
    Dataset V8 - usa nmr_dataset_v3 (2 canales):
      Canal 0: DEPT escalado por N_H (CH2 negativo, CH/CH3 positivo)
      Canal 1: tipo CH normalizado   (CH=0.33, CH2=0.67, CH3=1.0)

    Condicionante reducido: [total_senales, total_CH2]  <- 2 valores
    Sin formula molecular (se evalua su aporte por separado en V8b).
    """
    def __init__(self, h5_path, labels_path, smiles_path=None, transform=None):
        self.h5_path = h5_path
        self.labels  = np.load(labels_path).astype(np.float32)
        self.h5_file = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)

        # HSQC 2 canales -> shape (2, 256, 256), sin unsqueeze
        hsqc_raw = self.h5_file['hsqc'][idx]
        hsqc_raw = torch.tensor(hsqc_raw, dtype=torch.float32)

        vec_c = self.h5_file['vec_c'][idx]
        vec_h = self.h5_file['vec_h'][idx]
        vec_cat = np.concatenate((vec_c, vec_h))
        vec_cat = torch.tensor(vec_cat, dtype=torch.float32)

        target_vec = self.labels[idx]

        # Total de senales (suma de los 17 grupos)
        total_signals = np.sum(target_vec).astype(np.float32)

        # Total de metilenos sp3: CH2(1) + CH2-O(5) + CH2-N(9) + =CH2(12)
        total_ch2 = (target_vec[1] + target_vec[5] +
                     target_vec[9] + target_vec[12]).astype(np.float32)

        # Condicionante: [total_senales, total_CH2]  <- 2 valores
        cond_tensor = torch.tensor([total_signals, total_ch2], dtype=torch.float32)

        return (hsqc_raw, vec_cat, cond_tensor), torch.tensor(target_vec)