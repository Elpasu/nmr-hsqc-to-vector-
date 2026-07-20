# coding: ascii
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from rdkit import Chem

class NMRDataset(Dataset):
    """
    Dataset V10 - HSQC 2 canales (nmr_dataset_v3) + Formula Molecular.
      Canal 0: DEPT escalado por N_H
      Canal 1: tipo CH normalizado
    Condicionante: [total_senales, total_CH2, C,H,N,O,S,Hal] = 8 valores
    Labels: 19 clases
    """
    def __init__(self, h5_path, labels_path, smiles_path, transform=None):
        self.h5_path = h5_path
        self.labels  = np.load(labels_path).astype(np.float32)
        self.smiles  = np.load(smiles_path, allow_pickle=True)
        self.h5_file = None

        print("[INFO] Extrayendo formulas moleculares (C,H,N,O,S,Hal)...")
        self.formula_matrix = np.zeros((len(self.smiles), 6), dtype=np.float32)
        for i, smi in enumerate(self.smiles):
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                mol = Chem.AddHs(mol)
                c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
                h = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
                n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
                o = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
                s = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
                hal = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53])
                self.formula_matrix[i] = [c, h, n, o, s, hal]
        print("[INFO] Formulas moleculares cargadas.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)

        # HSQC 2 canales -> (2, 256, 256), SIN unsqueeze
        hsqc_raw = self.h5_file['hsqc'][idx]
        hsqc_raw = torch.tensor(hsqc_raw, dtype=torch.float32)

        vec_c = self.h5_file['vec_c'][idx]
        vec_h = self.h5_file['vec_h'][idx]
        vec_cat = np.concatenate((vec_c, vec_h))
        vec_cat = torch.tensor(vec_cat, dtype=torch.float32)

        target_vec = self.labels[idx]
        total_signals = np.sum(target_vec).astype(np.float32)
        # 19 dims: CH2 en indices 1, 5, 9, 12
        total_ch2 = (target_vec[1] + target_vec[5] +
                     target_vec[9] + target_vec[12]).astype(np.float32)

        cond_data = [total_signals, total_ch2] + self.formula_matrix[idx].tolist()
        cond_tensor = torch.tensor(cond_data, dtype=torch.float32)

        return (hsqc_raw, vec_cat, cond_tensor), torch.tensor(target_vec)
