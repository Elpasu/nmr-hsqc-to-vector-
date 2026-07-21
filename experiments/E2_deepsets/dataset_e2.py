# coding: ascii
import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem


class NMRPeaksDataset(Dataset):
    """
    Dataset Exp E Fase 2 -- picos extraidos del pkl original (Exp E Fase
    1b) en vez de imagen HSQC + proyecciones 1D. Carga
    peaks_pkl_202465.npz completo en memoria al iniciar (chico, ~100MB,
    sin h5py de por medio).
    Condicionante: [total_senales, total_CH2, C,H,N,O,S,Hal] = 8 valores,
    calculado EXACTO igual que dataset_v10.py (no reinventar).
    Labels: 19 clases.
    """
    def __init__(self, peaks_path, labels_path, smiles_path):
        self.labels = np.load(labels_path).astype(np.float32)
        self.smiles = np.load(smiles_path, allow_pickle=True)

        npz = np.load(peaks_path)
        self.peaks = npz["peaks"].astype(np.float32)          # (N, max_peaks, 4)
        self.peaks_mask = npz["peaks_mask"].astype(np.float32)  # (N, max_peaks)

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
        peaks = torch.tensor(self.peaks[idx], dtype=torch.float32)
        mask = torch.tensor(self.peaks_mask[idx], dtype=torch.float32)

        target_vec = self.labels[idx]
        total_signals = np.sum(target_vec).astype(np.float32)
        # 19 dims: CH2 en indices 1, 5, 9, 12
        total_ch2 = (target_vec[1] + target_vec[5] +
                     target_vec[9] + target_vec[12]).astype(np.float32)

        cond_data = [total_signals, total_ch2] + self.formula_matrix[idx].tolist()
        cond_tensor = torch.tensor(cond_data, dtype=torch.float32)

        return (peaks, mask, cond_tensor), torch.tensor(target_vec)
