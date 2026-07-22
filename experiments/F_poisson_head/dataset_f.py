# coding: ascii
"""Exp F -- dataset identico al de Exp E Fase 3 (dataset_e3.py), copiado sin
cambios de logica. Exp F no toca el dataset, solo el modelo (cabeza
softplus) y la loss (Poisson en vez de MSE)."""
import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem


class NMRTwoSetsDataset(Dataset):
    """Dos conjuntos de picos:
      - crosspeaks C-H (delta_c, delta_h, amp_ch0, amp_ch1), de peaks_pkl (Fase 1b).
      - 13C (delta_c,), de peaks_13c (Fase 3) -- incluye cuaternarios.
    Normaliza los desplazamientos min-max con la calibracion del config
    (norm_cfg). Condicionante FM identico a dataset_v10/dataset_e2/dataset_e3
    (8 valores).
    """
    def __init__(self, peaks_ch_path, peaks_13c_path, labels_path, smiles_path, norm_cfg):
        self.labels = np.load(labels_path).astype(np.float32)
        self.smiles = np.load(smiles_path, allow_pickle=True)

        npz_ch = np.load(peaks_ch_path)
        peaks_ch = npz_ch["peaks"].astype(np.float32)          # (N, 32, 4)
        self.mask_ch = npz_ch["peaks_mask"].astype(np.float32)  # (N, 32)

        npz_c13 = np.load(peaks_13c_path)
        peaks_13c = npz_c13["peaks_13c"].astype(np.float32)     # (N, M, 1)
        self.mask_13c = npz_c13["mask_13c"].astype(np.float32)  # (N, M)

        # --- normalizacion min-max desde el config (no hardcodear valores) ---
        c_min, c_max = float(norm_cfg["c13_ppm_min"]), float(norm_cfg["c13_ppm_max"])
        h_min, h_max = float(norm_cfg["h1_ppm_min"]), float(norm_cfg["h1_ppm_max"])
        amp0_scale = float(norm_cfg["amp_ch0_scale"])
        peaks_ch[:, :, 0] = (peaks_ch[:, :, 0] - c_min) / (c_max - c_min)
        peaks_ch[:, :, 1] = (peaks_ch[:, :, 1] - h_min) / (h_max - h_min)
        peaks_ch[:, :, 2] = peaks_ch[:, :, 2] / amp0_scale
        # amp_ch1 (col 3) se deja como esta.
        peaks_13c[:, :, 0] = (peaks_13c[:, :, 0] - c_min) / (c_max - c_min)
        self.peaks_ch = peaks_ch
        self.peaks_13c = peaks_13c

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
        peaks_ch = torch.tensor(self.peaks_ch[idx], dtype=torch.float32)
        mask_ch = torch.tensor(self.mask_ch[idx], dtype=torch.float32)
        peaks_13c = torch.tensor(self.peaks_13c[idx], dtype=torch.float32)
        mask_13c = torch.tensor(self.mask_13c[idx], dtype=torch.float32)

        target_vec = self.labels[idx]
        total_signals = np.sum(target_vec).astype(np.float32)
        total_ch2 = (target_vec[1] + target_vec[5] +
                     target_vec[9] + target_vec[12]).astype(np.float32)
        cond_data = [total_signals, total_ch2] + self.formula_matrix[idx].tolist()
        cond_tensor = torch.tensor(cond_data, dtype=torch.float32)

        return (peaks_ch, mask_ch, peaks_13c, mask_13c, cond_tensor), torch.tensor(target_vec)
