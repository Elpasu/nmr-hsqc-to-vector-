# coding: ascii
"""dataset_j.py -- Exp J: mismos dos conjuntos de picos que el E3, con dos
diferencias:

  1. Los crosspeaks traen una 5a feature, la DEGENERACION (cuantos carbonos
     comparten esa senal). peak_features=4 la recorta -> corrida de control.
  2. Los labels cuentan CARBONOS TOTALES, no senales. El codigo que arma el
     condicionante es IDENTICO al del E3 (sum del target y las 4 clases del
     cupo CH2); lo que cambia es el significado, porque cambiaron los labels:
     cond[0] pasa a ser C de la formula y cond[1] los carbonos CH2.
"""
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset

IDX_CH2 = [1, 5, 9, 12]   # CH2, CH2-O, CH2-N, =CH2 (orden de config/db.yaml)


class NMRTwoSetsDatasetJ(Dataset):
    def __init__(self, peaks_ch_path, peaks_13c_path, labels_path, smiles_path,
                 norm_cfg, peak_features=5):
        if int(peak_features) not in (4, 5):
            raise ValueError(
                f"peak_features debe ser 4 (control) o 5 (con degeneracion), "
                f"recibido: {peak_features!r}")
        self.peak_features = int(peak_features)

        self.labels = np.load(labels_path).astype(np.float32)
        self.smiles = np.load(smiles_path, allow_pickle=True)

        npz_ch = np.load(peaks_ch_path)
        peaks_ch = npz_ch["peaks"].astype(np.float32)
        self.mask_ch = npz_ch["peaks_mask"].astype(np.float32)

        n_cols = peaks_ch.shape[2]
        if n_cols < self.peak_features:
            raise ValueError(
                f"El .npz tiene {n_cols} columnas por pico pero se pidieron "
                f"peak_features={self.peak_features}. Con peak_features=5 hace "
                f"falta el archivo con degeneracion (peaks_pkl_deg_*.npz); "
                f"entrenar con una columna de ceros haciendose pasar por la "
                f"degeneracion daria un control disfrazado de experimento.")

        npz_c13 = np.load(peaks_13c_path)
        peaks_13c = npz_c13["peaks_13c"].astype(np.float32)
        self.mask_13c = npz_c13["mask_13c"].astype(np.float32)

        # --- normalizacion min-max desde el config (regla dura 3) ---
        c_min, c_max = float(norm_cfg["c13_ppm_min"]), float(norm_cfg["c13_ppm_max"])
        h_min, h_max = float(norm_cfg["h1_ppm_min"]), float(norm_cfg["h1_ppm_max"])
        amp0_scale = float(norm_cfg["amp_ch0_scale"])
        peaks_ch[:, :, 0] = (peaks_ch[:, :, 0] - c_min) / (c_max - c_min)
        peaks_ch[:, :, 1] = (peaks_ch[:, :, 1] - h_min) / (h_max - h_min)
        peaks_ch[:, :, 2] = peaks_ch[:, :, 2] / amp0_scale
        # amp_ch1 (col 3) se deja como esta, igual que en el E3.
        if self.peak_features == 5:
            # Sin default: si falta en el config tiene que romper, no elegir un
            # valor a dedo que despues nadie pueda rastrear (regla dura 3).
            deg_scale = float(norm_cfg["degeneracion_scale"])
            peaks_ch[:, :, 4] = peaks_ch[:, :, 4] / deg_scale

        self.peaks_ch = peaks_ch[:, :, :self.peak_features]
        peaks_13c[:, :, 0] = (peaks_13c[:, :, 0] - c_min) / (c_max - c_min)
        self.peaks_13c = peaks_13c

        print("[INFO] Extrayendo formulas moleculares (C,H,N,O,S,Hal)...")
        self.formula_matrix = np.zeros((len(self.smiles), 6), dtype=np.float32)
        for i, smi in enumerate(self.smiles):
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                mol = Chem.AddHs(mol)
                nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
                self.formula_matrix[i] = [
                    sum(1 for z in nums if z == 6),
                    sum(1 for z in nums if z == 1),
                    sum(1 for z in nums if z == 7),
                    sum(1 for z in nums if z == 8),
                    sum(1 for z in nums if z == 16),
                    sum(1 for z in nums if z in (9, 17, 35, 53)),
                ]
        print(f"[INFO] Formulas cargadas. peak_features={self.peak_features}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        peaks_ch = torch.tensor(self.peaks_ch[idx], dtype=torch.float32)
        mask_ch = torch.tensor(self.mask_ch[idx], dtype=torch.float32)
        peaks_13c = torch.tensor(self.peaks_13c[idx], dtype=torch.float32)
        mask_13c = torch.tensor(self.mask_13c[idx], dtype=torch.float32)

        target_vec = self.labels[idx]
        # Mismo codigo que el E3. Con los labels de carbonos totales, total_c
        # vale C de la formula y total_ch2 los carbonos CH2 -- ambos siguen
        # siendo observables: C sale de la FM (exacto, sin error de lectura) y
        # los CH2 de sumar integrales de los crosspeaks de fase negativa.
        total_c = np.sum(target_vec).astype(np.float32)
        total_ch2 = sum(target_vec[i] for i in IDX_CH2)
        cond_data = [total_c, np.float32(total_ch2)] + self.formula_matrix[idx].tolist()
        cond_tensor = torch.tensor(cond_data, dtype=torch.float32)

        return (peaks_ch, mask_ch, peaks_13c, mask_13c, cond_tensor), torch.tensor(target_vec)
