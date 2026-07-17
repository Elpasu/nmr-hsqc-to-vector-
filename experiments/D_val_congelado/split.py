# coding: ascii
"""
split.py -- Exp D: val set congelado + dedup interna por SMILES canonico.

Genera val_indices_frozen.npy: los indices (dentro del dataset de 202465)
de las 14428 moleculas "originales" de las 144k, usando el MISMO
random_split(seed=42, val_split=0.1) que corrio el training historico
(V6-V9) sobre las 144280 moleculas originales. Como el dataset de 202k se
construyo agregando las 58185 nuevas AL FINAL (sin reordenar las 144280
originales), esos indices historicos son validos directamente contra
smiles_202465.npy / nmr_dataset_v3_202465_fast.h5.

El val queda FIJO (nunca se toca). Cualquier fila de TRAIN cuyo SMILES
canonico coincida con una fila de val se elimina de train (leak=0). Las
58185 moleculas nuevas van todas a train (ninguna cae en el rango
[0, 144280) donde vive el val historico).

Uso (en el cluster, login node, sin GPU):
    python split.py --config config.yaml

Requiere: numpy, rdkit (dedup/leak), torch (reproducir el random_split
historico -- ver historical_val_indices_144k). No requiere h5py: split.py
solo toca smiles, no las imagenes HSQC.
"""
import numpy as np
from rdkit import Chem


def canonicalize_smiles(smiles_array):
    """Canonicaliza con RDKit. SMILES invalidos se conservan tal cual
    (no se descarta ninguna molecula del dataset)."""
    canonical = []
    n_invalid = 0
    for smi in smiles_array:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            canonical.append(str(smi))
            n_invalid += 1
        else:
            canonical.append(Chem.MolToSmiles(mol))
    return np.array(canonical, dtype=object), n_invalid


def find_duplicate_groups(canonical_smiles):
    """dict: SMILES canonico -> lista de indices, solo para grupos con >1 molecula."""
    groups = {}
    for idx, smi in enumerate(canonical_smiles):
        groups.setdefault(smi, []).append(idx)
    return {smi: idxs for smi, idxs in groups.items() if len(idxs) > 1}


if __name__ == "__main__":
    pass
