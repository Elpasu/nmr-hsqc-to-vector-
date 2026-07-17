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
import argparse
from pathlib import Path

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


def remove_leaking_from_train(train_idx, val_idx, canonical_smiles):
    """Elimina de train cualquier indice cuyo SMILES canonico tambien este
    en val. val_idx NUNCA se modifica (queda "congelado")."""
    val_smiles_set = set(canonical_smiles[i] for i in val_idx)
    clean_train_idx = np.array(
        [i for i in train_idx if canonical_smiles[i] not in val_smiles_set],
        dtype=np.int64,
    )
    n_removed = len(train_idx) - len(clean_train_idx)
    return clean_train_idx, n_removed


def historical_val_indices_144k(n_144k=144280, val_split=0.1, seed=42):
    """Reproduce el random_split(seed=42) que uso el training historico
    (V6-V9) sobre las 144280 moleculas originales. Requiere torch para
    igualar bit a bit el RNG que ya se uso (no se puede reproducir con
    numpy). Los indices devueltos son validos directamente contra el
    dataset de 202k porque las 58185 nuevas se agregaron al final, sin
    reordenar las 144280 originales."""
    import torch
    from torch.utils.data import random_split

    val_size = int(n_144k * val_split)
    train_size = n_144k - val_size
    generator = torch.Generator().manual_seed(seed)
    _, val_subset = random_split(range(n_144k), [train_size, val_size], generator=generator)
    return np.array(sorted(val_subset.indices), dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Exp D: genera val_indices_frozen.npy")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(cfg["paths"]["base_dir"])
    smiles_path = base_dir / cfg["paths"]["smiles_filename"]
    out_path = base_dir / cfg["paths"]["val_indices_filename"]

    n_144k = int(cfg["split"]["n_144k"])
    val_split_144k = float(cfg["split"]["val_split_144k"])
    seed = int(cfg["split"]["seed"])

    print("=" * 60)
    print("  EXP D: split congelado")
    print("=" * 60)

    smiles = np.load(smiles_path, allow_pickle=True)
    n_total = len(smiles)
    print(f"-> Moleculas totales: {n_total}")

    canonical, n_invalid = canonicalize_smiles(smiles)
    print(f"-> SMILES invalidos (no parsearon con RDKit): {n_invalid}")

    dup_groups = find_duplicate_groups(canonical)
    n_dup_mols = sum(len(idxs) - 1 for idxs in dup_groups.values())
    print(f"-> Grupos de duplicados canonicos: {len(dup_groups)}  "
          f"({n_dup_mols} moleculas 'de mas' respecto a canonicos unicos)")

    val_idx = historical_val_indices_144k(n_144k, val_split_144k, seed)
    print(f"-> Val congelado (historico 144k, seed={seed}): {len(val_idx)} moleculas")

    all_idx = np.arange(n_total)
    train_idx_raw = np.setdiff1d(all_idx, val_idx, assume_unique=False)

    train_idx, n_removed = remove_leaking_from_train(train_idx_raw, val_idx, canonical)
    print(f"-> Filas de train eliminadas por leak canonico contra val: {n_removed}")
    print(f"-> Train final: {len(train_idx)}   Val final (sin tocar): {len(val_idx)}")

    val_smiles_set = set(canonical[i] for i in val_idx)
    train_smiles_set = set(canonical[i] for i in train_idx)
    leak = val_smiles_set & train_smiles_set
    print(f"-> Verificacion leak=0: interseccion train/val = {len(leak)} SMILES canonicos")
    assert len(leak) == 0, "Leak residual tras remove_leaking_from_train (no deberia pasar)"

    np.save(out_path, val_idx)
    print(f"\n[SAVE] {out_path}  ({len(val_idx)} indices, dtype={val_idx.dtype})")
    print(">>> EXP D split.py OK <<<")


if __name__ == "__main__":
    main()
