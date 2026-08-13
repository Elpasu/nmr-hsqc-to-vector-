# coding: ascii
"""
split_utils.py -- Exp E Fase 2: funciones puras de dedup/leak, copiadas de
experiments/D_val_congelado/split.py (ya probadas ahi y en Exp B/C). Se usan
en train.py para reconstruir el mismo train set limpio a partir de
val_indices_frozen.npy (Exp D), sin volver a correr split.py completo ni
depender de otras carpetas de experimento (self-contained).
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


def subsample_train_idx(train_idx, fraction, seed=42):
    """Subsamplea train_idx de forma deterministica y anidada, para el
    estudio de escalado de datos (Parte 2 del spec de Exp F): la
    permutacion es la misma para cualquier fraccion (mismo seed), asi que
    fraccion=0.25 es subconjunto de fraccion=0.50, etc. -- la curva de
    escalado mide una progresion genuinamente incremental, no muestras
    independientes entre si. fraction >= 1.0 devuelve train_idx sin tocar."""
    if fraction >= 1.0:
        return train_idx
    rng = np.random.RandomState(seed)
    perm = rng.permutation(train_idx)
    n_keep = int(len(train_idx) * fraction)
    return np.sort(perm[:n_keep])
