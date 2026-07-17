# coding: ascii
"""
Tests locales (sin torch/h5py) de la logica pura de split.py -- Exp D.
Corren en cualquier maquina con numpy + rdkit (no requieren el cluster).

Uso:
    python tests/test_split.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from split import (
    canonicalize_smiles,
    find_duplicate_groups,
    remove_leaking_from_train,
)


def test_canonicalize_smiles_dedup_equivalent_forms():
    smiles = np.array(["CCO", "OCC", "C1=CC=CC=C1", "c1ccccc1"], dtype=object)
    canonical, n_invalid = canonicalize_smiles(smiles)
    assert n_invalid == 0
    assert canonical[0] == canonical[1], "etanol en dos formas debe canonicalizar igual"
    assert canonical[2] == canonical[3], "benceno kekulizado/aromatico debe canonicalizar igual"
    print("[OK] test_canonicalize_smiles_dedup_equivalent_forms")


def test_canonicalize_smiles_invalid_passthrough():
    smiles = np.array(["CCO", "not_a_smiles!!"], dtype=object)
    canonical, n_invalid = canonicalize_smiles(smiles)
    assert n_invalid == 1
    assert canonical[1] == "not_a_smiles!!"
    print("[OK] test_canonicalize_smiles_invalid_passthrough")


def test_find_duplicate_groups():
    canonical = np.array(["A", "B", "A", "C", "B", "A"], dtype=object)
    dups = find_duplicate_groups(canonical)
    assert dups == {"A": [0, 2, 5], "B": [1, 4]}
    assert "C" not in dups
    print("[OK] test_find_duplicate_groups")


def test_find_duplicate_groups_no_dups():
    canonical = np.array(["A", "B", "C"], dtype=object)
    assert find_duplicate_groups(canonical) == {}
    print("[OK] test_find_duplicate_groups_no_dups")


def test_remove_leaking_from_train():
    canonical = np.array(["A", "B", "A", "C", "D"], dtype=object)
    # idx 2 es val (canonical "A"); train excluye idx 2 (particion real disjunta).
    # idx 0 tambien es "A" -> duplicado interno del lado train
    # -> debe eliminarse de train por leak contra val.
    train_idx = np.array([0, 1, 3, 4])
    val_idx = np.array([2])
    clean_train, n_removed = remove_leaking_from_train(train_idx, val_idx, canonical)
    assert n_removed == 1
    assert 0 not in clean_train
    assert set(clean_train.tolist()) == {1, 3, 4}
    print("[OK] test_remove_leaking_from_train")


def test_remove_leaking_from_train_no_leak():
    canonical = np.array(["A", "B", "C"], dtype=object)
    train_idx = np.array([0, 1])
    val_idx = np.array([2])
    clean_train, n_removed = remove_leaking_from_train(train_idx, val_idx, canonical)
    assert n_removed == 0
    assert set(clean_train.tolist()) == {0, 1}
    print("[OK] test_remove_leaking_from_train_no_leak")


if __name__ == "__main__":
    test_canonicalize_smiles_dedup_equivalent_forms()
    test_canonicalize_smiles_invalid_passthrough()
    test_find_duplicate_groups()
    test_find_duplicate_groups_no_dups()
    test_remove_leaking_from_train()
    test_remove_leaking_from_train_no_leak()
    print("\n>>> SPLIT CORE TESTS OK <<<")
