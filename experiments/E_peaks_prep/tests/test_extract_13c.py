# coding: ascii
"""Test offline del extractor de picos 13C (Task 1). Verifica que:
 (1) un carbono cuaternario (sin H) SI entra al conjunto 13C,
 (2) carbonos equivalentes por simetria con el mismo shift colapsan a un pico,
 (3) el padding a (N, M, 1) y la mascara son consistentes.
Corre local, sin datos reales:  python tests/test_extract_13c.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rdkit import Chem

from extract_peaks_13c_pkl import (
    extract_13c_peaks_from_molecule,
    build_padded_arrays_13c,
)


def _shifts_for(smiles, per_carbon):
    """Arma un dict {atom_idx: shift} post-AddHs asignando a cada carbono
    (en orden de indice) el shift dado en per_carbon."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    shifts = {}
    ci = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            shifts[atom.GetIdx()] = per_carbon[ci]
            ci += 1
    return shifts


def test_quaternary_carbon_is_included():
    # Acetona CC(=O)C: 3 carbonos -> C(carbonilo, cuaternario) + 2 CH3.
    # Damos shifts distintos a los 3 -> deben salir 3 picos (el cuaternario incluido).
    smiles = "CC(=O)C"
    shifts = _shifts_for(smiles, [30.0, 205.0, 31.5])
    peaks = extract_13c_peaks_from_molecule(smiles, shifts)
    assert len(peaks) == 3, peaks
    # El shift del carbonilo (205.0, cuaternario, sin H) tiene que estar presente.
    assert any(abs(p[0] - 205.0) < 1e-6 for p in peaks), peaks
    print(f"[OK] cuaternario incluido: {peaks}")


def test_symmetry_dedup():
    # Dos carbonos con el MISMO shift (equivalentes por simetria) colapsan a 1.
    smiles = "CC(=O)C"
    shifts = _shifts_for(smiles, [30.0, 205.0, 30.0])  # los 2 CH3 iguales
    peaks = extract_13c_peaks_from_molecule(smiles, shifts)
    assert len(peaks) == 2, peaks   # {30.0, 205.0}
    print(f"[OK] dedup por simetria: {peaks}")


def test_padding_shape():
    per_mol = [[(30.0,), (205.0,)], [(10.0,)], []]
    peaks, mask = build_padded_arrays_13c(per_mol)
    assert peaks.shape == (3, 2, 1), peaks.shape
    assert mask.shape == (3, 2), mask.shape
    assert mask[0].tolist() == [True, True]
    assert mask[1].tolist() == [True, False]
    assert mask[2].tolist() == [False, False]
    print(f"[OK] padding -> {peaks.shape}, mask ok")


if __name__ == "__main__":
    test_quaternary_carbon_is_included()
    test_symmetry_dedup()
    test_padding_shape()
    print("\n>>> TEST EXTRACT 13C OK <<<")
