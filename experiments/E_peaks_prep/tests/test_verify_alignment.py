# coding: ascii
"""Tests de verify_smiles_alignment -- corre localmente, requiere rdkit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from extract_peaks_pkl import verify_smiles_alignment


def test_alignment_ok_with_different_but_equivalent_smiles():
    # Mismo orden, pero escritos distinto (canonico vs no-canonico) --
    # debe pasar igual porque se compara canonicalizado.
    local = np.array(["CCO", "c1ccccc1", "CC(=O)O"], dtype=object)
    real = np.array(["OCC", "C1=CC=CC=C1", "CC(O)=O"], dtype=object)
    ok, mismatch_idx = verify_smiles_alignment(local, real)
    assert ok is True, mismatch_idx
    assert mismatch_idx is None
    print("[OK] test_alignment_ok_with_different_but_equivalent_smiles")


def test_alignment_detects_mismatch_and_reports_index():
    local = np.array(["CCO", "c1ccccc1", "CC(=O)O"], dtype=object)
    real = np.array(["CCO", "CCN", "CC(=O)O"], dtype=object)   # indice 1 distinto
    ok, mismatch_idx = verify_smiles_alignment(local, real)
    assert ok is False
    assert mismatch_idx == 1, mismatch_idx
    print(f"[OK] test_alignment_detects_mismatch_and_reports_index -> idx={mismatch_idx}")


def test_alignment_detects_length_mismatch():
    local = np.array(["CCO", "CCN"], dtype=object)
    real = np.array(["CCO"], dtype=object)
    ok, mismatch_idx = verify_smiles_alignment(local, real)
    assert ok is False
    assert mismatch_idx is None   # no hay un indice puntual, es un mismatch de longitud
    print("[OK] test_alignment_detects_length_mismatch")


if __name__ == "__main__":
    test_alignment_ok_with_different_but_equivalent_smiles()
    test_alignment_detects_mismatch_and_reports_index()
    test_alignment_detects_length_mismatch()
    print("\n>>> test_verify_alignment.py OK <<<")
