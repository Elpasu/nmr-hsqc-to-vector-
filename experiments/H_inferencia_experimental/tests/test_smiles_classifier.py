# coding: utf-8
"""Tests del puerto fiel de Gen_vector.py (clasificacion de 19 clases desde SMILES).

Casos de verdad conocida elegidos para ejercitar cada rama de classify_carbon:
sp3 normal/O/N, sp2 normal/Aldeh/Imina, simetria (colapso de rangos), y las
clases nuevas C-2X/C-3X (heteroatomo = cualquier no-carbono, incluye halogenos
en el clasificador real, aunque el dataset de entrenamiento es CHON puro).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import smiles_classifier as sc  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..")))
import yaml  # noqa: E402


def _db_class_order():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    with open(os.path.join(repo_root, "config", "db.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["classes_19v"]


def test_class_order_matches_db_yaml():
    assert sc.SHORT_NAMES == _db_class_order()
    assert len(sc.CATEGORIES) == 19 == sc.N_CLASSES


def test_ethanol_ch3_ch2o():
    # CCO: CH3 (idx0) + CH2-O (idx5). Cruzado contra el forward real del
    # checkpoint E3 en esta misma sesion (dio exactamente este vector).
    vec = sc.true_vector_from_smiles("CCO")
    expected = np.zeros(19, dtype=np.int32)
    expected[0] = 1  # CH3
    expected[5] = 1  # CH2-O
    assert np.array_equal(vec, expected)


def test_benzene_symmetry_collapses_to_one():
    # Los 6 CH aromaticos son equivalentes por simetria -> un solo entorno.
    vec = sc.true_vector_from_smiles("c1ccccc1")
    expected = np.zeros(19, dtype=np.int32)
    expected[13] = 1  # =CH/Ar
    assert np.array_equal(vec, expected)


def test_acetaldehyde_has_aldeh():
    # CC=O: CH3 (idx0) + CH sp2 con doble enlace a O -> Aldeh (idx15).
    vec = sc.true_vector_from_smiles("CC=O")
    expected = np.zeros(19, dtype=np.int32)
    expected[0] = 1    # CH3
    expected[15] = 1   # Aldeh
    assert np.array_equal(vec, expected)


def test_imine_carbon():
    # CC=N: CH3 (idx0) + CH sp2 con doble enlace a N -> Imina (idx16).
    vec = sc.true_vector_from_smiles("CC=N")
    expected = np.zeros(19, dtype=np.int32)
    expected[0] = 1    # CH3
    expected[16] = 1   # Imina
    assert np.array_equal(vec, expected)


def test_dichloromethane_is_2x():
    # ClCCl (CH2Cl2): un carbono con 2 vecinos no-carbono (ambos Cl) -> C-2X (idx17).
    vec = sc.true_vector_from_smiles("ClCCl")
    expected = np.zeros(19, dtype=np.int32)
    expected[17] = 1   # C-2X
    assert np.array_equal(vec, expected)


def test_chloroform_is_3x():
    # ClC(Cl)Cl (CHCl3): 3 vecinos no-carbono -> C-3X (idx18).
    vec = sc.true_vector_from_smiles("ClC(Cl)Cl")
    expected = np.zeros(19, dtype=np.int32)
    expected[18] = 1   # C-3X
    assert np.array_equal(vec, expected)


def test_amine_carbon_is_n_class():
    # CCN: CH3 (idx0) + CH2-N (idx9).
    vec = sc.true_vector_from_smiles("CCN")
    expected = np.zeros(19, dtype=np.int32)
    expected[0] = 1   # CH3
    expected[9] = 1   # CH2-N
    assert np.array_equal(vec, expected)


def test_invalid_smiles_raises():
    try:
        sc.true_vector_from_smiles("no-es-un-smiles(((")
    except ValueError:
        return
    raise AssertionError("se esperaba ValueError ante SMILES invalido")


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n = 0
    for fn in fns:
        try:
            fn()
            n += 1
        except Exception:
            print(f"FALLO: {fn.__name__}")
            traceback.print_exc()
            sys.exit(1)
    print(f">>> {n} TESTS OK <<<")
