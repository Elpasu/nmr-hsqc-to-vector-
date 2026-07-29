# coding: utf-8
"""Exp H -- puerto fiel de Gen_vector.py (clasificador de 19 clases de carbono).

Portado desde E:\\Proyectos\\SciTrix\\MYGEN\\HSQC_a_Vector\\Gen_vector.py (fuente
de verdad de las labels de entrenamiento del dataset 202k). classify_carbon() es
COPIA EXACTA de la logica original -- NO modificar sin verificar contra el
original: un error aca corrompe el ground truth en silencio (regla dura 7).

Nota de fidelidad: nX (heteroatomo) cuenta CUALQUIER vecino no-carbono, incluidos
halogenos y S, no solo N/O. El dataset de entrenamiento es CHON puro asi que esa
rama nunca se ejerce en la practica, pero el clasificador -tal como esta escrito
en el original- no lo restringe. Se porta tal cual.

Diferencia deliberada respecto del original: Gen_vector.py devuelve un vector de
ceros ante un SMILES invalido (pensado para procesar un dataset completo sin
cortar el batch). Para inferencia interactiva un vector-cero silencioso
confundiria "SMILES con typo" con "molecula sin carbonos" -- aca se levanta
ValueError.
"""
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')  # silencia el parse-error esperado de SMILES invalidos

CATEGORIES = [
    "CH3_sp3_normal", "CH2_sp3_normal", "CH_sp3_normal", "Cq_sp3_normal",
    "CH3_sp3_O", "CH2_sp3_O", "CH_sp3_O", "Cq_sp3_O",
    "CH3_sp3_N", "CH2_sp3_N", "CH_sp3_N", "Cq_sp3_N",
    "CH2_sp2_normal", "CH_sp2_normal", "Cq_sp2_normal",
    "CH_sp2_Aldeh", "CH_sp2_Imina",
    "C_sp3_2X", "C_sp3_3X",
]
SHORT_NAMES = [
    "CH3", "CH2", "CH", "Cq",
    "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N",
    "=CH2", "=CH/Ar", "Cqsp2",
    "Aldeh", "Imina",
    "C-2X", "C-3X",
]
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
N_CLASSES = len(CATEGORIES)


def classify_carbon(atom):
    """Copia exacta de Gen_vector.classify_carbon. Devuelve el nombre largo de
    categoria (ver CATEGORIES)."""
    nH = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'H')
    heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() != 'H']
    nX = sum(1 for n in heavy_neighbors if n.GetAtomicNum() != 6)

    if nH >= 3:
        ch = "CH3"
    elif nH == 2:
        ch = "CH2"
    elif nH == 1:
        ch = "CH"
    else:
        ch = "Cq"

    sp2 = atom.GetHybridization() in (
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP,
    )

    if sp2:
        if nH >= 1:
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() >= 2.0:
                    other = bond.GetOtherAtom(atom)
                    if other.GetSymbol() == 'O':
                        return "CH_sp2_Aldeh"
                    if other.GetSymbol() == 'N':
                        return "CH_sp2_Imina"
            return "CH2_sp2_normal" if nH >= 2 else "CH_sp2_normal"
        else:
            return "Cq_sp2_normal"

    if nX == 2:
        return "C_sp3_2X"
    elif nX >= 3:
        return "C_sp3_3X"
    else:
        has_O = any(n.GetSymbol() == 'O' for n in heavy_neighbors)
        has_N = any(n.GetSymbol() == 'N' for n in heavy_neighbors)
        if has_O:
            return f"{ch}_sp3_O"
        elif has_N:
            return f"{ch}_sp3_N"
        else:
            return f"{ch}_sp3_normal"


def true_vector_from_smiles(smiles):
    """Puerto fiel de Gen_vector.nmr13C_vector_with_symmetry. Devuelve (19,)
    np.int32, en el orden de SHORT_NAMES (== classes_19v de db.yaml, verificado
    por test). Lanza ValueError ante SMILES invalido (ver nota de fidelidad)."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"SMILES invalido: {smiles!r}")
    mol = Chem.AddHs(mol)
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    vec = np.zeros(N_CLASSES, dtype=np.int32)
    seen = set()
    for atom, rank in zip(mol.GetAtoms(), ranks):
        if atom.GetSymbol() != 'C':
            continue
        if rank in seen:
            continue
        seen.add(rank)
        idx = CAT_INDEX.get(classify_carbon(atom))
        if idx is not None:
            vec[idx] += 1
    return vec
