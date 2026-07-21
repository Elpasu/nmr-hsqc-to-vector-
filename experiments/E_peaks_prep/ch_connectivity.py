# coding: ascii
"""
ch_connectivity.py -- reconstruccion de conectividad C-H via RDKit, copiada
tal cual de Genera_mapas_de_pkl_v2.py (lineas 128-148, script original del
dataset, fuera de este repo) -- no modificar la logica, solo se traslado el
codigo para reutilizarlo en la extraccion de picos desde el pkl.
"""


def get_carbon_multiplicity(mol, c_idx):
    atom = mol.GetAtomWithIdx(c_idx)
    if atom.GetAtomicNum() != 6:
        return -1
    return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)


def get_ch_connectivity_with_multiplicity(mol):
    ch_pairs = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        n1, n2 = a1.GetAtomicNum(), a2.GetAtomicNum()
        if n1 == 6 and n2 == 1:
            c_idx, h_idx = a1.GetIdx(), a2.GetIdx()
        elif n1 == 1 and n2 == 6:
            c_idx, h_idx = a2.GetIdx(), a1.GetIdx()
        else:
            continue
        mult = get_carbon_multiplicity(mol, c_idx)
        ch_pairs.append({"c_idx": c_idx, "h_idx": h_idx, "multiplicity": mult})
    return ch_pairs
