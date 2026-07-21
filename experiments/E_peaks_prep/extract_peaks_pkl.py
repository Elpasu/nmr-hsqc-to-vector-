# coding: ascii
"""
extract_peaks_pkl.py -- Exp E Fase 1b: extrae picos HSQC directamente de los
shifts DFT del pkl original (sin pasar por la imagen 256x256), agrupando por
CARBONO (no por par C-H) para que el conteo sea comparable con
visible_label_count.

Corre LOCAL en la maquina de Lucas (numpy + rdkit disponibles). Reutiliza
build_padded_arrays de extract_peaks.py y las funciones de validate_peaks.py
(ambos de Fase 1, en esta misma carpeta) sin modificarlas.

Uso:
    python extract_peaks_pkl.py --config config_pkl.yaml
"""
import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem

from ch_connectivity import get_ch_connectivity_with_multiplicity


def extract_peaks_from_pkl_molecule(smiles, nmr_shifts):
    """smiles: str. nmr_shifts: dict {atom_idx: float shift}, con indices de
    atomo POST AddHs. Devuelve lista de (delta_c, delta_h, amp_ch0, amp_ch1),
    un elemento por carbono con al menos un H con shift conocido."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)
    ch_pairs = get_ch_connectivity_with_multiplicity(mol)

    groups = {}
    for pair in ch_pairs:
        c_idx = pair["c_idx"]
        if c_idx not in groups:
            groups[c_idx] = {"mult": pair["multiplicity"], "h_idxs": []}
        groups[c_idx]["h_idxs"].append(pair["h_idx"])

    peaks = []
    for c_idx, group in groups.items():
        if c_idx not in nmr_shifts:
            continue
        h_shifts = [nmr_shifts[h_idx] for h_idx in group["h_idxs"] if h_idx in nmr_shifts]
        if not h_shifts:
            continue
        delta_c = float(nmr_shifts[c_idx])
        delta_h = float(sum(h_shifts) / len(h_shifts))
        mult = group["mult"]
        phase = -1.0 if mult == 2 else 1.0
        amp_ch0 = phase * float(mult)
        amp_ch1 = float(mult) / 3.0
        peaks.append((delta_c, delta_h, amp_ch0, amp_ch1))
    return peaks
