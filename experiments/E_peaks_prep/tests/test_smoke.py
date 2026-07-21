# coding: ascii
"""
Smoke test OFFLINE de Exp E Fase 1 (rule 5 de CLAUDE.md) -- construye 3
moleculas sinteticas en memoria (sin h5py, sin cluster), corre el pipeline
completo extract_peaks_from_molecule -> build_padded_arrays -> reporte de
validacion, y confirma que las formas y los conteos son consistentes.

Requiere scipy (via blob_detect.py) -- correr en el cluster (login node)
antes de correr extract_peaks.py sobre el h5 completo:
    python tests/test_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from extract_peaks import build_padded_arrays, extract_peaks_from_molecule
from validate_peaks import blob_counts_from_mask, validation_report, visible_label_counts

CALIB = {"c13_ppm_min": 0, "c13_ppm_max": 220, "h1_ppm_min": -1, "h1_ppm_max": 15, "resolution": 256}

CLASS_NAMES = [
    "CH3", "CH2", "CH", "Cq", "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N", "=CH2", "=CH/Ar", "Cqsp2",
    "Aldeh", "Imina", "C-2X", "C-3X",
]


def _gaussian_blob(matrix, row, col, sigma=0.5, intensity=1.0):
    size = matrix.shape[0]
    radius = int(6 * sigma) + 1
    r0, r1 = max(0, row - radius), min(size, row + radius + 1)
    c0, c1 = max(0, col - radius), min(size, col + radius + 1)
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    g = intensity * np.exp(-0.5 * (((rr - row) / sigma) ** 2 + ((cc - col) / sigma) ** 2))
    matrix[r0:r1, c0:c1] += g
    return matrix


def _make_molecule(peak_specs):
    """peak_specs: lista de (row, col, ch0_intensity, ch1_intensity)."""
    hsqc = np.zeros((2, 256, 256), dtype=np.float32)
    for row, col, i0, i1 in peak_specs:
        _gaussian_blob(hsqc[0], row, col, intensity=i0)
        _gaussian_blob(hsqc[1], row, col, intensity=i1)
    return hsqc


def test_pipeline_end_to_end_three_molecules():
    # Molecula 0: 1 CH3 (2 picos separados, imitando 2 carbonos distintos)
    mol0 = _make_molecule([(30, 40, 1.0, 1.0), (100, 120, 1.0, 1.0)])
    # Molecula 1: 1 CH2 (fase negativa) + colision deliberada (2 carbonos
    # a 1px de distancia se funden en un blob)
    mol1 = _make_molecule([(80, 90, -2.0, 0.67), (150, 150, 1.0, 1.0), (151, 151, 1.0, 1.0)])
    # Molecula 2: sin picos (imagen vacia)
    mol2 = np.zeros((2, 256, 256), dtype=np.float32)

    peaks_per_molecule = [
        extract_peaks_from_molecule(mol0, CALIB),
        extract_peaks_from_molecule(mol1, CALIB),
        extract_peaks_from_molecule(mol2, CALIB),
    ]
    assert len(peaks_per_molecule[0]) == 2
    assert len(peaks_per_molecule[1]) == 2  # 1 CH2 + 1 blob fusionado
    assert len(peaks_per_molecule[2]) == 0

    peaks_array, mask_array = build_padded_arrays(peaks_per_molecule)
    assert peaks_array.shape == (3, 2, 4)
    assert mask_array.tolist() == [[True, True], [True, True], [False, False]]

    # Validacion: labels sinteticos donde molecula 1 tiene 3 carbonos
    # visibles reales (el pipeline solo pudo recuperar 2 -> deficit=1)
    labels = np.zeros((3, 19), dtype=int)
    labels[0, CLASS_NAMES.index("CH3")] = 2
    labels[1, CLASS_NAMES.index("CH2")] = 1
    labels[1, CLASS_NAMES.index("CH")] = 2  # los 2 carbonos fusionados
    labels[2, CLASS_NAMES.index("Cq")] = 1  # invisible -> no cuenta

    visible_counts = visible_label_counts(labels, CLASS_NAMES)
    blob_counts = blob_counts_from_mask(mask_array)
    report = validation_report(blob_counts, visible_counts)

    assert report["n"] == 3
    assert report["n_collision"] == 1  # solo la molecula 1
    assert report["deficit"][1] == 1
    print(f"[OK] test_pipeline_end_to_end_three_molecules -> {report}")


if __name__ == "__main__":
    test_pipeline_end_to_end_three_molecules()
    print("\n>>> SMOKE EXP E FASE 1 OK - listo para correr extract_peaks.py sobre el h5 completo <<<")
