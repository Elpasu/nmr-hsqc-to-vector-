# coding: ascii
"""
blob_detect.py -- deteccion de picos HSQC via componentes conexos.

Un pico = un componente conexo (conectividad 8) de pixeles no-cero en el
canal 0 (DEPT). Si dos carbonos distintos caen tan cerca en (delta_C,
delta_H) que sus gaussianas se solapan, ndimage.label los funde en un solo
componente -- esa fusion es la "colision real" que valida Fase 1 (ver
validate_peaks.py). El centroide se pondera por |canal0| dentro del blob,
mas robusto que el centroide sin ponderar para blobs asimetricos/fusionados.
"""
import numpy as np
from scipy import ndimage

CONNECTIVITY_8 = np.ones((3, 3), dtype=int)


def detect_peaks(ch0, ch1):
    """ch0, ch1: arrays (H, W) float, canales de una sola molecula.
    Devuelve lista de (row_c, col_h, amp_ch0, amp_ch1) en coordenadas de
    pixel -- row_c/col_h son floats (centroide), amp_ch0/amp_ch1 son los
    valores de cada canal en el pixel entero mas cercano al centroide."""
    mask = ch0 != 0
    labeled, n_blobs = ndimage.label(mask, structure=CONNECTIVITY_8)
    if n_blobs == 0:
        return []

    indices = list(range(1, n_blobs + 1))
    centroids = ndimage.center_of_mass(np.abs(ch0), labeled, indices)

    h, w = ch0.shape
    peaks = []
    for row_c, col_h in centroids:
        r = min(max(int(round(row_c)), 0), h - 1)
        c = min(max(int(round(col_h)), 0), w - 1)
        peaks.append((float(row_c), float(col_h), float(ch0[r, c]), float(ch1[r, c])))
    return peaks
