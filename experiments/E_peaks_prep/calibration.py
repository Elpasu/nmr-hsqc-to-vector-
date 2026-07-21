# coding: ascii
"""
calibration.py -- conversion pixel (bin) -> ppm real para el HSQC.

Inversa exacta del binning uniforme usado en Genera_mapas_de_pkl_v2.py
(ppm_to_bin_uniform). bin_idx puede ser int (pixel entero) o float
(centroide de un blob).
"""


def bin_to_ppm(bin_idx, ppm_min, ppm_max, resolution=256):
    return bin_idx / (resolution - 1) * (ppm_max - ppm_min) + ppm_min
