# coding: utf-8
"""Exp H -- adaptador de picos experimentales al formato de entrada del E3.

numpy puro (sin torch, sin GUI): testeable local. Convierte FM + tabla de picos
a los tensores exactos que consume model_e3_settransformer, con la MISMA
normalizacion y las MISMAS amplitudes que extract_peaks_pkl.py (Fase 1b).
"""
import os
import re

import numpy as np
import yaml

MULT_H = {"CH3": 3, "CH2": 2, "CH": 1, "Cq": 0}
CH2_CLASS_IDX = [1, 5, 9, 12]
_HALOGENS = {"F", "Cl", "Br", "I"}


def parse_formula(formula):
    """'C10H12N2O' -> {'C':10,'H':12,'N':2,'O':1,'S':0,'Hal':0}. Elementos
    ausentes = 0; digito implicito = 1. Hal = F+Cl+Br+I."""
    counts = {"C": 0, "H": 0, "N": 0, "O": 0, "S": 0, "Hal": 0}
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        if elem == "":
            continue
        n = int(num) if num else 1
        if elem in _HALOGENS:
            counts["Hal"] += n
        elif elem in counts:
            counts[elem] += n
    return counts


def build_inputs(peaks, formula, norm_cfg):
    """peaks: lista de {delta_c, delta_h|None, mult}. formula: dict de parse_formula.
    Devuelve (peaks_ch, mask_ch, peaks_13c, mask_13c, cond), todos np.float32.
    Sin padding (batch=1): mascaras todo-1 sobre los picos reales."""
    c_min, c_max = float(norm_cfg["c13_ppm_min"]), float(norm_cfg["c13_ppm_max"])
    h_min, h_max = float(norm_cfg["h1_ppm_min"]), float(norm_cfg["h1_ppm_max"])
    amp0_scale = float(norm_cfg["amp_ch0_scale"])

    ch_rows, c13_rows = [], []
    total_ch2 = 0
    for p in peaks:
        mult = MULT_H[p["mult"]]
        dc = float(p["delta_c"])
        c13_rows.append([(dc - c_min) / (c_max - c_min)])
        if mult == 0:
            continue                       # Cq: sin crosspeak
        if p["mult"] == "CH2":
            total_ch2 += 1
        phase = -1.0 if mult == 2 else 1.0
        amp_ch0 = phase * mult
        amp_ch1 = mult / 3.0
        dh = float(p["delta_h"])
        ch_rows.append([
            (dc - c_min) / (c_max - c_min),
            (dh - h_min) / (h_max - h_min),
            amp_ch0 / amp0_scale,
            amp_ch1,
        ])

    peaks_ch = np.asarray(ch_rows, dtype=np.float32).reshape(-1, 4)
    peaks_13c = np.asarray(c13_rows, dtype=np.float32).reshape(-1, 1)
    mask_ch = np.ones(peaks_ch.shape[0], dtype=np.float32)
    mask_13c = np.ones(peaks_13c.shape[0], dtype=np.float32)

    cond = np.array([
        float(len(peaks)),          # total_senales = nro de picos 13C
        float(total_ch2),           # total_CH2
        formula["C"], formula["H"], formula["N"],
        formula["O"], formula["S"], formula["Hal"],
    ], dtype=np.float32)
    return peaks_ch, mask_ch, peaks_13c, mask_13c, cond


def true_vector(peaks, class_names):
    """Histograma de peak['clase'] sobre las 19 clases (solo modo evaluacion)."""
    idx = {name: i for i, name in enumerate(class_names)}
    vec = np.zeros(len(class_names), dtype=int)
    for p in peaks:
        if "clase" not in p or p["clase"] in (None, ""):
            raise ValueError("true_vector requiere 'clase' en toda fila")
        vec[idx[p["clase"]]] += 1
    return vec


def load_configs(repo_root):
    """Lee nombres de clase (db.yaml) y normalizacion (config del E3)."""
    with open(os.path.join(repo_root, "config", "db.yaml"), "r",
              encoding="utf-8") as f:
        class_names = yaml.safe_load(f)["classes_19v"]
    cfg_path = os.path.join(repo_root, "experiments", "E3_dos_conjuntos",
                            "config_settransformer.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        norm_cfg = yaml.safe_load(f)["normalization"]
    return class_names, norm_cfg
