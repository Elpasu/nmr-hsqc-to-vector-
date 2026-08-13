# coding: ascii
"""
Oraculo / post-procesamiento de conteos (Exp E Fase 3). numpy PURO (sin torch),
para que la logica sea testeable localmente y sea FUENTE UNICA de verdad,
importada por evaluate.py y dump_predictions.py (evita copias divergentes).

Modos:
  - crude_predict            : floor con clip a >=0 (EMA CRUDA).
  - ajustar_conteo_doble_exacto (v1): fuerza sum(pred)==total y
    sum(pred[IDX_CH2])==ch2.
  - ajustar_conteo_hetero (v2)      : v1 + zeroing airtight por ausencia de
    heteroatomo (N==0, O==0, N+O<2 -> C-2X, N+O<3 -> C-3X). Ver
    docs/superpowers/specs/2026-07-23-oraculo-v2-heteroatomos-design.md.
"""
import numpy as np

N_CLASSES = 19
IDX_CH2 = [1, 5, 9, 12]        # CH2, CH2-O, CH2-N, =CH2

# Clases que REQUIEREN el heteroatomo indicado (derivado de Gen_vector.py):
IDX_N = [8, 9, 10, 11, 16]     # CH3-N, CH2-N, CH-N, Cq-N, Imina
IDX_O = [4, 5, 6, 7, 15]       # CH3-O, CH2-O, CH-O, Cq-O, Aldeh
IDX_2X = 17                    # C-2X : sp3 con 2 heteroatomos (N+O >= 2)
IDX_3X = 18                    # C-3X : sp3 con >=3 heteroatomos (N+O >= 3)


def crude_predict(pred_cruda):
    """Modo CRUDO: floor con clip a >=0. Ignora el condicionante por completo."""
    return np.clip(np.floor(pred_cruda), 0, None).astype(int)


def _add_counts(pred_int, restos, faltantes, cand, robust):
    """Suma `faltantes` (>0) incrementos entre `cand`, por resto decimal
    descendente. robust=False replica el v1 original (+1 a las `faltantes`
    clases de mayor resto; si faltantes>len(cand) reparte a lo sumo len(cand)).
    robust=True (v2) hace round-robin para garantizar repartir TODO el faltante
    aun con pocos candidatos (evita que la suma no cierre al prohibir clases)."""
    if not cand or faltantes <= 0:
        return
    order = sorted(cand, key=lambda i: restos[i], reverse=True)
    if robust:
        for j in range(faltantes):
            pred_int[order[j % len(order)]] += 1
    else:
        for i in order[:faltantes]:
            pred_int[i] += 1


def ajustar_conteo_doble_exacto(pred_cruda, total_real, ch2_real, forbidden=None):
    """Modo ASISTIDO v1 (oraculo doble): fuerza sum(pred)==total_real y
    sum(pred[IDX_CH2])==ch2_real, repartiendo por el resto decimal.

    forbidden: set de indices que NO pueden recibir incrementos (los usa v2 para
    que una clase anulada por ausencia de heteroatomo no vuelva a poblarse al
    rellenar el cupo). Con forbidden=None/vacio el comportamiento es identico al
    v1 original (backward-compatible)."""
    forbidden = set() if forbidden is None else set(forbidden)
    robust = len(forbidden) > 0
    pred_int = np.floor(pred_cruda).astype(int)
    restos = pred_cruda - pred_int
    idx_resto = [i for i in range(N_CLASSES) if i not in IDX_CH2]

    # Candidatos para incrementar: excluyen las clases prohibidas.
    ch2_cand = [i for i in IDX_CH2 if i not in forbidden]
    resto_cand = [i for i in idx_resto if i not in forbidden]

    ch2_asignados = sum(pred_int[i] for i in IDX_CH2)
    ch2_faltantes = int(ch2_real - ch2_asignados)
    if ch2_faltantes > 0:
        _add_counts(pred_int, restos, ch2_faltantes, ch2_cand, robust)
    elif ch2_faltantes < 0:
        sobran = abs(ch2_faltantes)
        for i in sorted(IDX_CH2, key=lambda i: restos[i]):
            if pred_int[i] > 0:
                pred_int[i] -= 1
                sobran -= 1
                if sobran == 0:
                    break

    resto_real = total_real - ch2_real
    resto_asignados = sum(pred_int[i] for i in idx_resto)
    resto_faltantes = int(resto_real - resto_asignados)
    if resto_faltantes > 0:
        _add_counts(pred_int, restos, resto_faltantes, resto_cand, robust)
    elif resto_faltantes < 0:
        sobran = abs(resto_faltantes)
        for i in sorted(idx_resto, key=lambda i: restos[i]):
            if pred_int[i] > 0:
                pred_int[i] -= 1
                sobran -= 1
                if sobran == 0:
                    break

    return pred_int


def ajustar_conteo_hetero(pred_cruda, total_real, ch2_real, n_atoms, o_atoms):
    """Modo ASISTIDO v2: zeroing airtight por ausencia de heteroatomo antes del
    ajuste de doble restriccion. Si el elemento vale 0 en la FM, todas las clases
    que lo requieren son 0 (condicion necesaria exacta). La masa liberada se
    reparte automaticamente entre las clases permitidas por el algoritmo v1.

    No muta pred_cruda (trabaja sobre una copia)."""
    pred = np.array(pred_cruda, dtype=np.float64).copy()
    forbidden = set()
    if n_atoms == 0:
        forbidden.update(IDX_N)
    if o_atoms == 0:
        forbidden.update(IDX_O)
    if n_atoms + o_atoms < 2:
        forbidden.add(IDX_2X)
    if n_atoms + o_atoms < 3:
        forbidden.add(IDX_3X)
    # Zerea el pred_raw (mata el floor) y ademas excluye del relleno (que una
    # clase del cupo CH2, como CH2-N/CH2-O, no vuelva a poblarse).
    for i in forbidden:
        pred[i] = 0.0
    return ajustar_conteo_doble_exacto(pred, total_real, ch2_real, forbidden=forbidden)
