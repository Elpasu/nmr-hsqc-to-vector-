# coding: ascii
"""Generador de candidatos multi-vector (Exp G, cobertura@K). numpy PURO.

Desde el ancla FM-consistente (oraculo v2), enumera movimientos unitarios
INTRA-grupo-de-nH. Como el grupo 2H es exactamente IDX_CH2, todo movimiento
intra-grupo preserva total, totales por grupo y cupo CH2 -> todo candidato es
FM-consistente por construccion. Se rankea el resto por distancia L1 al crudo.

Ver docs/superpowers/specs/2026-07-24-exp-g-multivector-coverage-design.md
"""
import numpy as np
from oraculo import IDX_N, IDX_O, IDX_2X, IDX_3X, ajustar_conteo_hetero

# Grupos por multiplicidad (nH). Derivados de Gen_vector.py.
NH_GROUPS = {
    3: [0, 4, 8],
    2: [1, 5, 9, 12],
    1: [2, 6, 10, 13, 15, 16],
    0: [3, 7, 11, 14, 17, 18],
}


def _forbidden_set(n_atoms, o_atoms):
    """Clases que la FM prohibe (mismas reglas que el oraculo v2)."""
    f = set()
    if n_atoms == 0:
        f.update(IDX_N)
    if o_atoms == 0:
        f.update(IDX_O)
    if n_atoms + o_atoms < 2:
        f.add(IDX_2X)
    if n_atoms + o_atoms < 3:
        f.add(IDX_3X)
    return f


def _intra_group_moves(vec, forbidden):
    """Vectores a UN movimiento unitario intra-grupo: resta 1 a un donante
    (conteo>0) y suma 1 a un receptor permitido del mismo grupo (a != b)."""
    out = []
    for group in NH_GROUPS.values():
        allowed = [c for c in group if c not in forbidden]
        for a in group:
            if vec[a] <= 0:
                continue
            for b in allowed:
                if b == a:
                    continue
                nv = vec.copy()
                nv[a] -= 1
                nv[b] += 1
                out.append(nv)
    return out


def generate_candidates(raw, total, ch2, n_atoms, o_atoms, K, max_swaps=2):
    """Hasta K vectores int (largo 19), FM-consistentes. [0] = oraculo v2;
    el resto rankeado por sum|c-raw| ascendente; sin duplicados."""
    raw = np.asarray(raw, dtype=np.float64)
    anchor = ajustar_conteo_hetero(raw, total, ch2, n_atoms, o_atoms).astype(int)
    forbidden = _forbidden_set(n_atoms, o_atoms)

    seen = {anchor.tobytes()}
    all_vecs = [anchor]
    frontier = [anchor]
    for _ in range(max_swaps):
        nxt = []
        for v in frontier:
            for nv in _intra_group_moves(v, forbidden):
                key = nv.tobytes()
                if key not in seen:
                    seen.add(key)
                    all_vecs.append(nv)
                    nxt.append(nv)
        frontier = nxt
        if not frontier:
            break

    rest = sorted(all_vecs[1:], key=lambda c: float(np.abs(c - raw).sum()))
    return [anchor] + rest[:max(0, K - 1)]
