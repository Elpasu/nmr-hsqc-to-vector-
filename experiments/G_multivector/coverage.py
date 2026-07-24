# coding: utf-8
"""Metrica cobertura@K para Exp G (multi-vector). numpy puro.

cobertura@K = fraccion de moleculas con y_true dentro de los K candidatos.
total y ch2 se derivan de y_true (= condicionante del oraculo); N y O de la FM.

Uso (tras dumpear y_pred_raw):
  python coverage.py --parquet predictions_<exp>.parquet
"""
import argparse
import numpy as np
from oraculo import IDX_CH2
from candidates import generate_candidates, generate_candidates_uncertainty


def coverage_curve(y_true, y_pred_raw, n_atoms, o_atoms, Ks, max_swaps=2):
    y_true = np.asarray(y_true, dtype=int)
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)
    Kmax = max(Ks)
    M = len(y_true)
    hit_at = np.zeros(M, dtype=int)   # menor K que cubre (0 = no cubierta)
    n_emit = np.zeros(M, dtype=int)
    for m in range(M):
        total = int(y_true[m].sum())
        ch2 = int(sum(y_true[m, i] for i in IDX_CH2))
        cands = generate_candidates(y_pred_raw[m], total, ch2,
                                    int(n_atoms[m]), int(o_atoms[m]),
                                    K=Kmax, max_swaps=max_swaps)
        n_emit[m] = len(cands)
        for rank, c in enumerate(cands, start=1):
            if np.array_equal(c, y_true[m]):
                hit_at[m] = rank
                break
    res = {}
    for K in Ks:
        covered = (hit_at >= 1) & (hit_at <= K)
        emit_k = np.minimum(n_emit, K)
        res[K] = {
            "coverage": float(covered.mean() * 100),
            "k_mean": float(emit_k.mean()),
            "k_max": int(emit_k.max()),
        }
    return res


def coverage_uncertainty(y_true, y_pred_raw, n_atoms, o_atoms, tau, K_max, max_swaps=2):
    """Cobertura, K promedio y K maximo emitido para el generador guiado (Fase 1b)
    a un tau dado. total/ch2 se derivan de y_true."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)
    M = len(y_true)
    covered = np.zeros(M, dtype=bool)
    nemit = np.zeros(M, dtype=int)
    for m in range(M):
        total = int(y_true[m].sum())
        ch2 = int(sum(y_true[m, i] for i in IDX_CH2))
        cands = generate_candidates_uncertainty(
            y_pred_raw[m], total, ch2, int(n_atoms[m]), int(o_atoms[m]),
            tau, K_max, max_swaps=max_swaps)
        nemit[m] = len(cands)
        covered[m] = any(np.array_equal(c, y_true[m]) for c in cands)
    return {"coverage": float(covered.mean() * 100),
            "k_mean": float(nemit.mean()),
            "k_max": int(nemit.max())}


def _formulas(smiles):
    """Extrae N y O desde SMILES con rdkit.

    Caveat: si un SMILES no parsea (mol == None), N y O quedan en 0.
    Esto prohíbe las clases hetero de esa molécula y puede hacer que
    coverage@1 diverja levemente de EMA v2 (donde se usa la FM exacta).
    """
    from rdkit import Chem
    n = np.zeros(len(smiles), int); o = np.zeros(len(smiles), int)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            n[i] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
            o[i] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    return n, o


def main(parquet_path, Ks=(1, 2, 3, 4, 5), max_swaps=2):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    yt = np.vstack(df["y_true"].apply(lambda v: np.array(v, dtype=int)))
    raw = np.vstack(df["y_pred_raw"].apply(lambda v: np.array(v, dtype=float)))
    n, o = _formulas(df["smiles"].tolist())
    res = coverage_curve(yt, raw, n, o, list(Ks), max_swaps=max_swaps)
    print("=" * 56)
    print("  COBERTURA@K (Exp G multi-vector) -- val congelado")
    print("=" * 56)
    print(f"  moleculas: {len(df)}   max_swaps: {max_swaps}")
    print(f"\n  {'K':>3}  {'cobertura':>10}  {'K prom emit':>12}  {'K max emit':>11}")
    print("  " + "-" * 44)
    for K in Ks:
        r = res[K]
        print(f"  {K:>3}  {r['coverage']:>9.2f}%  {r['k_mean']:>12.2f}  {r['k_max']:>11d}")
    return res


def main_uncertainty(parquet_path, taus=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0), K_max=6):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    yt = np.vstack(df["y_true"].apply(lambda v: np.array(v, dtype=int)))
    raw = np.vstack(df["y_pred_raw"].apply(lambda v: np.array(v, dtype=float)))
    n, o = _formulas(df["smiles"].tolist())
    print("=" * 60)
    print("  COBERTURA guiada por incertidumbre (Exp G Fase 1b) -- val congelado")
    print("=" * 60)
    print(f"  moleculas: {len(df)}   K_max: {K_max}")
    print(f"\n  {'tau':>5}  {'cobertura':>10}  {'K prom emit':>12}  {'K max emit':>11}")
    print("  " + "-" * 46)
    res_all = {}
    for tau in taus:
        r = coverage_uncertainty(yt, raw, n, o, tau, K_max)
        res_all[tau] = r
        print(f"  {tau:>5.2f}  {r['coverage']:>9.2f}%  {r['k_mean']:>12.2f}  {r['k_max']:>11d}")
    return res_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cobertura@K Exp G")
    ap.add_argument("--parquet", required=True, help="parquet con y_true, y_pred_raw, smiles")
    ap.add_argument("--max-swaps", type=int, default=2)
    ap.add_argument("--uncertainty", action="store_true",
                    help="Corre el barrido de tau (Fase 1b) en vez de la curva de K (Fase 1).")
    args = ap.parse_args()
    if args.uncertainty:
        main_uncertainty(args.parquet, K_max=6)
    else:
        main(args.parquet, max_swaps=args.max_swaps)
