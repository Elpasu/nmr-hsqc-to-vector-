# coding: ascii
"""
extract_peaks_13c_pkl.py -- Exp E Fase 3: extrae el conjunto de picos 13C
desde el pkl DFT, un (delta_c,) por CARBONO con entorno quimico distinto
(TODOS los carbonos, con y sin H). A diferencia de extract_peaks_pkl.py
(Fase 1b, que arma crosspeaks C-H y descarta los carbonos sin H), aca los
cuaternarios (Cq, Cqsp2, Cq-O, Cq-N) SI entran -- son justamente los que el
HSQC no puede ver. Es el input que le faltaba al modelo en Fase 2.

Feature por pico: solo delta_c (posicion). NUNCA el numero de H del carbono
(eso es casi el label CH3/CH2/CH/Cq -> fuga).

Corre LOCAL en la maquina Windows de Lucas (numpy + rdkit). Reutiliza la
maquinaria de alineacion de extract_peaks_pkl.py sin modificarla.

Uso:
    python extract_peaks_13c_pkl.py --config config_pkl.yaml
"""
import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem

from extract_peaks_pkl import canonicalize_smiles, verify_smiles_alignment


def _dedupe_symmetric_13c(peaks):
    """Colapsa picos con delta_c identico (a 6 decimales) a uno solo.
    Carbonos equivalentes por simetria reciben el mismo shift DFT y el label
    de 19 clases los cuenta una vez (misma logica que _dedupe_symmetric_peaks
    de Fase 1b, pero sobre 1 sola feature)."""
    seen = set()
    out = []
    for p in peaks:
        key = round(p[0], 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def extract_13c_peaks_from_molecule(smiles, nmr_shifts):
    """smiles: str. nmr_shifts: dict {atom_idx: float shift}, indices POST
    AddHs. Devuelve lista de (delta_c,), un elemento por carbono con entorno
    quimico distinto (todos los carbonos que tengan shift en el pkl; los
    equivalentes por simetria colapsan)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)
    peaks = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        c_idx = atom.GetIdx()
        if c_idx not in nmr_shifts:
            continue
        peaks.append((float(nmr_shifts[c_idx]),))
    return _dedupe_symmetric_13c(peaks)


def build_padded_arrays_13c(peaks_per_molecule):
    """peaks_per_molecule: lista de N listas de tuplas de 1 float. Devuelve
    (peaks_array (N, max_peaks, 1) float32, mask_array (N, max_peaks) bool)."""
    n = len(peaks_per_molecule)
    max_peaks = max((len(p) for p in peaks_per_molecule), default=0)
    peaks_array = np.zeros((n, max_peaks, 1), dtype=np.float32)
    mask_array = np.zeros((n, max_peaks), dtype=bool)
    for i, peaks in enumerate(peaks_per_molecule):
        for j, peak in enumerate(peaks):
            peaks_array[i, j, 0] = peak[0]
            mask_array[i, j] = True
    return peaks_array, mask_array


def main(config_path):
    import pickle
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir_144k = Path(cfg["paths"]["base_dir_144k"])
    base_dir_58k = Path(cfg["paths"]["base_dir_58k"])
    base_dir_202k = Path(cfg["paths"]["base_dir_202k"])

    print("=" * 60)
    print("  EXP E FASE 3: picos 13C (todos los carbonos) desde el pkl")
    print("=" * 60)

    smiles_144 = np.load(base_dir_144k / cfg["paths"]["smiles_144k"], allow_pickle=True)
    mol_ids_144 = np.load(base_dir_144k / cfg["paths"]["mol_ids_144k"], allow_pickle=True)
    smiles_58 = np.load(base_dir_58k / cfg["paths"]["smiles_58k"], allow_pickle=True)
    mol_ids_58 = np.load(base_dir_58k / cfg["paths"]["mol_ids_58k"], allow_pickle=True)
    smiles_real = np.load(base_dir_202k / cfg["paths"]["smiles_202465"], allow_pickle=True)
    labels = np.load(base_dir_202k / cfg["paths"]["labels_202465"]).astype(int)

    smiles_local = np.concatenate([smiles_144, smiles_58])
    mol_ids_local = np.concatenate([mol_ids_144, mol_ids_58])

    print(f"-> Moleculas locales (144k+58k): {len(smiles_local)}")
    ok, mismatch_idx = verify_smiles_alignment(smiles_local, smiles_real)
    if not ok:
        print(f"[ERROR] desajuste de alineacion (idx={mismatch_idx}) -- abortando")
        return
    print("[OK] alineacion verificada")

    with open(base_dir_144k / cfg["paths"]["pkl_144k"], "rb") as f:
        pkl_144 = pickle.load(f)
    with open(base_dir_58k / cfg["paths"]["pkl_58k"], "rb") as f:
        pkl_58 = pickle.load(f)

    n_total = len(smiles_local)
    n_144 = len(smiles_144)
    peaks_per_molecule = []
    for i in range(n_total):
        smiles = str(smiles_local[i])
        mol_id = str(mol_ids_local[i])
        pkl = pkl_144 if i < n_144 else pkl_58
        nmr_shifts = pkl.get(mol_id, {})
        peaks_per_molecule.append(extract_13c_peaks_from_molecule(smiles, nmr_shifts))
        if (i + 1) % 20000 == 0:
            print(f"   procesadas {i + 1}/{n_total}")

    peaks_array, mask_array = build_padded_arrays_13c(peaks_per_molecule)
    n_counts = mask_array.sum(axis=1)
    print(f"-> max_peaks 13C: {peaks_array.shape[1]}")
    print(f"-> picos 13C por molecula: min={n_counts.min()} max={n_counts.max()} "
          f"promedio={n_counts.mean():.2f}")

    out_path = base_dir_202k / "peaks_13c_202465.npz"
    np.savez(out_path, peaks_13c=peaks_array, mask_13c=mask_array)
    print(f"\n[SAVE] {out_path}")

    # Validacion: #picos 13C vs total del label (TODOS los carbonos, incluidos
    # cuaternarios). Deberia dar ~100% (mucho mejor que el 97% de crosspeaks).
    total_label = labels.sum(axis=1).astype(int)
    deficit = total_label - n_counts.astype(int)
    n = len(deficit)
    pct_exact = int((deficit == 0).sum()) / n * 100.0
    n_coll = int((deficit > 0).sum())
    print(f"\nMoleculas evaluadas: {n}")
    print(f"Match exacto (picos_13C == total_label): {pct_exact:.2f}%")
    print(f"Con colision (total > picos): {n_coll} ({n_coll / n * 100:.2f}%)")
    if n_coll > 0:
        print(f"Deficit promedio en las que colisionan: {deficit[deficit > 0].mean():.2f}")
    print(">>> EXP E FASE 3 extract_peaks_13c_pkl.py OK <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp E Fase 3: picos 13C desde el pkl")
    parser.add_argument("--config", type=str, default="config_pkl.yaml")
    args = parser.parse_args()
    main(args.config)
