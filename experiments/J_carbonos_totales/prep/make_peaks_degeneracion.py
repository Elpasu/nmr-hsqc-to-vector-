# coding: ascii
"""make_peaks_degeneracion.py -- Exp J: crosspeaks con una 5a feature, la
DEGENERACION de cada senal (cuantos carbonos la comparten).

Es exactamente el pipeline de Fase 1b (extract_peaks_pkl.py) con un solo
cambio: donde _dedupe_symmetric_peaks DESCARTABA los picos con el mismo
(delta_c, delta_h), aca se los CUENTA. El dato ya estaba y se tiraba.

Experimentalmente esa degeneracion es lo que se lee de la integracion del 1H:
integracion_en_H = degeneracion x H_por_carbono. Para el benceno, 1 senal con
integracion 6H -> 6 carbonos equivalentes.

Uso:
    python make_peaks_degeneracion.py --config config_prep.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem, RDLogger

# Se reusa la maquinaria ya probada de Fase 1b sin tocarla.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "experiments" / "E_peaks_prep"))

from ch_connectivity import get_ch_connectivity_with_multiplicity  # noqa: E402
from extract_peaks_pkl import verify_smiles_alignment  # noqa: E402

RDLogger.DisableLog('rdApp.*')

N_FEATURES = 5   # delta_c, delta_h, amp_ch0, amp_ch1, degeneracion


def agrupar_con_degeneracion(peaks):
    """peaks: lista de (delta_c, delta_h, amp_ch0, amp_ch1). Agrupa por
    (delta_c, delta_h) redondeado a 6 decimales -- IDENTICO al criterio de
    _dedupe_symmetric_peaks de Fase 1b -- pero en vez de descartar los
    duplicados devuelve el tamano del grupo como 5a feature.

    Conserva el orden de aparicion del primer pico de cada grupo, igual que
    Fase 1b, para que los dos .npz sean comparables fila por fila.

    Nota: el agrupamiento es por COINCIDENCIA DE SHIFT, no por simetria de
    RDKit. Dos carbonos distintos con shifts accidentalmente iguales cuentan
    como degeneracion 2 -- y eso es lo correcto: en un espectro real esa
    coincidencia es indistinguible de la simetria (se ve una sola senal con el
    doble de integral). Es la colision del 2.19% ya documentada en Fase 1b."""
    orden = []
    grupos = {}
    for peak in peaks:
        clave = (round(peak[0], 6), round(peak[1], 6))
        if clave not in grupos:
            grupos[clave] = [peak, 0]
            orden.append(clave)
        grupos[clave][1] += 1
    return [tuple(grupos[c][0]) + (float(grupos[c][1]),) for c in orden]


def extract_peaks_deg_from_pkl_molecule(smiles, nmr_shifts):
    """Copia de extract_peaks_from_pkl_molecule (Fase 1b) con agrupacion que
    cuenta. smiles: str. nmr_shifts: dict {atom_idx: shift}, indices POST
    AddHs. Devuelve lista de (delta_c, delta_h, amp_ch0, amp_ch1, degeneracion)."""
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
        h_shifts = [nmr_shifts[h] for h in group["h_idxs"] if h in nmr_shifts]
        if not h_shifts:
            continue
        delta_c = float(nmr_shifts[c_idx])
        delta_h = float(sum(h_shifts) / len(h_shifts))
        mult = group["mult"]
        phase = -1.0 if mult == 2 else 1.0
        peaks.append((delta_c, delta_h, phase * float(mult), float(mult) / 3.0))
    return agrupar_con_degeneracion(peaks)


def build_padded_arrays_n(peaks_per_molecule, n_features):
    """Version generalizada de build_padded_arrays (Fase 1) que no hardcodea 4
    columnas. Devuelve (peaks (N, max, n_features) float32, mask (N, max) bool)."""
    n = len(peaks_per_molecule)
    max_peaks = max((len(p) for p in peaks_per_molecule), default=0)
    peaks_array = np.zeros((n, max_peaks, n_features), dtype=np.float32)
    mask_array = np.zeros((n, max_peaks), dtype=bool)
    for i, peaks in enumerate(peaks_per_molecule):
        for j, peak in enumerate(peaks):
            peaks_array[i, j] = peak
            mask_array[i, j] = True
    return peaks_array, mask_array


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    base_144 = Path(p["base_dir_144k"])
    base_58 = Path(p["base_dir_58k"])
    base_202 = Path(p["base_dir_202k"])

    print("=" * 62)
    print("  EXP J: crosspeaks con degeneracion (5a feature)")
    print("=" * 62)

    smiles_144 = np.load(base_144 / p["smiles_144k"], allow_pickle=True)
    mol_ids_144 = np.load(base_144 / p["mol_ids_144k"], allow_pickle=True)
    smiles_58 = np.load(base_58 / p["smiles_58k"], allow_pickle=True)
    mol_ids_58 = np.load(base_58 / p["mol_ids_58k"], allow_pickle=True)
    smiles_real = np.load(base_202 / p["smiles_202465"], allow_pickle=True)

    smiles_local = np.concatenate([smiles_144, smiles_58])
    mol_ids_local = np.concatenate([mol_ids_144, mol_ids_58])
    print(f"-> Moleculas locales: {len(smiles_local)} | reales: {len(smiles_real)}")

    ok, idx_malo = verify_smiles_alignment(smiles_local, smiles_real)
    if not ok:
        print(f"[ABORT] desajuste de alineacion (idx={idx_malo}). Los picos"
              f" quedarian pegados a la molecula equivocada.")
        sys.exit(1)
    print("[OK] alineacion verificada: SMILES canonicos coinciden fila por fila")

    with open(base_144 / p["pkl_144k"], "rb") as f:
        pkl_144 = pickle.load(f)
    with open(base_58 / p["pkl_58k"], "rb") as f:
        pkl_58 = pickle.load(f)

    n_total = len(smiles_local)
    n_144 = len(smiles_144)
    peaks_per_molecule = []
    for i in range(n_total):
        pkl = pkl_144 if i < n_144 else pkl_58
        shifts = pkl.get(str(mol_ids_local[i]), {})
        peaks_per_molecule.append(
            extract_peaks_deg_from_pkl_molecule(str(smiles_local[i]), shifts))
        if (i + 1) % 25000 == 0:
            print(f"   procesadas {i + 1}/{n_total}")

    peaks_array, mask_array = build_padded_arrays_n(peaks_per_molecule, N_FEATURES)
    n_picos = mask_array.sum(axis=1)
    deg = peaks_array[:, :, 4]
    deg_validas = deg[mask_array]

    print(f"\n-> shape: {peaks_array.shape}")
    print(f"-> picos por molecula: min={n_picos.min()} max={n_picos.max()} "
          f"promedio={n_picos.mean():.2f}")
    print(f"-> degeneracion: min={deg_validas.min():.0f} max={deg_validas.max():.0f} "
          f"promedio={deg_validas.mean():.2f}")
    print(f"-> senales con degeneracion > 1: "
          f"{100.0 * (deg_validas > 1).mean():.1f}%")

    # Validacion: Sum(degeneracion x mult) <= H sobre carbono. Igualdad solo si
    # el pkl tiene shift para TODOS los H (ver spec 9.1.4): un pkl incompleto da
    # estrictamente menos, y eso no es un test que haya que relajar sino un dato
    # sobre la calidad de los datos.
    mult = np.rint(peaks_array[:, :, 3] * 3.0)
    h_reconstruidos = (deg * mult * mask_array).sum(axis=1)

    h_reales = np.zeros(n_total, dtype=np.float64)
    for i, s in enumerate(smiles_local):
        mol = Chem.MolFromSmiles(str(s))       # una sola vez por molecula:
        if mol is None:                        # parsear dos veces sobre 202465
            continue                           # cuesta varios minutos de mas
        mol = Chem.AddHs(mol)
        h_reales[i] = sum(
            1 for a in mol.GetAtoms()
            if a.GetAtomicNum() == 1
            and any(nb.GetAtomicNum() == 6 for nb in a.GetNeighbors()))
    n_igual = int((h_reconstruidos == h_reales).sum())
    n_exceso = int((h_reconstruidos > h_reales).sum())
    print(f"\n-> Sum(degeneracion x mult) == H sobre carbono: "
          f"{n_igual}/{n_total} ({100.0 * n_igual / n_total:.2f}%)")
    print(f"-> casos con EXCESO (imposible, seria un bug): {n_exceso}")
    if n_exceso > 0:
        print("[ABORT] la degeneracion reconstruye MAS H de los que tiene la"
              " molecula: hay un error en el agrupamiento.")
        sys.exit(1)

    out = base_202 / p["peaks_deg_output"]
    np.savez(out, peaks=peaks_array, peaks_mask=mask_array)
    print(f"\n[SAVE] {out}")
    print(">>> EXP J make_peaks_degeneracion.py OK <<<")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exp J: crosspeaks con degeneracion")
    ap.add_argument("--config", type=str, default="config_prep.yaml")
    args = ap.parse_args()
    main(args.config)
