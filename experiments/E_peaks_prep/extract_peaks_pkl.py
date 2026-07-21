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


def _dedupe_symmetric_peaks(peaks):
    """Colapsa picos con (delta_c, delta_h) practicamente identicos (mismos
    hasta 6 decimales) a uno solo. Corresponden a carbonos quimicamente
    equivalentes por simetria molecular (ej. las dos posiciones orto de un
    anillo para-sustituido) -- el calculo DFT les asigna el mismo shift, y
    en un HSQC real son indistinguibles (una sola senal), por eso el label
    de 19 clases los cuenta una vez. Mantiene el primer pico de cada grupo.
    Confirmado con datos reales de produccion: sin este paso, el conteo de
    picos superaba sistematicamente al conteo visible del label en ~61% de
    las 202465 moleculas (exceso, no colision)."""
    seen = set()
    deduped = []
    for peak in peaks:
        key = (round(peak[0], 6), round(peak[1], 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(peak)
    return deduped


def extract_peaks_from_pkl_molecule(smiles, nmr_shifts):
    """smiles: str. nmr_shifts: dict {atom_idx: float shift}, con indices de
    atomo POST AddHs. Devuelve lista de (delta_c, delta_h, amp_ch0, amp_ch1),
    un elemento por CARBONO CON ENTORNO QUIMICO DISTINTO (carbonos
    equivalentes por simetria con shift identico colapsan a un solo pico,
    ver _dedupe_symmetric_peaks) con al menos un H con shift conocido."""
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
    return _dedupe_symmetric_peaks(peaks)


def canonicalize_smiles(smiles_array):
    """Copiada tal cual de experiments/D_val_congelado/split.py -- misma
    logica ya probada en Exp D, no reinventar. Canonicaliza con RDKit.
    SMILES invalidos se conservan tal cual (no se descarta ninguna molecula)."""
    canonical = []
    n_invalid = 0
    for smi in smiles_array:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            canonical.append(str(smi))
            n_invalid += 1
        else:
            canonical.append(Chem.MolToSmiles(mol))
    return np.array(canonical, dtype=object), n_invalid


def verify_smiles_alignment(local_smiles, real_smiles):
    """local_smiles, real_smiles: arrays de SMILES. Canonicaliza ambos y
    compara posicion por posicion. Devuelve (ok, primer_indice_en_conflicto)
    -- indice es None si el desajuste es de longitud, no posicional."""
    if len(local_smiles) != len(real_smiles):
        return False, None
    local_canonical, _ = canonicalize_smiles(local_smiles)
    real_canonical, _ = canonicalize_smiles(real_smiles)
    for i in range(len(local_canonical)):
        if local_canonical[i] != real_canonical[i]:
            return False, i
    return True, None


def main(config_path):
    import sys as _sys
    import pickle

    import yaml

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_peaks import build_padded_arrays
    from validate_peaks import blob_counts_from_mask, validation_report, visible_label_counts

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir_144k = Path(cfg["paths"]["base_dir_144k"])
    base_dir_58k = Path(cfg["paths"]["base_dir_58k"])
    base_dir_202k = Path(cfg["paths"]["base_dir_202k"])
    class_names = cfg["classes_19v"]

    print("=" * 60)
    print("  EXP E FASE 1b: picos desde el pkl original")
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
    print(f"-> Moleculas en smiles_202465 real: {len(smiles_real)}")

    ok, mismatch_idx = verify_smiles_alignment(smiles_local, smiles_real)
    if not ok:
        if mismatch_idx is None:
            print("[ERROR] longitudes distintas entre smiles_local y smiles_202465 -- abortando")
        else:
            print(f"[ERROR] desajuste de alineacion en indice {mismatch_idx}")
            print(f"  local: {smiles_local[mismatch_idx]!r}")
            print(f"  real:  {smiles_real[mismatch_idx]!r}")
        return
    print("[OK] alineacion verificada: SMILES canonicos coinciden posicion por posicion")

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
        peaks_per_molecule.append(extract_peaks_from_pkl_molecule(smiles, nmr_shifts))
        if (i + 1) % 20000 == 0:
            print(f"   procesadas {i + 1}/{n_total}")

    peaks_array, mask_array = build_padded_arrays(peaks_per_molecule)
    n_counts = mask_array.sum(axis=1)
    print(f"-> max_peaks detectado: {peaks_array.shape[1]}")
    print(f"-> picos por molecula: min={n_counts.min()} max={n_counts.max()} "
          f"promedio={n_counts.mean():.2f}")

    out_path = base_dir_202k / cfg["paths"]["peaks_output_filename"]
    np.savez(out_path, peaks=peaks_array, peaks_mask=mask_array)
    print(f"\n[SAVE] {out_path}")

    visible_counts = visible_label_counts(labels, class_names)
    blob_counts = blob_counts_from_mask(mask_array)
    report = validation_report(blob_counts, visible_counts)

    print(f"\nMoleculas evaluadas: {report['n']}")
    print(f"Match exacto (picos == visible): {report['pct_exact_match']:.2f}%")
    print(f"Con colision (visible > picos): {report['n_collision']} "
          f"({report['pct_collision']:.2f}%)")
    print(f"Deficit promedio en las que tienen colision: "
          f"{report['mean_deficit_positive']:.2f}")

    print(">>> EXP E FASE 1b extract_peaks_pkl.py OK <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp E Fase 1b: picos desde el pkl original")
    parser.add_argument("--config", type=str, default="config_pkl.yaml")
    args = parser.parse_args()
    main(args.config)
