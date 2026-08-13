# coding: ascii
"""make_labels_totales.py -- Exp J: genera el vector de CARBONOS TOTALES.

Diferencia unica respecto del esquema historico: no se colapsan los carbonos
equivalentes por simetria. classify_carbon es exactamente el mismo (puerto fiel
de Gen_vector.py que vive en experiments/H_inferencia_experimental), asi que la
clasificacion de cada carbono no cambia -- solo cuantas veces se cuenta.

Antes de escribir nada, main() corre un GATE: regenera los labels VIEJOS (con
colapso) y los compara contra vectors_13c_19v_202465.npy. Si no coinciden
exactamente, aborta. Sin ese gate, un puerto sutilmente distinto produciria un
ground truth nuevo corrupto sin tirar ningun error (regla dura 7).

Uso:
    python make_labels_totales.py --config config_prep.yaml
    python make_labels_totales.py --config config_prep.yaml --solo-verificar
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# El puerto fiel del clasificador vive en Exp H. Se IMPORTA en vez de copiarse:
# es la unica fuente de verdad de las 19 clases y una copia divergente
# corromperia el ground truth en silencio (regla dura 7).
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "experiments" / "H_inferencia_experimental"))

from smiles_classifier import CAT_INDEX, N_CLASSES, classify_carbon  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog('rdApp.*')


def _mol_con_h(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"SMILES invalido: {smiles!r}")
    return Chem.AddHs(mol)


def vector_carbonos_totales(smiles):
    """Vector de 19 clases contando TODOS los carbonos (sin colapso de
    simetria). sum(vector) == numero de carbonos de la formula."""
    mol = _mol_con_h(smiles)
    vec = np.zeros(N_CLASSES, dtype=np.int32)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C':
            continue
        idx = CAT_INDEX.get(classify_carbon(atom))
        if idx is not None:
            vec[idx] += 1
    return vec


def vector_con_simetria(smiles):
    """Vector del esquema HISTORICO: un carbono por clase de equivalencia
    (CanonicalRankAtoms con breakTies=False). Existe solo para el gate de
    verificacion -- es lo que tiene que reproducir vectors_13c_19v_202465.npy."""
    mol = _mol_con_h(smiles)
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    vec = np.zeros(N_CLASSES, dtype=np.int32)
    vistos = set()
    for atom, rank in zip(mol.GetAtoms(), ranks):
        if atom.GetSymbol() != 'C':
            continue
        if rank in vistos:
            continue
        vistos.add(rank)
        idx = CAT_INDEX.get(classify_carbon(atom))
        if idx is not None:
            vec[idx] += 1
    return vec


def verificar_puerto(smiles_array, labels_viejos, n_muestra=None):
    """GATE. Regenera los labels viejos con vector_con_simetria y los compara
    con los existentes. Devuelve (n_ok, n_total, primer_fallo) donde
    primer_fallo es None si no hubo ninguno, o (indice, smiles, viejo, nuevo)."""
    n = len(smiles_array)
    indices = range(n) if n_muestra is None else np.random.default_rng(7).choice(
        n, min(n_muestra, n), replace=False)
    n_ok = 0
    primer_fallo = None
    for i in indices:
        try:
            v = vector_con_simetria(smiles_array[i])
        except ValueError:
            v = np.zeros(N_CLASSES, dtype=np.int32)
        if np.array_equal(v, labels_viejos[i]):
            n_ok += 1
        elif primer_fallo is None:
            primer_fallo = (int(i), str(smiles_array[i]),
                            labels_viejos[i].tolist(), v.tolist())
    return n_ok, len(list(indices)) if n_muestra is not None else n, primer_fallo


def main(config_path, solo_verificar=False):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    base_202k = Path(p["base_dir_202k"])

    print("=" * 62)
    print("  EXP J: labels de CARBONOS TOTALES (sin colapso de simetria)")
    print("=" * 62)

    smiles = np.load(base_202k / p["smiles_202465"], allow_pickle=True)
    labels_viejos = np.load(base_202k / p["labels_viejos_202465"]).astype(int)
    print(f"-> Moleculas: {len(smiles)}")

    print("\n[GATE] Verificando que el clasificador reproduce los labels viejos...")
    n_ok, n_tot, fallo = verificar_puerto(smiles, labels_viejos)
    pct = 100.0 * n_ok / n_tot
    print(f"       coincidencia exacta: {n_ok}/{n_tot} ({pct:.4f}%)")
    if fallo is not None:
        i, smi, viejo, nuevo = fallo
        print(f"\n[ABORT] El clasificador NO reproduce el ground truth existente.")
        print(f"        primer fallo en idx={i}: {smi}")
        print(f"        label existente: {viejo}")
        print(f"        regenerado     : {nuevo}")
        print("        Generar labels nuevos con un clasificador que no reproduce")
        print("        los viejos corromperia el target sin avisar (regla dura 7).")
        sys.exit(1)
    print("[GATE OK] el clasificador es fiel; se puede confiar en la version sin colapso.")

    if solo_verificar:
        print("\n(--solo-verificar: no se escribe nada)")
        return

    print("\n-> Generando el vector de carbonos totales...")
    labels_totales = np.zeros((len(smiles), N_CLASSES), dtype=np.int32)
    n_invalidos = 0
    for i, smi in enumerate(smiles):
        try:
            labels_totales[i] = vector_carbonos_totales(smi)
        except ValueError:
            n_invalidos += 1
        if (i + 1) % 25000 == 0:
            print(f"   procesadas {i + 1}/{len(smiles)}")

    n_c = np.array([
        sum(1 for a in Chem.MolFromSmiles(str(s)).GetAtoms() if a.GetAtomicNum() == 6)
        if Chem.MolFromSmiles(str(s)) is not None else 0
        for s in smiles
    ])
    suma = labels_totales.sum(axis=1)
    n_suma_ok = int((suma == n_c).sum())
    escondidos = suma - labels_viejos.sum(axis=1)

    print(f"\n-> SMILES invalidos: {n_invalidos}")
    print(f"-> sum(vector) == C de la formula: {n_suma_ok}/{len(smiles)} "
          f"({100.0 * n_suma_ok / len(smiles):.4f}%)")
    print(f"-> carbonos escondidos por simetria: promedio {escondidos.mean():.2f}, "
          f"max {escondidos.max()}")
    print(f"-> moleculas con simetria: {100.0 * (escondidos > 0).mean():.1f}%")
    if int((escondidos < 0).sum()) != 0:
        print("[ABORT] hay moleculas con MENOS carbonos que senales: imposible.")
        sys.exit(1)

    out = base_202k / p["labels_totales_output"]
    np.save(out, labels_totales)
    print(f"\n[SAVE] {out}")
    print(">>> EXP J make_labels_totales.py OK <<<")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exp J: labels de carbonos totales")
    ap.add_argument("--config", type=str, default="config_prep.yaml")
    ap.add_argument("--solo-verificar", action="store_true",
                    help="corre solo el gate, no escribe el .npy")
    args = ap.parse_args()
    main(args.config, solo_verificar=args.solo_verificar)
