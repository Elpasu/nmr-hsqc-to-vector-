# Exp E — Fase 1b: Picos desde el pkl original — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reprocesar los picos HSQC directamente desde los shifts DFT del pkl
original (sin pasar por la imagen 256×256), agrupando por carbono, para medir
si la colisión cae respecto al 88.75% encontrado en Fase 1 (blob-detection).

**Architecture:** Un módulo de conectividad C-H (copiado del generador
original), un extractor que agrupa por carbono y promedia δH de H
diastereotópicos, una verificación de alineación SMILES antes de confiar en
el matching posicional pkl↔dataset, y un orquestador que reutiliza
`build_padded_arrays` (Fase 1) y las funciones puras de `validate_peaks.py`
(Fase 1) sin modificarlas.

**Tech Stack:** Python, numpy, RDKit, pyyaml. **Corre en la máquina local de
Lucas** (Windows) — a diferencia de Fase 1, acá numpy y RDKit SÍ están
disponibles localmente, así que la mayoría de los tests de este plan se
ejecutan de verdad, no solo se revisan.

## Global Constraints

- Un pico por **carbono**, no por par C-H — agrupar por `c_idx`, `delta_h` =
  promedio de los shifts de sus H asociados presentes en `nmr_shifts`.
- `amp_ch0 = (-1.0 if mult == 2 else +1.0) * mult`, `amp_ch1 = mult / 3.0` —
  misma convención DEPT/tipo-CH que `Genera_mapas_de_pkl_v2.py`, sin
  normalizar (no hay imagen).
- Conectividad C-H (`get_carbon_multiplicity`,
  `get_ch_connectivity_with_multiplicity`) copiada tal cual de
  `Genera_mapas_de_pkl_v2.py` líneas 128-148 — no reinventar la lógica.
- Verificación de alineación SMILES obligatoria antes de generar cualquier
  pico — si no coincide 100% posición por posición contra `smiles_202465.npy`
  real, el script debe frenar y reportar el primer índice en conflicto, no
  seguir con datos mal alineados.
- Salida en `.npz` (no `.h5` — sin dependencia de h5py, que no está instalado
  localmente).
- Reutilizar sin modificar: `build_padded_arrays` de
  `experiments/E_peaks_prep/extract_peaks.py`, y `visible_label_counts`,
  `blob_counts_from_mask`, `validation_report` de
  `experiments/E_peaks_prep/validate_peaks.py` (ambos ya existen en el repo,
  de Fase 1).
- Nada hardcodeado: rutas vía `config_pkl.yaml` propio de esta fase.

---

### Task 1: Scaffold + `ch_connectivity.py` + `config_pkl.yaml`

**Files:**
- Create: `experiments/E_peaks_prep/ch_connectivity.py`
- Create: `experiments/E_peaks_prep/config_pkl.yaml`
- Test: `experiments/E_peaks_prep/tests/test_ch_connectivity.py`

**Interfaces:**
- Produces: `get_carbon_multiplicity(mol, c_idx) -> int`,
  `get_ch_connectivity_with_multiplicity(mol) -> list[dict]` (cada dict con
  claves `c_idx`, `h_idx`, `multiplicity`) — usadas por Task 2.

- [ ] **Step 1: Escribir el test que falla**

```python
# experiments/E_peaks_prep/tests/test_ch_connectivity.py
# coding: ascii
"""Tests de ch_connectivity.py -- corre localmente, requiere rdkit
(disponible en la maquina local de Lucas)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdkit import Chem

from ch_connectivity import get_carbon_multiplicity, get_ch_connectivity_with_multiplicity


def test_ethanol_connectivity():
    # Etanol CCO -> AddHs: atomo 0=C(CH3), 1=C(CH2), 2=O, 3-5=H de CH3,
    # 6-7=H de CH2, 8=H del OH. Verificado corriendo RDKit directamente.
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))

    mult_ch3 = get_carbon_multiplicity(mol, 0)
    mult_ch2 = get_carbon_multiplicity(mol, 1)
    assert mult_ch3 == 3, f"esperado CH3 (mult=3), salio {mult_ch3}"
    assert mult_ch2 == 2, f"esperado CH2 (mult=2), salio {mult_ch2}"

    ch_pairs = get_ch_connectivity_with_multiplicity(mol)
    pairs_ch3 = [p for p in ch_pairs if p["c_idx"] == 0]
    pairs_ch2 = [p for p in ch_pairs if p["c_idx"] == 1]

    assert len(pairs_ch3) == 3, f"esperados 3 pares C-H para el CH3, salio {len(pairs_ch3)}"
    assert {p["h_idx"] for p in pairs_ch3} == {3, 4, 5}
    assert all(p["multiplicity"] == 3 for p in pairs_ch3)

    assert len(pairs_ch2) == 2, f"esperados 2 pares C-H para el CH2, salio {len(pairs_ch2)}"
    assert {p["h_idx"] for p in pairs_ch2} == {6, 7}
    assert all(p["multiplicity"] == 2 for p in pairs_ch2)
    print(f"[OK] test_ethanol_connectivity -> CH3 pairs={pairs_ch3} CH2 pairs={pairs_ch2}")


def test_oxygen_has_no_multiplicity():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    mult_o = get_carbon_multiplicity(mol, 2)
    assert mult_o == -1, f"esperado -1 para atomo no-carbono, salio {mult_o}"
    print("[OK] test_oxygen_has_no_multiplicity")


if __name__ == "__main__":
    test_ethanol_connectivity()
    test_oxygen_has_no_multiplicity()
    print("\n>>> test_ch_connectivity.py OK <<<")
```

- [ ] **Step 2: Correr el test y confirmar que falla**

Run: `python experiments/E_peaks_prep/tests/test_ch_connectivity.py`
Expected: `ModuleNotFoundError: No module named 'ch_connectivity'`

- [ ] **Step 3: Implementar `ch_connectivity.py`**

```python
# experiments/E_peaks_prep/ch_connectivity.py
# coding: ascii
"""
ch_connectivity.py -- reconstruccion de conectividad C-H via RDKit, copiada
tal cual de Genera_mapas_de_pkl_v2.py (lineas 128-148, script original del
dataset, fuera de este repo) -- no modificar la logica, solo se traslado el
codigo para reutilizarlo en la extraccion de picos desde el pkl.
"""


def get_carbon_multiplicity(mol, c_idx):
    atom = mol.GetAtomWithIdx(c_idx)
    if atom.GetAtomicNum() != 6:
        return -1
    return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)


def get_ch_connectivity_with_multiplicity(mol):
    ch_pairs = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        n1, n2 = a1.GetAtomicNum(), a2.GetAtomicNum()
        if n1 == 6 and n2 == 1:
            c_idx, h_idx = a1.GetIdx(), a2.GetIdx()
        elif n1 == 1 and n2 == 6:
            c_idx, h_idx = a2.GetIdx(), a1.GetIdx()
        else:
            continue
        mult = get_carbon_multiplicity(mol, c_idx)
        ch_pairs.append({"c_idx": c_idx, "h_idx": h_idx, "multiplicity": mult})
    return ch_pairs
```

- [ ] **Step 4: Correr el test y confirmar que pasa**

Run: `python experiments/E_peaks_prep/tests/test_ch_connectivity.py`
Expected:
```
[OK] test_ethanol_connectivity -> CH3 pairs=[...] CH2 pairs=[...]
[OK] test_oxygen_has_no_multiplicity

>>> test_ch_connectivity.py OK <<<
```

- [ ] **Step 5: Crear `experiments/E_peaks_prep/config_pkl.yaml`**

```yaml
# experiments/E_peaks_prep/config_pkl.yaml
#
# Exp E Fase 1b: picos desde el pkl original (sin binning). Corre LOCAL en
# la maquina Windows de Lucas -- rutas locales, no las del cluster.
# base_dir: carpeta con TODOS los archivos de abajo juntos (144k ya estan
# ahi; Lucas copia los de 58k desde snmgt01:DB_58K, y smiles_202465.npy +
# vectors_13c_19v_202465.npy desde donde esten disponibles).

paths:
  base_dir: "E:/Proyectos/SciTrix/ScitrixDB/DB-Batch0"
  pkl_144k: "nmr_calculated_data_scaled_144K.pkl"
  mol_ids_144k: "mol_ids_144280.npy"
  smiles_144k: "smiles_144280.npy"
  pkl_58k: "nmr_calculated_data_scaled_58k.pkl"
  mol_ids_58k: "mol_ids_58185.npy"
  smiles_58k: "smiles_58185.npy"
  smiles_202465: "smiles_202465.npy"
  labels_202465: "vectors_13c_19v_202465.npy"
  peaks_output_filename: "peaks_pkl_202465.npz"

classes_19v:
  - CH3
  - CH2
  - CH
  - Cq
  - CH3-O
  - CH2-O
  - CH-O
  - Cq-O
  - CH3-N
  - CH2-N
  - CH-N
  - Cq-N
  - "=CH2"
  - "=CH/Ar"
  - Cqsp2
  - Aldeh
  - Imina
  - C-2X
  - C-3X
```

- [ ] **Step 6: Commit**

```bash
git add experiments/E_peaks_prep/ch_connectivity.py experiments/E_peaks_prep/config_pkl.yaml experiments/E_peaks_prep/tests/test_ch_connectivity.py
git commit -m "exp-e-fase1b: ch_connectivity.py (copiado del generador) + config_pkl.yaml"
```

---

### Task 2: `extract_peaks_from_pkl_molecule` — agrupación por carbono (TDD)

**Files:**
- Create: `experiments/E_peaks_prep/extract_peaks_pkl.py`
- Test: `experiments/E_peaks_prep/tests/test_extract_peaks_pkl.py`

**Interfaces:**
- Consumes: `ch_connectivity.get_ch_connectivity_with_multiplicity` (Task 1).
- Produces: `extract_peaks_from_pkl_molecule(smiles, nmr_shifts) ->
  list[tuple[float, float, float, float]]` — lista de
  `(delta_c, delta_h, amp_ch0, amp_ch1)`, un elemento por carbono. Usada por
  Task 4 (`main()`).

- [ ] **Step 1: Escribir el test que falla**

```python
# experiments/E_peaks_prep/tests/test_extract_peaks_pkl.py
# coding: ascii
"""Tests de extract_peaks_pkl.py -- corre localmente, requiere rdkit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract_peaks_pkl import extract_peaks_from_pkl_molecule


def test_ethanol_one_peak_per_carbon_with_diastereotopic_average():
    # Etanol CCO -> AddHs: atomo 0=C(CH3, H en 3,4,5), 1=C(CH2, H en 6,7),
    # 2=O (indices verificados con RDKit directamente, ver Task 1).
    # H diastereotopicos en el CH2 (6 y 7) con shifts DISTINTOS a proposito,
    # para confirmar que se promedian en vez de generar 2 picos.
    nmr_shifts = {
        0: 18.0,   # C del CH3
        1: 58.0,   # C del CH2
        3: 1.2, 4: 1.2, 5: 1.2,   # H del CH3 (isocronos)
        6: 3.5, 7: 3.7,           # H del CH2 (diastereotopicos, shifts distintos)
    }

    peaks = extract_peaks_from_pkl_molecule("CCO", nmr_shifts)
    assert len(peaks) == 2, f"esperados 2 picos (1 por carbono), salieron {len(peaks)}"

    peaks_by_c = {round(p[0], 3): p for p in peaks}
    assert 18.0 in peaks_by_c, peaks_by_c
    assert 58.0 in peaks_by_c, peaks_by_c

    delta_c, delta_h, amp_ch0, amp_ch1 = peaks_by_c[18.0]
    assert abs(delta_h - 1.2) < 1e-9, delta_h
    assert amp_ch0 == 3.0   # CH3: fase +1 * mult 3
    assert abs(amp_ch1 - 1.0) < 1e-9   # mult 3 / 3

    delta_c, delta_h, amp_ch0, amp_ch1 = peaks_by_c[58.0]
    assert abs(delta_h - 3.6) < 1e-9, delta_h   # promedio de 3.5 y 3.7
    assert amp_ch0 == -2.0   # CH2: fase -1 * mult 2
    assert abs(amp_ch1 - (2.0 / 3.0)) < 1e-9

    print(f"[OK] test_ethanol_one_peak_per_carbon_with_diastereotopic_average -> {peaks}")


def test_carbon_without_shift_is_dropped():
    # Falta el shift del carbono del CH2 (atomo 1) -- ese carbono no debe
    # generar pico (sin delta_c no hay pico posible).
    nmr_shifts = {
        0: 18.0,
        3: 1.2, 4: 1.2, 5: 1.2,
        6: 3.5, 7: 3.7,
    }
    peaks = extract_peaks_from_pkl_molecule("CCO", nmr_shifts)
    assert len(peaks) == 1, f"esperado 1 pico (CH2 sin delta_c se descarta), salieron {len(peaks)}"
    assert abs(peaks[0][0] - 18.0) < 1e-9
    print(f"[OK] test_carbon_without_shift_is_dropped -> {peaks}")


def test_invalid_smiles_returns_empty():
    peaks = extract_peaks_from_pkl_molecule("no_es_un_smiles_valido()", {0: 18.0})
    assert peaks == []
    print("[OK] test_invalid_smiles_returns_empty")


if __name__ == "__main__":
    test_ethanol_one_peak_per_carbon_with_diastereotopic_average()
    test_carbon_without_shift_is_dropped()
    test_invalid_smiles_returns_empty()
    print("\n>>> test_extract_peaks_pkl.py OK <<<")
```

- [ ] **Step 2: Correr el test y confirmar que falla**

Run: `python experiments/E_peaks_prep/tests/test_extract_peaks_pkl.py`
Expected: `ModuleNotFoundError: No module named 'extract_peaks_pkl'`

- [ ] **Step 3: Implementar `extract_peaks_pkl.py` (parte 1: extracción)**

```python
# experiments/E_peaks_prep/extract_peaks_pkl.py
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


def extract_peaks_from_pkl_molecule(smiles, nmr_shifts):
    """smiles: str. nmr_shifts: dict {atom_idx: float shift}, con indices de
    atomo POST AddHs. Devuelve lista de (delta_c, delta_h, amp_ch0, amp_ch1),
    un elemento por carbono con al menos un H con shift conocido."""
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
    return peaks
```

- [ ] **Step 4: Correr el test y confirmar que pasa**

Run: `python experiments/E_peaks_prep/tests/test_extract_peaks_pkl.py`
Expected:
```
[OK] test_ethanol_one_peak_per_carbon_with_diastereotopic_average -> [...]
[OK] test_carbon_without_shift_is_dropped -> [...]
[OK] test_invalid_smiles_returns_empty

>>> test_extract_peaks_pkl.py OK <<<
```

- [ ] **Step 5: Commit**

```bash
git add experiments/E_peaks_prep/extract_peaks_pkl.py experiments/E_peaks_prep/tests/test_extract_peaks_pkl.py
git commit -m "exp-e-fase1b: extract_peaks_from_pkl_molecule (agrupacion por carbono, TDD)"
```

---

### Task 3: `verify_smiles_alignment` — verificación de orden (TDD)

**Files:**
- Modify: `experiments/E_peaks_prep/extract_peaks_pkl.py`
- Test: `experiments/E_peaks_prep/tests/test_verify_alignment.py`

**Interfaces:**
- Produces: `canonicalize_smiles(smiles_array) -> (np.ndarray, int)` (copiada
  de `experiments/D_val_congelado/split.py`), `verify_smiles_alignment(local_smiles,
  real_smiles) -> tuple[bool, int|None]` — usada por Task 4 (`main()`) antes
  de generar cualquier pico.

- [ ] **Step 1: Escribir el test que falla**

```python
# experiments/E_peaks_prep/tests/test_verify_alignment.py
# coding: ascii
"""Tests de verify_smiles_alignment -- corre localmente, requiere rdkit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from extract_peaks_pkl import verify_smiles_alignment


def test_alignment_ok_with_different_but_equivalent_smiles():
    # Mismo orden, pero escritos distinto (canonico vs no-canonico) --
    # debe pasar igual porque se compara canonicalizado.
    local = np.array(["CCO", "c1ccccc1", "CC(=O)O"], dtype=object)
    real = np.array(["OCC", "C1=CC=CC=C1", "CC(O)=O"], dtype=object)
    ok, mismatch_idx = verify_smiles_alignment(local, real)
    assert ok is True, mismatch_idx
    assert mismatch_idx is None
    print("[OK] test_alignment_ok_with_different_but_equivalent_smiles")


def test_alignment_detects_mismatch_and_reports_index():
    local = np.array(["CCO", "c1ccccc1", "CC(=O)O"], dtype=object)
    real = np.array(["CCO", "CCN", "CC(=O)O"], dtype=object)   # indice 1 distinto
    ok, mismatch_idx = verify_smiles_alignment(local, real)
    assert ok is False
    assert mismatch_idx == 1, mismatch_idx
    print(f"[OK] test_alignment_detects_mismatch_and_reports_index -> idx={mismatch_idx}")


def test_alignment_detects_length_mismatch():
    local = np.array(["CCO", "CCN"], dtype=object)
    real = np.array(["CCO"], dtype=object)
    ok, mismatch_idx = verify_smiles_alignment(local, real)
    assert ok is False
    assert mismatch_idx is None   # no hay un indice puntual, es un mismatch de longitud
    print("[OK] test_alignment_detects_length_mismatch")


if __name__ == "__main__":
    test_alignment_ok_with_different_but_equivalent_smiles()
    test_alignment_detects_mismatch_and_reports_index()
    test_alignment_detects_length_mismatch()
    print("\n>>> test_verify_alignment.py OK <<<")
```

- [ ] **Step 2: Correr el test y confirmar que falla**

Run: `python experiments/E_peaks_prep/tests/test_verify_alignment.py`
Expected: `ImportError: cannot import name 'verify_smiles_alignment' from 'extract_peaks_pkl'`

- [ ] **Step 3: Agregar `canonicalize_smiles` y `verify_smiles_alignment` a `extract_peaks_pkl.py`**

Agregar al final de `experiments/E_peaks_prep/extract_peaks_pkl.py` (después
de `extract_peaks_from_pkl_molecule`, antes de cualquier `main()`):

```python
def canonicalize_smiles(smiles_array):
    """Copiada tal cual de experiments/D_val_congelado/split.py -- misma
    logica ya probada en Exp D, no reinventar."""
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
```

- [ ] **Step 4: Correr el test y confirmar que pasa**

Run: `python experiments/E_peaks_prep/tests/test_verify_alignment.py`
Expected:
```
[OK] test_alignment_ok_with_different_but_equivalent_smiles
[OK] test_alignment_detects_mismatch_and_reports_index -> idx=1
[OK] test_alignment_detects_length_mismatch

>>> test_verify_alignment.py OK <<<
```

- [ ] **Step 5: Commit**

```bash
git add experiments/E_peaks_prep/extract_peaks_pkl.py experiments/E_peaks_prep/tests/test_verify_alignment.py
git commit -m "exp-e-fase1b: verify_smiles_alignment (canonicalize_smiles copiada de Exp D, TDD)"
```

---

### Task 4: `main()` orquestador + smoke test con datos sintéticos + README

**Files:**
- Modify: `experiments/E_peaks_prep/extract_peaks_pkl.py`
- Create: `experiments/E_peaks_prep/tests/test_smoke_pkl.py`
- Create: `experiments/E_peaks_prep/README_pkl.md`

**Interfaces:**
- Consumes: `extract_peaks_from_pkl_molecule`, `verify_smiles_alignment`
  (Task 2/3), `build_padded_arrays` de `extract_peaks.py` (Fase 1, ya en el
  repo), `visible_label_counts`/`blob_counts_from_mask`/`validation_report`
  de `validate_peaks.py` (Fase 1, ya en el repo).

- [ ] **Step 1: Escribir el smoke test (corre localmente con archivos
  sintéticos chicos, sin depender de los pkl reales de 202k)**

```python
# experiments/E_peaks_prep/tests/test_smoke_pkl.py
# coding: ascii
"""
Smoke test OFFLINE de Exp E Fase 1b (rule 5 de CLAUDE.md) -- construye un
mini pkl + mini arrays de smiles/mol_ids/labels en un directorio temporal
(3 moleculas), corre extract_peaks_pkl.main() sobre ellos, y confirma que
el .npz de salida y el reporte de validacion tienen las formas y valores
esperados. No toca ningun dato real del cluster.

Requiere rdkit + pyyaml -- correr local antes de procesar los pkl reales:
    python tests/test_smoke_pkl.py
"""
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from extract_peaks_pkl import main

CLASS_NAMES = [
    "CH3", "CH2", "CH", "Cq", "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N", "=CH2", "=CH/Ar", "Cqsp2",
    "Aldeh", "Imina", "C-2X", "C-3X",
]


def test_pipeline_end_to_end_with_synthetic_files():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # 2 moleculas "144k" (etanol, metano) + 1 molecula "58k" (etanol de nuevo)
        smiles_144 = np.array(["CCO", "C"], dtype=object)
        mol_ids_144 = np.array(["mol0", "mol1"], dtype=object)
        smiles_58 = np.array(["CCO"], dtype=object)
        mol_ids_58 = np.array(["mol2"], dtype=object)
        smiles_real = np.concatenate([smiles_144, smiles_58])   # mismo orden -> alineacion OK

        pkl_144 = {
            "mol0": {0: 18.0, 1: 58.0, 3: 1.2, 4: 1.2, 5: 1.2, 6: 3.5, 7: 3.7},  # etanol
            "mol1": {0: -2.0, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},  # metano (CH4)
        }
        pkl_58 = {
            "mol2": {0: 18.0, 1: 58.0, 3: 1.2, 4: 1.2, 5: 1.2, 6: 3.5, 7: 3.7},  # etanol otra vez
        }

        labels = np.zeros((3, 19), dtype=int)
        labels[0, CLASS_NAMES.index("CH3")] = 1
        labels[0, CLASS_NAMES.index("CH2")] = 1
        labels[1, CLASS_NAMES.index("CH3")] = 1   # metano cuenta como 1 entorno visible (aprox para el test)
        labels[2, CLASS_NAMES.index("CH3")] = 1
        labels[2, CLASS_NAMES.index("CH2")] = 1

        np.save(base / "smiles_144.npy", smiles_144, allow_pickle=True)
        np.save(base / "mol_ids_144.npy", mol_ids_144, allow_pickle=True)
        np.save(base / "smiles_58.npy", smiles_58, allow_pickle=True)
        np.save(base / "mol_ids_58.npy", mol_ids_58, allow_pickle=True)
        np.save(base / "smiles_real.npy", smiles_real, allow_pickle=True)
        np.save(base / "labels.npy", labels)
        with open(base / "pkl_144.pkl", "wb") as f:
            pickle.dump(pkl_144, f)
        with open(base / "pkl_58.pkl", "wb") as f:
            pickle.dump(pkl_58, f)

        config = {
            "paths": {
                "base_dir": str(base),
                "pkl_144k": "pkl_144.pkl",
                "mol_ids_144k": "mol_ids_144.npy",
                "smiles_144k": "smiles_144.npy",
                "pkl_58k": "pkl_58.pkl",
                "mol_ids_58k": "mol_ids_58.npy",
                "smiles_58k": "smiles_58.npy",
                "smiles_202465": "smiles_real.npy",
                "labels_202465": "labels.npy",
                "peaks_output_filename": "peaks_out.npz",
            },
            "classes_19v": CLASS_NAMES,
        }
        config_path = base / "config_pkl.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

        main(str(config_path))

        out = np.load(base / "peaks_out.npz")
        assert out["peaks"].shape[0] == 3
        assert out["peaks_mask"].shape[0] == 3
        # etanol (mol0 y mol2) -> 2 picos cada uno; metano -> 1 pico (un solo C)
        counts = out["peaks_mask"].sum(axis=1)
        assert counts.tolist() == [2, 1, 2], counts.tolist()
        print(f"[OK] test_pipeline_end_to_end_with_synthetic_files -> peaks_mask counts={counts.tolist()}")


if __name__ == "__main__":
    test_pipeline_end_to_end_with_synthetic_files()
    print("\n>>> SMOKE EXP E FASE 1b OK - listo para correr con los pkl reales <<<")
```

- [ ] **Step 2: Correr el test y confirmar que falla**

Run: `python experiments/E_peaks_prep/tests/test_smoke_pkl.py`
Expected: `ImportError: cannot import name 'main' from 'extract_peaks_pkl'`

- [ ] **Step 3: Agregar `main()` a `extract_peaks_pkl.py`**

Agregar al final de `experiments/E_peaks_prep/extract_peaks_pkl.py`:

```python
def main(config_path):
    import sys as _sys
    import pickle

    import yaml

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_peaks import build_padded_arrays
    from validate_peaks import blob_counts_from_mask, validation_report, visible_label_counts

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(cfg["paths"]["base_dir"])
    class_names = cfg["classes_19v"]

    print("=" * 60)
    print("  EXP E FASE 1b: picos desde el pkl original")
    print("=" * 60)

    smiles_144 = np.load(base_dir / cfg["paths"]["smiles_144k"], allow_pickle=True)
    mol_ids_144 = np.load(base_dir / cfg["paths"]["mol_ids_144k"], allow_pickle=True)
    smiles_58 = np.load(base_dir / cfg["paths"]["smiles_58k"], allow_pickle=True)
    mol_ids_58 = np.load(base_dir / cfg["paths"]["mol_ids_58k"], allow_pickle=True)
    smiles_real = np.load(base_dir / cfg["paths"]["smiles_202465"], allow_pickle=True)
    labels = np.load(base_dir / cfg["paths"]["labels_202465"]).astype(int)

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

    with open(base_dir / cfg["paths"]["pkl_144k"], "rb") as f:
        pkl_144 = pickle.load(f)
    with open(base_dir / cfg["paths"]["pkl_58k"], "rb") as f:
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

    out_path = base_dir / cfg["paths"]["peaks_output_filename"]
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
```

- [ ] **Step 4: Correr el test y confirmar que pasa**

Run: `python experiments/E_peaks_prep/tests/test_smoke_pkl.py`
Expected:
```
============================================================
  EXP E FASE 1b: picos desde el pkl original
============================================================
-> Moleculas locales (144k+58k): 3
-> Moleculas en smiles_202465 real: 3
[OK] alineacion verificada: SMILES canonicos coinciden posicion por posicion
-> max_peaks detectado: 2
-> picos por molecula: min=1 max=2 promedio=1.67

[SAVE] .../peaks_out.npz
...
>>> EXP E FASE 1b extract_peaks_pkl.py OK <<<
[OK] test_pipeline_end_to_end_with_synthetic_files -> peaks_mask counts=[2, 1, 2]

>>> SMOKE EXP E FASE 1b OK - listo para correr con los pkl reales <<<
```

- [ ] **Step 5: Crear `experiments/E_peaks_prep/README_pkl.md`**

```markdown
# Exp E — Fase 1b: Picos desde el pkl original

Corre LOCAL en tu máquina Windows (no en el cluster) — el pkl de 144k y sus
arrays ya están en `E:\Proyectos\SciTrix\ScitrixDB\DB-Batch0\`.

## Antes de correr

Copiá a esa misma carpeta, si todavía no están:
- Desde `snmgt01:/data/contrib/pci_78/Lucas/DB_58K/`:
  `nmr_calculated_data_scaled_58k.pkl`, `mol_ids_58185.npy`,
  `smiles_58185.npy`.
- Desde donde tengas `smiles_202465.npy` y `vectors_13c_19v_202465.npy`
  (login-1 `DB_200k` u otro lado) — son archivos chicos.

## Orden de comandos

1. Smoke test obligatorio (no toca ningún dato real, usa archivos
   sintéticos en un directorio temporal):
   ```bash
   cd experiments/E_peaks_prep
   python tests/test_ch_connectivity.py
   python tests/test_extract_peaks_pkl.py
   python tests/test_verify_alignment.py
   python tests/test_smoke_pkl.py
   ```
   Todos deben terminar en `>>> ... OK <<<`.
2. Correr sobre los datos reales (202465 moléculas):
   ```bash
   python extract_peaks_pkl.py --config config_pkl.yaml
   ```
   Si falla en "[ERROR] desajuste de alineacion en indice N" — pará ahí,
   avisá con el índice y los dos SMILES que reportó, no sigas.
3. Si termina OK, va a imprimir el mismo tipo de reporte que Fase 1 (match
   exacto %, % con colisión, déficit promedio) — comparalo contra el
   88.75% de colisión de Fase 1 y avisá los números.
```

- [ ] **Step 6: Commit**

```bash
git add experiments/E_peaks_prep/extract_peaks_pkl.py experiments/E_peaks_prep/tests/test_smoke_pkl.py experiments/E_peaks_prep/README_pkl.md
git commit -m "exp-e-fase1b: main() orquestador + smoke test sintetico + README"
```

---

## Al terminar

Correr el pipeline completo con los datos reales en la máquina local
(instrucciones en `README_pkl.md`), comparar el resultado contra el 88.75%
de colisión de Fase 1, agregar una entrada nueva a `docs/Runs/RESULTS.md`
("Exp E Fase 1b"), y decidir con esos números si se escribe el spec de
Fase 2 (DeepSets) o si hace falta seguir ajustando la representación.
