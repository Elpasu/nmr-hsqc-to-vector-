# Exp E — Fase 3: dos conjuntos de picos (crosspeaks C-H + ¹³C) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corregir el input del DeepSets de Fase 2 agregando un segundo conjunto de picos ¹³C (todos los carbonos, incluidos los cuaternarios que el HSQC no ve) y probar dos arquitecturas sobre esa representación: DeepSets de dos ramas y Set Transformer.

**Architecture:** Se extrae un nuevo conjunto de picos ¹³C `(δC,)` desde el pkl DFT (todos los carbonos, sin filtrar por H). El dataset carga los dos conjuntos (crosspeaks C-H de Fase 1b + ¹³C nuevo), normaliza los desplazamientos min-max, y alimenta un modelo intercambiable por config (`arch: deepsets | settransformer`). Todo lo demás (loss, split congelado, scheduler, épocas) es idéntico a E2/Exp C para que la comparación sea limpia.

**Tech Stack:** Python, PyTorch, NumPy, RDKit, PyYAML. Sin dependencias nuevas.

## Global Constraints

Copiadas verbatim del spec (`docs/superpowers/specs/2026-07-22-exp-e-fase3-dos-conjuntos-picos-design.md`) y de `CLAUDE.md`. Aplican a TODAS las tareas:

- `num_workers: 0` siempre (regla 1). No cambiar aunque no haya h5py.
- SLURM usa `#SBATCH --gres=gpu:1`, nunca `--gpus=1` (regla 2).
- Nada hardcodeado en los `.py`: rutas, nombres de archivo y constantes de normalización salen del config YAML (regla 3).
- Encoding: todos los `.py` empiezan con `# coding: ascii` y usan solo ASCII en el código (regla 4). Los archivos se crean con LF.
- Smoke test offline obligatorio antes de cualquier `sbatch` (regla 5).
- Scheduler `patience=8, factor=0.7` (regla 6). No cambiar.
- `num_classes=19` y el orden de clases de `config/db.yaml` es fijo (regla 7). No reordenar.
- Split congelado idéntico a Exp D/B/C/E2: `val_indices_frozen.npy` + `split_utils.py`. val NUNCA se toca (regla 8).
- El pico ¹³C lleva **solo δC**. Nunca el nº de H del carbono (sería filtrar el label).
- Normalización min-max desde el config: δC → `(δC-0)/(220-0)`; δH → `(δH-(-1))/(15-(-1))`; `amp_ch0 → amp_ch0/3`; `amp_ch1` sin tocar.
- Baseline a batir: Exp C (EMA cruda 0.89%). Reportar cruda + asistida (oráculo) + confusiones.

**Rutas de datos (cluster, login-1):** `base_dir = /home/lpassaglia.iquir/DB_200k`. Archivos que deben existir ahí: `peaks_pkl_202465.npz` (Fase 1b), `peaks_13c_202465.npz` (Task 1, hay que copiarlo), `vectors_13c_19v_202465.npy`, `smiles_202465.npy`, `val_indices_frozen.npy`.

**Máquina local (Windows, para Task 1):** paths en `experiments/E_peaks_prep/config_pkl.yaml` (144K/58K/202K_suma en `E:/Proyectos/SciTrix/ScitrixDB/DB_nmr_to_vector/`).

---

### Task 1: Extractor de picos ¹³C desde el pkl (`extract_peaks_13c_pkl.py`)

Extrae el segundo conjunto de picos: un `(δC,)` por carbono con entorno químico distinto (todos los carbonos, con y sin H — los cuaternarios ahora SÍ entran). Corre local en la máquina de Lucas, como Fase 1b. Reutiliza la maquinaria de alineación/dedup de `extract_peaks_pkl.py`.

**Files:**
- Create: `experiments/E_peaks_prep/extract_peaks_13c_pkl.py`
- Create: `experiments/E_peaks_prep/tests/test_extract_13c.py`
- Reference (no modificar): `experiments/E_peaks_prep/extract_peaks_pkl.py`, `ch_connectivity.py`, `extract_peaks.py` (`build_padded_arrays`), `config_pkl.yaml`

**Interfaces:**
- Produces:
  - `extract_13c_peaks_from_molecule(smiles: str, nmr_shifts: dict) -> list[tuple[float]]` — lista de `(δC,)`, un elemento por carbono con δC distinto (dedup por simetría).
  - `build_padded_arrays_13c(peaks_per_molecule: list[list[tuple]]) -> (np.ndarray (N,M,1) float32, np.ndarray (N,M) bool)`.
  - Archivo de salida `peaks_13c_202465.npz` con keys `peaks_13c (N,M,1)` y `mask_13c (N,M)`.

- [ ] **Step 1: Write the failing test**

```python
# experiments/E_peaks_prep/tests/test_extract_13c.py
# coding: ascii
"""Test offline del extractor de picos 13C (Task 1). Verifica que:
 (1) un carbono cuaternario (sin H) SI entra al conjunto 13C,
 (2) carbonos equivalentes por simetria con el mismo shift colapsan a un pico,
 (3) el padding a (N, M, 1) y la mascara son consistentes.
Corre local, sin datos reales:  python tests/test_extract_13c.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rdkit import Chem

from extract_peaks_13c_pkl import (
    extract_13c_peaks_from_molecule,
    build_padded_arrays_13c,
)


def _shifts_for(smiles, per_carbon):
    """Arma un dict {atom_idx: shift} post-AddHs asignando a cada carbono
    (en orden de indice) el shift dado en per_carbon."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    shifts = {}
    ci = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            shifts[atom.GetIdx()] = per_carbon[ci]
            ci += 1
    return shifts


def test_quaternary_carbon_is_included():
    # Acetona CC(=O)C: 3 carbonos -> C(carbonilo, cuaternario) + 2 CH3.
    # Damos shifts distintos a los 3 -> deben salir 3 picos (el cuaternario incluido).
    smiles = "CC(=O)C"
    shifts = _shifts_for(smiles, [30.0, 205.0, 31.5])
    peaks = extract_13c_peaks_from_molecule(smiles, shifts)
    assert len(peaks) == 3, peaks
    # El shift del carbonilo (205.0, cuaternario, sin H) tiene que estar presente.
    assert any(abs(p[0] - 205.0) < 1e-6 for p in peaks), peaks
    print(f"[OK] cuaternario incluido: {peaks}")


def test_symmetry_dedup():
    # Dos carbonos con el MISMO shift (equivalentes por simetria) colapsan a 1.
    smiles = "CC(=O)C"
    shifts = _shifts_for(smiles, [30.0, 205.0, 30.0])  # los 2 CH3 iguales
    peaks = extract_13c_peaks_from_molecule(smiles, shifts)
    assert len(peaks) == 2, peaks   # {30.0, 205.0}
    print(f"[OK] dedup por simetria: {peaks}")


def test_padding_shape():
    per_mol = [[(30.0,), (205.0,)], [(10.0,)], []]
    peaks, mask = build_padded_arrays_13c(per_mol)
    assert peaks.shape == (3, 2, 1), peaks.shape
    assert mask.shape == (3, 2), mask.shape
    assert mask[0].tolist() == [True, True]
    assert mask[1].tolist() == [True, False]
    assert mask[2].tolist() == [False, False]
    print(f"[OK] padding -> {peaks.shape}, mask ok")


if __name__ == "__main__":
    test_quaternary_carbon_is_included()
    test_symmetry_dedup()
    test_padding_shape()
    print("\n>>> TEST EXTRACT 13C OK <<<")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd experiments/E_peaks_prep && python tests/test_extract_13c.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'extract_peaks_13c_pkl'`.

- [ ] **Step 3: Write the extractor**

```python
# experiments/E_peaks_prep/extract_peaks_13c_pkl.py
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
    class_names = cfg["classes_19v"]

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd experiments/E_peaks_prep && python tests/test_extract_13c.py`
Expected: PASS — imprime `>>> TEST EXTRACT 13C OK <<<`.

- [ ] **Step 5: (Manual, cuando Lucas corra la extracción real) Generar y validar el npz**

Run (local, Windows): `cd experiments/E_peaks_prep && python extract_peaks_13c_pkl.py --config config_pkl.yaml`
Expected: imprime `Match exacto (...): ` cercano a ~100%. Si es bajo (< 95%), el pkl no tiene shifts de algún tipo de carbono → PARAR y avisar antes de entrenar. Copiar `peaks_13c_202465.npz` a `/home/lpassaglia.iquir/DB_200k/` en el cluster.

- [ ] **Step 6: Commit**

```bash
git add experiments/E_peaks_prep/extract_peaks_13c_pkl.py experiments/E_peaks_prep/tests/test_extract_13c.py
git commit -m "exp-e-fase3: extractor de picos 13C (todos los carbonos, con cuaternarios)"
```

---

### Task 2: Dataset de dos conjuntos con normalización (`dataset_e3.py`)

**Files:**
- Create: `experiments/E3_dos_conjuntos/dataset_e3.py`
- Create: `experiments/E3_dos_conjuntos/tests/test_dataset_e3.py`
- Reference (no modificar): `experiments/E2_deepsets/dataset_e2.py`

**Interfaces:**
- Consumes: `peaks_pkl_202465.npz` (keys `peaks (N,32,4)`, `peaks_mask (N,32)`), `peaks_13c_202465.npz` (keys `peaks_13c (N,M,1)`, `mask_13c (N,M)`) de Task 1.
- Produces: `NMRTwoSetsDataset(peaks_ch_path, peaks_13c_path, labels_path, smiles_path, norm_cfg)`. `__getitem__` devuelve `((peaks_ch (32,4), mask_ch (32,), peaks_13c (M,1), mask_13c (M,), cond (8,)), target (19,))`, todos `torch.float32` salvo target. Normalización aplicada en `__init__`.

- [ ] **Step 1: Write the failing test**

```python
# experiments/E3_dos_conjuntos/tests/test_dataset_e3.py
# coding: ascii
"""Test del dataset E3 con npz sinteticos chicos. Verifica shapes de salida
y que la normalizacion min-max mapea los extremos de la calibracion a [0,1]."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from dataset_e3 import NMRTwoSetsDataset

NORM = {"c13_ppm_min": 0.0, "c13_ppm_max": 220.0,
        "h1_ppm_min": -1.0, "h1_ppm_max": 15.0, "amp_ch0_scale": 3.0}


def _make_tmp(tmp):
    # 2 moleculas. crosspeaks: (2, 2, 4); 13c: (2, 3, 1).
    peaks_ch = np.zeros((2, 2, 4), dtype=np.float32)
    peaks_ch[0, 0] = [220.0, 15.0, 3.0, 1.0]   # extremos de calibracion
    peaks_ch[0, 1] = [0.0, -1.0, -3.0, 0.333]
    mask_ch = np.array([[True, True], [True, False]])
    np.savez(tmp / "ch.npz", peaks=peaks_ch, peaks_mask=mask_ch)

    peaks_13c = np.zeros((2, 3, 1), dtype=np.float32)
    peaks_13c[0, 0, 0] = 220.0
    peaks_13c[0, 1, 0] = 110.0
    mask_13c = np.array([[True, True, False], [True, False, False]])
    np.savez(tmp / "c13.npz", peaks_13c=peaks_13c, mask_13c=mask_13c)

    labels = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                      dtype=np.int64)
    np.save(tmp / "labels.npy", labels)
    np.save(tmp / "smiles.npy", np.array(["CCO", "CC(=O)C"], dtype=object))


def test_shapes_and_normalization(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    _make_tmp(tmp)
    ds = NMRTwoSetsDataset(str(tmp / "ch.npz"), str(tmp / "c13.npz"),
                           str(tmp / "labels.npy"), str(tmp / "smiles.npy"), NORM)
    (peaks_ch, mask_ch, peaks_13c, mask_13c, cond), target = ds[0]
    assert peaks_ch.shape == (2, 4), peaks_ch.shape
    assert peaks_13c.shape == (3, 1), peaks_13c.shape
    assert cond.shape == (8,), cond.shape
    assert target.shape == (19,), target.shape
    # delta_c=220 -> 1.0 ; delta_h=15 -> 1.0 ; amp0=3 -> 1.0
    assert abs(peaks_ch[0, 0].item() - 1.0) < 1e-5
    assert abs(peaks_ch[0, 1].item() - 1.0) < 1e-5
    assert abs(peaks_ch[0, 2].item() - 1.0) < 1e-5
    # 13c delta_c=220 -> 1.0 ; 110 -> 0.5
    assert abs(peaks_13c[0, 0].item() - 1.0) < 1e-5
    assert abs(peaks_13c[1, 0].item() - 0.5) < 1e-5
    print("[OK] shapes y normalizacion correctas")


if __name__ == "__main__":
    test_shapes_and_normalization()
    print("\n>>> TEST DATASET E3 OK <<<")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd experiments/E3_dos_conjuntos && python tests/test_dataset_e3.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'dataset_e3'`.

- [ ] **Step 3: Write the dataset**

```python
# experiments/E3_dos_conjuntos/dataset_e3.py
# coding: ascii
import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem


class NMRTwoSetsDataset(Dataset):
    """Exp E Fase 3 -- dos conjuntos de picos:
      - crosspeaks C-H (delta_c, delta_h, amp_ch0, amp_ch1), de peaks_pkl (Fase 1b).
      - 13C (delta_c,), de peaks_13c (Fase 3, Task 1) -- incluye cuaternarios.
    Normaliza los desplazamientos min-max con la calibracion del config
    (norm_cfg). Condicionante FM identico a dataset_v10/dataset_e2 (8 valores).
    """
    def __init__(self, peaks_ch_path, peaks_13c_path, labels_path, smiles_path, norm_cfg):
        self.labels = np.load(labels_path).astype(np.float32)
        self.smiles = np.load(smiles_path, allow_pickle=True)

        npz_ch = np.load(peaks_ch_path)
        peaks_ch = npz_ch["peaks"].astype(np.float32)          # (N, 32, 4)
        self.mask_ch = npz_ch["peaks_mask"].astype(np.float32)  # (N, 32)

        npz_c13 = np.load(peaks_13c_path)
        peaks_13c = npz_c13["peaks_13c"].astype(np.float32)     # (N, M, 1)
        self.mask_13c = npz_c13["mask_13c"].astype(np.float32)  # (N, M)

        # --- normalizacion min-max desde el config (no hardcodear valores) ---
        c_min, c_max = float(norm_cfg["c13_ppm_min"]), float(norm_cfg["c13_ppm_max"])
        h_min, h_max = float(norm_cfg["h1_ppm_min"]), float(norm_cfg["h1_ppm_max"])
        amp0_scale = float(norm_cfg["amp_ch0_scale"])
        peaks_ch[:, :, 0] = (peaks_ch[:, :, 0] - c_min) / (c_max - c_min)
        peaks_ch[:, :, 1] = (peaks_ch[:, :, 1] - h_min) / (h_max - h_min)
        peaks_ch[:, :, 2] = peaks_ch[:, :, 2] / amp0_scale
        # amp_ch1 (col 3) se deja como esta.
        peaks_13c[:, :, 0] = (peaks_13c[:, :, 0] - c_min) / (c_max - c_min)
        self.peaks_ch = peaks_ch
        self.peaks_13c = peaks_13c

        print("[INFO] Extrayendo formulas moleculares (C,H,N,O,S,Hal)...")
        self.formula_matrix = np.zeros((len(self.smiles), 6), dtype=np.float32)
        for i, smi in enumerate(self.smiles):
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                mol = Chem.AddHs(mol)
                c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
                h = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
                n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
                o = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
                s = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
                hal = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53])
                self.formula_matrix[i] = [c, h, n, o, s, hal]
        print("[INFO] Formulas moleculares cargadas.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        peaks_ch = torch.tensor(self.peaks_ch[idx], dtype=torch.float32)
        mask_ch = torch.tensor(self.mask_ch[idx], dtype=torch.float32)
        peaks_13c = torch.tensor(self.peaks_13c[idx], dtype=torch.float32)
        mask_13c = torch.tensor(self.mask_13c[idx], dtype=torch.float32)

        target_vec = self.labels[idx]
        total_signals = np.sum(target_vec).astype(np.float32)
        total_ch2 = (target_vec[1] + target_vec[5] +
                     target_vec[9] + target_vec[12]).astype(np.float32)
        cond_data = [total_signals, total_ch2] + self.formula_matrix[idx].tolist()
        cond_tensor = torch.tensor(cond_data, dtype=torch.float32)

        return (peaks_ch, mask_ch, peaks_13c, mask_13c, cond_tensor), torch.tensor(target_vec)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd experiments/E3_dos_conjuntos && python tests/test_dataset_e3.py`
Expected: PASS — `>>> TEST DATASET E3 OK <<<`.

- [ ] **Step 5: Commit**

```bash
git add experiments/E3_dos_conjuntos/dataset_e3.py experiments/E3_dos_conjuntos/tests/test_dataset_e3.py
git commit -m "exp-e-fase3: dataset de dos conjuntos con normalizacion min-max"
```

---

### Task 3: Modelo DeepSets de dos ramas (`model_e3_deepsets.py`)

**Files:**
- Create: `experiments/E3_dos_conjuntos/model_e3_deepsets.py`
- Create: `experiments/E3_dos_conjuntos/tests/test_forward_deepsets.py`

**Interfaces:**
- Produces: `NMR_DeepSets(num_classes=19)`. `forward(peaks_ch, mask_ch, peaks_13c, mask_13c, cond) -> (B, 19)`.

- [ ] **Step 1: Write the failing test**

```python
# experiments/E3_dos_conjuntos/tests/test_forward_deepsets.py
# coding: ascii
"""Smoke test offline del DeepSets de dos ramas (rule 5)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model_e3_deepsets import NMR_DeepSets

N_CLASSES, MAX_CH, MAX_13C = 19, 32, 40


def test_forward_shape():
    model = NMR_DeepSets(num_classes=N_CLASSES).eval()
    B = 4
    with torch.no_grad():
        out = model(torch.randn(B, MAX_CH, 4), torch.ones(B, MAX_CH),
                    torch.randn(B, MAX_13C, 1), torch.ones(B, MAX_13C),
                    torch.randn(B, 8))
    assert out.shape == (B, N_CLASSES), out.shape
    print(f"[OK] forward -> {tuple(out.shape)}")


def test_empty_molecule_no_nan():
    model = NMR_DeepSets(num_classes=N_CLASSES).eval()
    B = 2
    mask_ch = torch.ones(B, MAX_CH); mask_13c = torch.ones(B, MAX_13C)
    mask_ch[0] = 0.0; mask_13c[0] = 0.0   # molecula 0 sin picos en ambos
    with torch.no_grad():
        out = model(torch.randn(B, MAX_CH, 4), mask_ch,
                    torch.randn(B, MAX_13C, 1), mask_13c, torch.randn(B, 8))
    assert torch.isfinite(out).all(), out
    print("[OK] molecula sin picos -> sin NaN/Inf")


def test_param_count_small():
    model = NMR_DeepSets(num_classes=N_CLASSES)
    n = sum(p.numel() for p in model.parameters())
    assert n < 100_000, n   # chico por diseno (V10 ~8.6M)
    print(f"[OK] parametros = {n:,} (chico por diseno)")


if __name__ == "__main__":
    test_forward_shape(); test_empty_molecule_no_nan(); test_param_count_small()
    print("\n>>> SMOKE DEEPSETS E3 OK <<<")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd experiments/E3_dos_conjuntos && python tests/test_forward_deepsets.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'model_e3_deepsets'`.

- [ ] **Step 3: Write the model**

```python
# experiments/E3_dos_conjuntos/model_e3_deepsets.py
# coding: ascii
import torch
import torch.nn as nn
import torch.nn.functional as F


class NMR_DeepSets(nn.Module):
    """Exp E Fase 3 -- DeepSets de dos ramas:
      - Rama crosspeaks: MLP 4->64->64 por pico, promedio enmascarado -> aggA.
      - Rama 13C:        MLP 1->64->64 por pico, promedio enmascarado -> aggB.
      - Fusion: [aggA(64), aggB(64), cond(8)] -> 128 -> 64 -> num_classes.
    """
    def __init__(self, num_classes=19):
        super().__init__()
        self.ch_mlp1 = nn.Linear(4, 64)
        self.ch_mlp2 = nn.Linear(64, 64)
        self.c13_mlp1 = nn.Linear(1, 64)
        self.c13_mlp2 = nn.Linear(64, 64)

        fusion_dim = 64 + 64 + 8
        self.fc_fusion1 = nn.Linear(fusion_dim, 128)
        self.fc_fusion2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)

    @staticmethod
    def _masked_mean(x, mask):
        # x: (B, T, 64), mask: (B, T)
        m = mask.unsqueeze(-1)                    # (B, T, 1)
        counts = m.sum(dim=1).clamp(min=1.0)      # (B, 1)
        return (x * m).sum(dim=1) / counts        # (B, 64)

    def forward(self, peaks_ch, mask_ch, peaks_13c, mask_13c, cond):
        a = F.relu(self.ch_mlp1(peaks_ch))
        a = F.relu(self.ch_mlp2(a))
        aggA = self._masked_mean(a, mask_ch)

        b = F.relu(self.c13_mlp1(peaks_13c))
        b = F.relu(self.c13_mlp2(b))
        aggB = self._masked_mean(b, mask_13c)

        x = torch.cat((aggA, aggB, cond), dim=1)
        x = F.relu(self.fc_fusion1(x))
        x = F.relu(self.fc_fusion2(x))
        return self.fc_out(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd experiments/E3_dos_conjuntos && python tests/test_forward_deepsets.py`
Expected: PASS — `>>> SMOKE DEEPSETS E3 OK <<<`.

- [ ] **Step 5: Commit**

```bash
git add experiments/E3_dos_conjuntos/model_e3_deepsets.py experiments/E3_dos_conjuntos/tests/test_forward_deepsets.py
git commit -m "exp-e-fase3: modelo DeepSets de dos ramas"
```

---

### Task 4: Modelo Set Transformer (`model_e3_settransformer.py`)

Self-attention (SAB) sobre la unión de los dos conjuntos con embedding de tipo, pooling por atención (PMA). Implementación estándar de Set Transformer (Lee et al. 2019) con máscara de padding.

**Files:**
- Create: `experiments/E3_dos_conjuntos/model_e3_settransformer.py`
- Create: `experiments/E3_dos_conjuntos/tests/test_forward_settransformer.py`

**Interfaces:**
- Produces: `NMR_SetTransformer(num_classes=19, d_model=64, n_heads=4, n_layers=2, n_seeds=1)`. `forward(peaks_ch, mask_ch, peaks_13c, mask_13c, cond) -> (B, 19)`.

- [ ] **Step 1: Write the failing test**

```python
# experiments/E3_dos_conjuntos/tests/test_forward_settransformer.py
# coding: ascii
"""Smoke test offline del Set Transformer (rule 5). Verifica shape, ausencia
de NaN con molecula vacia, invariancia a permutacion, y tamano chico."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model_e3_settransformer import NMR_SetTransformer

N_CLASSES, MAX_CH, MAX_13C = 19, 32, 40


def test_forward_shape():
    model = NMR_SetTransformer(num_classes=N_CLASSES).eval()
    B = 4
    with torch.no_grad():
        out = model(torch.randn(B, MAX_CH, 4), torch.ones(B, MAX_CH),
                    torch.randn(B, MAX_13C, 1), torch.ones(B, MAX_13C),
                    torch.randn(B, 8))
    assert out.shape == (B, N_CLASSES), out.shape
    print(f"[OK] forward -> {tuple(out.shape)}")


def test_empty_molecule_no_nan():
    model = NMR_SetTransformer(num_classes=N_CLASSES).eval()
    B = 2
    mask_ch = torch.ones(B, MAX_CH); mask_13c = torch.ones(B, MAX_13C)
    mask_ch[0] = 0.0; mask_13c[0] = 0.0
    with torch.no_grad():
        out = model(torch.randn(B, MAX_CH, 4), mask_ch,
                    torch.randn(B, MAX_13C, 1), mask_13c, torch.randn(B, 8))
    assert torch.isfinite(out).all(), out
    print("[OK] molecula sin picos -> sin NaN/Inf")


def test_permutation_invariance():
    model = NMR_SetTransformer(num_classes=N_CLASSES).eval()
    peaks_ch = torch.randn(1, MAX_CH, 4); mask_ch = torch.ones(1, MAX_CH)
    peaks_13c = torch.randn(1, MAX_13C, 1); mask_13c = torch.ones(1, MAX_13C)
    cond = torch.randn(1, 8)
    perm = torch.randperm(MAX_CH)
    with torch.no_grad():
        o1 = model(peaks_ch, mask_ch, peaks_13c, mask_13c, cond)
        o2 = model(peaks_ch[:, perm], mask_ch[:, perm], peaks_13c, mask_13c, cond)
    assert torch.allclose(o1, o2, atol=1e-4), (o1 - o2).abs().max()
    print("[OK] invariante a permutacion de los picos")


def test_param_count_small():
    model = NMR_SetTransformer(num_classes=N_CLASSES)
    n = sum(p.numel() for p in model.parameters())
    assert n < 200_000, n   # chico por diseno (V10 ~8.6M)
    print(f"[OK] parametros = {n:,} (chico por diseno)")


if __name__ == "__main__":
    test_forward_shape(); test_empty_molecule_no_nan()
    test_permutation_invariance(); test_param_count_small()
    print("\n>>> SMOKE SET TRANSFORMER E3 OK <<<")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd experiments/E3_dos_conjuntos && python tests/test_forward_settransformer.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'model_e3_settransformer'`.

- [ ] **Step 3: Write the model**

```python
# experiments/E3_dos_conjuntos/model_e3_settransformer.py
# coding: ascii
"""Set Transformer (Lee et al. 2019) sobre la union de los dos conjuntos de
picos, con embedding de tipo (crosspeak / 13C) y mascara de padding. MAB/SAB/PMA
adaptados con key_padding_mask; nan_to_num tras el softmax evita NaN cuando una
fila queda totalmente enmascarada (molecula sin picos)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, valid_mask=None):
        # Q: (B, nq, dim_Q), K: (B, nk, dim_K)
        # valid_mask: (B, nk) float/bool, 1/True = token valido, 0/False = padding.
        Qp = self.fc_q(Q); Kp = self.fc_k(K); Vp = self.fc_v(K)
        H = self.num_heads
        d = self.dim_V // H
        Qh = torch.cat(Qp.split(d, 2), 0)   # (H*B, nq, d)
        Kh = torch.cat(Kp.split(d, 2), 0)
        Vh = torch.cat(Vp.split(d, 2), 0)
        logits = Qh.bmm(Kh.transpose(1, 2)) / math.sqrt(d)   # (H*B, nq, nk)
        if valid_mask is not None:
            vm = (valid_mask > 0.5)                          # (B, nk) bool
            vm = vm.repeat(H, 1).unsqueeze(1)                # (H*B, 1, nk)
            logits = logits.masked_fill(~vm, float("-inf"))
            A = torch.softmax(logits, dim=2)
            A = torch.nan_to_num(A, nan=0.0)                 # filas todo -inf -> 0
        else:
            A = torch.softmax(logits, dim=2)
        O = Qh + A.bmm(Vh)
        O = torch.cat(O.split(Q.size(0), 0), 2)              # (B, nq, dim_V)
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, X, valid_mask=None):
        return self.mab(X, X, valid_mask)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, X, valid_mask=None):
        S = self.S.repeat(X.size(0), 1, 1)
        return self.mab(S, X, valid_mask)   # (B, num_seeds, dim)


class NMR_SetTransformer(nn.Module):
    def __init__(self, num_classes=19, d_model=64, n_heads=4, n_layers=2, n_seeds=1):
        super().__init__()
        self.proj_ch = nn.Linear(4, d_model)
        self.proj_13c = nn.Linear(1, d_model)
        self.type_emb = nn.Embedding(2, d_model)   # 0=crosspeak, 1=13C
        self.encoder = nn.ModuleList([SAB(d_model, n_heads) for _ in range(n_layers)])
        self.pma = PMA(d_model, n_heads, n_seeds)

        fusion_dim = d_model * n_seeds + 8
        self.fc_fusion1 = nn.Linear(fusion_dim, 128)
        self.fc_fusion2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, peaks_ch, mask_ch, peaks_13c, mask_13c, cond):
        B = peaks_ch.size(0)
        tok_ch = self.proj_ch(peaks_ch) + self.type_emb.weight[0].view(1, 1, -1)
        tok_13c = self.proj_13c(peaks_13c) + self.type_emb.weight[1].view(1, 1, -1)
        tokens = torch.cat([tok_ch, tok_13c], dim=1)      # (B, T, d_model)
        valid = torch.cat([mask_ch, mask_13c], dim=1)     # (B, T)

        x = tokens
        for sab in self.encoder:
            x = sab(x, valid)
        pooled = self.pma(x, valid).reshape(B, -1)        # (B, d_model*n_seeds)

        h = torch.cat([pooled, cond], dim=1)
        h = F.relu(self.fc_fusion1(h))
        h = F.relu(self.fc_fusion2(h))
        return self.fc_out(h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd experiments/E3_dos_conjuntos && python tests/test_forward_settransformer.py`
Expected: PASS — `>>> SMOKE SET TRANSFORMER E3 OK <<<`.

- [ ] **Step 5: Commit**

```bash
git add experiments/E3_dos_conjuntos/model_e3_settransformer.py experiments/E3_dos_conjuntos/tests/test_forward_settransformer.py
git commit -m "exp-e-fase3: modelo Set Transformer (SAB+PMA sobre la union de conjuntos)"
```

---

### Task 5: `train.py` parametrizado + configs + `split_utils.py`

Un solo `train.py` que instancia el modelo según `model.arch`. Se copia `split_utils.py` de E2 sin cambios (self-contained). Dos configs que solo difieren en arch/nombres.

**Files:**
- Create: `experiments/E3_dos_conjuntos/train.py`
- Create: `experiments/E3_dos_conjuntos/config_deepsets.yaml`
- Create: `experiments/E3_dos_conjuntos/config_settransformer.yaml`
- Create: `experiments/E3_dos_conjuntos/split_utils.py` (copia byte-a-byte de `experiments/E2_deepsets/split_utils.py`)
- Reference (no modificar): `experiments/E2_deepsets/train.py`

**Interfaces:**
- Consumes: `NMRTwoSetsDataset` (Task 2), `NMR_DeepSets` (Task 3), `NMR_SetTransformer` (Task 4), `canonicalize_smiles`/`remove_leaking_from_train` (split_utils).
- Produces: checkpoint `{base_dir}/{checkpoint_dir}/{experiment_name}_best.pth`.

- [ ] **Step 1: Copiar split_utils.py**

Copiá `experiments/E2_deepsets/split_utils.py` a `experiments/E3_dos_conjuntos/split_utils.py` sin cambios (ya lo leíste; contiene `canonicalize_smiles` y `remove_leaking_from_train`).

- [ ] **Step 2: Escribir los dos configs**

```yaml
# experiments/E3_dos_conjuntos/config_deepsets.yaml
experiment_name: "nmr_202k_e3_deepsets_2sets_19v"

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  peaks_ch_filename: "peaks_pkl_202465.npz"
  peaks_13c_filename: "peaks_13c_202465.npz"
  labels_filename: "vectors_13c_19v_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_E3_deepsets"
  val_indices_filename: "val_indices_frozen.npy"

model:
  arch: "deepsets"

normalization:
  c13_ppm_min: 0
  c13_ppm_max: 220
  h1_ppm_min: -1
  h1_ppm_max: 15
  amp_ch0_scale: 3.0

hyperparameters:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  scheduler:
    patience: 8
    factor: 0.7

system:
  device: "cuda"
  num_workers: 0
  pin_memory: true
```

```yaml
# experiments/E3_dos_conjuntos/config_settransformer.yaml
experiment_name: "nmr_202k_e3_settransformer_2sets_19v"

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  peaks_ch_filename: "peaks_pkl_202465.npz"
  peaks_13c_filename: "peaks_13c_202465.npz"
  labels_filename: "vectors_13c_19v_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_E3_settransformer"
  val_indices_filename: "val_indices_frozen.npy"

model:
  arch: "settransformer"
  d_model: 64
  n_heads: 4
  n_layers: 2
  n_seeds: 1

normalization:
  c13_ppm_min: 0
  c13_ppm_max: 220
  h1_ppm_min: -1
  h1_ppm_max: 15
  amp_ch0_scale: 3.0

hyperparameters:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  scheduler:
    patience: 8
    factor: 0.7

system:
  device: "cuda"
  num_workers: 0
  pin_memory: true
```

- [ ] **Step 3: Escribir train.py**

```python
# experiments/E3_dos_conjuntos/train.py
# coding: ascii
"""
train.py -- Exp E Fase 3: entrena DeepSets o Set Transformer (segun
model.arch del config) sobre los dos conjuntos de picos (crosspeaks C-H +
13C). Sin regularizacion (misma decision que Exp C/E2). Split congelado de
Exp D (val_indices_frozen.npy). Todo lo demas identico a E2 salvo el modelo,
los dos conjuntos y la normalizacion (que vive en el dataset).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import time, os, yaml, argparse, random
import numpy as np
from pathlib import Path

from dataset_e3 import NMRTwoSetsDataset
from split_utils import canonicalize_smiles, remove_leaking_from_train


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConstrainedMSELoss(nn.Module):
    def __init__(self, lambda_sum=0.5):
        super().__init__()
        self.mse = nn.MSELoss(); self.lambda_sum = lambda_sum

    def forward(self, pred, target):
        li = self.mse(pred, target)
        ls = self.mse(torch.sum(pred, dim=1), torch.sum(target, dim=1))
        return li + self.lambda_sum * ls


def load_config(p):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(cfg, num_classes=19):
    arch = cfg['model']['arch']
    if arch == 'deepsets':
        from model_e3_deepsets import NMR_DeepSets
        return NMR_DeepSets(num_classes=num_classes)
    if arch == 'settransformer':
        from model_e3_settransformer import NMR_SetTransformer
        m = cfg['model']
        return NMR_SetTransformer(
            num_classes=num_classes,
            d_model=int(m.get('d_model', 64)),
            n_heads=int(m.get('n_heads', 4)),
            n_layers=int(m.get('n_layers', 2)),
            n_seeds=int(m.get('n_seeds', 1)),
        )
    raise ValueError(f"model.arch desconocido: {arch!r} (usar 'deepsets' o 'settransformer')")


def build_frozen_split(full_dataset, base_dir, cfg):
    val_indices_path = base_dir / cfg['paths']['val_indices_filename']
    if not os.path.exists(val_indices_path):
        raise FileNotFoundError(
            f"No se encontro el split congelado en: {val_indices_path}\n"
            "Corri primero experiments/D_val_congelado/split.py (Exp D)."
        )
    val_idx = np.load(val_indices_path)
    smiles_path = base_dir / cfg['paths']['smiles_filename']
    smiles = np.load(smiles_path, allow_pickle=True)
    canonical, n_invalid = canonicalize_smiles(smiles)

    all_idx = np.arange(len(full_dataset))
    train_idx_raw = np.setdiff1d(all_idx, val_idx, assume_unique=False)
    train_idx, n_removed = remove_leaking_from_train(train_idx_raw, val_idx, canonical)
    print(f"[INFO] Split congelado: SMILES invalidos={n_invalid} | "
          f"train={len(train_idx)} (leak removido={n_removed}) | val={len(val_idx)}")
    return train_idx, val_idx


def unpack(inputs, device):
    return (inputs[0].to(device), inputs[1].to(device),
            inputs[2].to(device), inputs[3].to(device), inputs[4].to(device))


def validate(model, loader, criterion, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            pch, mch, p13, m13, cond = unpack(inputs, device)
            targets = targets.to(device)
            total += criterion(model(pch, mch, p13, m13, cond), targets).item()
    return total / len(loader)


def train(config_path):
    set_seed(42)
    cfg = load_config(config_path)
    print(f"--- ENTRENAMIENTO EXP E FASE 3 ({cfg['model']['arch']}): {cfg['experiment_name']} ---")

    base_dir = Path(cfg['paths']['base_dir'])
    peaks_ch = base_dir / cfg['paths']['peaks_ch_filename']
    peaks_13c = base_dir / cfg['paths']['peaks_13c_filename']
    labels_path = base_dir / cfg['paths']['labels_filename']
    smiles_path = base_dir / cfg['paths']['smiles_filename']
    ckpt_dir = base_dir / cfg['paths']['checkpoint_dir']
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    full_dataset = NMRTwoSetsDataset(str(peaks_ch), str(peaks_13c), str(labels_path),
                                     str(smiles_path), cfg['normalization'])
    train_idx, val_idx = build_frozen_split(full_dataset, base_dir, cfg)
    train_ds = Subset(full_dataset, train_idx.tolist())
    val_ds = Subset(full_dataset, val_idx.tolist())

    use_pin = cfg['system'].get('pin_memory', False) and device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=cfg['hyperparameters']['batch_size'],
                              shuffle=True, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=cfg['hyperparameters']['batch_size'],
                            shuffle=False, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)

    model = build_model(cfg, num_classes=19).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parametros totales del modelo ({cfg['model']['arch']}): {n_params:,} "
          f"(chico por diseno; V10 ~8,603,299)")

    criterion = ConstrainedMSELoss(lambda_sum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=cfg['hyperparameters']['learning_rate'])
    sched_cfg = cfg['hyperparameters'].get('scheduler', {})
    patience = sched_cfg.get('patience', 8); factor = sched_cfg.get('factor', 0.7)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    print(f"[INFO] Scheduler: patience={patience}, factor={factor}")

    epochs = cfg['hyperparameters']['epochs']
    print(f"\n[START] {epochs} epochs...")
    start_time = time.time(); best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train(); running_loss = 0.0; epoch_start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            pch, mch, p13, m13, cond = unpack(inputs, device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(pch, mch, p13, m13, cond)
            loss = criterion(outputs, targets)
            loss.backward(); optimizer.step()
            running_loss += loss.item()
            if batch_idx % 200 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        if device.type == 'cuda':
            torch.cuda.synchronize()
        val_loss = validate(model, val_loader, criterion, device)
        avg_train = running_loss / len(train_loader)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f"[EPOCH {epoch+1}] Train: {avg_train:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f} | Time: {time.time()-epoch_start:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / f"{cfg['experiment_name']}_best.pth")
            print("[SAVE] Nuevo mejor modelo!")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"{cfg['experiment_name']}_ep{epoch+1}.pth")

    print(f"\n[DONE] {(time.time()-start_time)/60:.1f} min. Mejor Val: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_deepsets.yaml")
    args = parser.parse_args()
    train(args.config)
```

- [ ] **Step 4: Verificar la instanciación de ambos modelos (offline, sin datos)**

Run:
```bash
cd experiments/E3_dos_conjuntos && python -c "import train; \
d=train.build_model({'model':{'arch':'deepsets'}}); \
s=train.build_model({'model':{'arch':'settransformer','d_model':64,'n_heads':4,'n_layers':2,'n_seeds':1}}); \
print('deepsets params:', sum(p.numel() for p in d.parameters())); \
print('settransformer params:', sum(p.numel() for p in s.parameters()))"
```
Expected: imprime los dos conteos de parámetros sin error (ambos < 200k).

- [ ] **Step 5: Commit**

```bash
git add experiments/E3_dos_conjuntos/train.py experiments/E3_dos_conjuntos/config_deepsets.yaml experiments/E3_dos_conjuntos/config_settransformer.yaml experiments/E3_dos_conjuntos/split_utils.py
git commit -m "exp-e-fase3: train.py parametrizado por arch + configs + split_utils"
```

---

### Task 6: `evaluate.py` y `dump_predictions.py` adaptados

Adaptación de los de E2: mismos cálculos de EMA cruda/asistida y oráculo (idénticos), pero (a) desempaquetan 5 inputs, (b) instancian el modelo según `model.arch`, (c) leen los dos conjuntos + normalización.

**Files:**
- Create: `experiments/E3_dos_conjuntos/evaluate.py`
- Create: `experiments/E3_dos_conjuntos/dump_predictions.py`
- Create: `experiments/E3_dos_conjuntos/tests/test_oraculo.py`
- Reference (no modificar): `experiments/E2_deepsets/evaluate.py`, `experiments/E2_deepsets/dump_predictions.py`

**Interfaces:**
- Consumes: `NMRTwoSetsDataset` (Task 2), `build_model` (Task 5, importado de `train`).

- [ ] **Step 1: Write the failing test (funciones puras del oráculo)**

```python
# experiments/E3_dos_conjuntos/tests/test_oraculo.py
# coding: ascii
"""El oraculo de doble restriccion es identico al de E2/Exp C. Este test fija
su comportamiento: con la restriccion de suma total y de CH2, la prediccion
asistida debe respetar ambos totales."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from evaluate import ajustar_conteo_doble_exacto, crude_predict, IDX_CH2


def test_asistida_respeta_totales():
    pred_raw = np.array([0.6, 1.4, 0.2, 0.1, 0, 0, 0, 0, 0, 0.9, 0, 0,
                         0.3, 2.1, 1.2, 0, 0, 0, 0], dtype=float)
    total_real, ch2_real = 8, 3
    out = ajustar_conteo_doble_exacto(pred_raw, total_real, ch2_real)
    assert out.sum() == total_real, out.sum()
    assert sum(out[i] for i in IDX_CH2) == ch2_real, out
    print(f"[OK] asistida respeta total={total_real} y ch2={ch2_real}: {out}")


def test_crude_es_floor_no_negativo():
    pred_raw = np.array([-0.3, 1.9, 0.4] + [0.0] * 16, dtype=float)
    out = crude_predict(pred_raw)
    assert out[0] == 0 and out[1] == 1 and out[2] == 0, out
    print("[OK] crude = floor con clip a >=0")


if __name__ == "__main__":
    test_asistida_respeta_totales(); test_crude_es_floor_no_negativo()
    print("\n>>> TEST ORACULO E3 OK <<<")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd experiments/E3_dos_conjuntos && python tests/test_oraculo.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'evaluate'`.

- [ ] **Step 3: Escribir evaluate.py**

Copiá `experiments/E2_deepsets/evaluate.py` a `experiments/E3_dos_conjuntos/evaluate.py` y aplicá EXACTAMENTE estos cambios (todo lo demás — `GROUP_NAMES`, `IDX_CH2`, `crude_predict`, `ajustar_conteo_doble_exacto`, `compute_ema`, `report_mode`, `print_confusiones`, `print_tabla_comparativa` — queda idéntico):

1. En `evaluate(config_path, oraculo, eval_batch_size)`, reemplazá el bloque de imports perezosos y de carga del dataset/modelo:

```python
    # Import perezoso: dataset trae rdkit; el test de oraculo no lo necesita.
    from dataset_e3 import NMRTwoSetsDataset
    from train import build_model

    cfg = load_config(config_path)

    base_dir = Path(cfg["paths"]["base_dir"])
    peaks_ch = base_dir / cfg["paths"]["peaks_ch_filename"]
    peaks_13c = base_dir / cfg["paths"]["peaks_13c_filename"]
    labels_path = base_dir / cfg["paths"]["labels_filename"]
    smiles_path = base_dir / cfg["paths"]["smiles_filename"]
    ckpt_path = base_dir / cfg["paths"]["checkpoint_dir"] / f"{cfg['experiment_name']}_best.pth"
    val_indices_path = base_dir / cfg["paths"]["val_indices_filename"]
    num_workers = int(cfg["system"]["num_workers"])
```

2. Reemplazá la construcción del dataset y del modelo:

```python
    full_dataset = NMRTwoSetsDataset(str(peaks_ch), str(peaks_13c), str(labels_path),
                                     str(smiles_path), cfg["normalization"])
    val_indices = np.load(val_indices_path)
    val_ds = Subset(full_dataset, val_indices.tolist())
    ...
    model = build_model(cfg, num_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
```

3. En el loop de inferencia, reemplazá el desempaque y el forward:

```python
    with torch.no_grad():
        for inputs, targets in val_loader:
            pch = inputs[0].to(device); mch = inputs[1].to(device)
            p13 = inputs[2].to(device); m13 = inputs[3].to(device)
            cond = inputs[4].to(device)
            pred_raw = model(pch, mch, p13, m13, cond).cpu().numpy()
            targs = targets.cpu().numpy().astype(int)
            cond_np = cond.cpu().numpy()
            all_targets.append(targs)
            if run_off:
                all_pred_off.append(crude_predict(pred_raw))
            if run_on:
                batch_on = np.empty_like(targs)
                for k in range(len(targs)):
                    batch_on[k] = ajustar_conteo_doble_exacto(
                        pred_raw[k], int(cond_np[k, 0]), int(cond_np[k, 1]))
                all_pred_on.append(batch_on)
```

4. Cambiá el título impreso `EVALUACION EXP E FASE 2 ...` por `EVALUACION EXP E FASE 3 (dos conjuntos) - SPLIT CONGELADO` y el `argparse` default de `--config` a `"config_deepsets.yaml"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd experiments/E3_dos_conjuntos && python tests/test_oraculo.py`
Expected: PASS — `>>> TEST ORACULO E3 OK <<<`.

- [ ] **Step 5: Escribir dump_predictions.py**

Copiá `experiments/E2_deepsets/dump_predictions.py` a `experiments/E3_dos_conjuntos/dump_predictions.py` y aplicá estos cambios (el resto — `oraculo_doble`, la escritura del parquet — idéntico):

1. Imports y paths en `main`:

```python
    from dataset_e3 import NMRTwoSetsDataset
    from train import build_model

    cfg = load_config(config_path)
    base_dir = Path(cfg["paths"]["base_dir"])
    peaks_ch = base_dir / cfg["paths"]["peaks_ch_filename"]
    peaks_13c = base_dir / cfg["paths"]["peaks_13c_filename"]
    labels_path = base_dir / cfg["paths"]["labels_filename"]
    smiles_path = base_dir / cfg["paths"]["smiles_filename"]
    ckpt_path = base_dir / cfg["paths"]["checkpoint_dir"] / f"{cfg['experiment_name']}_best.pth"
    val_indices_path = base_dir / cfg["paths"]["val_indices_filename"]
    out_file = f"predictions_{cfg['experiment_name']}.parquet"
```

2. Construcción del dataset y modelo:

```python
    ds = NMRTwoSetsDataset(str(peaks_ch), str(peaks_13c), str(labels_path),
                           str(smiles_path), cfg["normalization"])
    ...
    model = build_model(cfg, num_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
```

3. Loop de inferencia:

```python
    with torch.no_grad():
        for inputs, targets in loader:
            pch = inputs[0].to(device); mch = inputs[1].to(device)
            p13 = inputs[2].to(device); m13 = inputs[3].to(device)
            cond = inputs[4].to(device)
            out = model(pch, mch, p13, m13, cond).cpu().numpy()
            t = targets.cpu().numpy().astype(int)
            c = cond.cpu().numpy()
            for k in range(len(t)):
                orig_idx = int(val_indices[ptr]); ptr += 1
                total = int(c[k, 0]); ch2 = int(c[k, 1])
                rows.append({
                    "idx": orig_idx,
                    "smiles": str(smiles_all[orig_idx]),
                    "y_true": t[k].tolist(),
                    "y_pred_crude": np.clip(np.floor(out[k]), 0, None).astype(int).tolist(),
                    "y_pred_assisted": oraculo_doble(out[k], total, ch2).tolist(),
                })
```

4. `argparse` default de `--config` a `"config_deepsets.yaml"`.

- [ ] **Step 6: Commit**

```bash
git add experiments/E3_dos_conjuntos/evaluate.py experiments/E3_dos_conjuntos/dump_predictions.py experiments/E3_dos_conjuntos/tests/test_oraculo.py
git commit -m "exp-e-fase3: evaluate + dump_predictions (dos conjuntos, modelo por arch)"
```

---

### Task 7: Scripts SLURM + README + RATIONALE

**Files:**
- Create: `experiments/E3_dos_conjuntos/run_train_deepsets.sh`
- Create: `experiments/E3_dos_conjuntos/run_train_settransformer.sh`
- Create: `experiments/E3_dos_conjuntos/run_eval.sh`
- Create: `experiments/E3_dos_conjuntos/README.md`
- Create: `experiments/E3_dos_conjuntos/RATIONALE.md`

- [ ] **Step 1: run_train_deepsets.sh**

```bash
#!/bin/bash
#SBATCH --job-name=expE3_ds_train
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE3_ds_train_%j.out
#SBATCH --error=expE3_ds_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

cd ~/nmr-hsqc-to-vector-/experiments/E3_dos_conjuntos
python -u train.py --config config_deepsets.yaml
```

- [ ] **Step 2: run_train_settransformer.sh** (idéntico salvo nombres y config)

```bash
#!/bin/bash
#SBATCH --job-name=expE3_st_train
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE3_st_train_%j.out
#SBATCH --error=expE3_st_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

cd ~/nmr-hsqc-to-vector-/experiments/E3_dos_conjuntos
python -u train.py --config config_settransformer.yaml
```

- [ ] **Step 3: run_eval.sh** (recibe el config como argumento para evaluar cualquiera de los dos)

```bash
#!/bin/bash
#SBATCH --job-name=expE3_eval
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE3_eval_%j.out
#SBATCH --error=expE3_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

cd ~/nmr-hsqc-to-vector-/experiments/E3_dos_conjuntos
CONFIG="${1:-config_deepsets.yaml}"
python -u evaluate.py --config "$CONFIG" --oraculo both --batch-size 256
```

- [ ] **Step 4: README.md** (checklist de ejecución en el cluster)

Contenido: orden de comandos — (1) `git pull`; (2) copiar `peaks_13c_202465.npz` (Task 1) a `/home/lpassaglia.iquir/DB_200k/`; (3) confirmar que existen `peaks_pkl_202465.npz`, `peaks_13c_202465.npz`, `vectors_13c_19v_202465.npy`, `smiles_202465.npy`, `val_indices_frozen.npy`; (4) smoke tests obligatorios (rule 5): `python tests/test_dataset_e3.py`, `python tests/test_forward_deepsets.py`, `python tests/test_forward_settransformer.py`, `python tests/test_oraculo.py`; (5) `sbatch run_train_deepsets.sh` y `sbatch run_train_settransformer.sh`; (6) revisar temprano los `.out` (entrena en minutos); (7) `sbatch run_eval.sh config_deepsets.yaml` y `sbatch run_eval.sh config_settransformer.yaml`; (8) comparar contra Exp C (0.89% cruda) y, sobre todo, mirar si las confusiones `Cqsp2`↔`=CH/Ar` y `CH2`↔`CH2-N` bajaron; (9) agregar dos filas a `docs/Runs/RESULTS.md`; (10) avisar a Claude Code con los números.

- [ ] **Step 5: RATIONALE.md**

Resumen del spec (`docs/superpowers/specs/2026-07-22-...`): por qué se agrega el conjunto ¹³C (los cuaternarios que faltaban en Fase 2), por qué se normaliza, por qué se corren dos arquitecturas (DeepSets aísla el efecto del input; Set Transformer el de la capacidad relacional), y el criterio de éxito (EMA cruda ≥ 0.89%, asistida > Exp C/E2, y caída de las confusiones de cuaternarios). Enlazar al spec y a `docs/Runs/RESULTS.md`.

- [ ] **Step 6: Commit**

```bash
git add experiments/E3_dos_conjuntos/run_train_deepsets.sh experiments/E3_dos_conjuntos/run_train_settransformer.sh experiments/E3_dos_conjuntos/run_eval.sh experiments/E3_dos_conjuntos/README.md experiments/E3_dos_conjuntos/RATIONALE.md
git commit -m "exp-e-fase3: scripts SLURM + README + RATIONALE"
```

---

## Notas de ejecución

- **Tasks 2-7 corren en la máquina de Lucas / cluster.** Task 1 (extracción real, Step 5) y todos los smoke tests corren offline sin GPU. El entrenamiento (`sbatch`) lo lanza Lucas — Claude Code no lo hace (CLAUDE.md).
- **Orden de dependencias:** Task 1 → Task 2 → (Task 3, Task 4 en paralelo) → Task 5 → Task 6 → Task 7.
- **Antes de cualquier `sbatch`:** correr los 4 smoke tests (rule 5). Si el conteo de parámetros de algún modelo no es "chico" (< 200k), revisar antes de gastar GPU.
