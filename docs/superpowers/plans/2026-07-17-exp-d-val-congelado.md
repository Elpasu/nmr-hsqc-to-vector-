# Exp D — Val Set Congelado — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Congelar un val set fijo y deduplicado (14428 moléculas originales de las 144k) en `experiments/D_val_congelado/`, y re-evaluar el checkpoint V10 ya entrenado sobre ese split para obtener una referencia "V10-on-frozen-val" comparable con los futuros Exp B y C.

**Architecture:** Carpeta autocontenida `experiments/D_val_congelado/` (copias exactas de `model_v10.py`/`dataset_v10.py`, sin `sys.path` hacia `src/`). `split.py` separa lógica pura (canonicalización RDKit, detección de duplicados, remoción de leak — testeable localmente sin torch) de la única función que necesita `torch` (reproducir el `random_split(seed=42)` histórico sobre las 144280 moléculas originales). `evaluate.py` es `evaluate_v10.py` adaptado para leer `val_indices_frozen.npy` vía `Subset` en vez de `random_split`. No se reentrena nada.

**Tech Stack:** Python, PyTorch, RDKit, NumPy, PyYAML, h5py (cluster only), SLURM.

## Global Constraints

- `num_workers: 0` en todo DataLoader que use `NMRDataset` (h5py no es fork-safe) — CLAUDE.md regla 1.
- SLURM: `#SBATCH --gres=gpu:1`, nunca `--gpus=1` — CLAUDE.md regla 2.
- Nada hardcodeado salvo lo ya documentado como excepción (nombres de clase, índices CH2, seed=42) — CLAUDE.md regla 3, memoria `exp-config-convention`.
- El h5 a usar es `nmr_dataset_v3_202465_fast.h5` (rechunkeado), nunca el original sin `_fast` — CLAUDE.md regla 4.
- Smoke test obligatorio (`tests/test_forward.py`) antes de proponer cualquier `sbatch` — CLAUDE.md regla 5.
- `num_classes=19`, orden de clases fijo — CLAUDE.md regla 7.
- Métrica primaria = EMA cruda; reportar siempre las dos (cruda y asistida) — PROMPT regla 9.
- Este entorno de desarrollo (Windows, máquina de Lucas) tiene `numpy`, `rdkit`, `pandas` pero **NO tiene `torch`, `h5py`, `pyyaml` ni `pytest`**. Todo lo que dependa de `torch`/`h5py` no se puede ejecutar aquí — se implementa y se documenta como "verificar en el cluster", nunca se afirma haberlo corrido sin haberlo corrido. Todo lo que dependa solo de `numpy`/`rdkit` se implementa con TDD real, ejecutado en este entorno.
- Ningún test usa `pytest` (no está instalado ni en este entorno ni, por convención ya establecida en `tests/test_eval_forward.py`, se asume en el cluster) — son funciones planas invocadas desde `if __name__ == "__main__":`, siguiendo el patrón ya existente en `tests/test_eval_forward.py`.
- No se modifica ningún archivo de `src/` (V10 es la referencia, CLAUDE.md).
- No se ejecuta nada en el cluster — el plan deja todo listo para que Lucas haga `git pull` + smoke test + `sbatch` manualmente.

---

### Task 1: Housekeeping — actualizar `docs/Runs/RESULTS.md` con los números reales del Exp A

**Files:**
- Modify: `docs/Runs/RESULTS.md`

**Interfaces:** N/A (edición de documentación, no código).

- [ ] **Step 1: Reemplazar los "TBD" de la fila V10 con los números reales del Exp A**

En `docs/Runs/RESULTS.md`, reemplazar la tabla superior:

```markdown
| Model | Ch | Cls | Data | Reg. | Best Val Loss (ep) | EMA crude | EMA assist | Notes |
|-------|----|----|------|------|--------------------|-----------|------------|-------|
| V10 baseline | 2 | 19 | 202k | none | 0.0303 (76) | 0.61% | 74.92% | overfits from ~ep48; assisted EMA inflated by oracle (Exp A) |
```

Y la sección `## V10 — baseline`, reemplazar la línea `- **EMA:** TBD → run Exp A (...)` por:

```markdown
- **EMA:** cruda 0.61% · asistida 74.92% (Exp A, oráculo de doble restricción). Δ = +74.3pp — la EMA asistida satura la métrica, no sirve para comparar versiones. Ver `docs/PROMPT_superpowers_mejoras.md`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/Runs/RESULTS.md
git commit -m "docs: registra los EMA reales del Exp A en RESULTS.md (eran TBD)"
```

---

### Task 2: Scaffold de `experiments/D_val_congelado/` + copias exactas del baseline

**Files:**
- Create: `experiments/D_val_congelado/model_v10.py` (copia exacta de `src/models/model_v10.py`)
- Create: `experiments/D_val_congelado/dataset_v10.py` (copia exacta de `src/data/dataset_v10.py`)

**Interfaces:**
- Produces: `NMR_Net(num_classes=19)` (de `model_v10.py`) y `NMRDataset(h5_path, labels_path, smiles_path)` (de `dataset_v10.py`), ambos sin cambios de interfaz respecto al baseline — los usan las Tasks 6 y 8.

- [ ] **Step 1: Copiar los archivos sin modificar nada**

```bash
mkdir -p experiments/D_val_congelado/tests
cp src/models/model_v10.py experiments/D_val_congelado/model_v10.py
cp src/data/dataset_v10.py experiments/D_val_congelado/dataset_v10.py
```

- [ ] **Step 2: Verificar que son copias byte-idénticas**

```bash
diff src/models/model_v10.py experiments/D_val_congelado/model_v10.py
diff src/data/dataset_v10.py experiments/D_val_congelado/dataset_v10.py
```

Expected: ambos comandos sin output (sin diferencias).

- [ ] **Step 3: Commit**

```bash
git add experiments/D_val_congelado/model_v10.py experiments/D_val_congelado/dataset_v10.py
git commit -m "exp D: copia autocontenida de model_v10.py y dataset_v10.py (sin cambios)"
```

---

### Task 3: `RATIONALE.md`

**Files:**
- Create: `experiments/D_val_congelado/RATIONALE.md`

**Interfaces:** N/A (documentación).

- [ ] **Step 1: Escribir RATIONALE.md**

```markdown
# RATIONALE — Exp D: Val set congelado

## Hipótesis

Los resultados de V6..V10 no son estrictamente comparables entre sí: cada
versión particiona un dataset de tamaño distinto con `random_split(seed=42)`,
así que el 10% de val cambia de composición en cada corrida. Además, las
144k moléculas originales tienen del orden de miles de duplicados internos
por SMILES canónico, que pueden caer uno en train y su gemelo en val (fuga
de datos silenciosa). Congelar un val fijo y deduplicado hace que las EMAs
de Exp B, Exp C y cualquier versión futura sean comparables pie a pie.

## Qué causa del diagnóstico ataca

Rigor de comparación (crítica #5 del diagnóstico del Exp A). No ataca el
overfitting (eso lo ataca Exp B) ni el modality collapse (Exp C) — es una
precondición metodológica para que esos dos experimentos den números
confiables entre sí.

## Qué cambia exactamente respecto al V10

- Nada en el modelo ni en el dataset: `model_v10.py` y `dataset_v10.py` se
  copian sin modificar.
- El split: en vez de `random_split(seed=42, val_split=0.1)` sobre las
  202465 moléculas (lo que hace `train_v10.py`), el val queda fijo en las
  14428 moléculas "originales" de las 144k — el mismo `random_split`
  histórico que usó el training de V6-V9, aplicado al rango `[0, 144280)`
  del dataset de 202k (las 58185 nuevas se agregaron al final, sin
  reordenar las originales). Se guarda en `val_indices_frozen.npy`
  (vive en `DB_200k/`, no se versiona en git — igual que los `.h5` y los
  checkpoints).
- Cualquier fila de train cuyo SMILES canónico coincide con una fila de val
  se elimina de train (leak = 0). Las 58185 moléculas nuevas van siempre a
  train (ninguna está en `[0, 144280)`, donde vive el val histórico).
- Se re-evalúa el checkpoint V10 **ya entrenado**
  (`nmr_202k_v10_2ch_fm_19v_best.pth`) sobre este split nuevo — no se
  reentrena nada — para tener una referencia "V10-on-frozen-val" contra la
  que se compararán Exp B y Exp C.

## Qué métrica esperás mover y cuánto

Ninguna EMA debería moverse por una razón de aprendizaje — es el mismo
checkpoint congelado. Se espera un cambio *pequeño* en el número (el val
ahora es más chico, sin los 58k scaffolds nuevos, sin duplicados con fuga)
respecto al 0.61% / 74.92% original. Un cambio grande (>5pp) sería señal de
que el val original y el nuevo tienen dificultad muy distinta, y vale la
pena investigarlo antes de seguir con B/C.

## Criterio de éxito/fracaso

- **Éxito:** se reporta el número de duplicados canónicos encontrados, se
  genera `val_indices_frozen.npy` con ~14428 índices, la verificación
  train ∩ val (por SMILES canónico) da 0, y se obtiene el EMA
  cruda/asistida del V10 sobre ese split nuevo.
- **Fracaso:** el script no logra reproducir un split de tamaño razonable
  (p. ej. por desalineación de orden entre el dataset de 144k histórico y
  el de 202k), o la verificación de leak falla y no se puede corregir sin
  tocar el val (que debe quedar fijo).
```

- [ ] **Step 2: Commit**

```bash
git add experiments/D_val_congelado/RATIONALE.md
git commit -m "exp D: agrega RATIONALE.md"
```

---

### Task 4: `split.py` — canonicalización y detección de duplicados (TDD, corre local)

**Files:**
- Create: `experiments/D_val_congelado/split.py`
- Test: `experiments/D_val_congelado/tests/test_split.py`

**Interfaces:**
- Produces: `canonicalize_smiles(smiles_array: np.ndarray) -> (np.ndarray[object], int)` — devuelve (SMILES canónicos, cantidad de inválidos). `find_duplicate_groups(canonical_smiles: np.ndarray) -> dict[str, list[int]]` — solo grupos con más de una molécula. Usados por la Task 5 y por `main()` (Task 6).

- [ ] **Step 1: Escribir el test (va a fallar: `split.py` no existe todavía)**

Crear `experiments/D_val_congelado/tests/test_split.py`:

```python
# coding: ascii
"""
Tests locales (sin torch/h5py) de la logica pura de split.py -- Exp D.
Corren en cualquier maquina con numpy + rdkit (no requieren el cluster).

Uso:
    python tests/test_split.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from split import (
    canonicalize_smiles,
    find_duplicate_groups,
    remove_leaking_from_train,
)


def test_canonicalize_smiles_dedup_equivalent_forms():
    smiles = np.array(["CCO", "OCC", "C1=CC=CC=C1", "c1ccccc1"], dtype=object)
    canonical, n_invalid = canonicalize_smiles(smiles)
    assert n_invalid == 0
    assert canonical[0] == canonical[1], "etanol en dos formas debe canonicalizar igual"
    assert canonical[2] == canonical[3], "benceno kekulizado/aromatico debe canonicalizar igual"
    print("[OK] test_canonicalize_smiles_dedup_equivalent_forms")


def test_canonicalize_smiles_invalid_passthrough():
    smiles = np.array(["CCO", "not_a_smiles!!"], dtype=object)
    canonical, n_invalid = canonicalize_smiles(smiles)
    assert n_invalid == 1
    assert canonical[1] == "not_a_smiles!!"
    print("[OK] test_canonicalize_smiles_invalid_passthrough")


def test_find_duplicate_groups():
    canonical = np.array(["A", "B", "A", "C", "B", "A"], dtype=object)
    dups = find_duplicate_groups(canonical)
    assert dups == {"A": [0, 2, 5], "B": [1, 4]}
    assert "C" not in dups
    print("[OK] test_find_duplicate_groups")


def test_find_duplicate_groups_no_dups():
    canonical = np.array(["A", "B", "C"], dtype=object)
    assert find_duplicate_groups(canonical) == {}
    print("[OK] test_find_duplicate_groups_no_dups")


def test_remove_leaking_from_train():
    canonical = np.array(["A", "B", "A", "C", "D"], dtype=object)
    # idx 2 es val (canonical "A"); train excluye idx 2 (particion real disjunta).
    # idx 0 tambien es "A" -> duplicado interno del lado train
    # -> debe eliminarse de train por leak contra val.
    train_idx = np.array([0, 1, 3, 4])
    val_idx = np.array([2])
    clean_train, n_removed = remove_leaking_from_train(train_idx, val_idx, canonical)
    assert n_removed == 1
    assert 0 not in clean_train
    assert set(clean_train.tolist()) == {1, 3, 4}
    print("[OK] test_remove_leaking_from_train")


def test_remove_leaking_from_train_no_leak():
    canonical = np.array(["A", "B", "C"], dtype=object)
    train_idx = np.array([0, 1])
    val_idx = np.array([2])
    clean_train, n_removed = remove_leaking_from_train(train_idx, val_idx, canonical)
    assert n_removed == 0
    assert set(clean_train.tolist()) == {0, 1}
    print("[OK] test_remove_leaking_from_train_no_leak")


if __name__ == "__main__":
    test_canonicalize_smiles_dedup_equivalent_forms()
    test_canonicalize_smiles_invalid_passthrough()
    test_find_duplicate_groups()
    test_find_duplicate_groups_no_dups()
    test_remove_leaking_from_train()
    test_remove_leaking_from_train_no_leak()
    print("\n>>> SPLIT CORE TESTS OK <<<")
```

- [ ] **Step 2: Correr el test para confirmar que falla**

Run: `python experiments/D_val_congelado/tests/test_split.py`
Expected: `ModuleNotFoundError: No module named 'split'` (todavía no existe `split.py`).

- [ ] **Step 3: Crear `split.py` con `canonicalize_smiles` y `find_duplicate_groups`**

Crear `experiments/D_val_congelado/split.py`:

```python
# coding: ascii
"""
split.py -- Exp D: val set congelado + dedup interna por SMILES canonico.

Genera val_indices_frozen.npy: los indices (dentro del dataset de 202465)
de las 14428 moleculas "originales" de las 144k, usando el MISMO
random_split(seed=42, val_split=0.1) que corrio el training historico
(V6-V9) sobre las 144280 moleculas originales. Como el dataset de 202k se
construyo agregando las 58185 nuevas AL FINAL (sin reordenar las 144280
originales), esos indices historicos son validos directamente contra
smiles_202465.npy / nmr_dataset_v3_202465_fast.h5.

El val queda FIJO (nunca se toca). Cualquier fila de TRAIN cuyo SMILES
canonico coincida con una fila de val se elimina de train (leak=0). Las
58185 moleculas nuevas van todas a train (ninguna cae en el rango
[0, 144280) donde vive el val historico).

Uso (en el cluster, login node, sin GPU):
    python split.py --config config.yaml

Requiere: numpy, rdkit (dedup/leak), torch (reproducir el random_split
historico -- ver historical_val_indices_144k). No requiere h5py: split.py
solo toca smiles, no las imagenes HSQC.
"""
import argparse
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem


def canonicalize_smiles(smiles_array):
    """Canonicaliza con RDKit. SMILES invalidos se conservan tal cual
    (no se descarta ninguna molecula del dataset)."""
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


def find_duplicate_groups(canonical_smiles):
    """dict: SMILES canonico -> lista de indices, solo para grupos con >1 molecula."""
    groups = {}
    for idx, smi in enumerate(canonical_smiles):
        groups.setdefault(smi, []).append(idx)
    return {smi: idxs for smi, idxs in groups.items() if len(idxs) > 1}


if __name__ == "__main__":
    pass
```

- [ ] **Step 4: Correr el test de nuevo — deben pasar los primeros 4**

Run: `python experiments/D_val_congelado/tests/test_split.py`
Expected:
```
[OK] test_canonicalize_smiles_dedup_equivalent_forms
[OK] test_canonicalize_smiles_invalid_passthrough
[OK] test_find_duplicate_groups
[OK] test_find_duplicate_groups_no_dups
Traceback (most recent call last):
  ...
ImportError: cannot import name 'remove_leaking_from_train' from 'split'
```
(Falla en el import de `remove_leaking_from_train`, que se implementa en la Task 5 — es el resultado esperado en este punto.)

- [ ] **Step 5: Commit**

```bash
git add experiments/D_val_congelado/split.py experiments/D_val_congelado/tests/test_split.py
git commit -m "exp D: split.py - canonicalize_smiles + find_duplicate_groups (TDD)"
```

---

### Task 5: `split.py` — `remove_leaking_from_train` (TDD, corre local)

**Files:**
- Modify: `experiments/D_val_congelado/split.py`

**Interfaces:**
- Consumes: `canonical_smiles: np.ndarray[object]` (de `canonicalize_smiles`, Task 4).
- Produces: `remove_leaking_from_train(train_idx: np.ndarray, val_idx: np.ndarray, canonical_smiles: np.ndarray) -> (np.ndarray[int64], int)`. Usado por `main()` (Task 6).

- [ ] **Step 1: Agregar `remove_leaking_from_train` a `split.py`**

En `experiments/D_val_congelado/split.py`, agregar después de `find_duplicate_groups` (antes del `if __name__ == "__main__":`):

```python
def remove_leaking_from_train(train_idx, val_idx, canonical_smiles):
    """Elimina de train cualquier indice cuyo SMILES canonico tambien este
    en val. val_idx NUNCA se modifica (queda "congelado")."""
    val_smiles_set = set(canonical_smiles[i] for i in val_idx)
    clean_train_idx = np.array(
        [i for i in train_idx if canonical_smiles[i] not in val_smiles_set],
        dtype=np.int64,
    )
    n_removed = len(train_idx) - len(clean_train_idx)
    return clean_train_idx, n_removed
```

- [ ] **Step 2: Correr el test completo — ahora todo debe pasar**

Run: `python experiments/D_val_congelado/tests/test_split.py`
Expected:
```
[OK] test_canonicalize_smiles_dedup_equivalent_forms
[OK] test_canonicalize_smiles_invalid_passthrough
[OK] test_find_duplicate_groups
[OK] test_find_duplicate_groups_no_dups
[OK] test_remove_leaking_from_train
[OK] test_remove_leaking_from_train_no_leak

>>> SPLIT CORE TESTS OK <<<
```
(RDKit puede imprimir líneas `SMILES Parse Error` a stderr por el SMILES inválido de prueba — es esperado y no afecta el resultado.)

- [ ] **Step 3: Commit**

```bash
git add experiments/D_val_congelado/split.py
git commit -m "exp D: split.py - remove_leaking_from_train (TDD)"
```

---

### Task 6: `split.py` — `historical_val_indices_144k` + CLI `main()` (requiere torch, no ejecutable en este entorno)

**Files:**
- Modify: `experiments/D_val_congelado/split.py`

**Interfaces:**
- Consumes: `canonicalize_smiles`, `find_duplicate_groups`, `remove_leaking_from_train` (Tasks 4-5).
- Produces: archivo `val_indices_frozen.npy` en `{base_dir}` (leído por `evaluate.py`, Task 8, y por futuros Exp B/C).

> **Nota de entorno:** esta tarea agrega la única función de `split.py` que
> necesita `torch` (para reproducir bit a bit el `random_split` histórico) y
> el `main()` que lee `config.yaml` (necesita `pyyaml`). Ninguno de los dos
> paquetes está instalado en este entorno de desarrollo — no se puede
> ejecutar `split.py --config config.yaml` aquí. El código se escribe
> completo y se verifica por revisión; la ejecución real la hace Lucas en
> el cluster (paso documentado en el Step 3, y repetido en el `README.md`
> de la Task 11).

- [ ] **Step 1: Agregar `historical_val_indices_144k` y `main()` a `split.py`**

Reemplazar el `if __name__ == "__main__": pass` final de
`experiments/D_val_congelado/split.py` por:

```python
def historical_val_indices_144k(n_144k=144280, val_split=0.1, seed=42):
    """Reproduce el random_split(seed=42) que uso el training historico
    (V6-V9) sobre las 144280 moleculas originales. Requiere torch para
    igualar bit a bit el RNG que ya se uso (no se puede reproducir con
    numpy). Los indices devueltos son validos directamente contra el
    dataset de 202k porque las 58185 nuevas se agregaron al final, sin
    reordenar las 144280 originales."""
    import torch
    from torch.utils.data import random_split

    val_size = int(n_144k * val_split)
    train_size = n_144k - val_size
    generator = torch.Generator().manual_seed(seed)
    _, val_subset = random_split(range(n_144k), [train_size, val_size], generator=generator)
    return np.array(sorted(val_subset.indices), dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Exp D: genera val_indices_frozen.npy")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(cfg["paths"]["base_dir"])
    smiles_path = base_dir / cfg["paths"]["smiles_filename"]
    out_path = base_dir / cfg["paths"]["val_indices_filename"]

    n_144k = int(cfg["split"]["n_144k"])
    val_split_144k = float(cfg["split"]["val_split_144k"])
    seed = int(cfg["split"]["seed"])

    print("=" * 60)
    print("  EXP D: split congelado")
    print("=" * 60)

    smiles = np.load(smiles_path, allow_pickle=True)
    n_total = len(smiles)
    print(f"-> Moleculas totales: {n_total}")

    canonical, n_invalid = canonicalize_smiles(smiles)
    print(f"-> SMILES invalidos (no parsearon con RDKit): {n_invalid}")

    dup_groups = find_duplicate_groups(canonical)
    n_dup_mols = sum(len(idxs) - 1 for idxs in dup_groups.values())
    print(f"-> Grupos de duplicados canonicos: {len(dup_groups)}  "
          f"({n_dup_mols} moleculas 'de mas' respecto a canonicos unicos)")

    val_idx = historical_val_indices_144k(n_144k, val_split_144k, seed)
    print(f"-> Val congelado (historico 144k, seed={seed}): {len(val_idx)} moleculas")

    all_idx = np.arange(n_total)
    train_idx_raw = np.setdiff1d(all_idx, val_idx, assume_unique=False)

    train_idx, n_removed = remove_leaking_from_train(train_idx_raw, val_idx, canonical)
    print(f"-> Filas de train eliminadas por leak canonico contra val: {n_removed}")
    print(f"-> Train final: {len(train_idx)}   Val final (sin tocar): {len(val_idx)}")

    val_smiles_set = set(canonical[i] for i in val_idx)
    train_smiles_set = set(canonical[i] for i in train_idx)
    leak = val_smiles_set & train_smiles_set
    print(f"-> Verificacion leak=0: interseccion train/val = {len(leak)} SMILES canonicos")
    assert len(leak) == 0, "Leak residual tras remove_leaking_from_train (no deberia pasar)"

    np.save(out_path, val_idx)
    print(f"\n[SAVE] {out_path}  ({len(val_idx)} indices, dtype={val_idx.dtype})")
    print(">>> EXP D split.py OK <<<")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verificación estática (lo único posible en este entorno)**

Run: `python -c "import ast; ast.parse(open('experiments/D_val_congelado/split.py', encoding='utf-8').read())"`
Expected: sin output (el archivo parsea como Python válido). Esto NO reemplaza correrlo de verdad — solo descarta errores de sintaxis antes de que Lucas lo use.

- [ ] **Step 3: Verificación real — la hace Lucas en el cluster (documentar, no ejecutar aquí)**

Una vez que exista `config.yaml` (Task 7), Lucas corre en el login node:
```bash
cd experiments/D_val_congelado
python split.py --config config.yaml
```
Expected (aproximado — los conteos reales de duplicados/leak dependen de los datos reales):
```
============================================================
  EXP D: split congelado
============================================================
-> Moleculas totales: 202465
-> SMILES invalidos (no parsearon con RDKit): 0
-> Grupos de duplicados canonicos: <N>  (<M> moleculas 'de mas' respecto a canonicos unicos)
-> Val congelado (historico 144k, seed=42): 14428 moleculas
-> Filas de train eliminadas por leak canonico contra val: <K>
-> Train final: <202465 - 14428 - K>   Val final (sin tocar): 14428
-> Verificacion leak=0: interseccion train/val = 0 SMILES canonicos

[SAVE] /home/lpassaglia.iquir/DB_200k/val_indices_frozen.npy  (14428 indices, dtype=int64)
>>> EXP D split.py OK <<<
```
Si `n_invalid > 0` o el val final no da ~14428, no seguir con Exp B/C sin
avisar — puede indicar que el orden de las 144280 originales no se preservó
al construir el dataset de 202k (la asunción central de `historical_val_indices_144k`).

- [ ] **Step 4: Commit**

```bash
git add experiments/D_val_congelado/split.py
git commit -m "exp D: split.py - historical_val_indices_144k + CLI main() (requiere cluster para correr)"
```

---

### Task 7: `config.yaml`

**Files:**
- Create: `experiments/D_val_congelado/config.yaml`

**Interfaces:**
- Consumes: valores canónicos de `config/db.yaml` (copiados a mano, convención ya establecida — memoria `exp-config-convention`).
- Produces: config leído por `split.py` (Task 6) y `evaluate.py` (Task 8).

- [ ] **Step 1: Escribir config.yaml**

```yaml
# experiments/D_val_congelado/config.yaml
#
# Exp D NO reentrena: reutiliza el checkpoint V10 ya entrenado.
# experiment_name + checkpoint_dir apuntan exactamente al V10 baseline para
# que evaluate.py encuentre {base_dir}/{checkpoint_dir}/{experiment_name}_best.pth
#
# OJO: configs/config_V10.yaml (en git) tiene h5_filename SIN "_fast" y
# num_workers=4 -- ambos violan reglas duras del proyecto (CLAUDE.md reglas
# 1 y 4) y no coinciden con lo que documenta docs/Runs/RESULTS.md para la
# corrida real de V10. Este config.yaml usa los valores correctos
# (con "_fast", num_workers=0). No copiar los valores de config_V10.yaml
# tal cual.

experiment_name: "nmr_202k_v10_2ch_fm_19v"

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  h5_filename: "nmr_dataset_v3_202465_fast.h5"
  labels_filename: "vectors_13c_19v_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_V10_202k"
  val_indices_filename: "val_indices_frozen.npy"

split:
  n_144k: 144280
  val_split_144k: 0.1
  seed: 42

system:
  device: "cuda"
  num_workers: 0
  pin_memory: true
```

- [ ] **Step 2: Commit**

```bash
git add experiments/D_val_congelado/config.yaml
git commit -m "exp D: agrega config.yaml"
```

---

### Task 8: `evaluate.py` — adaptar `evaluate_v10.py` para leer el split congelado

**Files:**
- Create: `experiments/D_val_congelado/evaluate.py` (basado en `src/evaluate_v10.py`)

**Interfaces:**
- Consumes: `NMR_Net` (de `model_v10.py`, Task 2), `NMRDataset` (de `dataset_v10.py`, Task 2), `val_indices_frozen.npy` (generado por `split.py`, Task 6) vía `cfg["paths"]["val_indices_filename"]` (Task 7).
- Produces: funciones `crude_predict`, `ajustar_conteo_doble_exacto`, `compute_ema`, `compute_mae`, `ema_entorno` — mismas firmas que `evaluate_v10.py`, consumidas por `tests/test_forward.py` (Task 9).

- [ ] **Step 1: Crear `evaluate.py`**

```python
# coding: ascii
"""
Evaluacion Exp D (HSQC 2 canales + FM + 19 clases, split CONGELADO) sobre el
checkpoint V10 YA ENTRENADO. No reentrena nada -- es un forward pass sobre
val_indices_frozen.npy (generado por split.py) en vez del random_split que
usa evaluate_v10.py.

  --oraculo on   -> ajustar_conteo_doble_exacto (EMA ASISTIDA).
  --oraculo off  -> np.clip(np.floor(pred_raw), 0, None) (EMA CRUDA).
  --oraculo both -> corre ambos e imprime la tabla comparativa (DEFAULT).

Config: UN SOLO config.yaml (ver Task 7). El checkpoint se deriva IGUAL que
el guardado del training: {base_dir}/{paths.checkpoint_dir}/{experiment_name}_best.pth.
Autocontenido (nombres de clase e indices CH2 hardcodeados aca, como
evaluate_v10.py) para no depender de rutas relativas a otros archivos del
repo que pueden no existir en la carpeta del cluster.

Reglas (CLAUDE.md):
  - num_workers=0 (h5py no es fork-safe; rule 1).
  - val_indices_frozen.npy debe existir (correr split.py primero).

NO ejecutar hasta tener val_indices_frozen.npy y el checkpoint del V10.
"""
import os
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset


# --- Clases 19v (orden EXACTO de config/db.yaml, no reordenar) --------------
GROUP_NAMES = [
    "CH3", "CH2", "CH", "Cq",
    "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N",
    "=CH2", "=CH/Ar", "Cqsp2", "Aldeh", "Imina",
    "C-2X", "C-3X",
]
N_CLASSES = 19
IDX_CH2 = [1, 5, 9, 12]   # CH2, CH2-O, CH2-N, =CH2

ENTORNOS = {
    "Alifaticos (sp3)":               ["CH3", "CH2", "CH", "Cq"],
    "Heteroatomicos O/N (sp3)":       ["CH3-O", "CH2-O", "CH-O", "Cq-O",
                                       "CH3-N", "CH2-N", "CH-N", "Cq-N"],
    "Carbonos sp2 (Olef/Arom/C=O)":   ["=CH2", "=CH/Ar", "Cqsp2", "Aldeh", "Imina"],
    "Sistemas X-Multiples (C-2X/3X)": ["C-2X", "C-3X"],
}
ENTORNOS_IDX = {ent: [GROUP_NAMES.index(c) for c in cls]
                for ent, cls in ENTORNOS.items()}


# --- Config ------------------------------------------------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Post-procesamiento ------------------------------------------------------
def crude_predict(pred_cruda):
    """Modo CRUDO: floor con clip a >=0. Ignora el condicionante por completo."""
    return np.clip(np.floor(pred_cruda), 0, None).astype(int)


def ajustar_conteo_doble_exacto(pred_cruda, total_real, ch2_real):
    """Modo ASISTIDO (oraculo): fuerza sum(pred)==total_real y
    sum(pred[IDX_CH2])==ch2_real, repartiendo por el resto decimal."""
    pred_int = np.floor(pred_cruda).astype(int)
    restos = pred_cruda - pred_int
    idx_resto = [i for i in range(N_CLASSES) if i not in IDX_CH2]

    ch2_asignados = sum(pred_int[i] for i in IDX_CH2)
    ch2_faltantes = int(ch2_real - ch2_asignados)
    if ch2_faltantes > 0:
        for i in sorted(IDX_CH2, key=lambda i: restos[i])[-ch2_faltantes:]:
            pred_int[i] += 1
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
        for i in sorted(idx_resto, key=lambda i: restos[i])[-resto_faltantes:]:
            pred_int[i] += 1
    elif resto_faltantes < 0:
        sobran = abs(resto_faltantes)
        for i in sorted(idx_resto, key=lambda i: restos[i]):
            if pred_int[i] > 0:
                pred_int[i] -= 1
                sobran -= 1
                if sobran == 0:
                    break

    return pred_int


# --- Metricas ----------------------------------------------------------------
def compute_ema(preds, targets):
    return float(np.mean(np.all(preds == targets, axis=1)) * 100)


def compute_mae(preds, targets):
    return np.mean(np.abs(targets - preds), axis=0)


def ema_entorno(preds, targets, indices):
    return float(np.mean(np.all(preds[:, indices] == targets[:, indices], axis=1)) * 100)


def estado(mae):
    if mae < 0.010:
        return "[PERFECTO]"
    if mae < 0.025:
        return "[EXCELENTE]"
    if mae < 0.040:
        return "[BUENO]"
    if mae < 0.060:
        return "[ACEPTABLE]"
    return "[MEJORABLE]"


def analizar_confusiones_cruzadas(all_preds, all_targets):
    error_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for i in range(len(all_targets)):
        diff = all_preds[i] - all_targets[i]
        under = [(g, -diff[g]) for g in range(N_CLASSES) if diff[g] < 0]
        over = [(g, diff[g]) for g in range(N_CLASSES) if diff[g] > 0]
        for gu, cu in under:
            for go, co in over:
                error_matrix[gu][go] += min(cu, co)
    return error_matrix


# --- Reporte por modo --------------------------------------------------------
def report_mode(label, preds, targets):
    n = len(targets)
    ema = compute_ema(preds, targets)
    mae = compute_mae(preds, targets)

    print("\n" + "=" * 60)
    print(f"  MODO {label}  ->  EXACT MATCH ACCURACY: {ema:.2f}%")
    print("=" * 60)

    print(f"\n{'GRUPO':<10} | {'MAE':>6}  Estado")
    print("-" * 40)
    for i, name in enumerate(GROUP_NAMES):
        print(f"{name:<10} | {mae[i]:.4f}  {estado(mae[i])}")

    print("\n  ERRORES POR ENTORNO")
    print("  " + "-" * 40)
    for entorno, indices in ENTORNOS_IDX.items():
        err_mask = np.any(preds[:, indices] != targets[:, indices], axis=1)
        n_err = int(err_mask.sum())
        mae_ent = float(np.mean(np.abs(targets[:, indices] - preds[:, indices])))
        ema_ent = ema_entorno(preds, targets, indices)
        print(f"\n  {entorno}")
        print(f"    Moleculas con error: {n_err} / {n}  ({n_err / n * 100:.1f}%)")
        print(f"    EMA del entorno:     {ema_ent:.2f}%   |   MAE promedio: {mae_ent:.4f}")
        for i in indices:
            err_g = int((preds[:, i] != targets[:, i]).sum())
            print(f"      {GROUP_NAMES[i]:<10}: MAE={mae[i]:.4f} | "
                  f"Mol. con error: {err_g} ({err_g / n * 100:.1f}%)")

    return ema, mae


def print_confusiones(preds, targets):
    n = len(targets)
    error_matrix = analizar_confusiones_cruzadas(preds, targets)
    print("\n" + "=" * 60)
    print("  MAPA DE CONFUSIONES CRUZADAS (solo modo asistido)")
    print("=" * 60)
    for i, name in enumerate(GROUP_NAMES):
        fila = error_matrix[i].copy()
        fila[i] = 0
        total = int(fila.sum())
        if total == 0:
            continue
        top3 = [(GROUP_NAMES[j], fila[j])
                for j in np.argsort(fila)[::-1][:3] if fila[j] > 0]
        pct_g = (preds[:, i] != targets[:, i]).sum() / n * 100
        print(f"  {name:<10} (falla en {pct_g:.1f}% mol) -> confunde con:")
        for dest, cnt in top3:
            print(f"      {dest:<10}: {cnt:>5} senales  ({cnt / total * 100:.1f}%)")


def print_tabla_comparativa(preds_on, preds_off, targets):
    ema_on = compute_ema(preds_on, targets)
    ema_off = compute_ema(preds_off, targets)
    print("\n" + "=" * 60)
    print("  TABLA COMPARATIVA: EMA CRUDA vs ASISTIDA")
    print("=" * 60)
    print(f"\n  {'':<32}{'CRUDA':>10}{'ASISTIDA':>12}{'Delta':>10}")
    print("  " + "-" * 62)
    print(f"  {'EMA GLOBAL':<32}{ema_off:>9.2f}%{ema_on:>11.2f}%{ema_on - ema_off:>+9.2f}")
    for entorno, indices in ENTORNOS_IDX.items():
        e_off = ema_entorno(preds_off, targets, indices)
        e_on = ema_entorno(preds_on, targets, indices)
        print(f"  {entorno:<32}{e_off:>9.2f}%{e_on:>11.2f}%{e_on - e_off:>+9.2f}")
    print("\n  Delta = ASISTIDA - CRUDA. Un Delta grande (>10 pp) indica que la")
    print("  EMA reportada depende fuertemente del oraculo de doble restriccion.")


# --- Main --------------------------------------------------------------------
def evaluate(config_path, oraculo, eval_batch_size):
    # Import perezoso: dataset_v10 trae rdkit/h5py; el smoke test no lo necesita.
    from dataset_v10 import NMRDataset
    from model_v10 import NMR_Net

    cfg = load_config(config_path)

    base_dir = Path(cfg["paths"]["base_dir"])
    h5_path = base_dir / cfg["paths"]["h5_filename"]
    labels_path = base_dir / cfg["paths"]["labels_filename"]
    smiles_path = base_dir / cfg["paths"]["smiles_filename"]
    ckpt_path = base_dir / cfg["paths"]["checkpoint_dir"] / f"{cfg['experiment_name']}_best.pth"
    val_indices_path = base_dir / cfg["paths"]["val_indices_filename"]

    num_workers = int(cfg["system"]["num_workers"])   # 0 (rule 1)

    modes = ["on", "off"] if oraculo == "both" else [oraculo]
    run_on = "on" in modes
    run_off = "off" in modes

    print("=" * 60)
    print("  EVALUACION EXP D (2CH + FM + 19v) - SPLIT CONGELADO")
    print("=" * 60)
    print(f"-> Experimento (checkpoint): {cfg['experiment_name']}")
    print(f"-> Modos: {modes}   | idx_ch2: {IDX_CH2}")
    print(f"-> num_workers={num_workers} (rule 1)  batch_size={eval_batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Dispositivo: {device.type.upper()}")

    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] No se encontro el checkpoint en:\n  {ckpt_path}")
        return
    if not os.path.exists(val_indices_path):
        print(f"\n[ERROR] No se encontro el split congelado en:\n  {val_indices_path}")
        print("        Corri primero split.py para generarlo.")
        return

    # Split CONGELADO (Exp D): val_indices_frozen.npy, no random_split.
    full_dataset = NMRDataset(str(h5_path), str(labels_path), str(smiles_path))
    val_indices = np.load(val_indices_path)
    val_ds = Subset(full_dataset, val_indices.tolist())

    use_pin = bool(cfg["system"].get("pin_memory", False)) and device.type == "cuda"
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin)

    model = NMR_Net(num_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_targets, all_pred_on, all_pred_off = [], [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            hsqc = inputs[0].to(device)
            proj = inputs[1].to(device)
            cond = inputs[2].to(device)
            pred_raw = model(hsqc, proj, cond).cpu().numpy()
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

    all_targets = np.vstack(all_targets)
    print(f"\n-> Set de validacion (congelado): {len(all_targets)} moleculas")

    preds_on = np.vstack(all_pred_on) if run_on else None
    preds_off = np.vstack(all_pred_off) if run_off else None

    if run_off:
        report_mode("CRUDO (--oraculo off)", preds_off, all_targets)
    if run_on:
        report_mode("ASISTIDO (--oraculo on)", preds_on, all_targets)
        print_confusiones(preds_on, all_targets)

    if run_on and run_off:
        print_tabla_comparativa(preds_on, preds_off, all_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Exp D (split congelado)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config unico (ver Task 7 del plan).")
    parser.add_argument("--oraculo", choices=["on", "off", "both"], default="both",
                        help="on=asistida, off=cruda, both=ambas + tabla (default).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size de evaluacion.")
    args = parser.parse_args()
    evaluate(args.config, args.oraculo, args.batch_size)
```

- [ ] **Step 2: Verificación estática (no requiere torch)**

Run: `python -c "import ast; ast.parse(open('experiments/D_val_congelado/evaluate.py', encoding='utf-8').read())"`
Expected: sin output.

- [ ] **Step 3: Commit**

```bash
git add experiments/D_val_congelado/evaluate.py
git commit -m "exp D: evaluate.py - evaluate_v10.py adaptado a Subset con val congelado"
```

---

### Task 9: `tests/test_forward.py` — smoke test (requiere torch, corre en el cluster)

**Files:**
- Create: `experiments/D_val_congelado/tests/test_forward.py`

**Interfaces:**
- Consumes: `NMR_Net` (Task 2), `crude_predict`, `ajustar_conteo_doble_exacto`, `compute_ema`, `compute_mae`, `ema_entorno` (Task 8).

> **Nota de entorno:** igual que la Task 6, este archivo necesita `torch` —
> no ejecutable en esta máquina. Sigue el patrón ya existente de
> `tests/test_eval_forward.py` (repo root), que tampoco se corrió nunca en
> este entorno. Se verifica por revisión + parseo estático; la ejecución
> real la hace Lucas en el login node antes de cualquier `sbatch` (regla
> dura 5 de CLAUDE.md).

- [ ] **Step 1: Crear `tests/test_forward.py`**

```python
# coding: ascii
"""
Smoke test OFFLINE del evaluador Exp D - rule 5 de CLAUDE.md.

NO depende del checkpoint ni del h5 real: usa tensores random y arrays
sinteticos. Valida:
  (1) el forward de model_v10 con HSQC de 2 canales -> (B, 19),
  (2) la prediccion CRUDA (floor + clip a >=0),
  (3) el ORACULO de doble restriccion,
  (4) el calculo de EMA / MAE,
  (5) que Subset(dataset, indices) respeta el orden de
      val_indices_frozen.npy -- el mecanismo que usa evaluate.py.

Correr en CPU (login node) antes de cualquier sbatch:
    python tests/test_forward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from model_v10 import NMR_Net
from evaluate import (
    crude_predict,
    ajustar_conteo_doble_exacto,
    compute_ema,
    compute_mae,
    ema_entorno,
)

N_CLASSES = 19
IDX_CH2 = [1, 5, 9, 12]   # CH2, CH2-O, CH2-N, =CH2


def test_forward_2ch():
    model = NMR_Net(num_classes=N_CLASSES)
    model.eval()
    B = 4
    hsqc = torch.randn(B, 2, 256, 256)
    proj = torch.randn(B, 512)
    cond = torch.randn(B, 8)
    with torch.no_grad():
        out = model(hsqc, proj, cond)
    assert out.shape == (B, N_CLASSES), out.shape
    print(f"[OK] forward 2ch -> {tuple(out.shape)} (esperado ({B}, {N_CLASSES}))")


def test_crude():
    pred_raw = np.array([0.9, -0.3, 2.7, 1.1] + [0.0] * 15, dtype=np.float32)
    p = crude_predict(pred_raw)
    assert p.dtype.kind == "i"
    assert (p >= 0).all(), "clip a >=0 fallo"
    assert p[0] == 0 and p[1] == 0 and p[2] == 2 and p[3] == 1, p[:4]
    print("[OK] crudo: floor + clip a >=0")


def test_oraculo_doble_restriccion():
    rng = np.random.default_rng(0)
    for _ in range(500):
        target = rng.integers(0, 3, size=N_CLASSES)
        total = int(target.sum())
        ch2 = int(sum(target[i] for i in IDX_CH2))
        pred_raw = np.clip(target + rng.uniform(-0.4, 0.4, size=N_CLASSES), 0, None)
        pred_raw = pred_raw.astype(np.float32)
        p = ajustar_conteo_doble_exacto(pred_raw, total, ch2)
        assert (p >= 0).all()
        assert int(p.sum()) == total, (int(p.sum()), total)
        assert int(sum(p[i] for i in IDX_CH2)) == ch2, (sum(p[i] for i in IDX_CH2), ch2)
        assert np.array_equal(p, target), (p, target)
    print("[OK] oraculo: sum==total_senales, sum(CH2)==total_CH2 y recupera el "
          "vector (500 casos)")


def test_metrics():
    t = np.array([[1, 0, 2], [3, 1, 0]])
    assert compute_ema(t.copy(), t) == 100.0
    p = t.copy()
    p[0, 0] += 1
    assert compute_ema(p, t) == 50.0
    mae = compute_mae(p, t)
    assert abs(mae[0] - 0.5) < 1e-9, mae
    assert ema_entorno(p, t, [1, 2]) == 100.0
    assert ema_entorno(p, t, [0]) == 50.0
    print("[OK] EMA / MAE / EMA por entorno")


def test_subset_from_frozen_indices():
    """Valida que Subset(dataset, indices) selecciona exactamente esos
    indices, en el mismo orden -- el mecanismo que usa evaluate.py para
    leer val_indices_frozen.npy."""

    class DummyDataset(Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return idx

    ds = DummyDataset()
    frozen = np.array([7, 2, 5])
    subset = Subset(ds, frozen.tolist())
    assert len(subset) == 3
    assert [subset[i] for i in range(3)] == [7, 2, 5]
    print("[OK] Subset respeta el orden de val_indices_frozen.npy")


if __name__ == "__main__":
    test_forward_2ch()
    test_crude()
    test_oraculo_doble_restriccion()
    test_metrics()
    test_subset_from_frozen_indices()
    print("\n>>> SMOKE EXP D OK - listo para sbatch run_eval.sh <<<")
```

- [ ] **Step 2: Verificación estática (no requiere torch)**

Run: `python -c "import ast; ast.parse(open('experiments/D_val_congelado/tests/test_forward.py', encoding='utf-8').read())"`
Expected: sin output.

- [ ] **Step 3: Verificación real — la hace Lucas en el cluster (documentar, no ejecutar aquí)**

```bash
cd experiments/D_val_congelado
python tests/test_forward.py
```
Expected:
```
[OK] forward 2ch -> (4, 19) (esperado (4, 19))
[OK] crudo: floor + clip a >=0
[OK] oraculo: sum==total_senales, sum(CH2)==total_CH2 y recupera el vector (500 casos)
[OK] EMA / MAE / EMA por entorno
[OK] Subset respeta el orden de val_indices_frozen.npy

>>> SMOKE EXP D OK - listo para sbatch run_eval.sh <<<
```

- [ ] **Step 4: Commit**

```bash
git add experiments/D_val_congelado/tests/test_forward.py
git commit -m "exp D: tests/test_forward.py - smoke test offline (requiere cluster para correr)"
```

---

### Task 10: `run_eval.sh` (SLURM)

**Files:**
- Create: `experiments/D_val_congelado/run_eval.sh`

**Interfaces:**
- Consumes: `evaluate.py` (Task 8), `config.yaml` (Task 7).

- [ ] **Step 1: Escribir `run_eval.sh`, siguiendo el patrón de `src/run_eval_v9.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=expD_eval_frozen
#SBATCH --partition=gpua10_hi
#SBATCH --output=expD_eval_%j.out
#SBATCH --error=expD_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector/experiments/D_val_congelado

python -u evaluate.py --config config.yaml --oraculo both --batch-size 256
```

- [ ] **Step 2: Commit**

```bash
git add experiments/D_val_congelado/run_eval.sh
git commit -m "exp D: run_eval.sh (sbatch, --gres=gpu:1)"
```

---

### Task 11: `README.md` — checklist para Lucas

**Files:**
- Create: `experiments/D_val_congelado/README.md`

**Interfaces:** N/A (documentación).

- [ ] **Step 1: Escribir README.md**

```markdown
# Exp D — Val set congelado

Checklist para correr esto en el cluster (`login-1`, env `NMR_env`). No
reentrena nada: reutiliza el checkpoint V10 ya entrenado.

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/D_val_congelado`
3. Generar el split congelado (no necesita GPU, corre en el login node,
   dura unos minutos):
   ```bash
   python split.py --config config.yaml
   ```
   Revisar el reporte impreso: cuántos duplicados canónicos aparecen,
   cuántas filas de train se eliminan por leak, y que el val final sea
   ~14428. Si `val final` se aleja mucho de 14428, o `SMILES invalidos` es
   grande, **parar y avisar** antes de seguir — puede indicar que el orden
   de las 144280 moléculas originales no se preservó al construir el
   dataset de 202k (ver `RATIONALE.md`, "Criterio de éxito/fracaso").
4. Confirmar que `val_indices_frozen.npy` quedó en
   `/home/lpassaglia.iquir/DB_200k/`.
5. Smoke test obligatorio antes de cualquier `sbatch` (CPU, sin checkpoint
   real):
   ```bash
   python tests/test_forward.py
   ```
6. Lanzar la re-evaluación del checkpoint V10 sobre el split nuevo:
   ```bash
   sbatch run_eval.sh
   ```
7. Cuando termine, revisar `expD_eval_<jobid>.out`: copiar la tabla
   "EMA CRUDA vs ASISTIDA" a `docs/Runs/RESULTS.md`, como fila nueva
   "V10-on-frozen-val (Exp D)".
8. Avisar a Claude Code con los números — con eso arrancamos Exp B.

## Nota

`configs/config_V10.yaml` (en la raíz del repo, el que se usó para
entrenar V10) tiene `h5_filename` sin `_fast` y `num_workers=4` — ambos
inconsistentes con las reglas duras del proyecto y con lo que documenta
`docs/Runs/RESULTS.md` sobre la corrida real. El `config.yaml` de esta
carpeta ya usa los valores correctos; no copiar `config_V10.yaml` tal cual
para futuros experimentos.
```

- [ ] **Step 2: Commit**

```bash
git add experiments/D_val_congelado/README.md
git commit -m "exp D: agrega README.md con el checklist de comandos"
```

---

## Self-Review

**Cobertura del spec (RATIONALE, config.yaml, split.py, evaluate.py, dump_predictions.py, run_train.sh/run_eval.sh, README.md — sección "Qué quiero que produzcas" del PROMPT):**
- RATIONALE.md ✓ (Task 3). config.yaml ✓ (Task 7). Scripts de dataset/modelo modificados ✓ (Task 2, copias sin cambios ya que D no los toca). evaluate.py ✓ (Task 8). run_eval.sh ✓ (Task 10). README.md ✓ (Task 11). `train.py` no aplica (D no entrena — documentado explícitamente). `dump_predictions.py` no aplica a D: no hay nada nuevo que volcar a la GUI (mismo checkpoint, mismas predicciones ya vistas en Exp A); se retoma en Exp B/C, que si generan checkpoints nuevos.
- Housekeeping de RESULTS.md (acordado en el diseño) ✓ (Task 1).
- Split congelado + dedup + leak=0 (spec del Exp D en `WORKFLOW_V11_para_ClaudeCode.md`) ✓ (Tasks 4-6).
- Re-evaluación del checkpoint V10 sobre el split nuevo ("V10-on-frozen-val", decisión del brainstorm) ✓ (Tasks 8, 10).

**Placeholders:** ninguno — cada step tiene código completo o el comando+output exacto documentado.

**Consistencia de tipos/nombres:** `canonicalize_smiles`, `find_duplicate_groups`, `remove_leaking_from_train`, `historical_val_indices_144k` se nombran igual en `split.py` (Tasks 4-6) y en `tests/test_split.py` (Task 4). `crude_predict`, `ajustar_conteo_doble_exacto`, `compute_ema`, `compute_mae`, `ema_entorno` se nombran igual en `evaluate.py` (Task 8) y `tests/test_forward.py` (Task 9). `cfg["paths"]["val_indices_filename"]` se define en `config.yaml` (Task 7) y se lee igual en `split.py` (Task 6) y `evaluate.py` (Task 8).

**Alcance:** un solo experimento (D), autocontenido. No toca Exp B/C ni el código de `src/`.
