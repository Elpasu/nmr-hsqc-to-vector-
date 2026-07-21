# Exp E — Fase 2: Modelo DeepSets sobre Picos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar `experiments/E2_deepsets/` lista para `sbatch` en el cluster:
un modelo DeepSets que predice el vector de 19 clases a partir del conjunto
de picos de Exp E Fase 1b (`peaks_pkl_202465.npz`), sin imagen HSQC ni
proyecciones 1D.

**Architecture:** Carpeta autocontenida (mismo patrón que `experiments/C_gap/`):
`model_e2.py` (DeepSets: MLP compartido por pico + promedio enmascarado +
fusión con FM), `dataset_e2.py` (carga el `.npz` de picos + calcula FM vía
RDKit, igual que `dataset_v10.py` pero sin imagen), `split_utils.py` (copia
de Exp D), `train.py`/`evaluate.py`/`dump_predictions.py` (mismo patrón que
Exp C, adaptado a la nueva firma de entrada), `config.yaml` único,
`run_train.sh`/`run_eval.sh` (SLURM).

**Tech Stack:** Python, PyTorch, numpy, RDKit, pyyaml.

## Global Constraints

- Entrada del modelo: `(peaks (B, max_peaks, 4), peaks_mask (B, max_peaks),
  cond (B, 8))` → salida `(B, 19)`. Sin imagen HSQC, sin `vec_c`/`vec_h`.
- DeepSets: MLP compartido `4 → 64 → 64` por pico (mismos pesos para todos
  los picos), agregación por **promedio enmascarado** (solo sobre picos
  válidos según `peaks_mask`; si una molécula no tiene picos válidos, el
  agregado es cero — nunca dividir por cero). Fusión:
  `(agregado 64 + cond 8) = 72 → 128 → 64 → 19`.
- `cond_tensor` (8 valores: total_señales, total_CH2, C,H,N,O,S,Hal) se
  calcula EXACTO igual que `experiments/D_val_congelado/dataset_v10.py`
  (total_señales/total_CH2 del label, fórmula molecular vía RDKit sobre
  SMILES) — no reinventar esa lógica.
- Capacidad del modelo deliberadamente chica (no agrandar "porque sobra GPU")
  — ver `RATIONALE.md`.
- Mismo split congelado que Exp D/B/C: `val_indices_frozen.npy` +
  `split_utils.py` copiado (mismas 2 funciones: `canonicalize_smiles`,
  `remove_leaking_from_train`).
- Mismos hiperparámetros que el resto: Adam lr=0.001, `ReduceLROnPlateau`
  patience=8/factor=0.7, batch=64, epochs=100, seed=42, `num_workers: 0`.
  Sin regularización (dropout/weight_decay) — misma decisión que Exp C.
- `peaks_pkl_202465.npz` vive solo en la máquina local de Lucas
  (`DB_nmr_to_vector/202K_suma/`) — hay que copiarlo a `DB_200k/` en el
  cluster antes de entrenar. Documentar en README.
- Entorno de desarrollo local: numpy, rdkit, pyyaml disponibles — **torch
  NO está instalado**. Todo lo que depende de torch (`model_e2.py`,
  `dataset_e2.py`, `train.py`, `evaluate.py`, `dump_predictions.py`) se
  verifica localmente solo por revisión de código (comparación directa
  contra los archivos ya probados de `experiments/C_gap/`), nunca se
  ejecuta ni se finge haberlo ejecutado. Lucas corre el smoke test real en
  el cluster antes de cualquier `sbatch`.
- Encoding UTF-8 en archivos nuevos (`# coding: ascii` en los `.py`, sin
  caracteres no-ASCII en el código — igual convención que el resto del repo).

---

### Task 1: Scaffold — carpeta + config.yaml + RATIONALE.md

**Files:**
- Create: `experiments/E2_deepsets/config.yaml`
- Create: `experiments/E2_deepsets/RATIONALE.md`

**Interfaces:**
- Produces: esquema de `config.yaml` (`experiment_name`, `paths.*`,
  `hyperparameters.*`, `system.*`) que consumen todos los tasks siguientes.

- [ ] **Step 1: Crear `experiments/E2_deepsets/config.yaml`**

```yaml
# experiments/E2_deepsets/config.yaml
#
# Exp E Fase 2: modelo DeepSets sobre picos extraidos del pkl original
# (Exp E Fase 1b, ver docs/Runs/RESULTS.md). Sin imagen HSQC, sin
# proyecciones 1D (vec_c/vec_h) -- ver RATIONALE.md.

experiment_name: "nmr_202k_e2_deepsets_picos_19v"

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  peaks_filename: "peaks_pkl_202465.npz"
  labels_filename: "vectors_13c_19v_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_E2_deepsets"
  val_indices_filename: "val_indices_frozen.npy"

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

- [ ] **Step 2: Crear `experiments/E2_deepsets/RATIONALE.md`**

```markdown
# RATIONALE — Exp E Fase 2: Modelo DeepSets sobre Picos

## Hipótesis

Fase 1b (`docs/Runs/RESULTS.md`) confirmó que los picos extraídos del pkl
original preservan 97.19% del conteo visible del label (2.19% de colisión
real, marginal) — mucho mejor que el 88.75% de colisión de la imagen 256×256
(Fase 1). Tres arquitecturas distintas sobre la imagen (V10, Exp B, Exp C)
mostraron las mismas confusiones de clase persistentes
(`Cqsp2`↔`=CH/Ar`, `CH2`↔`CH2-N`) — evidencia de que el problema es de
representación, no arquitectónico. Esta fase prueba si un modelo que
consume los picos directamente (sin imagen) mejora la EMA.

## Qué cambia exactamente respecto a Exp C

- **Entrada:** se elimina la imagen HSQC (2×256×256) y las proyecciones 1D
  (`vec_c`/`vec_h`, derivadas del mismo binning que falló en Fase 1). En su
  lugar: el conjunto de picos `(δC, δH, amp_ch0, amp_ch1)` de
  `peaks_pkl_202465.npz` (hasta 32 por molécula, con máscara de válidos).
- **Arquitectura:** DeepSets — MLP compartido `4 → 64 → 64` por pico
  (permutation-invariant), agregación por promedio enmascarado, fusión con
  `cond_tensor` (FM, 8 valores, se sigue calculando exactamente igual que
  siempre) → `72 → 128 → 64 → 19`.
- **Se mantiene igual:** `cond_tensor`, split congelado (Exp D), loss
  (`ConstrainedMSELoss`), scheduler, 100 épocas, sin regularización.

## Capacidad del modelo: deliberadamente chica

Este modelo tiene ~23k parámetros (por diseño, no por descuido) — bastante
menos que los ~223k de Exp C y muy por debajo de los ~8.6M de V10. Decisión
tomada con el usuario: V10 (8.6M parámetros) sobreajustó y dio peor
resultado que Exp C (223k, ~38x menos) — "más grande" ya perdió una vez en
este proyecto. Además, entrenar sobre picos + MLPs chicos es mucho más
rápido que sobre la imagen (sin convoluciones sobre 256×256) — el
presupuesto de GPU disponible (hasta 24h) va a sobrar largamente con este
tamaño de modelo. Se decidió NO usar ese presupuesto extra para agrandar el
modelo en esta misma corrida, para no mezclar dos variables (representación
de datos vs capacidad) en un solo experimento. Si el resultado es bueno, un
modelo más grande (o Set Transformer) queda para un experimento aparte.

## Qué métrica esperás mover y cuánto

- EMA cruda: objetivo mínimo ≥ 0.89% (Exp C, el mejor resultado limpio
  hasta ahora). El indicador real de éxito es si las confusiones
  `Cqsp2`↔`=CH/Ar` y `CH2`↔`CH2-N` (idénticas en V10/B/C) mejoran o
  desaparecen — eso confirmaría que eran un problema de representación.
- Val loss: sin referencia previa directa (arquitectura distinta) — se
  reporta igual para comparar convergencia general.

## Criterio de éxito/fracaso

- **Éxito:** EMA cruda ≥ 0.89% y las confusiones persistentes mejoran
  respecto a V10/B/C.
- **Fracaso (sin mejora):** EMA similar o peor, confusiones sin cambios —
  indicaría que el cuello de botella no era la representación de entrada
  (o que este tamaño de modelo es insuficiente para aprovecharla), y habría
  que revisar capacidad o probar Set Transformer.
```

- [ ] **Step 3: Commit**

```bash
git add experiments/E2_deepsets/config.yaml experiments/E2_deepsets/RATIONALE.md
git commit -m "exp-e-fase2: scaffold (config.yaml + RATIONALE.md)"
```

---

### Task 2: `split_utils.py` (copia de Exp D, TDD)

**Files:**
- Create: `experiments/E2_deepsets/split_utils.py`
- Test: `experiments/E2_deepsets/tests/test_split_utils.py`

**Interfaces:**
- Produces: `canonicalize_smiles(smiles_array) -> (np.ndarray, int)`,
  `remove_leaking_from_train(train_idx, val_idx, canonical_smiles) ->
  (np.ndarray, int)` — usadas por `train.py` (Task 6) para reconstruir el
  split congelado.

- [ ] **Step 1: Escribir el test (corre localmente, numpy+rdkit disponibles)**

```python
# experiments/E2_deepsets/tests/test_split_utils.py
# coding: ascii
"""
Tests locales (sin torch) de split_utils.py -- Exp E Fase 2. Mismas
funciones y mismos casos que experiments/C_gap/tests/test_split_utils.py:
split_utils.py es una copia autocontenida de esas 2 funciones puras, ya
probadas en Exp D/B/C. Corren en cualquier maquina con numpy + rdkit.

Uso:
    python tests/test_split_utils.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from split_utils import canonicalize_smiles, remove_leaking_from_train


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


def test_remove_leaking_from_train():
    canonical = np.array(["A", "B", "A", "C", "D"], dtype=object)
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
    test_remove_leaking_from_train()
    test_remove_leaking_from_train_no_leak()
    print("\n>>> SPLIT_UTILS TESTS OK <<<")
```

- [ ] **Step 2: Correr el test y confirmar que falla**

Run: `python experiments/E2_deepsets/tests/test_split_utils.py`
Expected: `ModuleNotFoundError: No module named 'split_utils'`

- [ ] **Step 3: Crear `experiments/E2_deepsets/split_utils.py`**

```python
# experiments/E2_deepsets/split_utils.py
# coding: ascii
"""
split_utils.py -- Exp E Fase 2: funciones puras de dedup/leak, copiadas de
experiments/D_val_congelado/split.py (ya probadas ahi y en Exp B/C). Se usan
en train.py para reconstruir el mismo train set limpio a partir de
val_indices_frozen.npy (Exp D), sin volver a correr split.py completo ni
depender de otras carpetas de experimento (self-contained).
"""
import numpy as np
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

- [ ] **Step 4: Correr el test y confirmar que pasa**

Run: `python experiments/E2_deepsets/tests/test_split_utils.py`
Expected:
```
[OK] test_canonicalize_smiles_dedup_equivalent_forms
[OK] test_canonicalize_smiles_invalid_passthrough
[OK] test_remove_leaking_from_train
[OK] test_remove_leaking_from_train_no_leak

>>> SPLIT_UTILS TESTS OK <<<
```

- [ ] **Step 5: Commit**

```bash
git add experiments/E2_deepsets/split_utils.py experiments/E2_deepsets/tests/test_split_utils.py
git commit -m "exp-e-fase2: split_utils.py (copia de Exp D, TDD)"
```

---

### Task 3: `model_e2.py` — arquitectura DeepSets

**Files:**
- Create: `experiments/E2_deepsets/model_e2.py`

**Interfaces:**
- Produces: `NMR_Net(num_classes=19)`, con `forward(x_peaks, x_mask,
  x_cond) -> (B, num_classes)` donde `x_peaks: (B, max_peaks, 4)`,
  `x_mask: (B, max_peaks)`, `x_cond: (B, 8)`. Consumido por `train.py`
  (Task 6), `evaluate.py` (Task 7), `dump_predictions.py` (Task 8),
  `tests/test_forward.py` (Task 9).

**Nota de entorno:** requiere `torch`, no instalado localmente. Se escribe
completo y se verifica por revisión de código (comparación directa contra
`experiments/C_gap/model_c.py`, que usa el mismo estilo `nn.Module` +
`F.relu` y ya está probado en el cluster) — no se afirma haberlo ejecutado.
Lucas lo corre de verdad en el cluster como parte del smoke test (Task 9).

- [ ] **Step 1: Implementar `experiments/E2_deepsets/model_e2.py`**

```python
# coding: ascii
import torch
import torch.nn as nn
import torch.nn.functional as F


class NMR_Net(nn.Module):
    """
    Modelo Exp E Fase 2 (DeepSets): reemplaza la imagen HSQC + proyecciones
    1D (V10/Exp B/Exp C) por un conjunto de picos (delta_c, delta_h,
    amp_ch0, amp_ch1) extraidos directamente del pkl original (Exp E Fase
    1b, ver docs/Runs/RESULTS.md), hasta 32 por molecula con mascara de
    validos.
      - Por pico: MLP compartido 4 -> 64 -> 64 (mismos pesos para todos los
        picos de todas las moleculas -> invariante a permutacion).
      - Agregacion: promedio enmascarado sobre los picos validos (si una
        molecula no tiene picos validos, el agregado queda en cero).
      - Fusion: agregado (64) + condicionante FM (8) = 72 -> 128 -> 64 -> 19.
    Capacidad deliberadamente chica (~23k parametros, ver RATIONALE.md) --
    no es un modelo grande, es una decision tomada con evidencia previa
    (V10 8.6M parametros peor que Exp C 223k).
    """
    def __init__(self, num_classes=19):
        super(NMR_Net, self).__init__()

        self.peak_mlp1 = nn.Linear(4, 64)
        self.peak_mlp2 = nn.Linear(64, 64)
        self.agg_dim = 64

        fusion_dim = self.agg_dim + 8  # + condicionante FM

        self.fc_fusion1 = nn.Linear(fusion_dim, 128)
        self.fc_fusion2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x_peaks, x_mask, x_cond):
        # x_peaks: (batch, max_peaks, 4)
        # x_mask:  (batch, max_peaks) -- 1.0 en picos validos, 0.0 en padding
        x = F.relu(self.peak_mlp1(x_peaks))   # (batch, max_peaks, 64)
        x = F.relu(self.peak_mlp2(x))         # (batch, max_peaks, 64)

        mask = x_mask.unsqueeze(-1)                     # (batch, max_peaks, 1)
        x_masked = x * mask
        counts = mask.sum(dim=1).clamp(min=1.0)         # (batch, 1)
        agg = x_masked.sum(dim=1) / counts               # (batch, 64)

        x = torch.cat((agg, x_cond), dim=1)
        x = F.relu(self.fc_fusion1(x))
        x = F.relu(self.fc_fusion2(x))
        return self.fc_out(x)
```

- [ ] **Step 2: Revisión de código (no ejecución local)**

Verificar a mano: `forward` acepta 3 tensores con las shapes documentadas,
la agregación divide por `counts.clamp(min=1.0)` (nunca por cero), y el
conteo de parámetros da ~23,315 (`4*64+64=320` + `64*64+64=4160` +
`72*128+128=9344` + `128*64+64=8256` + `64*19+19=1235` = 23,315) — Task 9
lo confirma con un assert real en el cluster.

- [ ] **Step 3: Commit**

```bash
git add experiments/E2_deepsets/model_e2.py
git commit -m "exp-e-fase2: model_e2.py (DeepSets: MLP por pico + promedio enmascarado)"
```

---

### Task 4: `dataset_e2.py` — dataset de picos + condicionante FM

**Files:**
- Create: `experiments/E2_deepsets/dataset_e2.py`

**Interfaces:**
- Produces: `NMRPeaksDataset(peaks_path, labels_path, smiles_path)`, un
  `torch.utils.data.Dataset` cuyo `__getitem__` devuelve
  `((peaks, mask, cond_tensor), target_vec)` con
  `peaks: (max_peaks, 4)`, `mask: (max_peaks,)`, `cond_tensor: (8,)`,
  `target_vec: (19,)`. Consumido por `train.py` (Task 6), `evaluate.py`
  (Task 7), `dump_predictions.py` (Task 8).

**Nota de entorno:** requiere `torch`, no instalado localmente — mismo
tratamiento que Task 3 (revisión de código, no ejecución).

- [ ] **Step 1: Implementar `experiments/E2_deepsets/dataset_e2.py`**

```python
# coding: ascii
import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem


class NMRPeaksDataset(Dataset):
    """
    Dataset Exp E Fase 2 -- picos extraidos del pkl original (Exp E Fase
    1b) en vez de imagen HSQC + proyecciones 1D. Carga
    peaks_pkl_202465.npz completo en memoria al iniciar (chico, ~100MB,
    sin h5py de por medio).
    Condicionante: [total_senales, total_CH2, C,H,N,O,S,Hal] = 8 valores,
    calculado EXACTO igual que dataset_v10.py (no reinventar).
    Labels: 19 clases.
    """
    def __init__(self, peaks_path, labels_path, smiles_path):
        self.labels = np.load(labels_path).astype(np.float32)
        self.smiles = np.load(smiles_path, allow_pickle=True)

        npz = np.load(peaks_path)
        self.peaks = npz["peaks"].astype(np.float32)          # (N, max_peaks, 4)
        self.peaks_mask = npz["peaks_mask"].astype(np.float32)  # (N, max_peaks)

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
        peaks = torch.tensor(self.peaks[idx], dtype=torch.float32)
        mask = torch.tensor(self.peaks_mask[idx], dtype=torch.float32)

        target_vec = self.labels[idx]
        total_signals = np.sum(target_vec).astype(np.float32)
        # 19 dims: CH2 en indices 1, 5, 9, 12
        total_ch2 = (target_vec[1] + target_vec[5] +
                     target_vec[9] + target_vec[12]).astype(np.float32)

        cond_data = [total_signals, total_ch2] + self.formula_matrix[idx].tolist()
        cond_tensor = torch.tensor(cond_data, dtype=torch.float32)

        return (peaks, mask, cond_tensor), torch.tensor(target_vec)
```

- [ ] **Step 2: Revisión de código (no ejecución local)**

Comparar línea por línea contra
`experiments/D_val_congelado/dataset_v10.py`: el cálculo de
`formula_matrix`/`total_signals`/`total_ch2`/`cond_tensor` es idéntico; lo
único que cambia es que `hsqc_raw`/`vec_cat` (imagen + proyecciones) se
reemplazan por `peaks`/`mask` leídos del `.npz`.

- [ ] **Step 3: Commit**

```bash
git add experiments/E2_deepsets/dataset_e2.py
git commit -m "exp-e-fase2: dataset_e2.py (picos + condicionante FM, igual que dataset_v10.py)"
```

---

### Task 5: `train.py`

**Files:**
- Create: `experiments/E2_deepsets/train.py`

**Interfaces:**
- Consumes: `NMRPeaksDataset` (Task 4), `NMR_Net` (Task 3),
  `canonicalize_smiles`/`remove_leaking_from_train` (Task 2).

**Nota de entorno:** requiere `torch` — revisión de código, no ejecución
(mismo tratamiento que Tasks 3/4).

- [ ] **Step 1: Implementar `experiments/E2_deepsets/train.py`**

```python
# coding: ascii
"""
train.py -- Exp E Fase 2: modelo DeepSets sobre picos (Exp E Fase 1b), sin
regularizacion (misma decision que Exp C tras la falla de Exp B). Usa el
split congelado de Exp D (val_indices_frozen.npy); el train set se
reconstruye con la misma logica de dedup/leak que uso split.py
originalmente (copiada a split_utils.py), sin regenerar el archivo ni
depender de otras carpetas de experimento.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import time, os, yaml, argparse, random
import numpy as np
from pathlib import Path

from dataset_e2 import NMRPeaksDataset
from model_e2 import NMR_Net
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


def build_frozen_split(full_dataset, base_dir, cfg):
    """Reconstruye train/val a partir de val_indices_frozen.npy (Exp D).
    val queda EXACTAMENTE como esta en el archivo (nunca se toca). train
    es todo lo demas, menos las filas que comparten SMILES canonico con
    alguna de val (misma logica de leak que uso split.py de Exp D)."""
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train(config_path):
    set_seed(42)
    cfg = load_config(config_path)
    print(f"--- ENTRENAMIENTO EXP E FASE 2 (DeepSets sobre picos): {cfg['experiment_name']} ---")

    base_dir    = Path(cfg['paths']['base_dir'])
    peaks_path  = base_dir / cfg['paths']['peaks_filename']
    labels_path = base_dir / cfg['paths']['labels_filename']
    smiles_path = base_dir / cfg['paths']['smiles_filename']
    ckpt_dir    = base_dir / cfg['paths']['checkpoint_dir']
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    full_dataset = NMRPeaksDataset(str(peaks_path), str(labels_path), str(smiles_path))
    train_idx, val_idx = build_frozen_split(full_dataset, base_dir, cfg)
    train_ds = Subset(full_dataset, train_idx.tolist())
    val_ds   = Subset(full_dataset, val_idx.tolist())

    use_pin = cfg['system'].get('pin_memory', False) and device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=cfg['hyperparameters']['batch_size'],
                              shuffle=True, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)
    val_loader   = DataLoader(val_ds, batch_size=cfg['hyperparameters']['batch_size'],
                              shuffle=False, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)

    model    = NMR_Net(num_classes=19).to(device)
    n_params = count_params(model)
    print(f"[INFO] Parametros totales del modelo: {n_params:,} (esperado ~23,315; "
          f"V10 original ~8,603,299, Exp C ~223,000)")

    criterion = ConstrainedMSELoss(lambda_sum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=cfg['hyperparameters']['learning_rate'])

    sched_cfg = cfg['hyperparameters'].get('scheduler', {})
    patience  = sched_cfg.get('patience', 8)
    factor    = sched_cfg.get('factor', 0.7)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    print(f"[INFO] Scheduler: patience={patience}, factor={factor}")

    epochs = cfg['hyperparameters']['epochs']
    print(f"\n[START] {epochs} epochs...")
    start_time = time.time(); best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train(); running_loss = 0.0; epoch_start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            peaks = inputs[0].to(device); mask = inputs[1].to(device)
            cond = inputs[2].to(device); targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(peaks, mask, cond)
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

        if (epoch + 1) in (10, 20, 30) and val_loss > 0.10:
            print(f"[WARN] Val loss ({val_loss:.4f}) todavia muy por encima de la referencia "
                  f"de V10 (0.031) en la epoca {epoch+1}. Podria ser underfitting -- ver "
                  f"RATIONALE.md antes de esperar a que termine el entrenamiento completo.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / f"{cfg['experiment_name']}_best.pth")
            print("[SAVE] Nuevo mejor modelo!")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"{cfg['experiment_name']}_ep{epoch+1}.pth")

    print(f"\n[DONE] {(time.time()-start_time)/60:.1f} min. Mejor Val: {best_val_loss:.4f}")


def validate(model, loader, criterion, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            peaks = inputs[0].to(device); mask = inputs[1].to(device)
            cond = inputs[2].to(device); targets = targets.to(device)
            total += criterion(model(peaks, mask, cond), targets).item()
    return total / len(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)
```

- [ ] **Step 2: Revisión de código (no ejecución local)**

Comparar contra `experiments/C_gap/train.py`: idéntico salvo
`dataset_v10.NMRDataset` -> `dataset_e2.NMRPeaksDataset`,
`model_c.NMR_Net` -> `model_e2.NMR_Net`, `h5_filename` -> `peaks_filename`,
y `model(hsqc, proj, cond)` -> `model(peaks, mask, cond)` en `train()` y
`validate()`. El resto (seed, loss, scheduler, guardado de checkpoints,
warning temprano) sin cambios.

- [ ] **Step 3: Commit**

```bash
git add experiments/E2_deepsets/train.py
git commit -m "exp-e-fase2: train.py (adaptado a picos, mismo patron que Exp C)"
```

---

### Task 6: `evaluate.py`

**Files:**
- Create: `experiments/E2_deepsets/evaluate.py`

**Interfaces:**
- Consumes: `NMRPeaksDataset` (Task 4), `NMR_Net` (Task 3).

**Nota de entorno:** requiere `torch` — revisión de código, no ejecución.

- [ ] **Step 1: Implementar `experiments/E2_deepsets/evaluate.py`**

```python
# coding: ascii
"""
Evaluacion Exp E Fase 2 (DeepSets sobre picos, split CONGELADO) sobre el
checkpoint de este experimento (entrenado por train.py, Task 5). Mismo
patron que experiments/C_gap/evaluate.py: Subset sobre
val_indices_frozen.npy.

  --oraculo on   -> ajustar_conteo_doble_exacto (EMA ASISTIDA).
  --oraculo off  -> np.clip(np.floor(pred_raw), 0, None) (EMA CRUDA).
  --oraculo both -> corre ambos e imprime la tabla comparativa (DEFAULT).

Config: UN SOLO config.yaml. El checkpoint se deriva IGUAL que el guardado
del training: {base_dir}/{paths.checkpoint_dir}/{experiment_name}_best.pth.

NO ejecutar hasta tener el checkpoint _best.pth de este experimento.
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
    # Import perezoso: dataset_e2 trae rdkit; el smoke test no lo necesita.
    from dataset_e2 import NMRPeaksDataset
    from model_e2 import NMR_Net

    cfg = load_config(config_path)

    base_dir = Path(cfg["paths"]["base_dir"])
    peaks_path = base_dir / cfg["paths"]["peaks_filename"]
    labels_path = base_dir / cfg["paths"]["labels_filename"]
    smiles_path = base_dir / cfg["paths"]["smiles_filename"]
    ckpt_path = base_dir / cfg["paths"]["checkpoint_dir"] / f"{cfg['experiment_name']}_best.pth"
    val_indices_path = base_dir / cfg["paths"]["val_indices_filename"]

    num_workers = int(cfg["system"]["num_workers"])   # 0 (rule 1)

    modes = ["on", "off"] if oraculo == "both" else [oraculo]
    run_on = "on" in modes
    run_off = "off" in modes

    print("=" * 60)
    print("  EVALUACION EXP E FASE 2 (DeepSets sobre picos) - SPLIT CONGELADO")
    print("=" * 60)
    print(f"-> Experimento (checkpoint): {cfg['experiment_name']}")
    print(f"-> Modos: {modes}   | idx_ch2: {IDX_CH2}")
    print(f"-> num_workers={num_workers} (rule 1)  batch_size={eval_batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Dispositivo: {device.type.upper()}")

    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] No se encontro el checkpoint en:\n  {ckpt_path}")
        print("        Corri primero train.py (Task 5).")
        return
    if not os.path.exists(val_indices_path):
        print(f"\n[ERROR] No se encontro el split congelado en:\n  {val_indices_path}")
        print("        Corri primero experiments/D_val_congelado/split.py (Exp D).")
        return

    full_dataset = NMRPeaksDataset(str(peaks_path), str(labels_path), str(smiles_path))
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
            peaks = inputs[0].to(device)
            mask = inputs[1].to(device)
            cond = inputs[2].to(device)
            pred_raw = model(peaks, mask, cond).cpu().numpy()
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
    parser = argparse.ArgumentParser(description="Eval Exp E Fase 2 (split congelado)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config unico.")
    parser.add_argument("--oraculo", choices=["on", "off", "both"], default="both",
                        help="on=asistida, off=cruda, both=ambas + tabla (default).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size de evaluacion.")
    args = parser.parse_args()
    evaluate(args.config, args.oraculo, args.batch_size)
```

- [ ] **Step 2: Revisión de código (no ejecución local)**

Comparar contra `experiments/C_gap/evaluate.py`: idéntico salvo
`dataset_v10`/`model_c` -> `dataset_e2`/`model_e2`, `h5_filename` ->
`peaks_filename`, y `model(hsqc, proj, cond)` -> `model(peaks, mask, cond)`
en el loop de evaluación. Las funciones de métricas/oráculo/reporte (todas
puras numpy, sin dependencia de la arquitectura) quedan sin ningún cambio.

- [ ] **Step 3: Commit**

```bash
git add experiments/E2_deepsets/evaluate.py
git commit -m "exp-e-fase2: evaluate.py (adaptado a picos, mismo patron que Exp C)"
```

---

### Task 7: `dump_predictions.py`

**Files:**
- Create: `experiments/E2_deepsets/dump_predictions.py`

**Interfaces:**
- Consumes: `NMRPeaksDataset` (Task 4), `NMR_Net` (Task 3).

**Nota de entorno:** requiere `torch` — revisión de código, no ejecución.

- [ ] **Step 1: Implementar `experiments/E2_deepsets/dump_predictions.py`**

```python
# coding: ascii
"""
dump_predictions.py -- Exp E Fase 2: vuelca las predicciones del checkpoint
de este experimento sobre el val congelado (Exp D), para la GUI
(src/gui/gui_inspector.py, corre en tu PC).

NO reentrena. Solo forward pass sobre val_indices_frozen.npy (~14k mols,
minutos -- mas rapido todavia que Exp C, sin CNN de por medio).

Salida: predictions_<experiment_name>.parquet con columnas:
  idx, smiles, y_true (19 ints), y_pred_crude (19 ints), y_pred_assisted (19 ints)

Uso:
  python dump_predictions.py --config config.yaml
"""
import os
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

N_CLASSES = 19
IDX_CH2 = [1, 5, 9, 12]   # CH2, CH2-O, CH2-N, =CH2


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def oraculo_doble(pred_raw, total_real, ch2_real):
    """Ajuste de doble restriccion (modo asistido), identico al evaluate."""
    pred = np.floor(pred_raw).astype(int)
    rest = pred_raw - pred
    idx_rest = [i for i in range(N_CLASSES) if i not in IDX_CH2]

    falt = int(ch2_real - sum(pred[i] for i in IDX_CH2))
    if falt > 0:
        for i in sorted(IDX_CH2, key=lambda i: rest[i])[-falt:]:
            pred[i] += 1
    elif falt < 0:
        s = -falt
        for i in sorted(IDX_CH2, key=lambda i: rest[i]):
            if pred[i] > 0:
                pred[i] -= 1; s -= 1
                if s == 0: break

    falt = int((total_real - ch2_real) - sum(pred[i] for i in idx_rest))
    if falt > 0:
        for i in sorted(idx_rest, key=lambda i: rest[i])[-falt:]:
            pred[i] += 1
    elif falt < 0:
        s = -falt
        for i in sorted(idx_rest, key=lambda i: rest[i]):
            if pred[i] > 0:
                pred[i] -= 1; s -= 1
                if s == 0: break
    return pred


def main(config_path):
    from model_e2 import NMR_Net
    from dataset_e2 import NMRPeaksDataset

    cfg = load_config(config_path)
    base_dir = Path(cfg["paths"]["base_dir"])
    peaks_path = base_dir / cfg["paths"]["peaks_filename"]
    labels_path = base_dir / cfg["paths"]["labels_filename"]
    smiles_path = base_dir / cfg["paths"]["smiles_filename"]
    ckpt_path = base_dir / cfg["paths"]["checkpoint_dir"] / f"{cfg['experiment_name']}_best.pth"
    val_indices_path = base_dir / cfg["paths"]["val_indices_filename"]
    out_file = f"predictions_{cfg['experiment_name']}.parquet"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No se encontro el checkpoint en:\n  {ckpt_path}")
        return
    if not os.path.exists(val_indices_path):
        print(f"[ERROR] No se encontro el split congelado en:\n  {val_indices_path}")
        print("        Corri primero experiments/D_val_congelado/split.py (Exp D).")
        return

    print("[INFO] Cargando dataset...")
    ds = NMRPeaksDataset(str(peaks_path), str(labels_path), str(smiles_path))
    smiles_all = np.load(smiles_path, allow_pickle=True)

    val_indices = np.load(val_indices_path)
    val_ds = Subset(ds, val_indices.tolist())
    loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    print(f"[INFO] Val set (congelado): {len(val_ds)} moleculas")
    model = NMR_Net(num_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    rows = []
    ptr = 0
    with torch.no_grad():
        for inputs, targets in loader:
            peaks = inputs[0].to(device)
            mask = inputs[1].to(device)
            cond = inputs[2].to(device)
            out = model(peaks, mask, cond).cpu().numpy()
            t = targets.cpu().numpy().astype(int)
            c = cond.cpu().numpy()
            for k in range(len(t)):
                orig_idx = int(val_indices[ptr]); ptr += 1
                y_true = t[k]
                total = int(c[k, 0]); ch2 = int(c[k, 1])
                y_crude = np.clip(np.floor(out[k]), 0, None).astype(int)
                y_assist = oraculo_doble(out[k], total, ch2)
                rows.append({
                    "idx": orig_idx,
                    "smiles": str(smiles_all[orig_idx]),
                    "y_true": y_true.tolist(),
                    "y_pred_crude": y_crude.tolist(),
                    "y_pred_assisted": y_assist.tolist(),
                })

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(out_file)
        print(f"[OK] Guardado {len(df)} filas -> {out_file}")
    except Exception:
        import json
        alt = out_file.replace(".parquet", ".json")
        with open(alt, "w") as f:
            json.dump(rows, f)
        print(f"[OK] (fallback JSON) Guardado {len(rows)} filas -> {alt}")
        print("     (instala pyarrow para parquet: pip install pyarrow)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp E Fase 2: dump de predicciones para la GUI")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
```

- [ ] **Step 2: Revisión de código (no ejecución local)**

Comparar contra `experiments/C_gap/dump_predictions.py`: idéntico salvo
`model_c`/`dataset_v10` -> `model_e2`/`dataset_e2`, `h5_filename` ->
`peaks_filename`, y `model(hsqc, proj, cond)` -> `model(peaks, mask,
cond)`. `oraculo_doble` sin cambios (no depende de la arquitectura).

- [ ] **Step 3: Commit**

```bash
git add experiments/E2_deepsets/dump_predictions.py
git commit -m "exp-e-fase2: dump_predictions.py (adaptado a picos, para la GUI)"
```

---

### Task 8: `tests/test_forward.py` — smoke test offline

**Files:**
- Create: `experiments/E2_deepsets/tests/test_forward.py`

**Interfaces:**
- Consumes: `NMR_Net` (Task 3).

**Nota de entorno:** requiere `torch`, no disponible localmente — se
escribe completo y se revisa a mano contra `experiments/C_gap/tests/test_forward.py`
(mismo estilo). Lucas lo corre de verdad en el cluster (login node, CPU,
sin GPU) antes de cualquier `sbatch` — es el paso obligatorio de la regla 5
de CLAUDE.md.

- [ ] **Step 1: Implementar `experiments/E2_deepsets/tests/test_forward.py`**

```python
# coding: ascii
"""
Smoke test OFFLINE de Exp E Fase 2 (DeepSets sobre picos) - rule 5 de
CLAUDE.md.

NO depende de checkpoint ni datos reales. Valida:
  (1) el forward de model_e2 con picos + mascara + condicionante ->
      (B, 19), mismo contrato de salida que V10/Exp C,
  (2) que una molecula SIN picos validos (mascara toda en cero) no rompe
      el forward (division por cero en la agregacion) y da salida finita,
  (3) el conteo de parametros: se espera ~23,315 (mucho menos que los
      ~223k de Exp C o los ~8.6M de V10 -- confirma que el modelo es
      chico a proposito, no por error).

Correr en CPU (login node) antes de cualquier sbatch:
    python tests/test_forward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model_e2 import NMR_Net

N_CLASSES = 19
MAX_PEAKS = 32
EXPECTED_PARAMS = 23_315


def test_forward_shape():
    model = NMR_Net(num_classes=N_CLASSES)
    model.eval()
    B = 4
    peaks = torch.randn(B, MAX_PEAKS, 4)
    mask = torch.ones(B, MAX_PEAKS)
    cond = torch.randn(B, 8)
    with torch.no_grad():
        out = model(peaks, mask, cond)
    assert out.shape == (B, N_CLASSES), out.shape
    print(f"[OK] forward -> {tuple(out.shape)} (esperado ({B}, {N_CLASSES}))")


def test_forward_with_empty_molecule_no_nan():
    # Una molecula sin picos validos (mascara toda en cero) no debe romper
    # la agregacion (division por cero) ni dar NaN/Inf en la salida.
    model = NMR_Net(num_classes=N_CLASSES)
    model.eval()
    B = 3
    peaks = torch.randn(B, MAX_PEAKS, 4)
    mask = torch.ones(B, MAX_PEAKS)
    mask[0] = 0.0   # molecula 0: sin picos validos
    cond = torch.randn(B, 8)
    with torch.no_grad():
        out = model(peaks, mask, cond)
    assert torch.isfinite(out).all(), out
    print(f"[OK] forward con molecula sin picos -> sin NaN/Inf: {out[0]}")


def test_param_count_is_small_by_design():
    model = NMR_Net(num_classes=N_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_PARAMS, (
        f"Parametros = {n_params}, esperado {EXPECTED_PARAMS}. Si cambio, "
        f"revisar que las dimensiones de model_e2.py no se hayan tocado "
        f"por error (el tamano chico es una decision deliberada, ver RATIONALE.md)."
    )
    print(f"[OK] parametros = {n_params:,} (esperado {EXPECTED_PARAMS:,}; "
          f"Exp C ~223,000, V10 original ~8,603,299)")


if __name__ == "__main__":
    test_forward_shape()
    test_forward_with_empty_molecule_no_nan()
    test_param_count_is_small_by_design()
    print("\n>>> SMOKE EXP E FASE 2 OK - listo para sbatch run_train.sh <<<")
```

- [ ] **Step 2: Revisión de código (no ejecución local)**

Verificar a mano el conteo de parámetros contra el cálculo de Task 3
(`4*64+64=320` + `64*64+64=4160` + `72*128+128=9344` + `128*64+64=8256` +
`64*19+19=1235` = `23,315`) — coincide con `EXPECTED_PARAMS`. Lucas
confirma la ejecución real en el cluster.

- [ ] **Step 3: Commit**

```bash
git add experiments/E2_deepsets/tests/test_forward.py
git commit -m "exp-e-fase2: tests/test_forward.py (smoke test offline)"
```

---

### Task 9: `run_train.sh` + `run_eval.sh` (SLURM)

**Files:**
- Create: `experiments/E2_deepsets/run_train.sh`
- Create: `experiments/E2_deepsets/run_eval.sh`

**Interfaces:** ninguna (scripts de shell, no código Python).

- [ ] **Step 1: Crear `experiments/E2_deepsets/run_train.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=expE2_train
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE2_train_%j.out
#SBATCH --error=expE2_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/E2_deepsets

python -u train.py --config config.yaml
```

Nota: `--time=14:00:00` es el mismo límite que Exp C (14h), aunque este
entrenamiento debería terminar mucho antes (sin CNN de por medio) — se deja
el mismo margen por las dudas, no hace falta ajustar el request de SLURM
para que corra más rápido.

- [ ] **Step 2: Crear `experiments/E2_deepsets/run_eval.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=expE2_eval
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE2_eval_%j.out
#SBATCH --error=expE2_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/E2_deepsets

python -u evaluate.py --config config.yaml --oraculo both --batch-size 256
```

- [ ] **Step 3: Commit**

```bash
git add experiments/E2_deepsets/run_train.sh experiments/E2_deepsets/run_eval.sh
git commit -m "exp-e-fase2: run_train.sh + run_eval.sh (SLURM)"
```

---

### Task 10: `README.md`

**Files:**
- Create: `experiments/E2_deepsets/README.md`

**Interfaces:** ninguna (documentación).

- [ ] **Step 1: Crear `experiments/E2_deepsets/README.md`**

```markdown
# Exp E — Fase 2: Modelo DeepSets sobre Picos

Checklist para correr esto en el cluster. A diferencia de Exp B/C, este
modelo entrena mucho más rápido (sin CNN sobre imágenes de 256×256) — no
esperes que tarde 10-14h como los anteriores, probablemente termine las 100
épocas en bastante menos de una hora.

## Antes de empezar: copiar el archivo de picos al cluster

`peaks_pkl_202465.npz` (generado en Exp E Fase 1b) está solo en tu máquina
local Windows (`E:\Proyectos\SciTrix\ScitrixDB\DB_nmr_to_vector\202K_suma\`).
Copialo a `/home/lpassaglia.iquir/DB_200k/` en el cluster (vía `scp` o lo
que uses normalmente) antes de seguir.

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/E2_deepsets`
3. Confirmar que existen en `/home/lpassaglia.iquir/DB_200k/`:
   - `peaks_pkl_202465.npz` (recién copiado)
   - `vectors_13c_19v_202465.npy`, `smiles_202465.npy` (ya deberían estar)
   - `val_indices_frozen.npy` (lo generó Exp D)
4. Smoke test obligatorio antes de cualquier `sbatch`:
   ```bash
   python tests/test_forward.py
   python tests/test_split_utils.py
   ```
   El primero debería mostrar ~23,315 parámetros (mucho menos que los
   ~223k de Exp C o los ~8.6M de V10) — si el número no coincide, algo
   está mal conectado en `model_e2.py`, avisá antes de entrenar.
5. Lanzar el entrenamiento:
   ```bash
   sbatch run_train.sh
   ```
6. **A diferencia de Exp B/C, revisá el log temprano** — como este modelo
   entrena mucho más rápido, probablemente veas las 100 épocas completas
   en minutos, no en horas. Mirá `expE2_train_<jobid>.out`.
7. Cuando termine, revisar el val loss final y compararlo contra V10
   (0.031) y Exp C (0.037) — no hay una referencia previa de picos, así
   que cualquier valor en ese orden de magnitud es razonable.
8. Evaluar el checkpoint sobre el mismo val congelado:
   ```bash
   sbatch run_eval.sh
   ```
9. (Opcional, para la GUI) Volcar las predicciones:
   ```bash
   python dump_predictions.py --config config.yaml
   ```
10. Revisar `expE2_eval_<jobid>.out`: copiar la tabla "EMA CRUDA vs
    ASISTIDA" y, sobre todo, si las confusiones `Cqsp2`↔`=CH/Ar` y
    `CH2`↔`CH2-N` (idénticas en V10/Exp B/Exp C) mejoraron o
    desaparecieron — es el indicador real de si la representación de
    picos resolvió el problema. Agregar los resultados a
    `docs/Runs/RESULTS.md`, fila "Exp E Fase 2".
11. Avisá a Claude Code con los números.

## Nota

Modelo deliberadamente chico (~23k parámetros) — no es un descuido, ver
`RATIONALE.md`. Si el resultado es bueno, el próximo paso (Set Transformer
o una variante más grande) es un experimento aparte, no una modificación
de este.
```

- [ ] **Step 2: Commit**

```bash
git add experiments/E2_deepsets/README.md
git commit -m "exp-e-fase2: README.md (checklist de ejecucion en cluster)"
```

---

## Al terminar

Cuando Lucas reporte los números reales del cluster (val loss, EMA
cruda/asistida, y si las confusiones persistentes mejoraron), actualizar
`docs/Runs/RESULTS.md` con una entrada "Exp E Fase 2" y decidir con esos
datos si se escribe el spec de Set Transformer, de una variante más grande
de DeepSets, o si el enfoque de conjuntos no resultó ser la respuesta.
