# Exp B — Regularización (dropout + weight_decay) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir `experiments/B_regularizacion/`, autocontenida, que repone `dropout=0.25` (en `fc_fusion1`/`fc_fusion2`) y `weight_decay=1e-5` (Adam) sobre el baseline V10, entrena un checkpoint nuevo, y lo evalúa sobre el mismo val congelado que generó Exp D.

**Architecture:** Copia de `model_v10.py` (`model_v11b.py`) con dos `nn.Dropout` agregados; `train.py` copia de `train_v10.py` + `weight_decay` en el optimizer + split reconstruido desde `val_indices_frozen.npy` (Exp D) usando `split_utils.py` (copia de las dos funciones puras de `split.py` de Exp D, ya testeadas). `evaluate.py` y `dump_predictions.py` siguen el mismo patrón `Subset`-sobre-val-congelado que Exp D.

**Tech Stack:** Python, PyTorch, RDKit, NumPy, PyYAML, h5py (cluster only), SLURM.

## Global Constraints

- `num_workers: 0` en todo DataLoader que use `NMRDataset` (h5py no es fork-safe) — CLAUDE.md regla 1.
- SLURM: `#SBATCH --gres=gpu:1`, nunca `--gpus=1` — CLAUDE.md regla 2.
- Nada hardcodeado salvo excepciones ya documentadas (nombres de clase, índices CH2, seed=42) — CLAUDE.md regla 3.
- El h5 a usar es `nmr_dataset_v3_202465_fast.h5` (rechunkeado) — CLAUDE.md regla 4.
- Smoke test obligatorio (`tests/test_forward.py`) antes de cualquier `sbatch` — CLAUDE.md regla 5.
- Scheduler `patience=8, factor=0.7` — CLAUDE.md regla 6. No cambia en este experimento.
- `num_classes=19`, orden de clases fijo — CLAUDE.md regla 7.
- **Split idéntico a Exp D:** usa `val_indices_frozen.npy` (ya generado en el cluster,
  `DB_200k/val_indices_frozen.npy`), NUNCA `random_split`. El train set se reconstruye con
  la misma lógica de dedup/leak que usó `split.py` de Exp D (`canonicalize_smiles` +
  `remove_leaking_from_train`, copiadas a `split_utils.py` — self-contained, no importar
  desde `experiments/D_val_congelado/`).
- Métrica primaria = EMA cruda; reportar siempre las dos (cruda y asistida) — PROMPT regla 9.
- Carpeta 100% autocontenida: todo archivo sin cambios respecto al baseline se copia
  verbatim (`dataset_v10.py`), nunca se importa vía `sys.path` — decisión de diseño ya
  tomada en Exp D.
- Este entorno de desarrollo (Windows, máquina de Lucas) tiene `numpy`, `rdkit`, `pandas`
  pero **NO tiene `torch`, `h5py`, `pyyaml` ni `pytest`**. Todo lo que dependa de
  `torch`/`h5py`/`yaml` (model, train.py, evaluate.py, dump_predictions.py, el smoke test)
  se implementa completo y se verifica solo con `ast.parse` — nunca se afirma haberlo
  ejecutado sin haberlo ejecutado. `split_utils.py` y su test SÍ son ejecutables de verdad
  aquí (numpy+rdkit) y deben correrse en serio, con evidencia TDD real.
- No se ejecuta nada en el cluster ni se lanza `sbatch` — el plan deja todo listo para que
  Lucas haga `git pull` + smoke test + `sbatch` manualmente.
- No se modifica ningún archivo de `src/` ni de `experiments/D_val_congelado/` (ambos son
  referencia cerrada).

---

### Task 1: Scaffold de `experiments/B_regularizacion/` + copia exacta de `dataset_v10.py`

**Files:**
- Create: `experiments/B_regularizacion/dataset_v10.py` (copia exacta de `src/data/dataset_v10.py`)

**Interfaces:**
- Produces: `NMRDataset(h5_path, labels_path, smiles_path)` — misma interfaz que en Exp D, la usan las Tasks 6, 7 y 8.

- [ ] **Step 1: Copiar el archivo sin modificar nada**

```bash
mkdir -p experiments/B_regularizacion/tests
cp src/data/dataset_v10.py experiments/B_regularizacion/dataset_v10.py
```

- [ ] **Step 2: Verificar que es una copia byte-idéntica**

```bash
diff src/data/dataset_v10.py experiments/B_regularizacion/dataset_v10.py
```

Expected: sin output (sin diferencias).

- [ ] **Step 3: Commit**

```bash
git add experiments/B_regularizacion/dataset_v10.py
git commit -m "exp B: copia autocontenida de dataset_v10.py (sin cambios)"
```

---

### Task 2: `RATIONALE.md`

**Files:**
- Create: `experiments/B_regularizacion/RATIONALE.md`

**Interfaces:** N/A (documentación).

- [ ] **Step 1: Escribir RATIONALE.md**

```markdown
# RATIONALE — Exp B: Regularización

## Hipótesis

El V10 tiene overfitting medido y documentado (train 0.013 vs val 0.031 en la corrida
real; val estancado desde ~ep48 mientras train sigue bajando — ver `docs/Runs/RESULTS.md`).
Los modelos V4-V6 tenían dropout y weight_decay; se perdieron entre V6 y V9 y nunca se
repusieron. Reponerlos debería achicar esa brecha train/val y, potencialmente, mejorar la
generalización (EMA en val).

## Qué causa del diagnóstico ataca

Causa #2 del diagnóstico del Exp A: overfitting sin regularización. No ataca el modality
collapse (desbalance de ramas conv/1D/FM) — eso lo ataca Exp C.

## Qué cambia exactamente respecto al V10

- `model_v11b.py`: copia de `model_v10.py` + `nn.Dropout(p=dropout)` después de la ReLU
  de `fc_fusion1` y de `fc_fusion2`. `dropout` es parámetro del constructor de `NMR_Net`,
  no un valor fijo — viene del config.
- `train.py`: copia de `train_v10.py` + `weight_decay=cfg['regularization']['weight_decay']`
  en el `optim.Adam`. `dropout=0.25` y `weight_decay=0.00001` (ya están en
  `config/db.yaml`, sección `regularization`, y se copian al `config.yaml` propio de este
  experimento).
- **Split de entrenamiento:** en vez de `random_split(seed=42)` (lo que hace
  `train_v10.py`), usa el val congelado que generó Exp D
  (`DB_200k/val_indices_frozen.npy`). El train set se reconstruye al arrancar el
  entrenamiento con la misma lógica de deduplicación/leak que usó `split.py` de Exp D
  (`canonicalize_smiles` + `remove_leaking_from_train`, copiadas a `split_utils.py` de
  este experimento — self-contained, no se reimporta Exp D). Esto no requiere que vuelvas
  a correr nada de Exp D; es un cálculo de menos de un minuto al arrancar el training, no
  por época.
- Todo lo demás queda igual al V10: mismo `ConstrainedMSELoss` (lambda_sum=0.5), mismo
  scheduler (`ReduceLROnPlateau`, patience=8, factor=0.7), 100 épocas, `num_workers=0`,
  mismo batch_size (64) y learning_rate (0.001).

## Qué métrica esperás mover y cuánto

El gap train/val (loss) debería achicarse respecto al de V10 (train 0.013 / val 0.031).
La EMA cruda/asistida sobre el val congelado no debería empeorar respecto a la referencia
"V10-on-frozen-val" (0.93% cruda / 90.66% asistida, `docs/Runs/RESULTS.md`); idealmente
mejora. Ojo: esa referencia está inflada por contaminación train/val (documentado en
`docs/Runs/RESULTS.md` y en el `RATIONALE.md` de Exp D) — Exp B, en cambio, SÍ excluye el
val congelado de su entrenamiento, así que su número es limpio. La comparación más honesta
no es solo "¿la EMA subió?", sino también "¿el gap train/val bajó?": eso es lo que prueba
que la regularización está haciendo su trabajo, independientemente de cuánto se mueva la
EMA en este val específico.

## Criterio de éxito/fracaso

- **Éxito:** el gap train/val final es menor que el de V10 (0.031-0.013=0.018), y la EMA
  cruda sobre el val congelado no cae por debajo de un margen razonable respecto a la
  referencia. Shapes del smoke test idénticos a V10 (dropout no cambia dimensiones).
- **Fracaso:** dropout/weight_decay tan agresivos que el modelo underfittea (val loss no
  baja, o EMA cruda cae fuerte) — señal de que 0.25/1e-5 son demasiado agresivos para este
  dataset y hace falta ajustar.
```

- [ ] **Step 2: Commit**

```bash
git add experiments/B_regularizacion/RATIONALE.md
git commit -m "exp B: agrega RATIONALE.md"
```

---

### Task 3: `model_v11b.py` — modelo con dropout

**Files:**
- Create: `experiments/B_regularizacion/model_v11b.py`

**Interfaces:**
- Produces: `NMR_Net(num_classes=19, dropout=0.25)` — forward `(x_img, x_proj, x_cond) -> (batch, num_classes)`. Consumido por Tasks 6 (train.py), 7 (evaluate.py), 8 (dump_predictions.py), 9 (tests/test_forward.py).

> **Nota de entorno:** este archivo importa `torch` — no ejecutable en este entorno de
> desarrollo (sin torch instalado). Verificación aquí: `ast.parse` + comparación manual
> línea por línea contra `model_v10.py` para confirmar que el ÚNICO cambio es el dropout.
> La verificación real (forward pass) la hace Lucas en el cluster vía el smoke test
> (Task 9).

- [ ] **Step 1: Crear `model_v11b.py`**

```python
# coding: ascii
import torch
import torch.nn as nn
import torch.nn.functional as F

class NMR_Net(nn.Module):
    """
    Modelo V11B: igual a V10 (HSQC 2 canales + Formula Molecular + 19 clases)
    + dropout en fc_fusion1/fc_fusion2 (Exp B: regularizacion).
      - Conv2d(2->16): 2 canales (de V8)
      - fusion_dim: flat_dim + 128 + 8  (8 = cond con FM, de V9)
      - dropout(p=dropout) despues de cada ReLU de fusion (Exp B)
    """
    def __init__(self, num_classes=19, dropout=0.25):
        super(NMR_Net, self).__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)  # 2 canales
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flat_dim = 64 * 32 * 32

        self.fc_proj1 = nn.Linear(512, 256)
        self.fc_proj2 = nn.Linear(256, 128)

        # Condicionante 8 valores (con FM)
        fusion_dim = self.flat_dim + 128 + 8

        self.fc_fusion1 = nn.Linear(fusion_dim, 128)
        self.drop1 = nn.Dropout(p=dropout)
        self.fc_fusion2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(p=dropout)
        self.fc_out     = nn.Linear(64, num_classes)

    def forward(self, x_img, x_proj, x_cond):
        # x_img: (batch, 2, 256, 256)
        x1 = self.pool1(F.relu(self.conv1(x_img)))
        x1 = self.pool2(F.relu(self.conv2(x1)))
        x1 = self.pool3(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, self.flat_dim)

        x2 = F.relu(self.fc_proj1(x_proj))
        x2 = F.relu(self.fc_proj2(x2))

        x = torch.cat((x1, x2, x_cond), dim=1)
        x = F.relu(self.fc_fusion1(x))
        x = self.drop1(x)
        x = F.relu(self.fc_fusion2(x))
        x = self.drop2(x)
        return self.fc_out(x)
```

- [ ] **Step 2: Verificación estática**

Run: `python -c "import ast; ast.parse(open('experiments/B_regularizacion/model_v11b.py', encoding='utf-8').read())"`
Expected: sin output.

- [ ] **Step 3: Confirmar por inspección que el único delta vs `model_v10.py` es el dropout**

Run: `diff src/models/model_v10.py experiments/B_regularizacion/model_v11b.py`
Expected: diff muestra únicamente: (a) el docstring de la clase actualizado, (b) el
parámetro `dropout=0.25` agregado al `__init__`, (c) las líneas `self.drop1 =
nn.Dropout(...)` y `self.drop2 = nn.Dropout(...)`, (d) las dos líneas `x =
self.drop1(x)` / `x = self.drop2(x)` en `forward`. Ningún otro cambio (conv, pooling,
fc_proj, fusion_dim, fc_out deben quedar idénticos).

- [ ] **Step 4: Commit**

```bash
git add experiments/B_regularizacion/model_v11b.py
git commit -m "exp B: model_v11b.py - model_v10.py + dropout en fc_fusion1/fc_fusion2"
```

---

### Task 4: `split_utils.py` — funciones puras de dedup/leak (TDD, corre local)

**Files:**
- Create: `experiments/B_regularizacion/split_utils.py`
- Test: `experiments/B_regularizacion/tests/test_split_utils.py`

**Interfaces:**
- Produces: `canonicalize_smiles(smiles_array: np.ndarray) -> (np.ndarray[object], int)`,
  `remove_leaking_from_train(train_idx: np.ndarray, val_idx: np.ndarray, canonical_smiles: np.ndarray) -> (np.ndarray[int64], int)`.
  Usados por Task 6 (`train.py`).

- [ ] **Step 1: Escribir el test (va a fallar: `split_utils.py` no existe todavía)**

Crear `experiments/B_regularizacion/tests/test_split_utils.py`:

```python
# coding: ascii
"""
Tests locales (sin torch/h5py) de split_utils.py -- Exp B.

Mismas funciones y mismos casos que
experiments/D_val_congelado/tests/test_split.py (Tasks 4-5 de ese plan):
split_utils.py es una copia autocontenida de esas dos funciones puras,
ya probadas en Exp D. Corren en cualquier maquina con numpy + rdkit.

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
    test_remove_leaking_from_train()
    test_remove_leaking_from_train_no_leak()
    print("\n>>> SPLIT_UTILS TESTS OK <<<")
```

- [ ] **Step 2: Correr el test para confirmar que falla**

Run: `python experiments/B_regularizacion/tests/test_split_utils.py`
Expected: `ModuleNotFoundError: No module named 'split_utils'` (todavía no existe el archivo).

- [ ] **Step 3: Crear `split_utils.py`**

```python
# coding: ascii
"""
split_utils.py -- Exp B: funciones puras de dedup/leak, copiadas de
experiments/D_val_congelado/split.py (Tasks 4-5 de ese plan, ya
probadas). Se usan en train.py para reconstruir el mismo train set
limpio a partir de val_indices_frozen.npy (Exp D), sin volver a correr
split.py completo ni depender de esa carpeta (self-contained).
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

- [ ] **Step 4: Correr el test de nuevo — todos deben pasar**

Run: `python experiments/B_regularizacion/tests/test_split_utils.py`
Expected:
```
[OK] test_canonicalize_smiles_dedup_equivalent_forms
[OK] test_canonicalize_smiles_invalid_passthrough
[OK] test_remove_leaking_from_train
[OK] test_remove_leaking_from_train_no_leak

>>> SPLIT_UTILS TESTS OK <<<
```
(RDKit puede imprimir líneas `SMILES Parse Error` a stderr por el SMILES inválido de
prueba — es esperado y no afecta el resultado.)

- [ ] **Step 5: Commit**

```bash
git add experiments/B_regularizacion/split_utils.py experiments/B_regularizacion/tests/test_split_utils.py
git commit -m "exp B: split_utils.py - canonicalize_smiles + remove_leaking_from_train (TDD)"
```

---

### Task 5: `config.yaml`

**Files:**
- Create: `experiments/B_regularizacion/config.yaml`

**Interfaces:**
- Consumes: valores canónicos de `config/db.yaml` (sección `regularization`, ya con
  `dropout: 0.25` y `weight_decay: 0.00001`), copiados a mano — convención ya establecida.
- Produces: config leído por `train.py` (Task 6), `evaluate.py` (Task 7),
  `dump_predictions.py` (Task 8).

- [ ] **Step 1: Escribir config.yaml**

```yaml
# experiments/B_regularizacion/config.yaml
#
# Exp B: reponer dropout + weight_decay sobre el baseline V10. Entrena un
# checkpoint NUEVO (no reutiliza el de V10). Usa el split congelado de
# Exp D (val_indices_frozen.npy) -- ver Global Constraints del plan.
#
# OJO: configs/config_V10.yaml (en git) tiene h5_filename SIN "_fast" y
# num_workers=4 -- ambos violan reglas duras del proyecto. Este config.yaml
# usa los valores correctos. No copiar config_V10.yaml tal cual.

experiment_name: "nmr_202k_v11b_reg_2ch_fm_19v"

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  h5_filename: "nmr_dataset_v3_202465_fast.h5"
  labels_filename: "vectors_13c_19v_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_V11B_reg"
  val_indices_filename: "val_indices_frozen.npy"

hyperparameters:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  scheduler:
    patience: 8
    factor: 0.7

regularization:
  dropout: 0.25
  weight_decay: 0.00001

system:
  device: "cuda"
  num_workers: 0
  pin_memory: true
```

- [ ] **Step 2: Commit**

```bash
git add experiments/B_regularizacion/config.yaml
git commit -m "exp B: agrega config.yaml"
```

---

### Task 6: `train.py`

**Files:**
- Create: `experiments/B_regularizacion/train.py`

**Interfaces:**
- Consumes: `NMR_Net(num_classes, dropout)` (Task 3), `NMRDataset` (Task 1),
  `canonicalize_smiles`, `remove_leaking_from_train` (Task 4), `config.yaml` (Task 5).
- Produces: checkpoint en `{base_dir}/{checkpoint_dir}/{experiment_name}_best.pth` —
  consumido por Task 7 (`evaluate.py`) y Task 8 (`dump_predictions.py`).

> **Nota de entorno:** requiere `torch`/`h5py`/`yaml` — no ejecutable aquí. Verificación:
> `ast.parse` + comparación manual contra `src/train_v10.py` para confirmar que los únicos
> deltas son los descritos en el Step 3.

- [ ] **Step 1: Crear `train.py`**

```python
# coding: ascii
"""
train.py -- Exp B: regularizacion (dropout + weight_decay) sobre el
baseline V10. Usa el split congelado de Exp D (val_indices_frozen.npy);
el train set se reconstruye con la misma logica de dedup/leak que uso
split.py originalmente (copiada a split_utils.py), sin regenerar el
archivo ni depender de experiments/D_val_congelado/.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import time, os, yaml, argparse, random
import numpy as np
from pathlib import Path

from dataset_v10 import NMRDataset
from model_v11b import NMR_Net
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


def train(config_path):
    set_seed(42)
    cfg = load_config(config_path)
    print(f"--- ENTRENAMIENTO EXP B (V10 + dropout + weight_decay): {cfg['experiment_name']} ---")

    base_dir    = Path(cfg['paths']['base_dir'])
    h5_path     = base_dir / cfg['paths']['h5_filename']
    labels_path = base_dir / cfg['paths']['labels_filename']
    smiles_path = base_dir / cfg['paths']['smiles_filename']
    ckpt_dir    = base_dir / cfg['paths']['checkpoint_dir']
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    full_dataset = NMRDataset(str(h5_path), str(labels_path), str(smiles_path))
    train_idx, val_idx = build_frozen_split(full_dataset, base_dir, cfg)
    train_ds = Subset(full_dataset, train_idx.tolist())
    val_ds   = Subset(full_dataset, val_idx.tolist())

    use_pin = cfg['system'].get('pin_memory', False) and device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=cfg['hyperparameters']['batch_size'],
                              shuffle=True, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)
    val_loader   = DataLoader(val_ds, batch_size=cfg['hyperparameters']['batch_size'],
                              shuffle=False, num_workers=cfg['system']['num_workers'], pin_memory=use_pin)

    dropout      = cfg['regularization']['dropout']
    weight_decay = cfg['regularization']['weight_decay']
    model     = NMR_Net(num_classes=19, dropout=dropout).to(device)
    criterion = ConstrainedMSELoss(lambda_sum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=cfg['hyperparameters']['learning_rate'],
                           weight_decay=weight_decay)
    print(f"[INFO] Regularizacion: dropout={dropout}, weight_decay={weight_decay}")

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
            hsqc = inputs[0].to(device); proj = inputs[1].to(device)
            cond = inputs[2].to(device); targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(hsqc, proj, cond)
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


def validate(model, loader, criterion, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            hsqc = inputs[0].to(device); proj = inputs[1].to(device)
            cond = inputs[2].to(device); targets = targets.to(device)
            total += criterion(model(hsqc, proj, cond), targets).item()
    return total / len(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)
```

- [ ] **Step 2: Verificación estática**

Run: `python -c "import ast; ast.parse(open('experiments/B_regularizacion/train.py', encoding='utf-8').read())"`
Expected: sin output.

- [ ] **Step 3: Confirmar por inspección los deltas vs `src/train_v10.py`**

Run: `diff src/train_v10.py experiments/B_regularizacion/train.py`
Expected: el diff debe mostrar únicamente: (a) imports (`Subset` en vez de `random_split`,
`model_v11b`/`split_utils` en vez de `model_v10`), (b) la función nueva
`build_frozen_split`, (c) en `train()`: `train_idx, val_idx = build_frozen_split(...)` +
`Subset(...)` en vez del bloque `random_split`, (d) `dropout`/`weight_decay` leídos de
`cfg['regularization']` y pasados a `NMR_Net(...)` / `optim.Adam(...)`. El resto (loop de
epochs, `ConstrainedMSELoss`, scheduler, guardado de checkpoints, `validate()`) debe ser
idéntico.

- [ ] **Step 4: Commit**

```bash
git add experiments/B_regularizacion/train.py
git commit -m "exp B: train.py - train_v10.py + weight_decay + split congelado (requiere cluster para correr)"
```

---

### Task 7: `evaluate.py`

**Files:**
- Create: `experiments/B_regularizacion/evaluate.py` (basado en `experiments/D_val_congelado/evaluate.py`, cambiando solo el import del modelo)

**Interfaces:**
- Consumes: `NMR_Net` (de `model_v11b.py`, Task 3), `NMRDataset` (Task 1),
  `val_indices_frozen.npy` (Exp D, vía `cfg["paths"]["val_indices_filename"]`, Task 5).
- Produces: funciones `crude_predict`, `ajustar_conteo_doble_exacto`, `compute_ema`,
  `compute_mae`, `ema_entorno` — consumidas por Task 9 (`tests/test_forward.py`).

- [ ] **Step 1: Crear `evaluate.py`**

```python
# coding: ascii
"""
Evaluacion Exp B (V10 + dropout + weight_decay, split CONGELADO) sobre el
checkpoint de este experimento (entrenado por train.py, Task 6). Mismo
patron que experiments/D_val_congelado/evaluate.py: Subset sobre
val_indices_frozen.npy en vez de random_split.

  --oraculo on   -> ajustar_conteo_doble_exacto (EMA ASISTIDA).
  --oraculo off  -> np.clip(np.floor(pred_raw), 0, None) (EMA CRUDA).
  --oraculo both -> corre ambos e imprime la tabla comparativa (DEFAULT).

Config: UN SOLO config.yaml (ver Task 5). El checkpoint se deriva IGUAL
que el guardado del training: {base_dir}/{paths.checkpoint_dir}/{experiment_name}_best.pth.

Reglas (CLAUDE.md):
  - num_workers=0 (h5py no es fork-safe; rule 1).
  - val_indices_frozen.npy debe existir (Exp D ya lo genero).

NO ejecutar hasta tener el checkpoint _best.pth de este experimento (Task 6).
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
    from model_v11b import NMR_Net

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
    print("  EVALUACION EXP B (V10 + dropout + weight_decay) - SPLIT CONGELADO")
    print("=" * 60)
    print(f"-> Experimento (checkpoint): {cfg['experiment_name']}")
    print(f"-> Modos: {modes}   | idx_ch2: {IDX_CH2}")
    print(f"-> num_workers={num_workers} (rule 1)  batch_size={eval_batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Dispositivo: {device.type.upper()}")

    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] No se encontro el checkpoint en:\n  {ckpt_path}")
        print("        Corri primero train.py (Task 6).")
        return
    if not os.path.exists(val_indices_path):
        print(f"\n[ERROR] No se encontro el split congelado en:\n  {val_indices_path}")
        print("        Corri primero experiments/D_val_congelado/split.py (Exp D).")
        return

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
    parser = argparse.ArgumentParser(description="Eval Exp B (split congelado)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config unico (ver Task 5 del plan).")
    parser.add_argument("--oraculo", choices=["on", "off", "both"], default="both",
                        help="on=asistida, off=cruda, both=ambas + tabla (default).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size de evaluacion.")
    args = parser.parse_args()
    evaluate(args.config, args.oraculo, args.batch_size)
```

- [ ] **Step 2: Verificación estática**

Run: `python -c "import ast; ast.parse(open('experiments/B_regularizacion/evaluate.py', encoding='utf-8').read())"`
Expected: sin output.

- [ ] **Step 3: Confirmar que el único delta vs `experiments/D_val_congelado/evaluate.py` es el import del modelo y el texto de los banners**

Run: `diff experiments/D_val_congelado/evaluate.py experiments/B_regularizacion/evaluate.py`
Expected: diff muestra solo: `from model_v10 import` → `from model_v11b import` (dos
apariciones: import perezoso dentro de `evaluate()` y el docstring), los textos de
banner/mensaje de error actualizados a "Exp B" / "Corri primero train.py", y el
`description` del argparse. La lógica de métricas, oráculo, reporte y el flujo de
`evaluate()` deben ser idénticos.

- [ ] **Step 4: Commit**

```bash
git add experiments/B_regularizacion/evaluate.py
git commit -m "exp B: evaluate.py - mismo patron Subset que Exp D, importa model_v11b"
```

---

### Task 8: `dump_predictions.py`

**Files:**
- Create: `experiments/B_regularizacion/dump_predictions.py`

**Interfaces:**
- Consumes: `NMR_Net` (Task 3), `NMRDataset` (Task 1), `config.yaml` (Task 5),
  `val_indices_frozen.npy` (Exp D).
- Produces: `predictions_nmr_202k_v11b_reg_2ch_fm_19v.parquet`, consumido por
  `src/gui/gui_inspector.py` (sin cambios, corre en la PC de Lucas).

- [ ] **Step 1: Crear `dump_predictions.py`**

```python
# coding: ascii
"""
dump_predictions.py -- Exp B: vuelca las predicciones del checkpoint de
este experimento sobre el val congelado (Exp D), para la GUI
(src/gui/gui_inspector.py, corre en tu PC).

NO reentrena. Solo forward pass sobre val_indices_frozen.npy (~14k mols,
minutos).

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
    from model_v11b import NMR_Net
    from dataset_v10 import NMRDataset

    cfg = load_config(config_path)
    base_dir = Path(cfg["paths"]["base_dir"])
    h5_path = base_dir / cfg["paths"]["h5_filename"]
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
    ds = NMRDataset(str(h5_path), str(labels_path), str(smiles_path))
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
            hsqc = inputs[0].to(device)
            proj = inputs[1].to(device)
            cond = inputs[2].to(device)
            out = model(hsqc, proj, cond).cpu().numpy()
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
    parser = argparse.ArgumentParser(description="Exp B: dump de predicciones para la GUI")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
```

- [ ] **Step 2: Verificación estática**

Run: `python -c "import ast; ast.parse(open('experiments/B_regularizacion/dump_predictions.py', encoding='utf-8').read())"`
Expected: sin output.

- [ ] **Step 3: Confirmar que el formato de salida coincide con `src/dump_predictions.py`**

Run: `diff src/dump_predictions.py experiments/B_regularizacion/dump_predictions.py`
Expected: el diff muestra que las columnas del parquet (`idx`, `smiles`, `y_true`,
`y_pred_crude`, `y_pred_assisted`) y la función `oraculo_doble` son idénticas; los
cambios son solo el mecanismo de split (`Subset`+`val_indices_frozen.npy` en vez de
`random_split`), la lectura de rutas desde `config.yaml` en vez de constantes
hardcodeadas, y el import de `model_v11b`.

- [ ] **Step 4: Commit**

```bash
git add experiments/B_regularizacion/dump_predictions.py
git commit -m "exp B: dump_predictions.py - mismo formato parquet que V10, lee config.yaml"
```

---

### Task 9: `tests/test_forward.py` — smoke test (requiere torch, corre en el cluster)

**Files:**
- Create: `experiments/B_regularizacion/tests/test_forward.py`

**Interfaces:**
- Consumes: `NMR_Net` (Task 3).

> **Nota de entorno:** requiere `torch` — no ejecutable aquí. Verificación: `ast.parse`.
> La ejecución real la hace Lucas en el login node antes de `sbatch run_train.sh` (regla
> dura 5 de CLAUDE.md).

- [ ] **Step 1: Crear `tests/test_forward.py`**

```python
# coding: ascii
"""
Smoke test OFFLINE de Exp B (V10 + dropout + weight_decay) - rule 5 de
CLAUDE.md.

NO depende de checkpoint ni h5 real. Valida:
  (1) el forward de model_v11b con HSQC 2 canales -> (B, 19), mismas
      dimensiones que V10 (dropout no cambia shapes),
  (2) dropout esta activo en train() (dos forwards con el mismo input dan
      resultados distintos) e inactivo en eval() (dos forwards dan el
      mismo resultado) -- catch para el bug clasico de olvidarse
      model.eval() antes de evaluar/predecir.

Correr en CPU (login node) antes de cualquier sbatch:
    python tests/test_forward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model_v11b import NMR_Net

N_CLASSES = 19


def test_forward_shape():
    model = NMR_Net(num_classes=N_CLASSES, dropout=0.25)
    model.eval()
    B = 4
    hsqc = torch.randn(B, 2, 256, 256)
    proj = torch.randn(B, 512)
    cond = torch.randn(B, 8)
    with torch.no_grad():
        out = model(hsqc, proj, cond)
    assert out.shape == (B, N_CLASSES), out.shape
    print(f"[OK] forward -> {tuple(out.shape)} (esperado ({B}, {N_CLASSES}), igual que V10)")


def test_dropout_active_in_train_mode():
    torch.manual_seed(0)
    model = NMR_Net(num_classes=N_CLASSES, dropout=0.5)
    model.train()
    hsqc = torch.randn(2, 2, 256, 256)
    proj = torch.randn(2, 512)
    cond = torch.randn(2, 8)
    out1 = model(hsqc, proj, cond)
    out2 = model(hsqc, proj, cond)
    assert not torch.allclose(out1, out2), "dropout deberia dar resultados distintos en train()"
    print("[OK] dropout activo en train(): dos forwards con el mismo input difieren")


def test_dropout_inactive_in_eval_mode():
    torch.manual_seed(0)
    model = NMR_Net(num_classes=N_CLASSES, dropout=0.5)
    model.eval()
    hsqc = torch.randn(2, 2, 256, 256)
    proj = torch.randn(2, 512)
    cond = torch.randn(2, 8)
    with torch.no_grad():
        out1 = model(hsqc, proj, cond)
        out2 = model(hsqc, proj, cond)
    assert torch.allclose(out1, out2), "dropout NO deberia afectar en eval()"
    print("[OK] dropout inactivo en eval(): dos forwards con el mismo input coinciden")


if __name__ == "__main__":
    test_forward_shape()
    test_dropout_active_in_train_mode()
    test_dropout_inactive_in_eval_mode()
    print("\n>>> SMOKE EXP B OK - listo para sbatch run_train.sh <<<")
```

- [ ] **Step 2: Verificación estática**

Run: `python -c "import ast; ast.parse(open('experiments/B_regularizacion/tests/test_forward.py', encoding='utf-8').read())"`
Expected: sin output.

- [ ] **Step 3: Verificación real — la hace Lucas en el cluster (documentar, no ejecutar aquí)**

```bash
cd experiments/B_regularizacion
python tests/test_forward.py
```
Expected:
```
[OK] forward -> (4, 19) (esperado (4, 19), igual que V10)
[OK] dropout activo en train(): dos forwards con el mismo input difieren
[OK] dropout inactivo en eval(): dos forwards con el mismo input coinciden

>>> SMOKE EXP B OK - listo para sbatch run_train.sh <<<
```

- [ ] **Step 4: Commit**

```bash
git add experiments/B_regularizacion/tests/test_forward.py
git commit -m "exp B: tests/test_forward.py - smoke test offline (requiere cluster para correr)"
```

---

### Task 10: `run_train.sh` (SLURM)

**Files:**
- Create: `experiments/B_regularizacion/run_train.sh`

**Interfaces:**
- Consumes: `train.py` (Task 6), `config.yaml` (Task 5).

- [ ] **Step 1: Escribir `run_train.sh`, siguiendo el patrón de `src/run_eval_v9.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=expB_train
#SBATCH --partition=gpua10_hi
#SBATCH --output=expB_train_%j.out
#SBATCH --error=expB_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

cd ~/nmr-hsqc-to-vector-/experiments/B_regularizacion

python -u train.py --config config.yaml
```

`--time=14:00:00`: V10 (100 epochs, mismo dataset) tardó 620 min (~10.3h); se deja margen
por el overhead extra de `Subset` + dropout.

- [ ] **Step 2: Commit**

```bash
git add experiments/B_regularizacion/run_train.sh
git commit -m "exp B: run_train.sh (sbatch, --gres=gpu:1, 14h)"
```

---

### Task 11: `run_eval.sh` (SLURM)

**Files:**
- Create: `experiments/B_regularizacion/run_eval.sh`

**Interfaces:**
- Consumes: `evaluate.py` (Task 7), `config.yaml` (Task 5).

- [ ] **Step 1: Escribir `run_eval.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=expB_eval
#SBATCH --partition=gpua10_hi
#SBATCH --output=expB_eval_%j.out
#SBATCH --error=expB_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

cd ~/nmr-hsqc-to-vector-/experiments/B_regularizacion

python -u evaluate.py --config config.yaml --oraculo both --batch-size 256
```

- [ ] **Step 2: Commit**

```bash
git add experiments/B_regularizacion/run_eval.sh
git commit -m "exp B: run_eval.sh (sbatch, --gres=gpu:1)"
```

---

### Task 12: `README.md` — checklist para Lucas

**Files:**
- Create: `experiments/B_regularizacion/README.md`

**Interfaces:** N/A (documentación).

- [ ] **Step 1: Escribir README.md**

```markdown
# Exp B — Regularización (dropout + weight_decay)

Checklist para correr esto en el cluster. A diferencia de Exp D, esto SÍ
entrena un modelo nuevo desde cero — consume horas de GPU.

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/B_regularizacion`
3. Confirmar que existe `/home/lpassaglia.iquir/DB_200k/val_indices_frozen.npy`
   (lo generó Exp D). Si no está, avisá antes de seguir — no se puede entrenar sin eso.
4. Smoke test obligatorio antes de cualquier `sbatch`:
   ```bash
   python tests/test_forward.py
   python tests/test_split_utils.py
   ```
5. Lanzar el entrenamiento (dura horas — V10 tardó ~10.3h para 100 épocas
   con el mismo dataset):
   ```bash
   sbatch run_train.sh
   ```
6. Cuando termine, revisar `expB_train_<jobid>.out`: comparar el gap
   train/val final contra el de V10 (train 0.013 vs val 0.031 —
   `docs/Runs/RESULTS.md`). Confirmar que `.err` está limpio.
7. Evaluar el checkpoint nuevo sobre el mismo val congelado:
   ```bash
   sbatch run_eval.sh
   ```
8. (Opcional, para inspeccionar en la GUI) Volcar las predicciones:
   ```bash
   python dump_predictions.py --config config.yaml
   ```
   Bajate el `.parquet` a tu PC y abrilo con `src/gui/gui_inspector.py`.
9. Revisar `expB_eval_<jobid>.out`: copiar la tabla "EMA CRUDA vs
   ASISTIDA" a `docs/Runs/RESULTS.md`, fila "Exp B — regularización".
   Compará contra "V10-on-frozen-val" (0.93% / 90.66%) sabiendo que esa
   referencia está inflada por contaminación train/val — la comparación
   más honesta es también el gap train/val del paso 6.
10. Avisá a Claude Code con los números — con eso decidimos si seguimos
    con Exp C o si hace falta ajustar dropout/weight_decay.
```

- [ ] **Step 2: Commit**

```bash
git add experiments/B_regularizacion/README.md
git commit -m "exp B: agrega README.md con el checklist de comandos"
```

---

## Self-Review

**Cobertura del spec (RATIONALE, scripts modificados, config.yaml, train.py, evaluate.py,
dump_predictions.py, run_train.sh/run_eval.sh, README.md — sección "Qué quiero que
produzcas" del PROMPT):**
- RATIONALE.md ✓ (Task 2). model_v11b.py ✓ (Task 3). config.yaml ✓ (Task 5). train.py ✓
  (Task 6). evaluate.py ✓ (Task 7). dump_predictions.py ✓ (Task 8, aplica esta vez porque
  SÍ hay checkpoint nuevo, a diferencia de Exp D). run_train.sh + run_eval.sh ✓ (Tasks 10,
  11). README.md ✓ (Task 12). `dataset_v10.py` copiado sin cambios ✓ (Task 1, ya que el
  dataset no cambia en Exp B).
- Regularización exacta del workflow (dropout=0.25 en fc_fusion1/fc_fusion2,
  weight_decay=1e-5 en Adam) ✓ (Tasks 3, 6).
- Split congelado de Exp D reutilizado sin reimportar esa carpeta (self-contained) ✓
  (Task 4: `split_utils.py` copia las 2 funciones puras necesarias).
- Comparación contra "V10-on-frozen-val" con la salvedad de contaminación ya documentada
  ✓ (RATIONALE.md Task 2, README.md Task 12).

**Placeholders:** ninguno — cada step tiene código completo o el comando+output exacto
documentado.

**Consistencia de tipos/nombres:** `canonicalize_smiles`/`remove_leaking_from_train` se
llaman igual en `split_utils.py` (Task 4) y en `train.py` (Task 6). `NMR_Net(num_classes,
dropout)` se llama igual en `model_v11b.py` (Task 3), `train.py` (Task 6), `evaluate.py`
(Task 7), `dump_predictions.py` (Task 8) y `tests/test_forward.py` (Task 9).
`cfg["paths"]["val_indices_filename"]` se define en `config.yaml` (Task 5) y se lee igual
en `train.py`, `evaluate.py` y `dump_predictions.py`. `crude_predict`,
`ajustar_conteo_doble_exacto`, `compute_ema`, `compute_mae`, `ema_entorno` en
`evaluate.py` (Task 7) tienen las mismas firmas que en Exp D (comparabilidad de reportes).

**Alcance:** un solo experimento (B), autocontenido. No toca Exp D, Exp C, ni el código de
`src/`.
