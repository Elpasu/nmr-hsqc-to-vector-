# Exp F: cabeza Poisson + entrenamiento extendido, y estudio de escalado de datos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar listo para `sbatch` un experimento nuevo (Exp F: cabeza Poisson + 250 épocas sobre el Set Transformer de Exp E Fase 3, que ataca la EMA cruda estancada en ~2-3%) y una ablación de escalado de datos sobre Fase 3 sin cambios (curva EMA vs tamaño de train, para decidir si ampliar el dataset otra vez rendiría).

**Architecture:** Parte 1 vive en una carpeta nueva `experiments/F_poisson_head/`, copia del Set Transformer de `E3_dos_conjuntos/` con dos cambios: activación `softplus` en la salida del modelo + `ConstrainedPoissonLoss` (Poisson NLL + término de restricción de suma) en vez de `ConstrainedMSELoss`, y `epochs: 250` en el config. Parte 2 no crea carpeta nueva: agrega un subsampleo determinístico y seedeado (`subsample_train_idx`, nuevo en `split_utils.py` de `E3_dos_conjuntos/`) más 5 configs que reusan el Set Transformer y la loss MSE de Fase 3 sin ningún cambio.

**Tech Stack:** Python, PyTorch, NumPy, RDKit, PyYAML. Sin dependencias nuevas.

## Global Constraints

Copiadas verbatim del spec (`docs/superpowers/specs/2026-07-22-exp-f-poisson-y-escalado-design.md`) y de `CLAUDE.md`. Aplican a TODAS las tareas:

- `num_workers: 0` siempre (regla 1). No cambiar.
- SLURM usa `#SBATCH --gres=gpu:1`, nunca `--gpus=1` (regla 2).
- Nada hardcodeado en los `.py`: rutas, nombres de archivo y epochs/train_fraction salen del config YAML (regla 3).
- Encoding: todos los `.py` empiezan con `# coding: ascii` y usan solo ASCII en el código (regla 4).
- Smoke test offline obligatorio antes de cualquier `sbatch` (regla 5). Los tests que importan `torch` no pueden correr en esta máquina local (sin torch instalado) — se verifican por revisión de código exhaustiva, como en Fase 2/3, y Lucas los corre de verdad en el login node antes de cualquier `sbatch`. Los tests que NO importan `torch` (el de `subsample_train_idx`, Task 6) SÍ deben ejecutarse y pasar localmente.
- Scheduler `patience=8, factor=0.7` (regla 6). No cambiar en ningún config, ni siquiera en Exp F pese a las 250 épocas.
- `num_classes=19` y el orden de clases de `config/db.yaml` es fijo (regla 7). No reordenar.
- Split congelado idéntico a Exp D/B/C/E2/E3: `val_indices_frozen.npy` + `split_utils.py`. val (14428 moléculas) NUNCA se toca, ni siquiera en la Parte 2 (el subsampleo actúa solo sobre `train_idx`, después de calcularlo).
- **Poisson (Parte 1):** el modelo devuelve `lambda = softplus(logits)` (garantiza `lambda >= 0`); `nn.PoissonNLLLoss(log_input=False, full=True)` porque el input ya es `lambda`, no `log(lambda)`. El oráculo de post-proceso (`ajustar_conteo_doble_exacto`) no cambia.
- **Comparabilidad de val loss (Parte 1):** el valor numérico del val loss de Exp F (Poisson NLL) NO es comparable contra el 0.0097 (MSE) de Fase 3. La comparación real es por EMA cruda/asistida y por el mapa de confusiones.
- **Escalado (Parte 2):** NO se aísla el efecto composicional de las 58k moléculas nuevas — el muestreo es aleatorio sobre el pool completo (144k+58k). Cada fracción usa la MISMA loss MSE que Fase 3 (nunca Poisson), para no mezclar esta pregunta con la de la Parte 1.

**Rutas de datos (cluster, login-1):** `base_dir = /home/lpassaglia.iquir/DB_200k`. Todos los archivos que hacen falta (`peaks_pkl_202465.npz`, `peaks_13c_202465.npz`, `vectors_13c_19v_202465.npy`, `smiles_202465.npy`, `val_indices_frozen.npy`) ya están ahí desde Fase 3 — esta fase no genera datos nuevos.

---

### Task 1: Dataset de dos conjuntos (`experiments/F_poisson_head/dataset_f.py`)

Copia exacta de `experiments/E3_dos_conjuntos/dataset_e3.py` (misma clase `NMRTwoSetsDataset`, misma normalización min-max) — solo cambia el nombre de archivo y el comentario de cabecera. Se copia (no se importa) para que la carpeta sea autocontenida, como en todas las fases anteriores.

**Files:**
- Create: `experiments/F_poisson_head/dataset_f.py`
- Create: `experiments/F_poisson_head/tests/test_dataset_f.py`
- Reference (no modificar): `experiments/E3_dos_conjuntos/dataset_e3.py`, `experiments/E3_dos_conjuntos/tests/test_dataset_e3.py`

**Interfaces:**
- Produces: `NMRTwoSetsDataset(peaks_ch_path, peaks_13c_path, labels_path, smiles_path, norm_cfg)`. `__getitem__` devuelve `((peaks_ch (32,4), mask_ch (32,), peaks_13c (M,1), mask_13c (M,), cond (8,)), target (19,))`, todos `torch.float32` salvo target.

- [ ] **Step 1: Write the failing test**

```python
# experiments/F_poisson_head/tests/test_dataset_f.py
# coding: ascii
"""Test del dataset de Exp F con npz sinteticos chicos. Identico al de Fase
3 (dataset_e3.py) -- Exp F no cambia el dataset, solo el modelo y la loss."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from dataset_f import NMRTwoSetsDataset

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
    assert abs(peaks_ch[0, 0].item() - 1.0) < 1e-5
    assert abs(peaks_ch[0, 1].item() - 1.0) < 1e-5
    assert abs(peaks_ch[0, 2].item() - 1.0) < 1e-5
    assert abs(peaks_13c[0, 0].item() - 1.0) < 1e-5
    assert abs(peaks_13c[1, 0].item() - 0.5) < 1e-5
    print("[OK] shapes y normalizacion correctas")


if __name__ == "__main__":
    test_shapes_and_normalization()
    print("\n>>> TEST DATASET F OK <<<")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd experiments/F_poisson_head && python tests/test_dataset_f.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'dataset_f'`. (Requiere torch — si esta maquina no lo tiene instalado, verificar por lectura que el ModuleNotFoundError sea por `dataset_f`, no por `torch` faltante primero; si es por torch, documentar en el commit que este paso se revisó por código, no se ejecutó, y que Lucas debe correrlo en el login node antes del primer `sbatch`.)

- [ ] **Step 3: Write the dataset**

```python
# experiments/F_poisson_head/dataset_f.py
# coding: ascii
"""Exp F -- dataset identico al de Exp E Fase 3 (dataset_e3.py), copiado sin
cambios de logica. Exp F no toca el dataset, solo el modelo (cabeza
softplus) y la loss (Poisson en vez de MSE)."""
import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem


class NMRTwoSetsDataset(Dataset):
    """Dos conjuntos de picos:
      - crosspeaks C-H (delta_c, delta_h, amp_ch0, amp_ch1), de peaks_pkl (Fase 1b).
      - 13C (delta_c,), de peaks_13c (Fase 3) -- incluye cuaternarios.
    Normaliza los desplazamientos min-max con la calibracion del config
    (norm_cfg). Condicionante FM identico a dataset_v10/dataset_e2/dataset_e3
    (8 valores).
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

Run: `cd experiments/F_poisson_head && python tests/test_dataset_f.py`
Expected: PASS — `>>> TEST DATASET F OK <<<`. (Revisión de código si torch no está disponible localmente.)

- [ ] **Step 5: Commit**

```bash
git add experiments/F_poisson_head/dataset_f.py experiments/F_poisson_head/tests/test_dataset_f.py
git commit -m "exp-f: dataset de dos conjuntos (copia identica de Fase 3)"
```

---

### Task 2: Modelo Set Transformer con cabeza softplus (`experiments/F_poisson_head/model_f_settransformer.py`)

Copia de `model_e3_settransformer.py` con un único cambio real: la salida final pasa por `softplus` para garantizar `lambda >= 0` (lo que exige `PoissonNLLLoss(log_input=False)`). No agrega parámetros.

**Files:**
- Create: `experiments/F_poisson_head/model_f_settransformer.py`
- Create: `experiments/F_poisson_head/tests/test_forward_settransformer.py`
- Reference (no modificar): `experiments/E3_dos_conjuntos/model_e3_settransformer.py`

**Interfaces:**
- Produces: `NMR_SetTransformer(num_classes=19, d_model=64, n_heads=4, n_layers=2, n_seeds=1)`. `forward(peaks_ch, mask_ch, peaks_13c, mask_13c, cond) -> (B, 19)`, siempre `>= 0`.

- [ ] **Step 1: Write the failing test**

```python
# experiments/F_poisson_head/tests/test_forward_settransformer.py
# coding: ascii
"""Smoke test offline del Set Transformer con cabeza softplus (rule 5).
Ademas de los checks de Fase 3 (shape, sin NaN, invariancia a permutacion,
tamano chico), agrega el check especifico de Exp F: la salida SIEMPRE es
>= 0 (lo exige PoissonNLLLoss con log_input=False)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model_f_settransformer import NMR_SetTransformer

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


def test_param_count_unchanged_vs_fase3():
    model = NMR_SetTransformer(num_classes=N_CLASSES)
    n = sum(p.numel() for p in model.parameters())
    assert n < 200_000, n   # chico por diseno (V10 ~8.6M); softplus no agrega parametros
    print(f"[OK] parametros = {n:,} (igual a Fase 3, softplus no agrega parametros)")


def test_output_always_nonnegative():
    model = NMR_SetTransformer(num_classes=N_CLASSES).eval()
    B = 4
    with torch.no_grad():
        # inputs escalados x100 para forzar logits grandes en ambos signos.
        out = model(torch.randn(B, MAX_CH, 4) * 100, torch.ones(B, MAX_CH),
                    torch.randn(B, MAX_13C, 1) * 100, torch.ones(B, MAX_13C),
                    torch.randn(B, 8) * 100)
    assert (out >= 0).all(), out
    print("[OK] salida siempre >= 0 (softplus)")


if __name__ == "__main__":
    test_forward_shape(); test_empty_molecule_no_nan()
    test_permutation_invariance(); test_param_count_unchanged_vs_fase3()
    test_output_always_nonnegative()
    print("\n>>> SMOKE SET TRANSFORMER F OK <<<")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd experiments/F_poisson_head && python tests/test_forward_settransformer.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'model_f_settransformer'`.

- [ ] **Step 3: Write the model**

```python
# experiments/F_poisson_head/model_f_settransformer.py
# coding: ascii
"""Set Transformer (Lee et al. 2019) sobre la union de los dos conjuntos de
picos, identico a experiments/E3_dos_conjuntos/model_e3_settransformer.py
salvo un cambio: la salida final pasa por softplus para garantizar
lambda >= 0 (lo exige PoissonNLLLoss con log_input=False en train.py). No
agrega parametros nuevos."""
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
        Qp = self.fc_q(Q); Kp = self.fc_k(K); Vp = self.fc_v(K)
        H = self.num_heads
        d = self.dim_V // H
        Qh = torch.cat(Qp.split(d, 2), 0)
        Kh = torch.cat(Kp.split(d, 2), 0)
        Vh = torch.cat(Vp.split(d, 2), 0)
        logits = Qh.bmm(Kh.transpose(1, 2)) / math.sqrt(d)
        if valid_mask is not None:
            vm = (valid_mask > 0.5)
            vm = vm.repeat(H, 1).unsqueeze(1)
            logits = logits.masked_fill(~vm, float("-inf"))
            A = torch.softmax(logits, dim=2)
            A = torch.nan_to_num(A, nan=0.0)
        else:
            A = torch.softmax(logits, dim=2)
        O = Qh + A.bmm(Vh)
        O = torch.cat(O.split(Q.size(0), 0), 2)
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
        return self.mab(S, X, valid_mask)


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
        tokens = torch.cat([tok_ch, tok_13c], dim=1)
        valid = torch.cat([mask_ch, mask_13c], dim=1)

        x = tokens
        for sab in self.encoder:
            x = sab(x, valid)
        pooled = self.pma(x, valid).reshape(B, -1)

        h = torch.cat([pooled, cond], dim=1)
        h = F.relu(self.fc_fusion1(h))
        h = F.relu(self.fc_fusion2(h))
        return F.softplus(self.fc_out(h))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd experiments/F_poisson_head && python tests/test_forward_settransformer.py`
Expected: PASS — `>>> SMOKE SET TRANSFORMER F OK <<<`.

- [ ] **Step 5: Commit**

```bash
git add experiments/F_poisson_head/model_f_settransformer.py experiments/F_poisson_head/tests/test_forward_settransformer.py
git commit -m "exp-f: Set Transformer con cabeza softplus (lambda >= 0 para Poisson)"
```

---

### Task 3: `ConstrainedPoissonLoss` + `train.py` + `config.yaml` + `split_utils.py`

`split_utils.py` se copia sin cambios. `train.py` reemplaza `ConstrainedMSELoss` por `ConstrainedPoissonLoss` (Poisson NLL + mismo término de restricción de suma), simplifica `build_model` a un solo modelo (sin el switch `arch` de Fase 3, porque Exp F solo entrena Set Transformer), y lee `epochs: 250` del config.

**Files:**
- Create: `experiments/F_poisson_head/split_utils.py` (copia byte-a-byte de `experiments/E3_dos_conjuntos/split_utils.py`)
- Create: `experiments/F_poisson_head/config.yaml`
- Create: `experiments/F_poisson_head/train.py`
- Create: `experiments/F_poisson_head/tests/test_poisson_loss.py`
- Reference (no modificar): `experiments/E3_dos_conjuntos/train.py`, `experiments/E3_dos_conjuntos/split_utils.py`

**Interfaces:**
- Consumes: `NMRTwoSetsDataset` (Task 1), `NMR_SetTransformer` (Task 2), `canonicalize_smiles`/`remove_leaking_from_train` (split_utils).
- Produces: `ConstrainedPoissonLoss(lambda_sum=0.5)` (nn.Module, `forward(pred, target) -> scalar tensor`); `build_model(cfg, num_classes=19) -> NMR_SetTransformer`; checkpoint `{base_dir}/{checkpoint_dir}/{experiment_name}_best.pth`.

- [ ] **Step 1: Copiar split_utils.py**

Copiá `experiments/E3_dos_conjuntos/split_utils.py` a `experiments/F_poisson_head/split_utils.py` sin cambios:

```python
# experiments/F_poisson_head/split_utils.py
# coding: ascii
"""
split_utils.py -- Exp F: funciones puras de dedup/leak, copiadas de
experiments/E3_dos_conjuntos/split_utils.py (self-contained, sin depender
de otras carpetas de experimento).
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

- [ ] **Step 2: Write the failing test (para la loss)**

```python
# experiments/F_poisson_head/tests/test_poisson_loss.py
# coding: ascii
"""Test offline de ConstrainedPoissonLoss (rule 5). No necesita el modelo
completo -- construye tensores sinteticos que simulan la salida del
softplus (siempre >= 0) para verificar que la loss es finita en casos
limite (target=0, lambda cercano a 0) y que penaliza mas cuando la
prediccion esta mas lejos del target."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from train import ConstrainedPoissonLoss


def test_finite_when_target_is_zero():
    criterion = ConstrainedPoissonLoss(lambda_sum=0.5)
    pred = torch.full((2, 19), 0.001)   # lambda chico pero > 0 (softplus nunca da 0 exacto)
    target = torch.zeros(2, 19)
    loss = criterion(pred, target)
    assert torch.isfinite(loss), loss
    print(f"[OK] loss finita con target=0: {loss.item():.4f}")


def test_penaliza_mas_lejos_del_target():
    criterion = ConstrainedPoissonLoss(lambda_sum=0.5)
    target = torch.full((1, 19), 3.0)
    cerca = torch.full((1, 19), 3.0)
    lejos = torch.full((1, 19), 0.1)
    loss_cerca = criterion(cerca, target)
    loss_lejos = criterion(lejos, target)
    assert loss_cerca.item() < loss_lejos.item(), (loss_cerca.item(), loss_lejos.item())
    print(f"[OK] loss(cerca)={loss_cerca.item():.4f} < loss(lejos)={loss_lejos.item():.4f}")


def test_termino_de_suma_penaliza_desbalance_de_totales():
    criterion_con_suma = ConstrainedPoissonLoss(lambda_sum=0.5)
    criterion_sin_suma = ConstrainedPoissonLoss(lambda_sum=0.0)
    target = torch.zeros(1, 19); target[0, 0] = 5.0   # total real = 5
    pred = torch.zeros(1, 19); pred[0, 0] = 0.001; pred[0, 1] = 4.999  # mismo total (~5), clase distinta
    # ambas dan Poisson NLL similar por clase 0 (target=5 vs lambda=0.001, mal) pero
    # el termino de suma es chico (total predicho ~ total real). Solo lo comparamos
    # contra una prediccion con el mismo error por clase pero total MUY distinto.
    pred_mal_total = torch.zeros(1, 19); pred_mal_total[0, 0] = 0.001; pred_mal_total[0, 1] = 0.001
    loss_total_ok = criterion_con_suma(pred, target)
    loss_total_mal = criterion_con_suma(pred_mal_total, target)
    assert loss_total_mal.item() > loss_total_ok.item(), (loss_total_mal.item(), loss_total_ok.item())
    print(f"[OK] termino de suma penaliza el desbalance de totales: "
          f"ok={loss_total_ok.item():.4f} < mal={loss_total_mal.item():.4f}")


if __name__ == "__main__":
    test_finite_when_target_is_zero()
    test_penaliza_mas_lejos_del_target()
    test_termino_de_suma_penaliza_desbalance_de_totales()
    print("\n>>> TEST POISSON LOSS OK <<<")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd experiments/F_poisson_head && python tests/test_poisson_loss.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'train'`.

- [ ] **Step 4: Write config.yaml**

```yaml
# experiments/F_poisson_head/config.yaml
experiment_name: "nmr_202k_f_poisson_settransformer_19v"

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  peaks_ch_filename: "peaks_pkl_202465.npz"
  peaks_13c_filename: "peaks_13c_202465.npz"
  labels_filename: "vectors_13c_19v_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_F_poisson"
  val_indices_filename: "val_indices_frozen.npy"

model:
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
  epochs: 250
  scheduler:
    patience: 8
    factor: 0.7

system:
  device: "cuda"
  num_workers: 0
  pin_memory: true
```

- [ ] **Step 5: Write train.py**

```python
# experiments/F_poisson_head/train.py
# coding: ascii
"""
train.py -- Exp F: cabeza Poisson + entrenamiento extendido sobre el Set
Transformer de Exp E Fase 3. Cambios respecto a Fase 3:
  (1) modelo con activacion softplus en la salida (model_f_settransformer.py) --
      garantiza lambda >= 0, que exige PoissonNLLLoss(log_input=False).
  (2) ConstrainedPoissonLoss en vez de ConstrainedMSELoss (Poisson NLL por
      clase + mismo termino de restriccion de suma, lambda_sum=0.5).
  (3) epochs=250 en vez de 100 (en Fase 3 el LR nunca bajo de 0.001 en 100
      epocas -- scheduler patience=8/factor=0.7 sin cambios, regla 6).
Dataset, split congelado (Exp D), condicionante FM y arquitectura del Set
Transformer (d_model/n_heads/n_layers/n_seeds) identicos a Fase 3.

CAVEAT: el val loss reportado es Poisson NLL, NO es comparable numericamente
contra el 0.0097 (MSE) de Fase 3. Comparar por EMA (evaluate.py).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import time, os, yaml, argparse, random
import numpy as np
from pathlib import Path

from dataset_f import NMRTwoSetsDataset
from model_f_settransformer import NMR_SetTransformer
from split_utils import canonicalize_smiles, remove_leaking_from_train


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConstrainedPoissonLoss(nn.Module):
    """Poisson NLL por clase (el modelo ya devuelve lambda >= 0 via
    softplus, por eso log_input=False) + el mismo termino de restriccion de
    suma que ConstrainedMSELoss (MSE sobre el total predicho vs el total
    real, lambda_sum=0.5). full=True agrega el termino de Stirling
    (aproximacion de log(target!)) para que la loss sea una NLL propiamente
    normalizada, no solo la parte que importa para el gradiente."""
    def __init__(self, lambda_sum=0.5):
        super().__init__()
        self.poisson = nn.PoissonNLLLoss(log_input=False, full=True)
        self.mse = nn.MSELoss()
        self.lambda_sum = lambda_sum

    def forward(self, pred, target):
        li = self.poisson(pred, target)
        ls = self.mse(torch.sum(pred, dim=1), torch.sum(target, dim=1))
        return li + self.lambda_sum * ls


def load_config(p):
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(cfg, num_classes=19):
    m = cfg['model']
    return NMR_SetTransformer(
        num_classes=num_classes,
        d_model=int(m.get('d_model', 64)),
        n_heads=int(m.get('n_heads', 4)),
        n_layers=int(m.get('n_layers', 2)),
        n_seeds=int(m.get('n_seeds', 1)),
    )


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
    print(f"--- ENTRENAMIENTO EXP F (Poisson + entrenamiento extendido): {cfg['experiment_name']} ---")

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
    print(f"[INFO] Parametros totales del modelo: {n_params:,} "
          f"(igual a Fase 3 Set Transformer, ~70k; softplus no agrega parametros)")

    criterion = ConstrainedPoissonLoss(lambda_sum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=cfg['hyperparameters']['learning_rate'])
    sched_cfg = cfg['hyperparameters'].get('scheduler', {})
    patience = sched_cfg.get('patience', 8); factor = sched_cfg.get('factor', 0.7)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    print(f"[INFO] Scheduler: patience={patience}, factor={factor}")
    print("[INFO] Loss: ConstrainedPoissonLoss -- el val loss NO es comparable "
          "numericamente contra el 0.0097 (MSE) de Fase 3. Comparar por EMA.")

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
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd experiments/F_poisson_head && python tests/test_poisson_loss.py`
Expected: PASS — `>>> TEST POISSON LOSS OK <<<`.

- [ ] **Step 7: Commit**

```bash
git add experiments/F_poisson_head/split_utils.py experiments/F_poisson_head/config.yaml experiments/F_poisson_head/train.py experiments/F_poisson_head/tests/test_poisson_loss.py
git commit -m "exp-f: ConstrainedPoissonLoss + train.py (250 epocas, single-arch) + config"
```

---

### Task 4: `evaluate.py` y `dump_predictions.py`

Copia de los de Fase 3, adaptados a un solo modelo (sin el print de `arch`) y a los módulos de Exp F (`dataset_f`, `train.build_model`). El oráculo de doble restricción (`ajustar_conteo_doble_exacto`) es idéntico — sigue recibiendo un `pred_raw` continuo (ahora `lambda`, siempre `>= 0`) con parte fraccionaria interpretable.

**Files:**
- Create: `experiments/F_poisson_head/evaluate.py`
- Create: `experiments/F_poisson_head/dump_predictions.py`
- Create: `experiments/F_poisson_head/tests/test_oraculo.py`
- Reference (no modificar): `experiments/E3_dos_conjuntos/evaluate.py`, `experiments/E3_dos_conjuntos/dump_predictions.py`

**Interfaces:**
- Consumes: `build_model` y `ConstrainedPoissonLoss` no se usan en evaluate (solo el forward del modelo), `NMRTwoSetsDataset` (Task 1).
- Produces: reporte de EMA cruda/asistida + confusiones por consola (`evaluate.py`); `predictions_<experiment_name>.parquet` con columnas `idx, smiles, y_true, y_pred_crude, y_pred_assisted` (`dump_predictions.py`), mismo schema que todas las fases anteriores (compatible con `src/gui/gui_inspector.py`).

- [ ] **Step 1: Write the failing test**

```python
# experiments/F_poisson_head/tests/test_oraculo.py
# coding: ascii
"""El oraculo de doble restriccion es identico al de Fase 3/E2/Exp C -- no
cambia con Poisson, porque sigue recibiendo un pred_raw continuo (ahora
lambda, siempre >= 0) con parte fraccionaria interpretable. Este test fija
su comportamiento."""
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
    pred_raw = np.array([0.0, 1.9, 0.4] + [0.0] * 16, dtype=float)
    out = crude_predict(pred_raw)
    assert out[0] == 0 and out[1] == 1 and out[2] == 0, out
    print("[OK] crude = floor con clip a >=0 (lambda ya es >=0 por softplus)")


if __name__ == "__main__":
    test_asistida_respeta_totales(); test_crude_es_floor_no_negativo()
    print("\n>>> TEST ORACULO F OK <<<")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd experiments/F_poisson_head && python tests/test_oraculo.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'evaluate'`.

- [ ] **Step 3: Write evaluate.py**

```python
# experiments/F_poisson_head/evaluate.py
# coding: ascii
"""
Evaluacion Exp F (cabeza Poisson + 250 epocas, split CONGELADO) sobre el
checkpoint de este experimento (entrenado por train.py). Mismo patron que
experiments/E3_dos_conjuntos/evaluate.py: Subset sobre val_indices_frozen.npy.
Un solo modelo (Set Transformer con softplus) -- sin el switch de arch de
Fase 3.

  --oraculo on   -> ajustar_conteo_doble_exacto (EMA ASISTIDA).
  --oraculo off  -> np.clip(np.floor(pred_raw), 0, None) (EMA CRUDA).
  --oraculo both -> corre ambos e imprime la tabla comparativa (DEFAULT).

CAVEAT: no compares el val loss de train.py (Poisson NLL) contra el 0.0097
(MSE) de Fase 3 -- son escalas distintas. Esta evaluacion es la comparacion
real, via EMA y confusiones.

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
    """Modo CRUDO: floor con clip a >=0. lambda ya es >=0 por softplus, el
    clip queda como salvaguarda sin efecto practico."""
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
    from dataset_f import NMRTwoSetsDataset
    from train import build_model

    cfg = load_config(config_path)

    base_dir = Path(cfg["paths"]["base_dir"])
    peaks_ch = base_dir / cfg["paths"]["peaks_ch_filename"]
    peaks_13c = base_dir / cfg["paths"]["peaks_13c_filename"]
    labels_path = base_dir / cfg["paths"]["labels_filename"]
    smiles_path = base_dir / cfg["paths"]["smiles_filename"]
    ckpt_path = base_dir / cfg["paths"]["checkpoint_dir"] / f"{cfg['experiment_name']}_best.pth"
    val_indices_path = base_dir / cfg["paths"]["val_indices_filename"]

    num_workers = int(cfg["system"]["num_workers"])   # 0 (rule 1)

    modes = ["on", "off"] if oraculo == "both" else [oraculo]
    run_on = "on" in modes
    run_off = "off" in modes

    print("=" * 60)
    print("  EVALUACION EXP F (Poisson + entrenamiento extendido) - SPLIT CONGELADO")
    print("=" * 60)
    print(f"-> Experimento (checkpoint): {cfg['experiment_name']}")
    print(f"-> Modos: {modes}   | idx_ch2: {IDX_CH2}")
    print(f"-> num_workers={num_workers} (rule 1)  batch_size={eval_batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Dispositivo: {device.type.upper()}")

    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] No se encontro el checkpoint en:\n  {ckpt_path}")
        print("        Corri primero train.py.")
        return
    if not os.path.exists(val_indices_path):
        print(f"\n[ERROR] No se encontro el split congelado en:\n  {val_indices_path}")
        print("        Corri primero experiments/D_val_congelado/split.py (Exp D).")
        return

    full_dataset = NMRTwoSetsDataset(str(peaks_ch), str(peaks_13c), str(labels_path),
                                     str(smiles_path), cfg["normalization"])
    val_indices = np.load(val_indices_path)
    val_ds = Subset(full_dataset, val_indices.tolist())

    use_pin = bool(cfg["system"].get("pin_memory", False)) and device.type == "cuda"
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin)

    model = build_model(cfg, num_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_targets, all_pred_on, all_pred_off = [], [], []
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
    parser = argparse.ArgumentParser(description="Eval Exp F (split congelado)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config del experimento.")
    parser.add_argument("--oraculo", choices=["on", "off", "both"], default="both",
                        help="on=asistida, off=cruda, both=ambas + tabla (default).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size de evaluacion.")
    args = parser.parse_args()
    evaluate(args.config, args.oraculo, args.batch_size)
```

- [ ] **Step 4: Write dump_predictions.py**

```python
# experiments/F_poisson_head/dump_predictions.py
# coding: ascii
"""
dump_predictions.py -- Exp F: vuelca las predicciones del checkpoint de
este experimento sobre el val congelado (Exp D), para la GUI
(src/gui/gui_inspector.py, corre en tu PC). Mismo patron que
experiments/E3_dos_conjuntos/dump_predictions.py, adaptado a un solo modelo.

NO reentrena. Solo forward pass sobre val_indices_frozen.npy (~14k mols).

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
    from dataset_f import NMRTwoSetsDataset
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
    ds = NMRTwoSetsDataset(str(peaks_ch), str(peaks_13c), str(labels_path),
                           str(smiles_path), cfg["normalization"])
    smiles_all = np.load(smiles_path, allow_pickle=True)

    val_indices = np.load(val_indices_path)
    val_ds = Subset(ds, val_indices.tolist())
    loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    print(f"[INFO] Val set (congelado): {len(val_ds)} moleculas")
    model = build_model(cfg, num_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    rows = []
    ptr = 0
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
    parser = argparse.ArgumentParser(description="Exp F: dump de predicciones para la GUI")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd experiments/F_poisson_head && python tests/test_oraculo.py`
Expected: PASS — `>>> TEST ORACULO F OK <<<`. (Este test no importa torch — puede correr localmente ya mismo, verificalo.)

- [ ] **Step 6: Commit**

```bash
git add experiments/F_poisson_head/evaluate.py experiments/F_poisson_head/dump_predictions.py experiments/F_poisson_head/tests/test_oraculo.py
git commit -m "exp-f: evaluate + dump_predictions (single-arch, mismo oraculo)"
```

---

### Task 5: Scripts SLURM + README + RATIONALE (`experiments/F_poisson_head/`)

**Files:**
- Create: `experiments/F_poisson_head/run_train.sh`
- Create: `experiments/F_poisson_head/run_eval.sh`
- Create: `experiments/F_poisson_head/README.md`
- Create: `experiments/F_poisson_head/RATIONALE.md`
- Reference (no modificar): `experiments/E3_dos_conjuntos/run_train_settransformer.sh`, `experiments/E3_dos_conjuntos/run_eval.sh`

**Interfaces:**
- No produce interfaces de código — son artefactos operativos (scripts SLURM y documentación) para que Lucas corra el experimento.

- [ ] **Step 1: Write run_train.sh**

```bash
#!/bin/bash
#SBATCH --job-name=expF_train
#SBATCH --partition=gpua10_hi
#SBATCH --output=expF_train_%j.out
#SBATCH --error=expF_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/F_poisson_head

python -u train.py --config config.yaml
```

(Nota: `--time=06:00:00` es un techo de seguridad amplio — Fase 3 tardó 39 min en 100 épocas con el mismo Set Transformer; 250 épocas deberían rondar ~1.5-2h.)

- [ ] **Step 2: Write run_eval.sh**

```bash
#!/bin/bash
#SBATCH --job-name=expF_eval
#SBATCH --partition=gpua10_hi
#SBATCH --output=expF_eval_%j.out
#SBATCH --error=expF_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/F_poisson_head

python -u evaluate.py --config config.yaml --oraculo both --batch-size 256
```

- [ ] **Step 3: Write README.md**

```markdown
# Exp F: cabeza Poisson + entrenamiento extendido

Checklist para correr esto en el cluster. Un solo modelo (Set Transformer,
igual al ganador de Fase 3) con dos cambios: cabeza Poisson en vez de MSE,
y 250 épocas en vez de 100 (el LR nunca bajó de 0.001 en Fase 3— 100
épocas puede haberse quedado corto).

## Antes de empezar

No hace falta generar datos nuevos — se reusan `peaks_pkl_202465.npz`,
`peaks_13c_202465.npz`, `vectors_13c_19v_202465.npy`, `smiles_202465.npy`
y `val_indices_frozen.npy`, ya en `/home/lpassaglia.iquir/DB_200k/` desde
Fase 3.

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/F_poisson_head`
3. **Smoke tests obligatorios antes de cualquier `sbatch` (regla 5):**
   ```bash
   python tests/test_dataset_f.py
   python tests/test_forward_settransformer.py
   python tests/test_poisson_loss.py
   python tests/test_oraculo.py
   ```
   El de forward imprime el conteo de parámetros: debería dar ~70k, igual
   que Fase 3 (softplus no agrega parámetros). Si no es chico (< 200k),
   avisá antes de entrenar.
4. Lanzar el entrenamiento:
   ```bash
   sbatch run_train.sh
   ```
5. **Revisá el log temprano**: `expF_train_<jobid>.out`. A diferencia de
   Fase 3, el val loss es Poisson NLL, no MSE — **no lo compares
   numéricamente contra el 0.0097 de Fase 3**. Mirá en cambio si el LR
   baja de 0.001 antes de la época 250 (en Fase 3 nunca bajó en 100).
6. Evaluar el checkpoint sobre el val congelado:
   ```bash
   sbatch run_eval.sh
   ```
7. (Opcional, para la GUI) Volcar predicciones:
   ```bash
   python dump_predictions.py --config config.yaml
   ```
8. Revisar `expF_eval_<jobid>.out`: comparar EMA cruda y asistida contra
   Fase 3 (2.26% / 91.35%), y el mapa de confusiones cruzadas — sobre todo
   si `CH2`↔`CH2-N`, `C-2X` y `=CH/Ar`↔`Imina` bajaron.
9. Agregar una fila "Exp F" a `docs/Runs/RESULTS.md`.
10. Avisá a Claude Code con los números.
```

- [ ] **Step 4: Write RATIONALE.md**

```markdown
# RATIONALE — Exp F: cabeza Poisson + entrenamiento extendido

## Hipótesis

La EMA cruda nunca superó ~2-3% en ningún experimento del proyecto
(V10 0.61%, Exp C 0.89%, E2 0.74%, DeepSets F3 2.28%, Set Transformer F3
2.26%), pese al salto de Fase 3 en la asistida (91.35%). `WORKFLOW_V11`
(sección "Notas de dominio") ya señalaba que la loss actual (MSE sobre
conteos + redondeo) es conceptualmente floja para conteos: "la
incertidumbre de un conteo escala con su magnitud; MSE la trata como
constante". Nunca se probó una cabeza Poisson. Además, en Fase 3 el LR
nunca bajó de 0.001 en 100 épocas (el scheduler no hizo plateau) — señal
de que el presupuesto de épocas pudo quedarse corto.

## Qué cambia respecto a Fase 3 (Set Transformer)

- **Cabeza:** activación `softplus` en la salida (`model_f_settransformer.py`)
  en vez de una salida lineal cruda. Garantiza `lambda >= 0` sin la
  inestabilidad de `exp` en logits grandes al inicio del entrenamiento.
- **Loss:** `ConstrainedPoissonLoss` (Poisson NLL por clase,
  `log_input=False` porque el input ya es `lambda`, más el mismo término
  de restricción de suma que `ConstrainedMSELoss`, `lambda_sum=0.5`) en
  vez de MSE.
- **Épocas:** 100 → 250. Scheduler sin cambios (`patience=8, factor=0.7`,
  regla dura del proyecto).
- El oráculo de post-proceso, el dataset, el split congelado y la
  arquitectura del Set Transformer (`d_model=64/n_heads=4/n_layers=2/n_seeds=1`)
  no cambian.

## Por qué Poisson y no clasificación ordinal

Ambas se mencionan en `WORKFLOW_V11` como alternativas a MSE. Poisson es
un cambio quirúrgico: no toca la forma de la salida (`(B, 19)`, no
`(B, 19, K)`), no toca el dataset, y el oráculo sigue funcionando sin
rediseño porque sigue recibiendo un valor continuo con parte fraccionaria
interpretable. Ordinal exigiría rediseñar la salida y el oráculo — se deja
como línea futura si Poisson no rinde.

## Caveat de comparabilidad

El val loss (Poisson NLL) no es comparable numéricamente contra el 0.0097
(MSE) de Fase 3 — son escalas distintas. La comparación real es por EMA
cruda/asistida y por el mapa de confusiones cruzadas (`evaluate.py`).

## Criterio de éxito / fracaso

- **EMA asistida:** ≥ 91.35% (Fase 3), apuntando a acercarse a 95%+.
- **EMA cruda:** mejora real y medible por sobre el techo histórico ~2-3%.
  Cualquier valor que se quede en ese rango es evidencia de que la cabeza
  Poisson no era el cuello de botella.
- **Confusiones:** `CH2`↔`CH2-N`, `C-2X` y `=CH/Ar`↔`Imina` deben bajar
  respecto a Fase 3.
- **Si falla:** ni la cabeza de salida ni el presupuesto de entrenamiento
  eran el cuello — apunta a un problema de información (HMBC simulado,
  fuera de alcance de este experimento) más que de modelado.

Spec completo: `docs/superpowers/specs/2026-07-22-exp-f-poisson-y-escalado-design.md`.
Plan: `docs/superpowers/plans/2026-07-22-exp-f-poisson-y-escalado.md`.
```

- [ ] **Step 5: Commit**

```bash
git add experiments/F_poisson_head/run_train.sh experiments/F_poisson_head/run_eval.sh experiments/F_poisson_head/README.md experiments/F_poisson_head/RATIONALE.md
git commit -m "exp-f: scripts SLURM + README + RATIONALE"
```

---

### Task 6: Subsampleo determinístico de train (`experiments/E3_dos_conjuntos/split_utils.py`, `train.py`)

Parte 2 (estudio de escalado). Agrega `subsample_train_idx` a `split_utils.py` de `E3_dos_conjuntos` (función pura, sin torch — se puede correr y verificar en esta máquina local ya mismo, a diferencia de todo lo demás en este plan) y la conecta en `train.py` vía un nuevo campo opcional de config `hyperparameters.train_fraction` (default `1.0`, no rompe los configs existentes de Fase 3). El val congelado nunca se toca — el subsampleo actúa solo sobre `train_idx`, después de calcularlo.

**Files:**
- Modify: `experiments/E3_dos_conjuntos/split_utils.py` — agregar `subsample_train_idx`
- Modify: `experiments/E3_dos_conjuntos/train.py` — usar `subsample_train_idx` en `build_frozen_split`
- Create: `experiments/E3_dos_conjuntos/tests/test_train_fraction.py`

**Interfaces:**
- Produces: `subsample_train_idx(train_idx: np.ndarray, fraction: float, seed: int = 42) -> np.ndarray`. Determinístico: mismo `seed` + mismo `train_idx` → mismos índices. Anidado: la fracción 0.10 es subconjunto de la 0.25, que es subconjunto de la 0.50, etc. (misma permutación, distinto corte). `fraction >= 1.0` devuelve `train_idx` sin alterar.
- Consumes en `train.py`: `cfg['hyperparameters'].get('train_fraction', 1.0)`.

- [ ] **Step 1: Write the failing test**

```python
# experiments/E3_dos_conjuntos/tests/test_train_fraction.py
# coding: ascii
"""Test puro (sin torch) del subsampleo deterministico de train_idx, para
el estudio de escalado de datos (Parte 2 del spec de Exp F). Corre local
ahora mismo -- no necesita torch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from split_utils import subsample_train_idx


def test_reproducible_same_seed():
    idx = np.arange(1000)
    a = subsample_train_idx(idx, 0.25, seed=42)
    b = subsample_train_idx(idx, 0.25, seed=42)
    assert np.array_equal(a, b)
    print("[OK] mismo seed -> mismos indices")


def test_fraction_size():
    idx = np.arange(1000)
    out = subsample_train_idx(idx, 0.25, seed=42)
    assert len(out) == 250, len(out)
    print(f"[OK] fraccion 0.25 de 1000 -> {len(out)} indices")


def test_fractions_are_nested():
    idx = np.arange(1000)
    f10 = set(subsample_train_idx(idx, 0.10, seed=42).tolist())
    f25 = set(subsample_train_idx(idx, 0.25, seed=42).tolist())
    f50 = set(subsample_train_idx(idx, 0.50, seed=42).tolist())
    assert f10.issubset(f25), "10% deberia ser subconjunto de 25%"
    assert f25.issubset(f50), "25% deberia ser subconjunto de 50%"
    print("[OK] fracciones anidadas (10% subset 25% subset 50%)")


def test_fraction_one_returns_all_unchanged():
    idx = np.arange(1000)
    out = subsample_train_idx(idx, 1.0, seed=42)
    assert np.array_equal(out, idx), "fraccion 1.0 no debe alterar train_idx"
    print("[OK] fraccion 1.0 devuelve train_idx intacto")


if __name__ == "__main__":
    test_reproducible_same_seed()
    test_fraction_size()
    test_fractions_are_nested()
    test_fraction_one_returns_all_unchanged()
    print("\n>>> TEST TRAIN_FRACTION OK <<<")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd experiments/E3_dos_conjuntos && python tests/test_train_fraction.py`
Expected: FAIL con `ImportError: cannot import name 'subsample_train_idx' from 'split_utils'`.

- [ ] **Step 3: Add subsample_train_idx to split_utils.py**

Modificar `experiments/E3_dos_conjuntos/split_utils.py` agregando esta función al final del archivo (después de `remove_leaking_from_train`, sin tocar las dos funciones existentes):

```python
def subsample_train_idx(train_idx, fraction, seed=42):
    """Subsamplea train_idx de forma deterministica y anidada, para el
    estudio de escalado de datos (Parte 2 del spec de Exp F): la
    permutacion es la misma para cualquier fraccion (mismo seed), asi que
    fraccion=0.25 es subconjunto de fraccion=0.50, etc. -- la curva de
    escalado mide una progresion genuinamente incremental, no muestras
    independientes entre si. fraction >= 1.0 devuelve train_idx sin tocar."""
    if fraction >= 1.0:
        return train_idx
    rng = np.random.RandomState(seed)
    perm = rng.permutation(train_idx)
    n_keep = int(len(train_idx) * fraction)
    return np.sort(perm[:n_keep])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd experiments/E3_dos_conjuntos && python tests/test_train_fraction.py`
Expected: PASS — `>>> TEST TRAIN_FRACTION OK <<<`. **Este test corre localmente ahora mismo (no requiere torch) — ejecutalo de verdad y confirmá el PASS antes de seguir.**

- [ ] **Step 5: Wire train_fraction into train.py**

En `experiments/E3_dos_conjuntos/train.py`:

1. Cambiar el import de `split_utils` (línea `from split_utils import canonicalize_smiles, remove_leaking_from_train`) a:

```python
from split_utils import canonicalize_smiles, remove_leaking_from_train, subsample_train_idx
```

2. Modificar `build_frozen_split` para que, después de calcular `train_idx` limpio, aplique el subsampleo si `train_fraction < 1.0`:

```python
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

    fraction = float(cfg['hyperparameters'].get('train_fraction', 1.0))
    if fraction < 1.0:
        train_idx = subsample_train_idx(train_idx, fraction, seed=42)

    print(f"[INFO] Split congelado: SMILES invalidos={n_invalid} | "
          f"train={len(train_idx)} (leak removido={n_removed}, train_fraction={fraction}) | val={len(val_idx)}")
    return train_idx, val_idx
```

(El resto de `train.py` no cambia: `hyperparameters.train_fraction` es opcional, `cfg['hyperparameters'].get('train_fraction', 1.0)` no rompe `config_deepsets.yaml` ni `config_settransformer.yaml`, que no tienen ese campo.)

- [ ] **Step 6: Commit**

```bash
git add experiments/E3_dos_conjuntos/split_utils.py experiments/E3_dos_conjuntos/train.py experiments/E3_dos_conjuntos/tests/test_train_fraction.py
git commit -m "exp-e-fase3: subsampleo deterministico de train (estudio de escalado, Parte 2 de Exp F)"
```

---

### Task 7: Configs de escalado + script SLURM + README (`experiments/E3_dos_conjuntos/`)

5 configs (10/25/50/75/100% del train) que reusan el Set Transformer y `ConstrainedMSELoss` de Fase 3 sin ningún cambio (deliberado — no usan Poisson, para no mezclar esta pregunta con la de la Parte 1). Un script SLURM parametrizado por `--config` (mismo patrón que `run_eval.sh` de Fase 3, que ya acepta el config como argumento). Checkpoints en carpetas separadas para no pisar el de Fase 3.

**Files:**
- Create: `experiments/E3_dos_conjuntos/config_scaling_10.yaml`
- Create: `experiments/E3_dos_conjuntos/config_scaling_25.yaml`
- Create: `experiments/E3_dos_conjuntos/config_scaling_50.yaml`
- Create: `experiments/E3_dos_conjuntos/config_scaling_75.yaml`
- Create: `experiments/E3_dos_conjuntos/config_scaling_100.yaml`
- Create: `experiments/E3_dos_conjuntos/run_train_scaling.sh`
- Modify: `experiments/E3_dos_conjuntos/README.md` — agregar sección del estudio de escalado al final

**Interfaces:**
- Consumes: `train.py` de Task 6 (lee `hyperparameters.train_fraction` de cada config), `evaluate.py`/`dump_predictions.py` de Fase 3 (sin cambios — ya aceptan `--config` como argumento).

- [ ] **Step 1: Write the 5 scaling configs**

```yaml
# experiments/E3_dos_conjuntos/config_scaling_10.yaml
experiment_name: "nmr_202k_e3_settransformer_scaling_10pct"

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  peaks_ch_filename: "peaks_pkl_202465.npz"
  peaks_13c_filename: "peaks_13c_202465.npz"
  labels_filename: "vectors_13c_19v_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_E3_scaling_10"
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
  train_fraction: 0.10
  scheduler:
    patience: 8
    factor: 0.7

system:
  device: "cuda"
  num_workers: 0
  pin_memory: true
```

```yaml
# experiments/E3_dos_conjuntos/config_scaling_25.yaml
experiment_name: "nmr_202k_e3_settransformer_scaling_25pct"

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  peaks_ch_filename: "peaks_pkl_202465.npz"
  peaks_13c_filename: "peaks_13c_202465.npz"
  labels_filename: "vectors_13c_19v_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_E3_scaling_25"
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
  train_fraction: 0.25
  scheduler:
    patience: 8
    factor: 0.7

system:
  device: "cuda"
  num_workers: 0
  pin_memory: true
```

```yaml
# experiments/E3_dos_conjuntos/config_scaling_50.yaml
experiment_name: "nmr_202k_e3_settransformer_scaling_50pct"

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  peaks_ch_filename: "peaks_pkl_202465.npz"
  peaks_13c_filename: "peaks_13c_202465.npz"
  labels_filename: "vectors_13c_19v_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_E3_scaling_50"
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
  train_fraction: 0.50
  scheduler:
    patience: 8
    factor: 0.7

system:
  device: "cuda"
  num_workers: 0
  pin_memory: true
```

```yaml
# experiments/E3_dos_conjuntos/config_scaling_75.yaml
experiment_name: "nmr_202k_e3_settransformer_scaling_75pct"

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  peaks_ch_filename: "peaks_pkl_202465.npz"
  peaks_13c_filename: "peaks_13c_202465.npz"
  labels_filename: "vectors_13c_19v_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_E3_scaling_75"
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
  train_fraction: 0.75
  scheduler:
    patience: 8
    factor: 0.7

system:
  device: "cuda"
  num_workers: 0
  pin_memory: true
```

```yaml
# experiments/E3_dos_conjuntos/config_scaling_100.yaml
experiment_name: "nmr_202k_e3_settransformer_scaling_100pct"

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  peaks_ch_filename: "peaks_pkl_202465.npz"
  peaks_13c_filename: "peaks_13c_202465.npz"
  labels_filename: "vectors_13c_19v_202465.npy"
  smiles_filename: "smiles_202465.npy"
  checkpoint_dir: "checkpoints_E3_scaling_100"
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
  train_fraction: 1.0
  scheduler:
    patience: 8
    factor: 0.7

system:
  device: "cuda"
  num_workers: 0
  pin_memory: true
```

- [ ] **Step 2: Write run_train_scaling.sh**

```bash
#!/bin/bash
#SBATCH --job-name=expE3_scaling
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE3_scaling_%j.out
#SBATCH --error=expE3_scaling_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/E3_dos_conjuntos

# Pasar el config de la fraccion a entrenar como argumento:
#   sbatch run_train_scaling.sh config_scaling_10.yaml
#   sbatch run_train_scaling.sh config_scaling_25.yaml
#   sbatch run_train_scaling.sh config_scaling_50.yaml
#   sbatch run_train_scaling.sh config_scaling_75.yaml
#   sbatch run_train_scaling.sh config_scaling_100.yaml
CONFIG="${1:?Falta el config, ej: sbatch run_train_scaling.sh config_scaling_10.yaml}"
python -u train.py --config "$CONFIG"
```

- [ ] **Step 3: Append scaling section to README.md**

Agregar al final de `experiments/E3_dos_conjuntos/README.md` (después de la sección "Nota" existente, sin modificar nada de lo que ya hay):

```markdown

## Estudio de escalado de datos (ablación, post Fase 3)

Curva EMA vs tamaño de train (10/25/50/75/100% de las ~187314 moléculas de
train de Exp D), mismo Set Transformer y misma loss MSE de Fase 3 — **sin
Poisson**, para no mezclar con Exp F (`experiments/F_poisson_head/`). El
val congelado (14428) es idéntico en las 5 corridas.

1. Confirmar que los 5 configs existen: `config_scaling_10.yaml` ...
   `config_scaling_100.yaml`.
2. Smoke test del subsampleo (corre local, sin torch — ejecutalo de
   verdad, no requiere el cluster):
   ```bash
   python tests/test_train_fraction.py
   ```
3. Lanzar las 5 corridas (no dependen entre sí, se pueden lanzar todas de una):
   ```bash
   sbatch run_train_scaling.sh config_scaling_10.yaml
   sbatch run_train_scaling.sh config_scaling_25.yaml
   sbatch run_train_scaling.sh config_scaling_50.yaml
   sbatch run_train_scaling.sh config_scaling_75.yaml
   sbatch run_train_scaling.sh config_scaling_100.yaml
   ```
4. Evaluar cada checkpoint (reusa `run_eval.sh` de Fase 3, que ya acepta
   el config como argumento):
   ```bash
   sbatch run_eval.sh config_scaling_10.yaml
   sbatch run_eval.sh config_scaling_25.yaml
   sbatch run_eval.sh config_scaling_50.yaml
   sbatch run_eval.sh config_scaling_75.yaml
   sbatch run_eval.sh config_scaling_100.yaml
   ```
5. Armar una tabla EMA cruda/asistida vs tamaño de train (10/25/50/75/100%)
   y agregarla a `docs/Runs/RESULTS.md`. Avisá a Claude Code con los 5
   números — si la pendiente entre 75% y 100% sigue siendo grande, vale la
   pena ampliar el dataset de nuevo; si se aplanó, no.
```

- [ ] **Step 4: Commit**

```bash
git add experiments/E3_dos_conjuntos/config_scaling_10.yaml experiments/E3_dos_conjuntos/config_scaling_25.yaml experiments/E3_dos_conjuntos/config_scaling_50.yaml experiments/E3_dos_conjuntos/config_scaling_75.yaml experiments/E3_dos_conjuntos/config_scaling_100.yaml experiments/E3_dos_conjuntos/run_train_scaling.sh experiments/E3_dos_conjuntos/README.md
git commit -m "exp-e-fase3: configs de escalado (10/25/50/75/100%) + script SLURM + README"
```

---

## Al terminar todas las tareas

Usar `superpowers:finishing-a-development-branch` para decidir cómo integrar la rama `exp-f-poisson-escalado` a `main` (mismo patrón que Fase 1b/2/3: merge fast-forward local, worktree removido, rama borrada). No hacer `sbatch` de nada — eso lo corre Lucas manualmente por SSH, siguiendo los README de `experiments/F_poisson_head/` y la sección nueva de `experiments/E3_dos_conjuntos/README.md`.
