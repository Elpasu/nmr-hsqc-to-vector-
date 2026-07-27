# Inferencia experimental con GUI (Exp H) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Una interfaz Streamlit local que predice el vector de 19 clases a partir de picos NMR experimentales leídos a mano + la fórmula molecular, reusando el modelo E3 congelado + oráculo v2 + Fase 1b.

**Architecture:** Lógica separada de la GUI. `adapter.py` (numpy puro, testeable) convierte FM + tabla de picos a los tensores exactos del modelo. `predict_core.py` (torch CPU) carga el checkpoint y corre el forward, y delega el post-proceso a `oraculo.py` (E3) + `generate_candidates_uncertainty` (G). `app_inferencia.py` (Streamlit) es I/O fino.

**Tech Stack:** Python, numpy, PyYAML (ya presentes); torch CPU + streamlit (setup local de Lucas). Reusa `experiments/E3_dos_conjuntos/{model_e3_settransformer.py, oraculo.py}` y `experiments/G_multivector/candidates.py`.

## Global Constraints

- **Regla 3 (nada hardcodeado):** nombres de clase salen de `config/db.yaml` (`classes_19v`); normalización sale de `experiments/E3_dos_conjuntos/config_settransformer.yaml` (`normalization`). No repetir esos valores en el código.
- **Regla 7 (orden de 19 clases fijo):** no reordenar; el orden es el de `db.yaml`. Por eso se **importa** el oráculo/candidatos en vez de copiarlos (evita una segunda copia que diverja).
- **Normalización exacta:** `δC → δC/220`, `δH → (δH+1)/16`, `amp_ch0 → amp_ch0/3.0`, `amp_ch1` tal cual, `peaks_13c: δC → δC/220`. Valores desde el config, no literales en el código.
- **Amplitudes (idéntico a `extract_peaks_pkl.py`):** `phase = -1.0 if mult==2 else 1.0`; `amp_ch0 = phase*mult`; `amp_ch1 = mult/3.0`. `Cq` (mult 0) → sin crosspeak.
- **Índice CH2 = clases [1,5,9,12]** (`total_CH2` = nº de filas `mult==CH2`).
- **Checkpoint:** Clementina XPU, `nmr_202k_e3_settransformer_2sets_19v_best.pth` (EMA v2 92.14%). Se carga con `map_location="cpu"`.
- **Encoding:** archivos en UTF-8, cabecera `# coding: utf-8`.
- **Entorno de ejecución del plan:** la PC de dev tiene numpy/rdkit/pandas/pyarrow/PyYAML pero **NO torch ni streamlit**. Solo la Task 1 (`adapter.py`, numpy puro) es testeable por el implementador. Las Tasks 2 y 3 se entregan como código completo + `python -m py_compile` (chequeo de sintaxis) + un bloque de **verificación manual que corre Lucas** tras el setup.

---

### Task 1: `adapter.py` — parser de FM + armado de tensores + vector verdadero (numpy puro, TDD)

**Files:**
- Create: `experiments/H_inferencia_experimental/adapter.py`
- Create: `experiments/H_inferencia_experimental/tests/test_adapter.py`
- Create: `experiments/H_inferencia_experimental/__init__.py` (vacío)
- Create: `experiments/H_inferencia_experimental/tests/__init__.py` (vacío)

**Interfaces:**
- Produces:
  - `MULT_H: dict` = `{"CH3":3, "CH2":2, "CH":1, "Cq":0}`
  - `parse_formula(formula: str) -> dict` con claves `C,H,N,O,S,Hal` (int; ausentes = 0).
  - `build_inputs(peaks: list[dict], formula: dict, norm_cfg: dict) -> tuple(peaks_ch, mask_ch, peaks_13c, mask_13c, cond)` — todos `np.ndarray` float32. Cada `peak` = `{"delta_c": float, "delta_h": float|None, "mult": str}`. `peaks_ch` shape `(n_ch,4)`, `mask_ch` `(n_ch,)` (todo 1), `peaks_13c` `(n_13c,1)`, `mask_13c` `(n_13c,)` (todo 1), `cond` `(8,)` = `[total_señales, total_CH2, C,H,N,O,S,Hal]`.
  - `true_vector(peaks: list[dict], class_names: list[str]) -> np.ndarray` shape `(19,)` int — histograma de `peak["clase"]`.
  - `load_configs(repo_root: str) -> tuple(class_names: list[str], norm_cfg: dict)` — lee `config/db.yaml` (`classes_19v`) y `config_settransformer.yaml` (`normalization`). (No unit-testeado; I/O.)

- [ ] **Step 1: Escribir el test que falla — `parse_formula`**

Crear `experiments/H_inferencia_experimental/tests/test_adapter.py`:

```python
# coding: utf-8
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from adapter import MULT_H, build_inputs, parse_formula, true_vector  # noqa: E402

CLASS_NAMES = [
    "CH3", "CH2", "CH", "Cq", "CH3-O", "CH2-O", "CH-O", "Cq-O", "CH3-N",
    "CH2-N", "CH-N", "Cq-N", "=CH2", "=CH/Ar", "Cqsp2", "Aldeh", "Imina",
    "C-2X", "C-3X",
]
NORM = {"c13_ppm_min": 0, "c13_ppm_max": 220, "h1_ppm_min": -1,
        "h1_ppm_max": 15, "amp_ch0_scale": 3.0}


def test_parse_formula_chon():
    f = parse_formula("C10H12N2O")
    assert f == {"C": 10, "H": 12, "N": 2, "O": 1, "S": 0, "Hal": 0}


def test_parse_formula_implicit_one_and_missing():
    f = parse_formula("CH4")
    assert f["C"] == 1 and f["H"] == 4 and f["N"] == 0 and f["O"] == 0


def test_parse_formula_halogens_sum():
    f = parse_formula("C2H4ClBr")
    assert f["Hal"] == 2 and f["C"] == 2


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n = 0
    for fn in fns:
        try:
            fn(); n += 1
        except Exception:
            print(f"FALLO: {fn.__name__}"); traceback.print_exc(); sys.exit(1)
    print(f">>> {n} TESTS OK <<<")
```

- [ ] **Step 2: Correr el test — debe fallar**

Run: `python experiments/H_inferencia_experimental/tests/test_adapter.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'adapter'`.

- [ ] **Step 3: Implementar `parse_formula` + constantes en `adapter.py`**

Crear `experiments/H_inferencia_experimental/adapter.py`:

```python
# coding: utf-8
"""Exp H -- adaptador de picos experimentales al formato de entrada del E3.

numpy puro (sin torch, sin GUI): testeable local. Convierte FM + tabla de picos
a los tensores exactos que consume model_e3_settransformer, con la MISMA
normalizacion y las MISMAS amplitudes que extract_peaks_pkl.py (Fase 1b).
"""
import os
import re

import numpy as np
import yaml

MULT_H = {"CH3": 3, "CH2": 2, "CH": 1, "Cq": 0}
CH2_CLASS_IDX = [1, 5, 9, 12]
_HALOGENS = {"F", "Cl", "Br", "I"}


def parse_formula(formula):
    """'C10H12N2O' -> {'C':10,'H':12,'N':2,'O':1,'S':0,'Hal':0}. Elementos
    ausentes = 0; digito implicito = 1. Hal = F+Cl+Br+I."""
    counts = {"C": 0, "H": 0, "N": 0, "O": 0, "S": 0, "Hal": 0}
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        if elem == "":
            continue
        n = int(num) if num else 1
        if elem in _HALOGENS:
            counts["Hal"] += n
        elif elem in counts:
            counts[elem] += n
    return counts
```

- [ ] **Step 4: Correr el test — debe pasar**

Run: `python experiments/H_inferencia_experimental/tests/test_adapter.py`
Expected: `>>> 3 TESTS OK <<<`

- [ ] **Step 5: Escribir tests que fallan — `build_inputs` y `true_vector`**

Agregar a `test_adapter.py` (antes del bloque `__main__`):

```python
def _mol_etanol():
    # CH3-CH2-OH: CH3 (d 1.2), CH2-O (d 3.7), 2 carbonos protonados, sin Cq.
    return [
        {"delta_c": 18.0, "delta_h": 1.2, "mult": "CH3", "clase": "CH3"},
        {"delta_c": 58.0, "delta_h": 3.7, "mult": "CH2", "clase": "CH2-O"},
    ]


def test_amp_ch2_is_negative_two():
    peaks = [{"delta_c": 58.0, "delta_h": 3.7, "mult": "CH2"}]
    peaks_ch, _, _, _, _ = build_inputs(peaks, parse_formula("C2H6O"), NORM)
    # amp_ch0 crudo = -2, normalizado /3.0 -> -0.6667
    assert np.isclose(peaks_ch[0, 2], -2.0 / 3.0, atol=1e-4)
    assert np.isclose(peaks_ch[0, 3], 2.0 / 3.0, atol=1e-4)  # amp_ch1 = mult/3


def test_amp_ch3_is_plus_three():
    peaks = [{"delta_c": 18.0, "delta_h": 1.2, "mult": "CH3"}]
    peaks_ch, _, _, _, _ = build_inputs(peaks, parse_formula("C2H6O"), NORM)
    assert np.isclose(peaks_ch[0, 2], 3.0 / 3.0, atol=1e-4)   # +3 /3 = 1.0
    assert np.isclose(peaks_ch[0, 3], 3.0 / 3.0, atol=1e-4)


def test_normalization_matches_config():
    peaks = [{"delta_c": 110.0, "delta_h": 7.0, "mult": "CH"}]
    peaks_ch, _, _, _, _ = build_inputs(peaks, parse_formula("C6H6"), NORM)
    assert np.isclose(peaks_ch[0, 0], 110.0 / 220.0, atol=1e-4)   # dC/220
    assert np.isclose(peaks_ch[0, 1], (7.0 + 1.0) / 16.0, atol=1e-4)  # (dH+1)/16


def test_cq_no_crosspeak_but_in_13c():
    peaks = [
        {"delta_c": 40.0, "delta_h": 1.5, "mult": "CH2"},
        {"delta_c": 150.0, "delta_h": None, "mult": "Cq"},
    ]
    peaks_ch, mask_ch, peaks_13c, mask_13c, _ = build_inputs(
        peaks, parse_formula("C3H6"), NORM)
    assert peaks_ch.shape[0] == 1          # solo el CH2 genera crosspeak
    assert peaks_13c.shape[0] == 2         # ambos carbonos en 13C
    assert mask_ch.sum() == 1 and mask_13c.sum() == 2


def test_cond_derived_from_spectrum_and_formula():
    _, _, _, _, cond = build_inputs(_mol_etanol(), parse_formula("C2H6O"), NORM)
    # [total_senales, total_CH2, C,H,N,O,S,Hal]
    assert cond[0] == 2      # total senales = nro filas
    assert cond[1] == 1      # un CH2 (el CH2-O del etanol) -> mult==CH2
    assert cond[2] == 2 and cond[4] == 0 and cond[5] == 1   # C=2, N=0, O=1


def test_true_vector_histogram():
    tv = true_vector(_mol_etanol(), CLASS_NAMES)
    assert tv.sum() == 2
    assert tv[0] == 1        # CH3 en indice 0
    assert tv[5] == 1        # CH2-O en indice 5
```

> Nota para el implementador: el test `test_cond_derived_from_spectrum_and_formula` tiene a propósito una aserción rota (`assert False`) para que la corrijas al implementar: el etanol tiene **un** carbono con `mult==CH2` (el CH2-O), así que `cond[1]` (total_CH2) debe ser **1**. Reemplazá las dos líneas de `cond[1]` por `assert cond[1] == 1` y borrá el `assert False`.

- [ ] **Step 6: Correr — debe fallar**

Run: `python experiments/H_inferencia_experimental/tests/test_adapter.py`
Expected: FAIL con `ImportError` (o `NameError`) porque `build_inputs`/`true_vector` aún no existen.

- [ ] **Step 7: Corregir la aserción marcada e implementar `build_inputs` + `true_vector`**

En `test_adapter.py`, dentro de `test_cond_derived_from_spectrum_and_formula`, reemplazar las dos líneas del comentario y el `assert False` por:

```python
    assert cond[1] == 1      # un CH2 (el CH2-O del etanol)
```

Agregar a `adapter.py`:

```python
def build_inputs(peaks, formula, norm_cfg):
    """peaks: lista de {delta_c, delta_h|None, mult}. formula: dict de parse_formula.
    Devuelve (peaks_ch, mask_ch, peaks_13c, mask_13c, cond), todos np.float32.
    Sin padding (batch=1): mascaras todo-1 sobre los picos reales."""
    c_min, c_max = float(norm_cfg["c13_ppm_min"]), float(norm_cfg["c13_ppm_max"])
    h_min, h_max = float(norm_cfg["h1_ppm_min"]), float(norm_cfg["h1_ppm_max"])
    amp0_scale = float(norm_cfg["amp_ch0_scale"])

    ch_rows, c13_rows = [], []
    total_ch2 = 0
    for p in peaks:
        mult = MULT_H[p["mult"]]
        dc = float(p["delta_c"])
        c13_rows.append([(dc - c_min) / (c_max - c_min)])
        if mult == 0:
            continue                       # Cq: sin crosspeak
        if p["mult"] == "CH2":
            total_ch2 += 1
        phase = -1.0 if mult == 2 else 1.0
        amp_ch0 = phase * mult
        amp_ch1 = mult / 3.0
        dh = float(p["delta_h"])
        ch_rows.append([
            (dc - c_min) / (c_max - c_min),
            (dh - h_min) / (h_max - h_min),
            amp_ch0 / amp0_scale,
            amp_ch1,
        ])

    peaks_ch = np.asarray(ch_rows, dtype=np.float32).reshape(-1, 4)
    peaks_13c = np.asarray(c13_rows, dtype=np.float32).reshape(-1, 1)
    mask_ch = np.ones(peaks_ch.shape[0], dtype=np.float32)
    mask_13c = np.ones(peaks_13c.shape[0], dtype=np.float32)

    cond = np.array([
        float(len(peaks)),          # total_senales = nro de picos 13C
        float(total_ch2),           # total_CH2
        formula["C"], formula["H"], formula["N"],
        formula["O"], formula["S"], formula["Hal"],
    ], dtype=np.float32)
    return peaks_ch, mask_ch, peaks_13c, mask_13c, cond


def true_vector(peaks, class_names):
    """Histograma de peak['clase'] sobre las 19 clases (solo modo evaluacion)."""
    idx = {name: i for i, name in enumerate(class_names)}
    vec = np.zeros(len(class_names), dtype=int)
    for p in peaks:
        if "clase" not in p or p["clase"] in (None, ""):
            raise ValueError("true_vector requiere 'clase' en toda fila")
        vec[idx[p["clase"]]] += 1
    return vec


def load_configs(repo_root):
    """Lee nombres de clase (db.yaml) y normalizacion (config del E3)."""
    with open(os.path.join(repo_root, "config", "db.yaml"), "r",
              encoding="utf-8") as f:
        class_names = yaml.safe_load(f)["classes_19v"]
    cfg_path = os.path.join(repo_root, "experiments", "E3_dos_conjuntos",
                            "config_settransformer.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        norm_cfg = yaml.safe_load(f)["normalization"]
    return class_names, norm_cfg
```

- [ ] **Step 8: Correr — todo verde**

Run: `python experiments/H_inferencia_experimental/tests/test_adapter.py`
Expected: `>>> 8 TESTS OK <<<`

- [ ] **Step 9: Commit**

```bash
git add experiments/H_inferencia_experimental/adapter.py experiments/H_inferencia_experimental/__init__.py experiments/H_inferencia_experimental/tests/
git commit -m "exp H task 1: adapter.py (FM+picos -> tensores E3) + tests numpy"
```

---

### Task 2: `predict_core.py` — carga del modelo, forward y candidatos (torch CPU)

**Files:**
- Create: `experiments/H_inferencia_experimental/predict_core.py`

**Interfaces:**
- Consumes: `build_inputs` (Task 1); `NMR_SetTransformer` de `E3_dos_conjuntos/model_e3_settransformer.py`; `ajustar_conteo_hetero` de `E3_dos_conjuntos/oraculo.py`; `generate_candidates_uncertainty` de `G_multivector/candidates.py`.
- Produces:
  - `load_model(checkpoint_path: str, model_cfg: dict) -> torch.nn.Module` (en `eval()`, CPU).
  - `predict_raw(model, inputs: tuple) -> np.ndarray` shape `(19,)` float — salida cruda (pre-redondeo). `inputs` = la 5-tupla de `build_inputs`.
  - `candidatos(raw, formula: dict, total: int, ch2: int, tau: float, k_max: int) -> list[np.ndarray]` — `[0]` es el ancla v2; el resto, alternativas Fase 1b. Todos `(19,)` int.

> **Nota de entorno:** esta task usa torch, que **no está instalado en la PC de dev**. El implementador escribe el código y corre `python -m py_compile` (chequeo de sintaxis). La verificación funcional la hace Lucas (Step 4).

- [ ] **Step 1: Escribir `predict_core.py`**

```python
# coding: utf-8
"""Exp H -- forward local (torch CPU) del E3 congelado + post-proceso.

Carga el checkpoint Clementina XPU con map_location='cpu'. El post-proceso
(oraculo v2 + Fase 1b) se IMPORTA de E3/G para no duplicar el oraculo (regla 7).
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_E3 = os.path.abspath(os.path.join(_HERE, "..", "E3_dos_conjuntos"))
_G = os.path.abspath(os.path.join(_HERE, "..", "G_multivector"))
for _p in (_E3, _G):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_e3_settransformer import NMR_SetTransformer  # noqa: E402
from oraculo import ajustar_conteo_hetero  # noqa: E402
from candidates import generate_candidates_uncertainty  # noqa: E402


def load_model(checkpoint_path, model_cfg):
    """Instancia NMR_SetTransformer con los hiperparametros del config y carga
    el state_dict en CPU. Devuelve el modelo en eval()."""
    model = NMR_SetTransformer(
        num_classes=19,
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        n_seeds=int(model_cfg["n_seeds"]),
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_raw(model, inputs):
    """inputs = (peaks_ch, mask_ch, peaks_13c, mask_13c, cond) de build_inputs
    (numpy). Agrega dimension de batch, corre el forward y devuelve el crudo (19,)."""
    peaks_ch, mask_ch, peaks_13c, mask_13c, cond = inputs

    def _b(a):
        return torch.tensor(a, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out = model(_b(peaks_ch), _b(mask_ch), _b(peaks_13c), _b(mask_13c), _b(cond))
    return out.squeeze(0).cpu().numpy().astype(np.float64)


def candidatos(raw, formula, total, ch2, tau, k_max):
    """Ancla v2 + alternativas Fase 1b. [0] = oraculo v2. FM-consistentes."""
    return generate_candidates_uncertainty(
        raw, int(total), int(ch2), int(formula["N"]), int(formula["O"]),
        float(tau), int(k_max),
    )
```

- [ ] **Step 2: Chequeo de sintaxis (lo que sí corre sin torch)**

Run: `python -m py_compile experiments/H_inferencia_experimental/predict_core.py`
Expected: sin salida (exit 0). Si torch no está instalado, NO importar el módulo (solo compilar).

- [ ] **Step 3: Commit**

```bash
git add experiments/H_inferencia_experimental/predict_core.py
git commit -m "exp H task 2: predict_core.py (forward torch CPU + oraculo v2 + Fase 1b)"
```

- [ ] **Step 4: Verificación manual (la corre Lucas tras el setup)**

Requiere `pip install torch` (CPU) y el checkpoint `..._best.pth` en local. Correr en un intérprete:

```python
import numpy as np, yaml, os
from experiments.H_inferencia_experimental import adapter, predict_core
cfg = yaml.safe_load(open("experiments/E3_dos_conjuntos/config_settransformer.yaml", encoding="utf-8"))
class_names, norm = adapter.load_configs(".")
peaks = [{"delta_c":18.0,"delta_h":1.2,"mult":"CH3"},
         {"delta_c":58.0,"delta_h":3.7,"mult":"CH2"}]
fm = adapter.parse_formula("C2H6O")
inp = adapter.build_inputs(peaks, fm, norm)
model = predict_core.load_model("RUTA/AL/checkpoint_best.pth", cfg["model"])
raw = predict_core.predict_raw(model, inp)
cands = predict_core.candidatos(raw, fm, total=inp[4][0], ch2=inp[4][1], tau=1.5, k_max=6)
print("raw:", np.round(raw,2)); print("ancla v2:", cands[0]); print("K:", len(cands))
```
Expected: `raw` shape `(19,)`; `cands[0]` (ancla) suma == total; `len(cands) >= 1`.

---

### Task 3: `app_inferencia.py` — interfaz Streamlit

**Files:**
- Create: `experiments/H_inferencia_experimental/app_inferencia.py`

**Interfaces:**
- Consumes: `adapter` (Task 1), `predict_core` (Task 2).

> **Nota de entorno:** usa streamlit + torch (no instalados en la PC de dev). El implementador escribe el código y corre `python -m py_compile`. La verificación es el lanzamiento que hace Lucas (Step 3).

- [ ] **Step 1: Escribir `app_inferencia.py`**

```python
# coding: utf-8
"""Exp H -- interfaz Streamlit local para predecir el vector desde picos
experimentales + FM. Corre en TU PC (no en el cluster).

Requisitos:  pip install streamlit torch pandas pyyaml numpy
Uso:         streamlit run experiments/H_inferencia_experimental/app_inferencia.py
Ajusta CHECKPOINT_PATH a la ruta local del checkpoint (scp desde Clementina).
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import adapter  # noqa: E402
import predict_core  # noqa: E402

# Ruta al checkpoint local (scp desde Clementina). Portable: hardcodeada si
# existe, si no relativa al repo.
_HARDCODED = r"E:\Proyectos\SciTrix\nmr-hsqc-to-vector\checkpoints_local\nmr_202k_e3_settransformer_2sets_19v_best.pth"
_REL = os.path.join(_REPO, "checkpoints_local",
                    "nmr_202k_e3_settransformer_2sets_19v_best.pth")
CHECKPOINT_PATH = _HARDCODED if os.path.exists(_HARDCODED) else _REL

st.set_page_config(page_title="NMR Inferencia experimental", layout="wide")


@st.cache_resource
def _load():
    class_names, norm = adapter.load_configs(_REPO)
    cfg = yaml.safe_load(open(
        os.path.join(_REPO, "experiments", "E3_dos_conjuntos",
                     "config_settransformer.yaml"), encoding="utf-8"))
    model = predict_core.load_model(CHECKPOINT_PATH, cfg["model"])
    return class_names, norm, model


st.title("NMR HSQC -> vector: inferencia sobre datos experimentales")

if not os.path.exists(CHECKPOINT_PATH):
    st.error(f"No encuentro el checkpoint en:\n{CHECKPOINT_PATH}\n"
             "Copialo desde Clementina (ver README).")
    st.stop()

class_names, norm, model = _load()

col1, col2 = st.columns([1, 1])
with col1:
    formula = st.text_input("Fórmula molecular (ej. C10H12N2O)", value="C2H6O")
with col2:
    tau = st.slider("τ (Fase 1b)", 0.0, 3.0, 1.5, 0.25)
    k_max = st.slider("K_max", 1, 10, 6, 1)

st.caption("Una fila por carbono. δH vacío si es Cq. 'clase' es opcional "
           "(solo para evaluar moléculas conocidas).")
plantilla = pd.DataFrame([
    {"delta_c": 18.0, "delta_h": 1.2, "mult": "CH3", "clase": "CH3"},
    {"delta_c": 58.0, "delta_h": 3.7, "mult": "CH2", "clase": "CH2-O"},
])
edited = st.data_editor(
    plantilla, num_rows="dynamic", use_container_width=True,
    column_config={
        "mult": st.column_config.SelectboxColumn(
            "mult", options=list(adapter.MULT_H.keys()), required=True),
        "clase": st.column_config.SelectboxColumn(
            "clase", options=[""] + list(class_names)),
    },
)

if st.button("Predecir", type="primary"):
    rows = edited.to_dict("records")
    peaks = []
    for r in rows:
        if r.get("delta_c") is None or (isinstance(r.get("delta_c"), float)
                                        and np.isnan(r["delta_c"])):
            continue
        dh = r.get("delta_h")
        dh = None if (dh is None or (isinstance(dh, float) and np.isnan(dh))) else float(dh)
        peaks.append({"delta_c": float(r["delta_c"]), "delta_h": dh,
                      "mult": r["mult"], "clase": r.get("clase") or None})

    fm = adapter.parse_formula(formula)
    inp = adapter.build_inputs(peaks, fm, norm)
    raw = predict_core.predict_raw(model, inp)
    total, ch2 = int(inp[4][0]), int(inp[4][1])
    cands = predict_core.candidatos(raw, fm, total, ch2, tau, k_max)

    st.subheader(f"Candidatos emitidos: K = {len(cands)}")
    df = pd.DataFrame({class_names[i]: [c[i] for c in cands]
                       for i in range(19)},
                      index=[f"cand {j}" + (" (ancla v2)" if j == 0 else "")
                             for j in range(len(cands))])
    st.dataframe(df.T, use_container_width=True)
    st.caption(f"crudo (redondeado): {list(np.round(raw, 2))}")

    have_clase = all(p["clase"] for p in peaks)
    if have_clase:
        yt = adapter.true_vector(peaks, class_names)
        cubierto = any(np.array_equal(yt, c) for c in cands)
        st.success("y_true CUBIERTO en K ✅") if cubierto else st.warning(
            "y_true NO cubierto en K ❌")
        diff = yt - cands[0]
        confus = [(class_names[i], int(diff[i])) for i in range(19) if diff[i] != 0]
        if confus:
            st.write("Diferencia ancla v2 vs verdadero (qué confunde):", confus)
    else:
        st.info("Sin columna 'clase' -> modo predicción real (sin evaluación).")
```

- [ ] **Step 2: Chequeo de sintaxis**

Run: `python -m py_compile experiments/H_inferencia_experimental/app_inferencia.py`
Expected: sin salida (exit 0).

- [ ] **Step 3: Verificación manual (la corre Lucas) + Commit**

Lucas: `streamlit run experiments/H_inferencia_experimental/app_inferencia.py`, carga el etanol de la plantilla, aprieta Predecir, confirma que aparece el/los vector(es) candidato(s) y el estado de cobertura.

```bash
git add experiments/H_inferencia_experimental/app_inferencia.py
git commit -m "exp H task 3: app_inferencia.py (Streamlit: FM + tabla de picos -> vectores)"
```

---

### Task 4: `README.md` + sanity de extremo a extremo

**Files:**
- Create: `experiments/H_inferencia_experimental/README.md`

- [ ] **Step 1: Escribir el README**

Crear `experiments/H_inferencia_experimental/README.md` con:
- **Qué es:** inferencia del vector sobre datos experimentales, interfaz local. Modelo E3 congelado (EMA v2 92.14%, cobertura Fase 1b ~97%).
- **Setup (una vez):**
  - `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - `scp` del checkpoint desde Clementina:
    `/data/contrib/pci_78/Lucas/DB_202K/checkpoints_E3_settransformer/nmr_202k_e3_settransformer_2sets_19v_best.pth`
    → `checkpoints_local/` en el repo (o ajustar `CHECKPOINT_PATH` en `app_inferencia.py`).
- **Uso:** `streamlit run experiments/H_inferencia_experimental/app_inferencia.py`. Formato de la tabla (una fila por carbono; `mult` real del espectro editado; `clase` opcional solo para evaluar).
- **Distinción entrada/eval:** `mult` = entrada real (del HSQC editado/DEPT); `clase` = solo evaluación (es lo que la red predice). FM reemplaza al SMILES (que es la respuesta buscada).
- **Tests:** `python experiments/H_inferencia_experimental/tests/test_adapter.py`.
- **Sanity de extremo a extremo:** tomar una molécula del val congelado del parquet (`docs/Runs/E3_settransformer/predictions_...parquet`, columnas `crosspeaks`, `c13_shifts`, `y_pred_assisted_v2`), cargar sus picos en la GUI y verificar que el vector predicho reproduce `y_pred_assisted_v2` de esa fila → prueba que el adaptador es fiel al formato de entrenamiento.

- [ ] **Step 2: Commit**

```bash
git add experiments/H_inferencia_experimental/README.md
git commit -m "exp H task 4: README (setup, uso, sanity end-to-end)"
```

---

## Notas de cierre (fuera del alcance de este plan)

- **Parte (b) — acople al generador:** `predict.py` fino `predict_vectors(picos, formula, tau, K_max) -> [vectores]` (reusa `adapter` + `predict_core`, sin GUI). Tarea siguiente.
- **Peak-picking automático** desde espectros crudos (hoy la entrada es la tabla leída a mano).
- **Portar `Gen_vector.py` local** para auto-computar `y_true` (hoy sale de la columna `clase`).
