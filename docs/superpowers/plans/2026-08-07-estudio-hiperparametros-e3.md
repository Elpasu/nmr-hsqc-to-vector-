# Exp I — Estudio de hiperparámetros del Set Transformer (E3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar listos 23 configs de entrenamiento, un `.sh` de SLURM, un recolector de resultados,
una figura y sus tests, para que Lucas pueda correr con `sbatch` un estudio de hiperparámetros
controlado del Set Transformer y justificar ante un revisor la arquitectura elegida.

**Architecture:** Todo vive dentro de `experiments/E3_dos_conjuntos/` (subcarpeta `hp_sweep/`), a
propósito: el estudio solo vale si el código de entrenamiento es idéntico al que produjo el
checkpoint congelado. Un YAML declarativo (`sweep_grid.yaml`) describe el diseño; un generador
(`make_configs.py`) lo expande a 23 configs commiteados; un `.sh` parametrizado corre train+eval en
un solo job; un recolector parsea los `.out` a tabla markdown + CSV.

**Tech Stack:** Python 3, PyYAML, numpy, torch (solo para el smoke test de forward), matplotlib
(solo para la figura), SLURM (fuera de alcance de la ejecución, es de Lucas).

**Spec:** `docs/superpowers/specs/2026-08-07-estudio-hiperparametros-e3-design.md` (commit `95a915b`).

## Global Constraints

- **Regla dura 1:** `num_workers: 0` en todos los configs. Nunca subirlo.
- **Regla dura 2:** SLURM usa `#SBATCH --gres=gpu:1`, **NO** `--gpus=1`.
- **Regla dura 3:** nada hardcodeado — rutas y constantes salen de los configs.
- **Regla dura 4:** encoding UTF-8; los `.py` de este repo abren cabecera `# coding: ascii` y usan
  texto sin tildes en comentarios/código (seguir el estilo de los archivos vecinos de
  `experiments/E3_dos_conjuntos/`). Los `.md` sí llevan tildes.
- **Regla dura 5:** smoke test local antes de cualquier `sbatch`. La Task 3 es exactamente eso.
- **Regla dura 6:** scheduler `patience=8, factor=0.7` — invariante en las 23 corridas.
- **Regla dura 7:** `num_classes=19` y el orden de clases de `db.yaml` — no se tocan.
- **Regla dura 8:** val congelado (`val_indices_frozen.npy`) y `epochs: 100` idénticos en las 23, o
  las EMAs no son comparables.
- **Cambios de código aditivos con default idéntico al actual.** El `state_dict` del checkpoint
  congelado debe seguir cargando sin cambios.
- **Claude Code no lanza SLURM ni lee logs del cluster.** El entregable es "listo para `sbatch`".
- **Entorno local disponible:** torch 2.13.0+cpu, numpy, PyYAML, rdkit, streamlit. Los tests de las
  Tasks 1, 2, 3 y 5 se ejecutan localmente de verdad. Task 6 requiere matplotlib (verificar con
  `python -c "import matplotlib"`; si falta, `pip install matplotlib`).
- **Convención de tests del repo:** los tests NO usan pytest. Son scripts con funciones `test_*` y un
  bloque `if __name__ == "__main__":` que las corre en orden e imprime `>>> ... OK <<<`. Se ejecutan
  con `python tests/test_x.py`. Seguir esa convención exactamente.

---

## Desviación respecto del spec (aprobada al escribir el plan)

El spec §3.3 dice 2 corridas de piso de ruido (seeds 43 y 44), reutilizando la corrida histórica de
seed 42 (A10, Exp E Fase 3, 91.35 %) como tercera réplica. **El plan usa 3 corridas (seeds 42, 43,
44), total 23 en vez de 22.**

Motivo: la Task 1 modifica `train.py` y `model_e3_settransformer.py`. Los cambios son aditivos y
preservan el comportamiento por default, pero la corrida histórica se hizo con la versión anterior
del código y hace tres semanas. Poder decirle a un revisor *"tres réplicas, mismo código, mismo
cluster, misma semana"* es materialmente más fuerte que *"dos nuevas más una histórica"*, y cuesta 39
minutos de GPU. La Task 7 actualiza el spec para reflejarlo.

Total final: **16 OFAT + 4 grid 2D + 3 réplicas = 23 corridas.**

---

## File Structure

**Se modifican (Task 1):**

| Archivo | Cambio |
|---|---|
| `experiments/E3_dos_conjuntos/model_e3_settransformer.py` | `NMR_SetTransformer` acepta `fusion_hidden=(128, 64)` |
| `experiments/E3_dos_conjuntos/train.py` | seed sale del config; `build_model` propaga `fusion_hidden` |

**Se crean:**

| Archivo | Responsabilidad | Task |
|---|---|---|
| `experiments/E3_dos_conjuntos/tests/test_hp_config_knobs.py` | Verifica que los dos knobs nuevos funcionan y que los defaults no cambiaron | 1 |
| `experiments/E3_dos_conjuntos/hp_sweep/sweep_grid.yaml` | El diseño del estudio, declarativo. Fuente única. | 2 |
| `experiments/E3_dos_conjuntos/hp_sweep/make_configs.py` | Expande `sweep_grid.yaml` a los 23 configs | 2 |
| `experiments/E3_dos_conjuntos/hp_sweep/configs/*.yaml` | Los 23 configs generados (commiteados) | 2 |
| `experiments/E3_dos_conjuntos/hp_sweep/tests/test_make_configs.py` | Invariantes del diseño experimental | 2 |
| `experiments/E3_dos_conjuntos/hp_sweep/tests/test_all_archs_forward.py` | 1 forward en CPU por cada uno de los 23 configs | 3 |
| `experiments/E3_dos_conjuntos/run_sweep.sh` | 1 job SLURM = train + eval de un config | 4 |
| `experiments/E3_dos_conjuntos/hp_sweep/README.md` | Cómo correr el estudio, paso a paso | 4 |
| `experiments/E3_dos_conjuntos/hp_sweep/collect_results.py` | Parsea los `.out` → CSV + tabla markdown | 5 |
| `experiments/E3_dos_conjuntos/hp_sweep/tests/test_collect_results.py` | Parsing contra un `.out` sintético | 5 |
| `experiments/E3_dos_conjuntos/hp_sweep/make_plot.py` | Figura OFAT con banda de ruido | 6 |
| `experiments/E3_dos_conjuntos/hp_sweep/tests/test_make_plot.py` | La figura se genera desde un CSV sintético | 6 |
| `docs/Runs/RESULTS.md` | Sección placeholder del Exp I (se modifica) | 7 |

---

## Task 1: Hacer configurables `seed` y `fusion_hidden`

**Files:**
- Modify: `experiments/E3_dos_conjuntos/model_e3_settransformer.py:71-83`
- Modify: `experiments/E3_dos_conjuntos/train.py:46-61` (`build_model`) y `train.py:104-107` (`train`)
- Test: `experiments/E3_dos_conjuntos/tests/test_hp_config_knobs.py`

**Interfaces:**
- Consumes: nada de tareas anteriores.
- Produces:
  - `NMR_SetTransformer(num_classes=19, d_model=64, n_heads=4, n_layers=2, n_seeds=1, fusion_hidden=(128, 64))`
  - `build_model(cfg, num_classes=19)` lee `cfg['model']['fusion_hidden']` (default `(128, 64)`).
  - `train.py` lee `cfg['hyperparameters']['seed']` (default `42`).

- [ ] **Step 1: Escribir el test que falla**

Crear `experiments/E3_dos_conjuntos/tests/test_hp_config_knobs.py`:

```python
# coding: ascii
"""Verifica los dos knobs que el estudio de hiperparametros (Exp I) necesita:
fusion_hidden parametrizable en el modelo y seed configurable en train.py.
Ambos son ADITIVOS: el default debe reproducir exactamente el comportamiento
historico, o el checkpoint congelado deja de cargar."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model_e3_settransformer import NMR_SetTransformer
from train import build_model

N_CLASSES, MAX_CH, MAX_13C = 19, 32, 40


def _forward(model, B=2):
    with torch.no_grad():
        return model(torch.randn(B, MAX_CH, 4), torch.ones(B, MAX_CH),
                     torch.randn(B, MAX_13C, 1), torch.ones(B, MAX_13C),
                     torch.randn(B, 8))


def test_default_fusion_shapes_unchanged():
    """El default (128, 64) tiene que dar EXACTAMENTE las dimensiones
    historicas, o el state_dict del checkpoint congelado no carga."""
    m = NMR_SetTransformer(num_classes=N_CLASSES)
    assert m.fc_fusion1.out_features == 128, m.fc_fusion1.out_features
    assert m.fc_fusion2.in_features == 128, m.fc_fusion2.in_features
    assert m.fc_fusion2.out_features == 64, m.fc_fusion2.out_features
    assert m.fc_out.in_features == 64, m.fc_out.in_features
    print("[OK] default fusion_hidden = (128, 64), dimensiones historicas")


def test_default_param_count_unchanged():
    """70,163 parametros: el numero exacto que RESULTS.md atribuye al
    checkpoint congelado. Si esto cambia, el cambio NO fue aditivo."""
    n = sum(p.numel() for p in NMR_SetTransformer(num_classes=N_CLASSES).parameters())
    assert n == 70163, n
    print(f"[OK] parametros con defaults = {n:,} (identico al historico)")


def test_custom_fusion_hidden_applies():
    m = NMR_SetTransformer(num_classes=N_CLASSES, fusion_hidden=(256, 128))
    assert m.fc_fusion1.out_features == 256
    assert m.fc_fusion2.in_features == 256 and m.fc_fusion2.out_features == 128
    assert m.fc_out.in_features == 128
    assert _forward(m).shape == (2, N_CLASSES)
    print("[OK] fusion_hidden=(256, 128) aplicado y forward OK")


def test_small_fusion_hidden_applies():
    m = NMR_SetTransformer(num_classes=N_CLASSES, fusion_hidden=(64, 32))
    assert m.fc_fusion1.out_features == 64 and m.fc_fusion2.out_features == 32
    assert _forward(m).shape == (2, N_CLASSES)
    print("[OK] fusion_hidden=(64, 32) aplicado y forward OK")


def _cfg(model_extra=None):
    m = {"arch": "settransformer", "d_model": 64, "n_heads": 4,
         "n_layers": 2, "n_seeds": 1}
    if model_extra:
        m.update(model_extra)
    return {"model": m}


def test_build_model_default_fusion():
    m = build_model(_cfg(), num_classes=N_CLASSES)
    assert m.fc_fusion1.out_features == 128 and m.fc_fusion2.out_features == 64
    print("[OK] build_model sin fusion_hidden -> default historico")


def test_build_model_reads_fusion_hidden_from_config():
    """Una lista de YAML (no una tupla) tiene que funcionar: yaml.safe_load
    devuelve listas, nunca tuplas."""
    m = build_model(_cfg({"fusion_hidden": [256, 128]}), num_classes=N_CLASSES)
    assert m.fc_fusion1.out_features == 256 and m.fc_fusion2.out_features == 128
    print("[OK] build_model lee fusion_hidden (lista de YAML) del config")


def test_train_reads_seed_from_config():
    """train.py no debe tener el 42 hardcodeado: el estudio de ruido (Exp I)
    necesita correr la MISMA config con seeds distintas."""
    src = (Path(__file__).resolve().parent.parent / "train.py").read_text(encoding="utf-8")
    assert "set_seed(42)" not in src, "train.py todavia hardcodea set_seed(42)"
    assert "'seed'" in src or '"seed"' in src, "train.py no lee 'seed' del config"
    print("[OK] train.py toma la seed del config")


if __name__ == "__main__":
    test_default_fusion_shapes_unchanged()
    test_default_param_count_unchanged()
    test_custom_fusion_hidden_applies()
    test_small_fusion_hidden_applies()
    test_build_model_default_fusion()
    test_build_model_reads_fusion_hidden_from_config()
    test_train_reads_seed_from_config()
    print("\n>>> HP CONFIG KNOBS OK <<<")
```

- [ ] **Step 2: Correr el test y verificar que falla**

```bash
cd experiments/E3_dos_conjuntos && python tests/test_hp_config_knobs.py
```

Esperado: FALLA en `test_custom_fusion_hidden_applies` con
`TypeError: __init__() got an unexpected keyword argument 'fusion_hidden'`.

Nota: si `test_default_param_count_unchanged` fallara con un número distinto de 70163, **parar y
avisar** — significaría que el modelo ya divergió del checkpoint congelado antes de empezar, y todo
el estudio quedaría inválido.

- [ ] **Step 3: Parametrizar `fusion_hidden` en el modelo**

En `experiments/E3_dos_conjuntos/model_e3_settransformer.py`, reemplazar el `__init__` de
`NMR_SetTransformer` (líneas 72-83) por:

```python
    def __init__(self, num_classes=19, d_model=64, n_heads=4, n_layers=2, n_seeds=1,
                 fusion_hidden=(128, 64)):
        super().__init__()
        self.proj_ch = nn.Linear(4, d_model)
        self.proj_13c = nn.Linear(1, d_model)
        self.type_emb = nn.Embedding(2, d_model)   # 0=crosspeak, 1=13C
        self.encoder = nn.ModuleList([SAB(d_model, n_heads) for _ in range(n_layers)])
        self.pma = PMA(d_model, n_heads, n_seeds)

        # fusion_hidden parametrizable (Exp I, estudio de hiperparametros). El
        # default (128, 64) reproduce EXACTAMENTE las dimensiones con las que se
        # entreno el checkpoint congelado: su state_dict sigue cargando igual.
        # Llega como lista desde YAML, por eso el int() explicito.
        h1, h2 = int(fusion_hidden[0]), int(fusion_hidden[1])
        fusion_dim = d_model * n_seeds + 8
        self.fc_fusion1 = nn.Linear(fusion_dim, h1)
        self.fc_fusion2 = nn.Linear(h1, h2)
        self.fc_out = nn.Linear(h2, num_classes)
```

- [ ] **Step 4: Propagar `fusion_hidden` en `build_model`**

En `experiments/E3_dos_conjuntos/train.py`, dentro de `build_model`, reemplazar el bloque
`settransformer` (líneas 51-60) por:

```python
    if arch == 'settransformer':
        from model_e3_settransformer import NMR_SetTransformer
        m = cfg['model']
        return NMR_SetTransformer(
            num_classes=num_classes,
            d_model=int(m.get('d_model', 64)),
            n_heads=int(m.get('n_heads', 4)),
            n_layers=int(m.get('n_layers', 2)),
            n_seeds=int(m.get('n_seeds', 1)),
            fusion_hidden=tuple(m.get('fusion_hidden', (128, 64))),
        )
```

- [ ] **Step 5: Hacer la seed configurable**

En `experiments/E3_dos_conjuntos/train.py`, la función `train()` empieza (líneas 104-107) con:

```python
def train(config_path):
    set_seed(42)
    cfg = load_config(config_path)
    print(f"--- ENTRENAMIENTO EXP E FASE 3 ({cfg['model']['arch']}): {cfg['experiment_name']} ---")
```

Reemplazarlo por (el orden se invierte: hay que leer el config ANTES de sembrar; `load_config` no
usa el RNG, así que invertirlo no cambia nada del comportamiento):

```python
def train(config_path):
    cfg = load_config(config_path)
    # La seed sale del config (default 42 = comportamiento historico). El Exp I
    # la varia para medir cuanta EMA se mueve por puro azar, sin cambiar nada mas.
    seed = int(cfg['hyperparameters'].get('seed', 42))
    set_seed(seed)
    print(f"--- ENTRENAMIENTO EXP E FASE 3 ({cfg['model']['arch']}): {cfg['experiment_name']} ---")
    print(f"[INFO] Seed: {seed}")
```

- [ ] **Step 6: Correr el test y verificar que pasa**

```bash
cd experiments/E3_dos_conjuntos && python tests/test_hp_config_knobs.py
```

Esperado: `>>> HP CONFIG KNOBS OK <<<`.

- [ ] **Step 7: Correr los tests existentes del E3 para verificar que no se rompió nada**

```bash
cd experiments/E3_dos_conjuntos && python tests/test_forward_settransformer.py && python tests/test_config_utils.py && python tests/test_oraculo.py && python tests/test_oraculo_hetero.py
```

Esperado: los cuatro imprimen su `>>> ... OK <<<`. En particular
`test_forward_settransformer.py::test_param_count_small` confirma de nuevo que el conteo de
parámetros no se movió.

- [ ] **Step 8: Commit**

```bash
git add experiments/E3_dos_conjuntos/model_e3_settransformer.py experiments/E3_dos_conjuntos/train.py experiments/E3_dos_conjuntos/tests/test_hp_config_knobs.py
git commit -m "exp I: seed y fusion_hidden configurables (aditivo, defaults intactos)"
```

---

## Task 2: `sweep_grid.yaml` + generador + los 23 configs

**Files:**
- Create: `experiments/E3_dos_conjuntos/hp_sweep/sweep_grid.yaml`
- Create: `experiments/E3_dos_conjuntos/hp_sweep/make_configs.py`
- Create: `experiments/E3_dos_conjuntos/hp_sweep/configs/` (23 `.yaml` generados)
- Test: `experiments/E3_dos_conjuntos/hp_sweep/tests/test_make_configs.py`

**Interfaces:**
- Consumes: de la Task 1, la clave `model.fusion_hidden` y `hyperparameters.seed` que `train.py`
  ahora entiende.
- Produces:
  - `make_configs.py::build_all(grid, baseline) -> dict[str, dict]` — mapea nombre de archivo
    (ej. `"hp_dmodel_32.yaml"`) a config completo.
  - `make_configs.py::load_baseline(path) -> dict` — carga el baseline SIN expandir variables de
    entorno.
  - `make_configs.py::slug(value) -> str`.
  - Los 23 archivos en `configs/`, consumidos por las Tasks 3, 4, 5 y 6.

- [ ] **Step 1: Crear el YAML del diseño**

Crear `experiments/E3_dos_conjuntos/hp_sweep/sweep_grid.yaml`:

```yaml
# sweep_grid.yaml -- el DISENO del estudio de hiperparametros (Exp I).
#
# Fuente unica de verdad: para cambiar el estudio se edita ESTE archivo y se
# corre `python make_configs.py`. Los YAML de configs/ son GENERADOS -- editarlos
# a mano se pierde en la proxima regeneracion.
#
# Spec: docs/superpowers/specs/2026-08-07-estudio-hiperparametros-e3-design.md

# El baseline real (no una copia): la config del checkpoint congelado.
baseline_config: "../config_settransformer.yaml"
prefix: "nmr_202k_e3_hp"

# --- OFAT: una dimension a la vez desde el baseline (16 corridas) -------------
# 'path' es la ruta de la clave dentro del YAML, separada por puntos.
ofat:
  - name: dmodel
    path: model.d_model
    values: [32, 128, 256]
  - name: layers
    path: model.n_layers
    values: [1, 3, 4]
  - name: heads
    path: model.n_heads
    values: [2, 8]
  - name: pma                      # n_seeds = semillas del pooling PMA
    path: model.n_seeds
    values: [2, 4]
  - name: lr
    path: hyperparameters.learning_rate
    values: [0.0003, 0.003]
  - name: bs
    path: hyperparameters.batch_size
    values: [32, 128]
  - name: fusion
    path: model.fusion_hidden
    values: [[64, 32], [256, 128]]

# --- Grid 2D capacidad x profundidad (4 corridas nuevas) ---------------------
# Producto cartesiano de los dos ejes. make_configs.py descarta automaticamente
# las celdas que ya cubren el baseline (64x2) o el OFAT (32x2, 128x2, 64x1,
# 64x4) -- no hay que listarlas a mano.
grid2d:
  name: grid
  axes:
    model.d_model: [32, 64, 128]
    model.n_layers: [1, 2, 4]

# --- Replicas del baseline para medir el piso de ruido (3 corridas) ----------
# Misma config exacta, distinta semilla de RNG. Da la media +- desvio con la que
# se decide si una diferencia de EMA es senal o azar.
noise:
  name: rep
  path: hyperparameters.seed
  values: [42, 43, 44]
```

- [ ] **Step 2: Escribir el test que falla**

Crear `experiments/E3_dos_conjuntos/hp_sweep/tests/test_make_configs.py`:

```python
# coding: ascii
"""Invariantes del diseno experimental del Exp I. Estos tests son la barrera
que impide que un descuido invalide 15 horas de GPU: si dos configs comparten
checkpoint_dir, la segunda corrida pisa a la primera SIN tirar error y despues
se reporta la EMA equivocada atribuida al modelo equivocado."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import make_configs

GRID_PATH = _HERE.parent / "sweep_grid.yaml"
N_EXPECTED = 23

EXPECTED_FILES = {
    "hp_dmodel_32.yaml", "hp_dmodel_128.yaml", "hp_dmodel_256.yaml",
    "hp_layers_1.yaml", "hp_layers_3.yaml", "hp_layers_4.yaml",
    "hp_heads_2.yaml", "hp_heads_8.yaml",
    "hp_pma_2.yaml", "hp_pma_4.yaml",
    "hp_lr_0p0003.yaml", "hp_lr_0p003.yaml",
    "hp_bs_32.yaml", "hp_bs_128.yaml",
    "hp_fusion_64x32.yaml", "hp_fusion_256x128.yaml",
    "hp_grid_32x1.yaml", "hp_grid_32x4.yaml",
    "hp_grid_128x1.yaml", "hp_grid_128x4.yaml",
    "hp_rep_42.yaml", "hp_rep_43.yaml", "hp_rep_44.yaml",
}

# Claves de identidad: son unicas por construccion y NO forman parte del diseno
# experimental, asi que se excluyen de la comparacion "difiere en una sola clave".
IDENTITY_KEYS = {"experiment_name", "paths.checkpoint_dir"}


def _built():
    grid = make_configs.load_grid(GRID_PATH)
    baseline = make_configs.load_baseline(GRID_PATH.parent / grid["baseline_config"])
    return make_configs.build_all(grid, baseline), baseline


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _diff_keys(a, b):
    fa, fb = _flat(a), _flat(b)
    keys = set(fa) | set(fb)
    return {k for k in keys
            if k not in IDENTITY_KEYS and fa.get(k, "<ausente>") != fb.get(k, "<ausente>")}


def test_exactly_23_configs_with_expected_names():
    built, _ = _built()
    assert len(built) == N_EXPECTED, f"{len(built)} configs, se esperaban {N_EXPECTED}"
    assert set(built) == EXPECTED_FILES, set(built) ^ EXPECTED_FILES
    print(f"[OK] {N_EXPECTED} configs con los nombres esperados")


def test_experiment_name_and_checkpoint_dir_unique():
    built, _ = _built()
    names = [c["experiment_name"] for c in built.values()]
    dirs = [c["paths"]["checkpoint_dir"] for c in built.values()]
    assert len(set(names)) == N_EXPECTED, "experiment_name repetido"
    assert len(set(dirs)) == N_EXPECTED, "checkpoint_dir repetido -- se pisarian entre si"
    print("[OK] experiment_name y checkpoint_dir unicos en las 23")


def test_ofat_configs_differ_in_exactly_one_key():
    built, baseline = _built()
    ofat_prefixes = ("hp_dmodel_", "hp_layers_", "hp_heads_", "hp_pma_",
                     "hp_lr_", "hp_bs_", "hp_fusion_")
    n = 0
    for fname, cfg in built.items():
        if not fname.startswith(ofat_prefixes):
            continue
        d = _diff_keys(cfg, baseline)
        assert len(d) == 1, f"{fname} difiere en {sorted(d)} (deberia ser 1 sola clave)"
        n += 1
    assert n == 16, f"{n} configs OFAT, se esperaban 16"
    print("[OK] las 16 corridas OFAT varian exactamente una dimension")


def test_grid2d_configs_differ_in_exactly_two_keys():
    built, baseline = _built()
    cells = set()
    for fname, cfg in built.items():
        if not fname.startswith("hp_grid_"):
            continue
        d = _diff_keys(cfg, baseline)
        assert d == {"model.d_model", "model.n_layers"}, f"{fname}: {sorted(d)}"
        cells.add((cfg["model"]["d_model"], cfg["model"]["n_layers"]))
    assert cells == {(32, 1), (32, 4), (128, 1), (128, 4)}, cells
    print("[OK] el grid 2D emite las 4 celdas no cubiertas por baseline/OFAT")


def test_noise_configs_differ_only_in_seed():
    built, baseline = _built()
    seeds = set()
    for fname, cfg in built.items():
        if not fname.startswith("hp_rep_"):
            continue
        assert _diff_keys(cfg, baseline) == {"hyperparameters.seed"}, fname
        seeds.add(cfg["hyperparameters"]["seed"])
    assert seeds == {42, 43, 44}, seeds
    print("[OK] las 3 replicas difieren solo en la seed")


def test_dmodel_divisible_by_nheads():
    """MAB.forward hace d = dim_V // num_heads: si no divide, el modelo se
    construye igual y entrena BASURA en silencio con dimensiones truncadas."""
    built, _ = _built()
    for fname, cfg in built.items():
        d, h = cfg["model"]["d_model"], cfg["model"]["n_heads"]
        assert d % h == 0, f"{fname}: d_model={d} no divisible por n_heads={h}"
    print("[OK] d_model divisible por n_heads en las 23")


def test_invariants_identical_in_all_configs():
    """Regla dura 8: si alguno de estos varia, las EMAs dejan de ser comparables
    y el estudio entero no vale nada."""
    built, baseline = _built()
    fb = _flat(baseline)
    invariants = [
        "hyperparameters.epochs",
        "hyperparameters.scheduler.patience",
        "hyperparameters.scheduler.factor",
        "system.num_workers",
        "system.device",
        "paths.base_dir",
        "paths.val_indices_filename",
        "paths.peaks_ch_filename",
        "paths.peaks_13c_filename",
        "paths.labels_filename",
        "paths.smiles_filename",
        "model.arch",
        "normalization.c13_ppm_max",
        "normalization.h1_ppm_max",
        "normalization.amp_ch0_scale",
    ]
    for fname, cfg in built.items():
        fc = _flat(cfg)
        for key in invariants:
            assert fc[key] == fb[key], f"{fname}: {key} = {fc[key]!r} != {fb[key]!r}"
    assert fb["hyperparameters.epochs"] == 100
    assert fb["hyperparameters.scheduler.patience"] == 8
    assert fb["hyperparameters.scheduler.factor"] == 0.7
    assert fb["system.num_workers"] == 0
    print("[OK] invariantes (epocas, scheduler, workers, dataset, split) identicos")


def test_base_dir_env_var_not_expanded():
    """load_baseline NO debe usar config_utils.load_config: eso expandiria
    ${NMR_DATA_DIR:-...} al valor de login-1 y los configs dejarian de servir en
    Clementina. Tienen que quedar con la variable literal."""
    built, _ = _built()
    for fname, cfg in built.items():
        assert "${NMR_DATA_DIR" in cfg["paths"]["base_dir"], f"{fname}: base_dir expandido"
    print("[OK] base_dir conserva la variable de entorno sin expandir")


def test_slug_formats():
    assert make_configs.slug(32) == "32"
    assert make_configs.slug(0.0003) == "0p0003"
    assert make_configs.slug(0.003) == "0p003"
    assert make_configs.slug([64, 32]) == "64x32"
    print("[OK] slug() formatea enteros, floats y listas sin caracteres raros")


def test_written_configs_on_disk_match_generated():
    """Los configs commiteados en configs/ tienen que ser EXACTAMENTE lo que
    make_configs.py genera hoy. Si alguien edito uno a mano, este test lo caza."""
    import yaml
    built, _ = _built()
    cfg_dir = GRID_PATH.parent / "configs"
    on_disk = {p.name for p in cfg_dir.glob("*.yaml")}
    assert on_disk == set(built), on_disk ^ set(built)
    for fname, cfg in built.items():
        with open(cfg_dir / fname, "r", encoding="utf-8") as f:
            assert yaml.safe_load(f) == cfg, f"{fname} en disco != generado"
    print("[OK] los 23 configs en disco coinciden con el generador")


if __name__ == "__main__":
    test_exactly_23_configs_with_expected_names()
    test_experiment_name_and_checkpoint_dir_unique()
    test_ofat_configs_differ_in_exactly_one_key()
    test_grid2d_configs_differ_in_exactly_two_keys()
    test_noise_configs_differ_only_in_seed()
    test_dmodel_divisible_by_nheads()
    test_invariants_identical_in_all_configs()
    test_base_dir_env_var_not_expanded()
    test_slug_formats()
    test_written_configs_on_disk_match_generated()
    print("\n>>> MAKE_CONFIGS OK <<<")
```

- [ ] **Step 3: Correr el test y verificar que falla**

```bash
cd experiments/E3_dos_conjuntos/hp_sweep && python tests/test_make_configs.py
```

Esperado: FALLA con `ModuleNotFoundError: No module named 'make_configs'`.

- [ ] **Step 4: Escribir el generador**

Crear `experiments/E3_dos_conjuntos/hp_sweep/make_configs.py`:

```python
# coding: ascii
"""make_configs.py -- expande sweep_grid.yaml a los 23 configs del Exp I.

Uso:  python make_configs.py           (escribe/actualiza configs/)
      python make_configs.py --check   (falla si configs/ esta desactualizado)

Los configs generados se commitean: un revisor tiene que poder leer exactamente
que se entreno sin ejecutar nada.
"""
import argparse
import copy
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
GRID_PATH = os.path.join(_HERE, "sweep_grid.yaml")
OUT_DIR = os.path.join(_HERE, "configs")

_BANNER = (
    "# ARCHIVO GENERADO por hp_sweep/make_configs.py -- NO editar a mano.\n"
    "# Para cambiar el estudio: editar hp_sweep/sweep_grid.yaml y regenerar.\n"
    "# Exp I -- estudio de hiperparametros del Set Transformer (E3).\n"
)


def slug(value):
    """Convierte un valor a un fragmento de nombre de archivo seguro:
    0.0003 -> '0p0003' (el punto rompe la lectura de la extension),
    [64, 32] -> '64x32'."""
    if isinstance(value, (list, tuple)):
        return "x".join(slug(v) for v in value)
    if isinstance(value, float):
        return ("%g" % value).replace(".", "p").replace("-", "m")
    return str(value)


def load_grid(path=GRID_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_baseline(path):
    """Carga el config baseline con yaml.safe_load CRUDO, a proposito.

    NO usar config_utils.load_config(): expande ${NMR_DATA_DIR:-...} al valor de
    login-1 y los configs generados dejarian de funcionar en Clementina. La
    variable tiene que sobrevivir literal hasta el YAML de salida.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_path(cfg, dotted, value):
    """Escribe cfg['a']['b'] = value a partir de 'a.b'. Falla si el tramo
    intermedio no existe: un typo en sweep_grid.yaml debe romper fuerte, no
    crear una clave nueva que train.py ignoraria en silencio."""
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"Ruta invalida en sweep_grid.yaml: {dotted!r} "
                           f"(no existe la seccion {p!r} en el baseline)")
        node = node[p]
    node[parts[-1]] = value


def _get_path(cfg, dotted):
    node = cfg
    for p in dotted.split("."):
        node = node[p]
    return node


def _make_variant(baseline, prefix, name, tag, overrides):
    """Copia del baseline con los overrides aplicados y una identidad unica."""
    cfg = copy.deepcopy(baseline)
    for dotted, value in overrides.items():
        _set_path(cfg, dotted, value)
    ident = f"{name}_{tag}"
    cfg["experiment_name"] = f"{prefix}_{ident}"
    cfg["paths"]["checkpoint_dir"] = f"checkpoints_E3_hp_{ident}"
    return f"hp_{ident}.yaml", cfg


def build_all(grid, baseline):
    """Devuelve {nombre_de_archivo: config}. 16 OFAT + 4 grid 2D + 3 replicas."""
    prefix = grid["prefix"]
    out = {}

    # --- OFAT ---------------------------------------------------------------
    ofat_values = {}          # path -> valores probados, para validar el grid
    for axis in grid["ofat"]:
        ofat_values[axis["path"]] = list(axis["values"])
        for value in axis["values"]:
            fname, cfg = _make_variant(baseline, prefix, axis["name"],
                                       slug(value), {axis["path"]: value})
            out[fname] = cfg

    # --- Grid 2D ------------------------------------------------------------
    g = grid["grid2d"]
    paths = list(g["axes"].keys())
    if len(paths) != 2:
        raise ValueError("grid2d espera exactamente 2 ejes")
    pa, pb = paths
    for va in g["axes"][pa]:
        for vb in g["axes"][pb]:
            changed = [p for p, v in ((pa, va), (pb, vb)) if _get_path(baseline, p) != v]
            if len(changed) == 0:
                continue                      # es el baseline mismo
            if len(changed) == 1:
                # Ya cubierta por el OFAT. Validar que efectivamente lo este:
                # si no, la celda se perderia en silencio.
                p = changed[0]
                v = va if p == pa else vb
                if v not in ofat_values.get(p, []):
                    raise ValueError(
                        f"La celda del grid ({pa}={va}, {pb}={vb}) difiere del "
                        f"baseline solo en {p}={v}, pero ese valor NO esta en el "
                        f"eje OFAT correspondiente: quedaria sin correr.")
                continue
            fname, cfg = _make_variant(baseline, prefix, g["name"],
                                       f"{slug(va)}x{slug(vb)}", {pa: va, pb: vb})
            out[fname] = cfg

    # --- Replicas (piso de ruido) -------------------------------------------
    n = grid["noise"]
    for value in n["values"]:
        fname, cfg = _make_variant(baseline, prefix, n["name"], slug(value),
                                   {n["path"]: value})
        out[fname] = cfg

    # --- Validaciones globales ----------------------------------------------
    for fname, cfg in out.items():
        d, h = cfg["model"]["d_model"], cfg["model"]["n_heads"]
        if d % h != 0:
            raise ValueError(f"{fname}: d_model={d} no es divisible por n_heads={h}")
    if len({c["paths"]["checkpoint_dir"] for c in out.values()}) != len(out):
        raise ValueError("checkpoint_dir repetido: las corridas se pisarian entre si")
    return out


def _dump(cfg):
    return _BANNER + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False,
                                    allow_unicode=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="no escribe; falla si configs/ esta desactualizado")
    args = ap.parse_args()

    grid = load_grid()
    baseline = load_baseline(os.path.join(_HERE, grid["baseline_config"]))
    built = build_all(grid, baseline)

    if args.check:
        stale = []
        for fname, cfg in built.items():
            path = os.path.join(OUT_DIR, fname)
            if not os.path.exists(path):
                stale.append(f"{fname} (falta)")
                continue
            with open(path, "r", encoding="utf-8") as f:
                if yaml.safe_load(f) != cfg:
                    stale.append(f"{fname} (distinto)")
        extra = sorted(set(os.listdir(OUT_DIR)) - set(built)) if os.path.isdir(OUT_DIR) else []
        if stale or extra:
            print("[FAIL] configs/ desactualizado:")
            for s in stale + [f"{e} (sobra)" for e in extra]:
                print(f"  - {s}")
            sys.exit(1)
        print(f"[OK] los {len(built)} configs en disco estan al dia")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, cfg in sorted(built.items()):
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write(_dump(cfg))
    print(f"[OK] {len(built)} configs escritos en {OUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Generar los configs**

```bash
cd experiments/E3_dos_conjuntos/hp_sweep && python make_configs.py
```

Esperado: `[OK] 23 configs escritos en .../hp_sweep/configs`.

- [ ] **Step 6: Correr el test y verificar que pasa**

```bash
cd experiments/E3_dos_conjuntos/hp_sweep && python tests/test_make_configs.py
```

Esperado: `>>> MAKE_CONFIGS OK <<<` (10 líneas `[OK]`).

- [ ] **Step 7: Inspeccionar dos configs a mano**

```bash
cd experiments/E3_dos_conjuntos/hp_sweep && cat configs/hp_dmodel_256.yaml && cat configs/hp_rep_43.yaml
```

Verificar a ojo: `hp_dmodel_256.yaml` tiene `d_model: 256` y todo lo demás igual al baseline;
`hp_rep_43.yaml` tiene `seed: 43` dentro de `hyperparameters`; **ambos** conservan
`base_dir: ${NMR_DATA_DIR:-/home/lpassaglia.iquir/DB_200k}` sin expandir, y tienen
`checkpoint_dir` distintos entre sí.

- [ ] **Step 8: Commit**

```bash
git add experiments/E3_dos_conjuntos/hp_sweep/
git commit -m "exp I: sweep_grid.yaml + generador + los 23 configs del estudio"
```

---

## Task 3: Smoke test de forward para las 23 arquitecturas

**Files:**
- Create: `experiments/E3_dos_conjuntos/hp_sweep/tests/test_all_archs_forward.py`

**Interfaces:**
- Consumes: `build_model(cfg, num_classes)` de `train.py` (Task 1) y los 23 configs de `configs/`
  (Task 2).
- Produces: nada que consuman tareas posteriores. Es la barrera de la regla dura 5.

- [ ] **Step 1: Escribir el test**

Crear `experiments/E3_dos_conjuntos/hp_sweep/tests/test_all_archs_forward.py`:

```python
# coding: ascii
"""Regla dura 5, aplicada a las 23 corridas de una sola vez: construye el modelo
de CADA config del sweep y le hace un forward en CPU con shapes reales.

Un mismatch de dimensiones se descubre en segundos en la PC local, no despues de
40 minutos de cola de GPU. Corre sin datos: solo necesita torch."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_E3 = _HERE.parent.parent                      # experiments/E3_dos_conjuntos
sys.path.insert(0, str(_E3))
sys.path.insert(0, str(_HERE.parent))          # hp_sweep (para make_configs)

import torch
import yaml

from train import build_model

CONFIG_DIR = _HERE.parent / "configs"
N_CLASSES, MAX_CH, MAX_13C, B = 19, 32, 40, 3
N_EXPECTED = 23


def _batch():
    """Batch sintetico con mascaras MIXTAS: una molecula llena, una parcial y
    una totalmente enmascarada (el caso que hace NaN si el softmax no esta
    protegido)."""
    mask_ch = torch.ones(B, MAX_CH)
    mask_13c = torch.ones(B, MAX_13C)
    mask_ch[1, 10:] = 0.0
    mask_13c[1, 12:] = 0.0
    mask_ch[2] = 0.0
    mask_13c[2] = 0.0
    return (torch.randn(B, MAX_CH, 4), mask_ch,
            torch.randn(B, MAX_13C, 1), mask_13c,
            torch.randn(B, 8))


def test_all_configs_present():
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(files) == N_EXPECTED, f"{len(files)} configs en {CONFIG_DIR}"
    print(f"[OK] {N_EXPECTED} configs encontrados")


def test_forward_every_config():
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    peaks_ch, mask_ch, peaks_13c, mask_13c, cond = _batch()
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg, num_classes=N_CLASSES).eval()
        with torch.no_grad():
            out = model(peaks_ch, mask_ch, peaks_13c, mask_13c, cond)
        assert out.shape == (B, N_CLASSES), f"{path.name}: shape {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{path.name}: NaN/Inf en la salida"
        n = sum(p.numel() for p in model.parameters())
        print(f"  [OK] {path.name:<24} params={n:>9,}  out={tuple(out.shape)}")
    print(f"[OK] forward correcto en los {len(files)} configs")


def test_param_counts_span_a_range():
    """Sanity del diseno: si todas las variantes tuvieran el mismo tamano, el
    estudio no estaria variando la capacidad realmente."""
    counts = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        counts.append(sum(p.numel() for p in build_model(cfg, N_CLASSES).parameters()))
    assert max(counts) > 3 * min(counts), (min(counts), max(counts))
    print(f"[OK] rango de parametros: {min(counts):,} - {max(counts):,}")


if __name__ == "__main__":
    test_all_configs_present()
    test_forward_every_config()
    test_param_counts_span_a_range()
    print("\n>>> SMOKE 23 ARQUITECTURAS OK <<<")
```

- [ ] **Step 2: Correr el test**

```bash
cd experiments/E3_dos_conjuntos/hp_sweep && python tests/test_all_archs_forward.py
```

Esperado: 23 líneas `[OK] hp_*.yaml params=... out=(3, 19)` y `>>> SMOKE 23 ARQUITECTURAS OK <<<`.

Si alguna falla con un error de dimensiones, **no arreglar el test**: arreglar los valores en
`sweep_grid.yaml` y regenerar con `python make_configs.py`.

- [ ] **Step 3: Commit**

```bash
git add experiments/E3_dos_conjuntos/hp_sweep/tests/test_all_archs_forward.py
git commit -m "exp I: smoke test de forward para las 23 arquitecturas (regla dura 5)"
```

---

## Task 4: `run_sweep.sh` + README del estudio

**Files:**
- Create: `experiments/E3_dos_conjuntos/run_sweep.sh`
- Create: `experiments/E3_dos_conjuntos/hp_sweep/README.md`

**Interfaces:**
- Consumes: los configs de `hp_sweep/configs/` (Task 2).
- Produces: archivos `expE3_hp_<jobid>.out` en el cluster, que la Task 5 parsea.

- [ ] **Step 1: Escribir el `.sh`**

Crear `experiments/E3_dos_conjuntos/run_sweep.sh` (modelado sobre `run_train_scaling.sh` y
`run_eval.sh`, que ya funcionan en login-1):

```bash
#!/bin/bash
#SBATCH --job-name=expE3_hp
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE3_hp_%j.out
#SBATCH --error=expE3_hp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1

# Exp I -- una corrida del estudio de hiperparametros: entrena Y evalua el
# mismo config en UN solo job. Que sean uno solo (y no dos) es deliberado:
# 23 sbatch en vez de 46, y elimina la clase de error "evalue un checkpoint
# que todavia no habia terminado de entrenarse".
#
# Uso:
#   sbatch run_sweep.sh hp_sweep/configs/hp_dmodel_128.yaml
#
# Los 23 de una:
#   for cfg in hp_sweep/configs/*.yaml; do sbatch run_sweep.sh "$cfg"; done

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/E3_dos_conjuntos

CONFIG="${1:?Falta el config, ej: sbatch run_sweep.sh hp_sweep/configs/hp_dmodel_128.yaml}"

echo "=== EXP I | CONFIG: $CONFIG ==="

echo "=== FASE 1/2: TRAIN ==="
python -u train.py --config "$CONFIG"
if [ $? -ne 0 ]; then
    echo "[ABORT] train.py fallo -- no se evalua (evitar reportar la EMA de un checkpoint viejo)"
    exit 1
fi

echo "=== FASE 2/2: EVAL ==="
# --oraculo all: cruda + asistida v1 + asistida v2, con la tabla de 3 vias que
# collect_results.py parsea.
python -u evaluate.py --config "$CONFIG" --oraculo all --batch-size 256
```

- [ ] **Step 2: Verificar la sintaxis del `.sh`**

```bash
bash -n experiments/E3_dos_conjuntos/run_sweep.sh && echo "SINTAXIS OK"
```

Esperado: `SINTAXIS OK`. (No se puede ejecutar de verdad: requiere SLURM y el cluster.)

- [ ] **Step 3: Verificar que cumple la regla dura 2**

```bash
grep -n "gres=gpu:1" experiments/E3_dos_conjuntos/run_sweep.sh && grep -c "gpus=" experiments/E3_dos_conjuntos/run_sweep.sh
```

Esperado: la primera línea muestra `#SBATCH --gres=gpu:1`; el `grep -c` devuelve `0` (y sale con
código 1, lo cual es correcto acá — significa que NO hay ningún `--gpus=`).

- [ ] **Step 4: Escribir el README**

Crear `experiments/E3_dos_conjuntos/hp_sweep/README.md`:

````markdown
# Exp I — estudio de hiperparámetros del Set Transformer

**Qué es:** 23 corridas controladas que varían los hiperparámetros de arquitectura y optimización
del Set Transformer del Exp E Fase 3, con el mismo val congelado y el mismo presupuesto de épocas,
para poder justificar la arquitectura elegida ante un revisor.

**Qué NO es:** una búsqueda para reemplazar el checkpoint congelado. Si aparece un ganador real
(mejora mayor a la banda de ruido medida), eso se documenta y se decide aparte.

Spec: `docs/superpowers/specs/2026-08-07-estudio-hiperparametros-e3-design.md`.

## Por qué vive acá y no en `experiments/I_.../`

El estudio solo vale si el código de entrenamiento y evaluación es **bit a bit el mismo** que produjo
el checkpoint congelado. Copiar `train.py` a otra carpeta abriría el riesgo de divergencia silenciosa
que invalidaría toda la comparación. Mismo criterio que el estudio de escalado de datos, que también
vivió en esta carpeta.

## El diseño

Baseline (config congelada): `d_model=64, n_heads=4, n_layers=2, n_seeds=1, lr=1e-3, bs=64,
fusion=128→64`.

| Bloque | Qué varía | Corridas |
|---|---|---|
| OFAT | `d_model` {32,128,256}, `n_layers` {1,3,4}, `n_heads` {2,8}, `n_seeds` {2,4}, `lr` {3e-4,3e-3}, `batch_size` {32,128}, fusión {64→32, 256→128} | 16 |
| Grid 2D | `d_model` × `n_layers`, las 4 celdas que el OFAT no cubre | 4 |
| Réplicas | mismo config, seeds 42/43/44 → mide el piso de ruido | 3 |

Todo lo demás es invariante en las 23 (regla dura 8): val congelado, 100 épocas,
`ConstrainedMSELoss`, scheduler `patience=8/factor=0.7`, `num_workers=0`, 19 clases, dataset completo.

**Correr las 23 en el mismo cluster.** Recomendado login-1/A10 (~39 min por corrida ⇒ ~15 h de GPU).

## Cómo se corre

**1. Antes de cualquier `sbatch` — smoke test local (regla dura 5):**

```bash
python hp_sweep/tests/test_make_configs.py
```

```bash
python hp_sweep/tests/test_all_archs_forward.py
```

**2. Lanzar las 23 (desde `experiments/E3_dos_conjuntos/` en el cluster):**

```bash
for cfg in hp_sweep/configs/*.yaml; do sbatch run_sweep.sh "$cfg"; done
```

Cada job entrena Y evalúa el mismo config, y deja todo en un `expE3_hp_<jobid>.out`.

**3. Bajar los `.out` a la PC local y recolectar:**

```bash
python hp_sweep/collect_results.py --out-dir ruta/a/los/out
```

Escribe `hp_sweep/results.csv` e imprime la tabla markdown lista para pegar en
`docs/Runs/RESULTS.md`.

**4. Generar la figura:**

```bash
python hp_sweep/make_plot.py
```

## Cómo se cambia el diseño

Editar `sweep_grid.yaml` (la fuente única) y regenerar:

```bash
python hp_sweep/make_configs.py
```

Los YAML de `configs/` son **generados**: editarlos a mano se pierde en la próxima regeneración.
Para verificar que están al día sin escribir nada:

```bash
python hp_sweep/make_configs.py --check
```

## Métrica de decisión

**Primaria: EMA asistida v2.** Desempate: best val loss. La EMA cruda se reporta pero **no decide**
— `RESULTS.md` ya documenta que es ruidosa (0.9–2.3 % a lo largo del scaling study, sin tendencia
monótona pese a que la asistida subía monótonamente).

**Regla fijada antes de ver los resultados:** una diferencia menor a la banda de ruido de las 3
réplicas se declara **no significativa**.
````

- [ ] **Step 5: Commit**

```bash
git add experiments/E3_dos_conjuntos/run_sweep.sh experiments/E3_dos_conjuntos/hp_sweep/README.md
git commit -m "exp I: run_sweep.sh (train+eval en un job) + README del estudio"
```

---

## Task 5: `collect_results.py` + su test

**Files:**
- Create: `experiments/E3_dos_conjuntos/hp_sweep/collect_results.py`
- Test: `experiments/E3_dos_conjuntos/hp_sweep/tests/test_collect_results.py`

**Interfaces:**
- Consumes: los `.out` que produce `run_sweep.sh` (Task 4).
- Produces:
  - `collect_results.py::parse_out(text) -> dict` con las claves
    `experiment_name, n_params, minutes, best_val_loss, ema_crude, ema_assist_v1, ema_assist_v2`
    (valor `None` en las que no aparezcan).
  - `collect_results.py::to_markdown(rows) -> str`.
  - `hp_sweep/results.csv`, consumido por `make_plot.py` (Task 6), con columnas
    `run, experiment_name, n_params, best_val_loss, ema_crude, ema_assist_v2, minutes`.

- [ ] **Step 1: Escribir el test que falla**

Crear `experiments/E3_dos_conjuntos/hp_sweep/tests/test_collect_results.py`:

```python
# coding: ascii
"""Parsing de los .out de SLURM. El .out sintetico de abajo copia LITERALMENTE
el formato que imprimen train.py y evaluate.py --oraculo all (verificado contra
train.py:134/175 y evaluate.py:171-181)."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import collect_results

FAKE_OUT = """=== EXP I | CONFIG: hp_sweep/configs/hp_dmodel_128.yaml ===
=== FASE 1/2: TRAIN ===
--- ENTRENAMIENTO EXP E FASE 3 (settransformer): nmr_202k_e3_hp_dmodel_128 ---
[INFO] Seed: 42
[INFO] Dispositivo: cuda
[INFO] Split congelado: SMILES invalidos=0 | train=187314 (leak removido=12, train_fraction=1.0) | val=14428
[INFO] Parametros totales del modelo (settransformer): 241,555 (chico por diseno; V10 ~8,603,299)
[INFO] Scheduler: patience=8, factor=0.7

[START] 100 epochs...
[EPOCH 1] Train: 0.9012 | Val: 0.4211 | LR: 0.001000 | Time: 23.4s
[EPOCH 100] Train: 0.0081 | Val: 0.0093 | LR: 0.001000 | Time: 23.1s
[SAVE] Nuevo mejor modelo!

[DONE] 41.7 min. Mejor Val: 0.0093
=== FASE 2/2: EVAL ===
============================================================
  EVALUACION EXP E FASE 3 (dos conjuntos) - SPLIT CONGELADO
============================================================
-> Experimento (checkpoint): nmr_202k_e3_hp_dmodel_128  | arch: settransformer
-> Modos: ['off', 'on', 'v2']   | idx_ch2: [1, 5, 9, 12]

============================================================
  MODO CRUDO  ->  EXACT MATCH ACCURACY: 2.11%
============================================================

====================================================================
  TABLA COMPARATIVA: CRUDA vs ASISTIDA v1 (doble) vs ASISTIDA v2 (hetero)
====================================================================

                                  CRUDA  ASIST v1  ASIST v2    v2-v1
  ------------------------------------------------------------------
  EMA GLOBAL                       2.11%    91.48%    91.50%    +0.02
  Alifaticos                      96.50%    96.55%    96.55%    +0.00
"""

TRUNCATED_OUT = """--- ENTRENAMIENTO EXP E FASE 3 (settransformer): nmr_202k_e3_hp_bs_32 ---
[INFO] Parametros totales del modelo (settransformer): 70,163 (chico por diseno; V10 ~8,603,299)
[START] 100 epochs...
[EPOCH 3] Train: 0.4001 | Val: 0.2010 | LR: 0.001000 | Time: 40.2s
slurmstepd: error: *** JOB 2377999 CANCELLED AT 2026-08-08T04:00:00 DUE TO TIME LIMIT ***
"""


def test_parses_complete_run():
    r = collect_results.parse_out(FAKE_OUT)
    assert r["experiment_name"] == "nmr_202k_e3_hp_dmodel_128", r["experiment_name"]
    assert r["n_params"] == 241555, r["n_params"]
    assert r["minutes"] == 41.7, r["minutes"]
    assert r["best_val_loss"] == 0.0093, r["best_val_loss"]
    assert r["ema_crude"] == 2.11, r["ema_crude"]
    assert r["ema_assist_v1"] == 91.48, r["ema_assist_v1"]
    assert r["ema_assist_v2"] == 91.50, r["ema_assist_v2"]
    print("[OK] parsea una corrida completa (train + eval)")


def test_truncated_run_gives_none_not_crash():
    """Un job cortado por TIME LIMIT no debe romper la recoleccion ni, peor,
    aparecer como si tuviera resultados."""
    r = collect_results.parse_out(TRUNCATED_OUT)
    assert r["experiment_name"] == "nmr_202k_e3_hp_bs_32"
    assert r["n_params"] == 70163
    assert r["best_val_loss"] is None and r["minutes"] is None
    assert r["ema_crude"] is None and r["ema_assist_v2"] is None
    print("[OK] corrida truncada -> None en vez de excepcion o dato inventado")


def test_empty_text_gives_all_none():
    r = collect_results.parse_out("")
    assert all(v is None for v in r.values()), r
    print("[OK] texto vacio -> todo None")


def test_markdown_marks_missing_runs_as_pending():
    rows = [
        collect_results.parse_out(FAKE_OUT),
        collect_results.parse_out(TRUNCATED_OUT),
    ]
    rows[0]["run"] = "dmodel_128"
    rows[1]["run"] = "bs_32"
    md = collect_results.to_markdown(rows)
    assert "| dmodel_128 |" in md
    assert "91.50" in md
    assert "PENDIENTE" in md, "una corrida sin EMA debe marcarse, no dejarse en blanco"
    print("[OK] la tabla markdown marca las corridas incompletas")


def test_noise_band_from_replicas():
    """Las filas hp_rep_* dan la banda de ruido: media y desvio de la EMA v2."""
    rows = [
        {"run": "rep_42", "ema_assist_v2": 91.35},
        {"run": "rep_43", "ema_assist_v2": 91.95},
        {"run": "rep_44", "ema_assist_v2": 92.14},
        {"run": "dmodel_128", "ema_assist_v2": 91.50},
    ]
    mean, std = collect_results.noise_band(rows)
    assert abs(mean - 91.8133) < 1e-3, mean
    assert abs(std - 0.3369) < 1e-3, std
    print(f"[OK] banda de ruido = {mean:.2f} +- {std:.2f} pp")


def test_noise_band_none_without_replicas():
    assert collect_results.noise_band([{"run": "dmodel_128", "ema_assist_v2": 91.5}]) is None
    print("[OK] sin replicas -> None (no se inventa una banda)")


if __name__ == "__main__":
    test_parses_complete_run()
    test_truncated_run_gives_none_not_crash()
    test_empty_text_gives_all_none()
    test_markdown_marks_missing_runs_as_pending()
    test_noise_band_from_replicas()
    test_noise_band_none_without_replicas()
    print("\n>>> COLLECT_RESULTS OK <<<")
```

- [ ] **Step 2: Correr el test y verificar que falla**

```bash
cd experiments/E3_dos_conjuntos/hp_sweep && python tests/test_collect_results.py
```

Esperado: FALLA con `ModuleNotFoundError: No module named 'collect_results'`.

- [ ] **Step 3: Escribir el recolector**

Crear `experiments/E3_dos_conjuntos/hp_sweep/collect_results.py`:

```python
# coding: ascii
"""collect_results.py -- junta los .out de SLURM del Exp I en un CSV y una tabla
markdown lista para pegar en docs/Runs/RESULTS.md.

Corre en la PC local sobre los .out que Lucas baje del cluster; no necesita torch
ni acceso al cluster.

Uso:  python collect_results.py --out-dir ruta/a/los/out
"""
import argparse
import csv
import os
import re
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "results.csv")

FIELDS = ["run", "experiment_name", "n_params", "best_val_loss",
          "ema_crude", "ema_assist_v1", "ema_assist_v2", "minutes"]

# train.py:107 y evaluate.py:212 imprimen el nombre; sirve cualquiera de los dos.
_RE_NAME = re.compile(r"ENTRENAMIENTO EXP E FASE 3 \([^)]*\): (\S+)")
_RE_NAME_EVAL = re.compile(r"Experimento \(checkpoint\): (\S+)")
# train.py:134
_RE_PARAMS = re.compile(r"Parametros totales del modelo \([^)]*\): ([\d,]+)")
# train.py:175
_RE_DONE = re.compile(r"\[DONE\] ([\d.]+) min\. Mejor Val: ([\d.]+)")
# evaluate.py:175 (tabla de 3 vias que emite --oraculo all)
_RE_EMA3 = re.compile(r"EMA GLOBAL\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%")


def _first(regex, text, group=1):
    m = regex.search(text)
    return m.group(group) if m else None


def parse_out(text):
    """Extrae las metricas de un .out. Todo lo que no aparezca queda en None:
    una corrida truncada NO debe aparecer como si tuviera resultados."""
    row = {k: None for k in FIELDS}

    row["experiment_name"] = _first(_RE_NAME, text) or _first(_RE_NAME_EVAL, text)

    params = _first(_RE_PARAMS, text)
    if params:
        row["n_params"] = int(params.replace(",", ""))

    m = _RE_DONE.search(text)
    if m:
        row["minutes"] = float(m.group(1))
        row["best_val_loss"] = float(m.group(2))

    m = _RE_EMA3.search(text)
    if m:
        row["ema_crude"] = float(m.group(1))
        row["ema_assist_v1"] = float(m.group(2))
        row["ema_assist_v2"] = float(m.group(3))

    return row


def run_label(experiment_name, prefix="nmr_202k_e3_hp_"):
    """'nmr_202k_e3_hp_dmodel_128' -> 'dmodel_128'."""
    if not experiment_name:
        return None
    return experiment_name[len(prefix):] if experiment_name.startswith(prefix) else experiment_name


def collect(out_dir):
    rows = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".out"):
            continue
        with open(os.path.join(out_dir, name), "r", encoding="utf-8", errors="replace") as f:
            row = parse_out(f.read())
        if row["experiment_name"] is None:
            continue                       # .out de otro experimento
        row["run"] = run_label(row["experiment_name"])
        rows.append(row)
    rows.sort(key=lambda r: r["run"] or "")
    return rows


def noise_band(rows):
    """(media, desvio muestral) de la EMA v2 sobre las replicas rep_*. None si
    hay menos de 2: con una sola replica no hay banda que reportar."""
    vals = [r["ema_assist_v2"] for r in rows
            if (r.get("run") or "").startswith("rep_") and r.get("ema_assist_v2") is not None]
    if len(vals) < 2:
        return None
    return statistics.mean(vals), statistics.stdev(vals)


def _fmt(v, spec):
    return "PENDIENTE" if v is None else format(v, spec)


def to_markdown(rows):
    lines = [
        "| Corrida | Params | Best Val Loss | EMA cruda | EMA asist v2 | min |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('run') or '?'} "
            f"| {_fmt(r.get('n_params'), ',d')} "
            f"| {_fmt(r.get('best_val_loss'), '.4f')} "
            f"| {_fmt(r.get('ema_crude'), '.2f')} "
            f"| {_fmt(r.get('ema_assist_v2'), '.2f')} "
            f"| {_fmt(r.get('minutes'), '.1f')} |"
        )
    band = noise_band(rows)
    if band:
        lines.append("")
        lines.append(f"**Piso de ruido (replicas rep_*):** EMA asistida v2 = "
                     f"{band[0]:.2f} +- {band[1]:.2f} pp. Las diferencias por debajo de "
                     f"{band[1]:.2f} pp NO son significativas.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="carpeta con los expE3_hp_*.out")
    args = ap.parse_args()

    rows = collect(args.out_dir)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})

    done = sum(1 for r in rows if r["ema_assist_v2"] is not None)
    print(f"[OK] {len(rows)} corridas leidas ({done} con EMA) -> {CSV_PATH}\n")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Correr el test y verificar que pasa**

```bash
cd experiments/E3_dos_conjuntos/hp_sweep && python tests/test_collect_results.py
```

Esperado: `>>> COLLECT_RESULTS OK <<<`.

- [ ] **Step 5: Commit**

```bash
git add experiments/E3_dos_conjuntos/hp_sweep/collect_results.py experiments/E3_dos_conjuntos/hp_sweep/tests/test_collect_results.py
git commit -m "exp I: collect_results.py -- .out de SLURM a CSV + tabla markdown"
```

---

## Task 6: `make_plot.py` + su test

**Files:**
- Create: `experiments/E3_dos_conjuntos/hp_sweep/make_plot.py`
- Test: `experiments/E3_dos_conjuntos/hp_sweep/tests/test_make_plot.py`

**Interfaces:**
- Consumes: `hp_sweep/results.csv` (Task 5), con las columnas de `collect_results.FIELDS`.
- Produces: `experiments/E3_dos_conjuntos/plots/hp_sweep_ofat.png`.
  - `make_plot.py::read_rows(csv_path) -> list[dict]`
  - `make_plot.py::group_by_axis(rows) -> dict[str, list[tuple[str, float]]]`
  - `make_plot.py::make_figure(rows, out_path) -> str`

- [ ] **Step 1: Verificar matplotlib**

```bash
python -c "import matplotlib; print(matplotlib.__version__)"
```

Si falla: `pip install matplotlib`.

- [ ] **Step 2: Escribir el test que falla**

Crear `experiments/E3_dos_conjuntos/hp_sweep/tests/test_make_plot.py`:

```python
# coding: ascii
"""La figura tiene que poder generarse ANTES de que existan los resultados
reales (con corridas a medias) y no romperse. Se testea con un CSV sintetico en
un directorio temporal: no toca plots/ del repo."""
import csv
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import make_plot

ROWS = [
    {"run": "rep_42", "ema_assist_v2": "91.35", "n_params": "70163", "best_val_loss": "0.0097"},
    {"run": "rep_43", "ema_assist_v2": "91.95", "n_params": "70163", "best_val_loss": "0.0094"},
    {"run": "rep_44", "ema_assist_v2": "92.14", "n_params": "70163", "best_val_loss": "0.0091"},
    {"run": "dmodel_32", "ema_assist_v2": "90.10", "n_params": "22019", "best_val_loss": "0.0121"},
    {"run": "dmodel_128", "ema_assist_v2": "91.50", "n_params": "241555", "best_val_loss": "0.0093"},
    {"run": "dmodel_256", "ema_assist_v2": "", "n_params": "", "best_val_loss": ""},   # pendiente
    {"run": "layers_1", "ema_assist_v2": "90.80", "n_params": "45000", "best_val_loss": "0.0110"},
    {"run": "layers_4", "ema_assist_v2": "91.60", "n_params": "120000", "best_val_loss": "0.0092"},
    {"run": "grid_32x1", "ema_assist_v2": "89.90", "n_params": "18000", "best_val_loss": "0.0130"},
]


def _write_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "n_params", "best_val_loss", "ema_assist_v2"])
        w.writeheader()
        for r in ROWS:
            w.writerow(r)


def test_read_rows_parses_and_skips_empty():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "results.csv"
        _write_csv(p)
        rows = make_plot.read_rows(p)
    assert len(rows) == len(ROWS)
    got = {r["run"]: r["ema_assist_v2"] for r in rows}
    assert got["dmodel_32"] == 90.10
    assert got["dmodel_256"] is None, "una celda vacia debe ser None, no 0.0"
    print("[OK] read_rows convierte numeros y deja None las celdas vacias")


def test_group_by_axis_splits_ofat_axes():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "results.csv"
        _write_csv(p)
        groups = make_plot.group_by_axis(make_plot.read_rows(p))
    assert "dmodel" in groups and "layers" in groups
    assert "rep" not in groups, "las replicas son la banda de ruido, no un eje OFAT"
    assert "grid" not in groups, "el grid 2D no va en los paneles OFAT"
    assert [v for _, v in groups["dmodel"]] == [90.10, 91.50], groups["dmodel"]
    print("[OK] group_by_axis separa los ejes OFAT y excluye rep_/grid_")


def test_make_figure_writes_png():
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "results.csv"
        png_path = Path(td) / "hp_sweep_ofat.png"
        _write_csv(csv_path)
        make_plot.make_figure(make_plot.read_rows(csv_path), png_path)
        assert png_path.exists() and png_path.stat().st_size > 5000, png_path.stat().st_size
    print("[OK] make_figure escribe un PNG no trivial")


def test_make_figure_survives_empty_csv():
    """Antes de correr los jobs el CSV esta vacio; el script no debe explotar."""
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["run", "n_params", "best_val_loss",
                                          "ema_assist_v2"]).writeheader()
        png_path = Path(td) / "vacio.png"
        make_plot.make_figure(make_plot.read_rows(csv_path), png_path)
        assert png_path.exists()
    print("[OK] CSV vacio -> figura vacia, sin excepcion")


if __name__ == "__main__":
    test_read_rows_parses_and_skips_empty()
    test_group_by_axis_splits_ofat_axes()
    test_make_figure_writes_png()
    test_make_figure_survives_empty_csv()
    print("\n>>> MAKE_PLOT OK <<<")
```

- [ ] **Step 3: Correr el test y verificar que falla**

```bash
cd experiments/E3_dos_conjuntos/hp_sweep && python tests/test_make_plot.py
```

Esperado: FALLA con `ModuleNotFoundError: No module named 'make_plot'`.

- [ ] **Step 4: Escribir el generador de la figura**

Crear `experiments/E3_dos_conjuntos/hp_sweep/make_plot.py`:

```python
# coding: ascii
"""make_plot.py -- figura del Exp I: EMA asistida v2 vs cada eje OFAT, con la
banda de ruido de las replicas sombreada.

Es la figura que contesta visualmente "por que esta combinacion y no otra":
si todos los puntos caen dentro de la banda, la respuesta es "porque da igual, y
esta es la mas chica".

Uso:  python make_plot.py [--csv results.csv] [--out ../plots/hp_sweep_ofat.png]
"""
import argparse
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")            # sin display: corre igual en el cluster
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "results.csv")
OUT_PATH = os.path.join(_HERE, "..", "plots", "hp_sweep_ofat.png")

# Titulo legible por eje OFAT (las claves son el 'name' de sweep_grid.yaml).
AXIS_TITLES = {
    "dmodel": "d_model (ancho)",
    "layers": "n_layers (profundidad)",
    "heads": "n_heads (cabezas de atencion)",
    "pma": "n_seeds (semillas del PMA)",
    "lr": "learning rate",
    "bs": "batch size",
    "fusion": "cabeza de fusion",
}


def _num(s):
    if s is None or str(s).strip() == "":
        return None
    return float(s)


def read_rows(csv_path=CSV_PATH):
    """Lee el CSV de collect_results.py. Celdas vacias -> None (NO 0.0: un cero
    se dibujaria como un resultado catastrofico inexistente)."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "run": r.get("run"),
                "n_params": _num(r.get("n_params")),
                "best_val_loss": _num(r.get("best_val_loss")),
                "ema_assist_v2": _num(r.get("ema_assist_v2")),
            })
    return rows


def group_by_axis(rows):
    """{'dmodel': [('32', 90.1), ('128', 91.5)], ...} solo con los ejes OFAT.
    Excluye rep_* (son la banda de ruido) y grid_* (van en su propia tabla).
    Las corridas sin EMA todavia (pendientes) se omiten del panel."""
    groups = {}
    for r in rows:
        run = r.get("run") or ""
        if "_" not in run:
            continue
        axis, tag = run.split("_", 1)
        if axis in ("rep", "grid") or axis not in AXIS_TITLES:
            continue
        if r["ema_assist_v2"] is None:
            continue
        groups.setdefault(axis, []).append((tag, r["ema_assist_v2"]))
    for axis in groups:
        groups[axis].sort(key=lambda t: t[1])
    return groups


def _noise_band(rows):
    vals = [r["ema_assist_v2"] for r in rows
            if (r.get("run") or "").startswith("rep_") and r["ema_assist_v2"] is not None]
    if len(vals) < 2:
        return None
    return statistics.mean(vals), statistics.stdev(vals)


def make_figure(rows, out_path=OUT_PATH):
    groups = group_by_axis(rows)
    band = _noise_band(rows)
    axes_names = [a for a in AXIS_TITLES if a in groups]

    n = max(len(axes_names), 1)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axarr = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                              squeeze=False)

    for i in range(nrows * ncols):
        ax = axarr[i // ncols][i % ncols]
        if i >= len(axes_names):
            ax.axis("off")
            continue
        name = axes_names[i]
        labels = [t for t, _ in groups[name]]
        values = [v for _, v in groups[name]]
        ax.bar(range(len(values)), values, color="#4878a8")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(AXIS_TITLES[name], fontsize=10)
        ax.set_ylabel("EMA asistida v2 (%)", fontsize=8)
        if band:
            mean, std = band
            ax.axhspan(mean - std, mean + std, color="#c8a020", alpha=0.25,
                       label="baseline +- 1 sigma")
            ax.axhline(mean, color="#c8a020", lw=1.2)
            lo = min(values + [mean - std]) - 0.5
            hi = max(values + [mean + std]) + 0.5
            ax.set_ylim(lo, hi)
            if i == 0:
                ax.legend(fontsize=7)

    title = "Exp I -- EMA asistida v2 por eje de hiperparametro"
    if band:
        title += f"  (baseline {band[0]:.2f} +- {band[1]:.2f} pp, n=3 replicas)"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    path = make_figure(read_rows(args.csv), args.out)
    print(f"[OK] figura escrita en {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Correr el test y verificar que pasa**

```bash
cd experiments/E3_dos_conjuntos/hp_sweep && python tests/test_make_plot.py
```

Esperado: `>>> MAKE_PLOT OK <<<`.

- [ ] **Step 6: Commit**

```bash
git add experiments/E3_dos_conjuntos/hp_sweep/make_plot.py experiments/E3_dos_conjuntos/hp_sweep/tests/test_make_plot.py
git commit -m "exp I: make_plot.py -- figura OFAT con banda de ruido"
```

---

## Task 7: Sección placeholder en `RESULTS.md` + corrección del spec

**Files:**
- Modify: `docs/Runs/RESULTS.md` (tabla de arriba, línea ~23; sección nueva al final, antes del
  comentario `<!-- Template for next entries`)
- Modify: `docs/superpowers/specs/2026-08-07-estudio-hiperparametros-e3-design.md`

**Interfaces:**
- Consumes: la tabla que produce `collect_results.to_markdown` (Task 5) — el placeholder tiene que
  tener las mismas columnas para que pegarla sea reemplazar un bloque.
- Produces: nada. Es documentación.

- [ ] **Step 1: Agregar la fila a la tabla resumen**

En `docs/Runs/RESULTS.md`, después de la fila de `Exp G Fase 1b` (línea 23), agregar:

```markdown
| Exp I — estudio de hiperparametros (Set Transformer) | n/a (sin imagen) | 19 | 202k | none | PENDIENTE | PENDIENTE | PENDIENTE | 23 corridas (16 OFAT + 4 grid 2D d_model x n_layers + 3 replicas de seed), val congelado y 100 epocas en las 23. **Jobs NO lanzados todavia.** Ver seccion |
```

- [ ] **Step 2: Agregar la sección al final**

En `docs/Runs/RESULTS.md`, justo **antes** de la línea `<!-- Template for next entries`, insertar:

```markdown
## Exp I — estudio de hiperparámetros del Set Transformer

- **Fecha:** 2026-08-07 (preparación) · **SLURM train+eval:** PENDIENTE ·
  **Config:** `experiments/E3_dos_conjuntos/hp_sweep/configs/*.yaml` (23) ·
  **Diseño:** `experiments/E3_dos_conjuntos/hp_sweep/sweep_grid.yaml`.
- **Estado: los jobs NO se lanzaron todavía.** Esta sección está preparada; los números se completan
  con `python hp_sweep/collect_results.py --out-dir <ruta>`, que emite la tabla con estas mismas
  columnas.
- **Qué es:** 23 corridas controladas que varían hiperparámetros de arquitectura y optimización sobre
  el Set Transformer de Fase 3. **No busca reemplazar el checkpoint congelado** — busca poder
  justificar la arquitectura elegida ante un revisor.
- **Diseño:** 16 corridas OFAT (una dimensión a la vez desde el baseline: `d_model` {32,128,256},
  `n_layers` {1,3,4}, `n_heads` {2,8}, `n_seeds` {2,4}, `lr` {3e-4,3e-3}, `batch_size` {32,128},
  cabeza de fusión {64→32, 256→128}), 4 celdas del grid 2D `d_model × n_layers` que el OFAT no
  cubre, y 3 réplicas del baseline (seeds 42/43/44) para **medir el piso de ruido**.
- **Invariantes en las 23 (regla dura 8):** val congelado (`val_indices_frozen.npy`, 14 428),
  100 épocas, `ConstrainedMSELoss(λ=0.5)`, scheduler `patience=8/factor=0.7`, `num_workers=0`,
  19 clases, dataset completo, y **un solo cluster** (login-1/A10).
- **Métrica de decisión (fijada ANTES de ver los resultados):** EMA asistida v2 como primaria, best
  val loss como desempate. La EMA cruda se reporta pero no decide (ya documentado más arriba: es
  ruidosa, 0.9–2.3 % sin tendencia monótona en el scaling study). **Una diferencia menor a la banda
  de ruido de las 3 réplicas se declara no significativa.**

| Corrida | Params | Best Val Loss | EMA cruda | EMA asist v2 | min |
|---|---|---|---|---|---|
| (23 filas — completar con `collect_results.py`) | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |

- **Predicción registrada de antemano (para que no se pueda ajustar después):** lo más probable es
  una **meseta**. El Exp F ya mostró que el cuello no es de optimización (cabeza Poisson + 250
  épocas con el scheduler llegando a saturación real, LR 0.001 → 0.00002, no mejoró) y el estudio de
  escalado mostró meseta de datos (75 % → 100 % = −0.20 pp). Si el sweep también da meseta, las tres
  evidencias convergen: el límite es de información/dominio, no de modelo.
- **Takeaway:** PENDIENTE. Spec:
  `docs/superpowers/specs/2026-08-07-estudio-hiperparametros-e3-design.md`. Plan:
  `docs/superpowers/plans/2026-08-07-estudio-hiperparametros-e3.md`. Código y cómo correrlo:
  `experiments/E3_dos_conjuntos/hp_sweep/README.md`.

---
```

- [ ] **Step 3: Corregir el conteo en el spec (22 → 23)**

En `docs/superpowers/specs/2026-08-07-estudio-hiperparametros-e3-design.md` hay que reflejar la
desviación documentada al principio de este plan. Cambios exactos:

1. En **§3.3**, reemplazar el párrafo que empieza `Baseline reentrenado con \`seed = 43\` y
   \`seed = 44\`. La \`seed = 42\` ya existe:` y su justificación de reutilización, por:

```markdown
Baseline reentrenado con `seed = 42`, `seed = 43` y `seed = 44` — **3 corridas nuevas**.

Nota sobre un cambio respecto de la versión original de este spec: inicialmente se planeó reutilizar
la corrida histórica de seed 42 (A10, Exp E Fase 3, EMA asistida 91.35 %) como tercera réplica. Se
descartó al escribir el plan de implementación: la §6 modifica `train.py` y
`model_e3_settransformer.py`, y aunque los cambios son aditivos y preservan el comportamiento por
default, la corrida histórica se hizo con la versión anterior del código y tres semanas antes. Poder
afirmar *"tres réplicas, mismo código, mismo cluster, misma semana"* es materialmente más fuerte ante
un revisor, y cuesta 39 minutos de GPU.
```

2. En **§3.3**, mantener el párrafo sobre la evidencia previa A10 vs XPU (~0.8 pp) tal cual: sigue
   siendo válido como motivación de por qué hay que medir la banda.

3. En **§3.4**, reemplazar `**22 corridas nuevas**` por `**23 corridas nuevas**` y `≈ **14 h de
   GPU**` por `≈ **15 h de GPU**`.

4. Reemplazar **todas** las apariciones restantes de "22" que se refieran al número de corridas o
   configs por "23". Están en: §2 (comentario sobre `configs/`), §4 (encabezado "lo que NO cambia en
   ninguna de las 22" y el texto de `epochs`), §4.1, §7.2, §7.4, §8.1 (varias) y §8.2. Verificar con:

```bash
grep -n "22\|las 22" docs/superpowers/specs/2026-08-07-estudio-hiperparametros-e3-design.md
```

Esperado tras el cambio: ninguna línea que hable de "22 corridas", "las 22" o "22 configs". (Puede
quedar algún "22" que sea parte de otra cosa, como una fecha — revisar cada match antes de cambiarlo.)

5. En **§11**, criterio 1, reemplazar `con 22 corridas controladas` por `con 23 corridas
   controladas`.

- [ ] **Step 4: Verificar que la documentación es consistente**

```bash
grep -rn "23 corridas\|las 23" docs/superpowers/specs/2026-08-07-estudio-hiperparametros-e3-design.md docs/Runs/RESULTS.md experiments/E3_dos_conjuntos/hp_sweep/README.md | head -30
```

Esperado: menciones consistentes de 23 en los tres archivos, sin ningún "22 corridas" sobreviviente.

- [ ] **Step 5: Correr toda la batería de tests una última vez**

```bash
cd experiments/E3_dos_conjuntos && python tests/test_hp_config_knobs.py && python tests/test_forward_settransformer.py && python hp_sweep/tests/test_make_configs.py && python hp_sweep/tests/test_all_archs_forward.py && python hp_sweep/tests/test_collect_results.py && python hp_sweep/tests/test_make_plot.py
```

Esperado: los seis imprimen su `>>> ... OK <<<`.

- [ ] **Step 6: Commit**

```bash
git add docs/Runs/RESULTS.md docs/superpowers/specs/2026-08-07-estudio-hiperparametros-e3-design.md
git commit -m "exp I: seccion placeholder en RESULTS.md + spec actualizado a 23 corridas"
```

---

## Checklist final para Lucas (lo que queda por hacer a mano)

Nada de esto lo puede hacer Claude Code — requiere SSH al cluster.

1. Clonar/actualizar el repo en login-1 y ajustar la ruta del `cd` en `run_sweep.sh` si difiere.
2. Correr los dos smoke tests locales (`test_make_configs.py`, `test_all_archs_forward.py`) — ya
   corridos en la PC, repetirlos en el cluster solo si el entorno difiere.
3. Lanzar: `for cfg in hp_sweep/configs/*.yaml; do sbatch run_sweep.sh "$cfg"; done`
4. Bajar los `expE3_hp_*.out` y correr `python hp_sweep/collect_results.py --out-dir <ruta>`.
5. Pegar la tabla en la sección del Exp I de `RESULTS.md` y correr `python hp_sweep/make_plot.py`.
6. Escribir el takeaway con lo que salga — incluido si es una meseta.
