# Exp E — Fase 1: Extracción y Validación de Picos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convertir el HSQC de imagen (2×256×256, 99.2% vacía) a un conjunto de
picos `(δC, δH, amp_ch0, amp_ch1)` por molécula vía blob-detection sobre el h5
existente, y validar cuánta información se pierde en la conversión (colisiones
reales entre carbonos distintos) comparando contra el label de 19 clases.

**Architecture:** Pipeline en 3 pasos sobre `experiments/E_peaks_prep/`:
(1) calibración pixel↔ppm y detección de blobs por conectividad-8, funciones
puras testeables con numpy sintético; (2) orquestador que recorre el h5 en
chunks, extrae picos por molécula y escribe un h5 nuevo (`peaks_202465.h5`) con
padding + máscara; (3) reporte de validación que compara conteo de blobs vs
conteo visible del label. No se entrena ningún modelo en esta fase.

**Tech Stack:** Python, numpy, scipy.ndimage (blob detection), h5py (I/O del
dataset), pyyaml (config). Sin torch — esta fase es puro procesamiento de datos,
corre en el login node sin GPU.

## Global Constraints

- `num_workers: 0` no aplica acá (no hay DataLoader/training en esta fase).
- Nada hardcodeado: calibración (`C13_PPM_MIN/MAX`, `H1_PPM_MIN/MAX`) y rutas
  salen de config, con los valores exactos copiados de
  `E:\Proyectos\SciTrix\ScitrixDB\DB-Batch0\Genera mapas de pkl v2.py`
  (`C13_PPM_MIN=0, C13_PPM_MAX=220, H1_PPM_MIN=-1, H1_PPM_MAX=15`, resolución 256).
- Encoding UTF-8 en todo archivo nuevo del cluster (heredoc con comillas).
- Smoke test obligatorio antes de correr sobre el h5 completo (202465 moléculas)
  — ver Task 6.
- `classes_19v` y el orden de `config/db.yaml` son fijos — `validate_peaks.py`
  los usa tal cual, sin reordenar.
- Carpeta `experiments/E_peaks_prep/` autocontenida: config propio (un solo
  `config.yaml`, mismo esquema que usan los experimentos anteriores — ver
  `experiments/C_gap/config.yaml` como referencia), no depende de imports
  relativos a `config/db.yaml` en tiempo de ejecución en el cluster.
- Entorno de desarrollo local: tiene numpy, rdkit, pandas — **NO tiene scipy,
  h5py, pyyaml, pytest**. Las tareas que solo usan numpy (Task 2, partes de
  Task 5) se pueden correr y verificar de verdad localmente. Las que dependen
  de scipy/h5py (Task 3, Task 4, partes de Task 5 y Task 6) se verifican
  localmente solo por revisión de código (`ast.parse` + lectura manual) y las
  corre Lucas en el cluster — esto se declara explícito en cada task, no se
  finge una ejecución que no ocurrió.

---

### Task 1: Scaffold de la carpeta + calibración en config/db.yaml + RATIONALE.md

**Files:**
- Create: `experiments/E_peaks_prep/` (carpeta)
- Create: `experiments/E_peaks_prep/RATIONALE.md`
- Create: `experiments/E_peaks_prep/config.yaml`
- Modify: `config/db.yaml` (agregar bloque `hsqc_calibration`)

**Interfaces:**
- Produces: bloque `hsqc_calibration` en `config/db.yaml` con claves
  `c13_ppm_min`, `c13_ppm_max`, `h1_ppm_min`, `h1_ppm_max`, `resolution` — estas
  claves son las que leen todos los scripts de tasks siguientes vía su propio
  `config.yaml` (self-contained, valores copiados acá, no importado en runtime).

- [ ] **Step 1: Agregar calibración a `config/db.yaml`**

Insertar después del bloque `model:` (línea 28, antes de `classes_19v:`):

```yaml
hsqc_calibration:          # copiado de Genera_mapas_de_pkl_v2.py — no inferido
  c13_ppm_min: 0
  c13_ppm_max: 220
  h1_ppm_min: -1
  h1_ppm_max: 15
  resolution: 256
```

- [ ] **Step 2: Crear `experiments/E_peaks_prep/config.yaml`**

```yaml
# experiments/E_peaks_prep/config.yaml
#
# Exp E Fase 1: extraccion de picos por blob-detection desde el h5 de
# imagenes existente + validacion contra el label de 19 clases. No entrena
# ningun modelo. Calibracion copiada de Genera_mapas_de_pkl_v2.py (fuera de
# este repo, script original del dataset) -- ver RATIONALE.md.

paths:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  h5_filename: "nmr_dataset_v3_202465_fast.h5"
  labels_filename: "vectors_13c_19v_202465.npy"
  peaks_output_filename: "peaks_202465.h5"

hsqc_calibration:
  c13_ppm_min: 0
  c13_ppm_max: 220
  h1_ppm_min: -1
  h1_ppm_max: 15
  resolution: 256

extraction:
  chunk_size: 1000   # moleculas por lectura del h5 (evita cargar 202465x2x256x256 en RAM)

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

- [ ] **Step 3: Crear `experiments/E_peaks_prep/RATIONALE.md`**

```markdown
# Exp E — Fase 1: Extracción y Validación de Picos — Rationale

## Por qué

Exp C (GAP) mejoró la EMA cruda levemente pero las confusiones de clase
(`Cqsp2`↔`=CH/Ar`, `CH2`↔`CH2-N`) persisten idénticas en V10, Exp B y Exp C —
tres arquitecturas distintas. Evidencia de que el cuello de botella es de
representación, no arquitectónico. La auditoría de pipeline
(`scripts/audit_data_pipeline.py`) mostró que la imagen HSQC es 99.2% espacio
vacío. Esta fase reemplaza la imagen por una lista compacta de picos reales.

## Calibración

Encontrada en `E:\Proyectos\SciTrix\ScitrixDB\DB-Batch0\Genera mapas de pkl v2.py`
(script original del dataset, fuera de este repo, solo consultado como
referencia — no se ejecuta ni se modifica):

- δC: `[0, 220]` ppm, binning uniforme, 256 bins.
- δH: `[-1, 15]` ppm, binning uniforme, 256 bins.
- Canal 0 (fila=δC, columna=δH) = gaussiana DEPT, sigma=0.5, escalada por N_H
  (CH2 negativo, CH/CH3 positivo).
- Canal 1 = tipo de carbono normalizado (CH=0.33, CH2=0.67, CH3=1.0).
- Los H de un mismo carbono se pintan en el mismo pixel (no es colisión real).
  La colisión real es cuando dos carbonos DISTINTOS caen tan cerca que sus
  gaussianas se funden en un blob — eso es lo que valida esta fase.

## Alcance

Solo extracción + validación. No hay modelo de conjuntos todavía (Fase 2,
spec separado, depende de qué tan limpia salga esta extracción).

Ver el spec completo: `docs/superpowers/specs/2026-07-21-exp-e-fase1-extraccion-picos-design.md`.
```

- [ ] **Step 4: Commit**

```bash
git add config/db.yaml experiments/E_peaks_prep/
git commit -m "exp-e: scaffold fase 1 (calibracion + config + rationale)"
```

---

### Task 2: `calibration.py` — conversión pixel↔ppm (TDD, corre localmente)

**Files:**
- Create: `experiments/E_peaks_prep/calibration.py`
- Test: `experiments/E_peaks_prep/tests/test_calibration.py`

**Interfaces:**
- Produces: `bin_to_ppm(bin_idx, ppm_min, ppm_max, resolution=256) -> float`.
  Usado por `extract_peaks.py` (Task 4) para convertir centroides de pixel a
  ppm reales.

- [ ] **Step 1: Escribir el test que falla**

```python
# experiments/E_peaks_prep/tests/test_calibration.py
# coding: ascii
"""Tests de calibration.py -- corre localmente, solo depende de numpy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration import bin_to_ppm


def test_bin_to_ppm_extremes():
    # C13: [0, 220], resolution=256 -> bin 0 = 0 ppm, bin 255 = 220 ppm
    assert abs(bin_to_ppm(0, 0, 220, 256) - 0.0) < 1e-6
    assert abs(bin_to_ppm(255, 0, 220, 256) - 220.0) < 1e-6
    print("[OK] test_bin_to_ppm_extremes")


def test_bin_to_ppm_midpoint():
    # H1: [-1, 15], bin 127.5 (centroide fraccionario) -> punto medio del rango
    mid = bin_to_ppm(127.5, -1, 15, 256)
    expected = -1 + (127.5 / 255) * 16
    assert abs(mid - expected) < 1e-6
    print(f"[OK] test_bin_to_ppm_midpoint ({mid:.4f})")


def test_bin_to_ppm_roundtrip_against_generator_forward():
    # ppm_to_bin_uniform tal como esta en Genera_mapas_de_pkl_v2.py (copiado
    # aca SOLO para este test, no es parte del script de produccion).
    import numpy as np

    def ppm_to_bin_uniform(ppm, ppm_min, ppm_max, resolution):
        ppm_clamped = max(ppm_min, min(ppm_max, ppm))
        normalized = (ppm_clamped - ppm_min) / (ppm_max - ppm_min)
        return int(np.clip(normalized * (resolution - 1), 0, resolution - 1))

    for ppm in (0.0, 55.3, 110.0, 219.9):
        b = ppm_to_bin_uniform(ppm, 0, 220, 256)
        back = bin_to_ppm(b, 0, 220, 256)
        # El roundtrip no es exacto (binning pierde resolucion sub-bin), pero
        # debe caer dentro de un bin de distancia (~0.86 ppm para C13/256).
        assert abs(back - ppm) < (220 / 255) + 1e-6, (ppm, b, back)
    print("[OK] test_bin_to_ppm_roundtrip_against_generator_forward")


if __name__ == "__main__":
    test_bin_to_ppm_extremes()
    test_bin_to_ppm_midpoint()
    test_bin_to_ppm_roundtrip_against_generator_forward()
    print("\n>>> test_calibration.py OK <<<")
```

- [ ] **Step 2: Correr el test y confirmar que falla**

Run: `python experiments/E_peaks_prep/tests/test_calibration.py`
Expected: `ModuleNotFoundError: No module named 'calibration'` (el archivo
todavía no existe).

- [ ] **Step 3: Implementar `calibration.py`**

```python
# experiments/E_peaks_prep/calibration.py
# coding: ascii
"""
calibration.py -- conversion pixel (bin) -> ppm real para el HSQC.

Inversa exacta del binning uniforme usado en Genera_mapas_de_pkl_v2.py
(ppm_to_bin_uniform). bin_idx puede ser int (pixel entero) o float
(centroide de un blob).
"""


def bin_to_ppm(bin_idx, ppm_min, ppm_max, resolution=256):
    return bin_idx / (resolution - 1) * (ppm_max - ppm_min) + ppm_min
```

- [ ] **Step 4: Correr el test y confirmar que pasa**

Run: `python experiments/E_peaks_prep/tests/test_calibration.py`
Expected:
```
[OK] test_bin_to_ppm_extremes
[OK] test_bin_to_ppm_midpoint (7.0000)
[OK] test_bin_to_ppm_roundtrip_against_generator_forward

>>> test_calibration.py OK <<<
```

- [ ] **Step 5: Commit**

```bash
git add experiments/E_peaks_prep/calibration.py experiments/E_peaks_prep/tests/test_calibration.py
git commit -m "exp-e: calibration.py (bin_to_ppm) con tests locales"
```

---

### Task 3: `blob_detect.py` — detección de picos por conectividad (TDD, requiere scipy)

**Files:**
- Create: `experiments/E_peaks_prep/blob_detect.py`
- Test: `experiments/E_peaks_prep/tests/test_blob_detect.py`

**Interfaces:**
- Consumes: nada de tasks anteriores (función pura sobre arrays numpy).
- Produces: `detect_peaks(ch0, ch1) -> list[tuple[float, float, float, float]]`
  — lista de `(row_c, col_h, amp_ch0, amp_ch1)` en coordenadas de pixel. Usado
  por `extract_peaks.py` (Task 4).

**Nota de entorno:** este test requiere `scipy` — no disponible en la máquina
local. Se escribe completo igual, y se verifica localmente solo por revisión
de código (no se afirma haberlo corrido). Lucas lo corre en el cluster como
parte del smoke test de Task 6.

- [ ] **Step 1: Escribir el test (no se puede correr localmente — falta scipy)**

```python
# experiments/E_peaks_prep/tests/test_blob_detect.py
# coding: ascii
"""Tests de blob_detect.py -- requiere scipy, correr en el cluster (login node)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from blob_detect import detect_peaks


def _gaussian_blob(matrix, row, col, sigma=0.5, intensity=1.0):
    size = matrix.shape[0]
    radius = int(6 * sigma) + 1
    r0, r1 = max(0, row - radius), min(size, row + radius + 1)
    c0, c1 = max(0, col - radius), min(size, col + radius + 1)
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    g = intensity * np.exp(-0.5 * (((rr - row) / sigma) ** 2 + ((cc - col) / sigma) ** 2))
    matrix[r0:r1, c0:c1] += g
    return matrix


def test_two_separated_peaks_detected_individually():
    ch0 = np.zeros((256, 256), dtype=np.float32)
    ch1 = np.zeros((256, 256), dtype=np.float32)
    _gaussian_blob(ch0, row=50, col=60, intensity=1.0)
    _gaussian_blob(ch1, row=50, col=60, intensity=0.33)
    _gaussian_blob(ch0, row=200, col=210, intensity=-2.0)  # CH2, fase negativa
    _gaussian_blob(ch1, row=200, col=210, intensity=0.67)

    peaks = detect_peaks(ch0, ch1)
    assert len(peaks) == 2, f"esperados 2 picos, salieron {len(peaks)}"

    peaks_sorted = sorted(peaks, key=lambda p: p[0])
    r0, c0, a0_ch0, a0_ch1 = peaks_sorted[0]
    assert abs(r0 - 50) < 1.0 and abs(c0 - 60) < 1.0
    assert a0_ch0 > 0  # CH/CH3, fase positiva

    r1, c1, a1_ch0, a1_ch1 = peaks_sorted[1]
    assert abs(r1 - 200) < 1.0 and abs(c1 - 210) < 1.0
    assert a1_ch0 < 0  # CH2, fase negativa
    print(f"[OK] test_two_separated_peaks_detected_individually -> {peaks_sorted}")


def test_two_adjacent_peaks_merge_into_one_blob():
    # Dos carbonos distintos a 1 pixel de distancia: sus gaussianas (radio
    # ~4px) se solapan y scipy.ndimage.label los ve como UN componente
    # conexo -- esto es la "colision real" que Fase 1 tiene que poder medir.
    ch0 = np.zeros((256, 256), dtype=np.float32)
    ch1 = np.zeros((256, 256), dtype=np.float32)
    _gaussian_blob(ch0, row=100, col=100, intensity=1.0)
    _gaussian_blob(ch0, row=101, col=101, intensity=1.0)

    peaks = detect_peaks(ch0, ch1)
    assert len(peaks) == 1, f"esperado 1 blob fusionado, salieron {len(peaks)}"
    print(f"[OK] test_two_adjacent_peaks_merge_into_one_blob -> {peaks}")


def test_empty_image_returns_no_peaks():
    ch0 = np.zeros((256, 256), dtype=np.float32)
    ch1 = np.zeros((256, 256), dtype=np.float32)
    peaks = detect_peaks(ch0, ch1)
    assert peaks == []
    print("[OK] test_empty_image_returns_no_peaks")


if __name__ == "__main__":
    test_two_separated_peaks_detected_individually()
    test_two_adjacent_peaks_merge_into_one_blob()
    test_empty_image_returns_no_peaks()
    print("\n>>> test_blob_detect.py OK <<<")
```

- [ ] **Step 2: Confirmar que fallaría (revisión, no ejecución local)**

`blob_detect.py` todavía no existe -> `ModuleNotFoundError`. No se ejecuta
localmente (falta scipy); se documenta el resultado esperado para que Lucas
lo confirme en el cluster.

- [ ] **Step 3: Implementar `blob_detect.py`**

```python
# experiments/E_peaks_prep/blob_detect.py
# coding: ascii
"""
blob_detect.py -- deteccion de picos HSQC via componentes conexos.

Un pico = un componente conexo (conectividad 8) de pixeles no-cero en el
canal 0 (DEPT). Si dos carbonos distintos caen tan cerca en (delta_C,
delta_H) que sus gaussianas se solapan, ndimage.label los funde en un solo
componente -- esa fusion es la "colision real" que valida Fase 1 (ver
validate_peaks.py). El centroide se pondera por |canal0| dentro del blob,
mas robusto que el centroide sin ponderar para blobs asimetricos/fusionados.
"""
import numpy as np
from scipy import ndimage

CONNECTIVITY_8 = np.ones((3, 3), dtype=int)


def detect_peaks(ch0, ch1):
    """ch0, ch1: arrays (H, W) float, canales de una sola molecula.
    Devuelve lista de (row_c, col_h, amp_ch0, amp_ch1) en coordenadas de
    pixel -- row_c/col_h son floats (centroide), amp_ch0/amp_ch1 son los
    valores de cada canal en el pixel entero mas cercano al centroide."""
    mask = ch0 != 0
    labeled, n_blobs = ndimage.label(mask, structure=CONNECTIVITY_8)
    if n_blobs == 0:
        return []

    indices = list(range(1, n_blobs + 1))
    centroids = ndimage.center_of_mass(np.abs(ch0), labeled, indices)
    if n_blobs == 1:
        centroids = [centroids]

    h, w = ch0.shape
    peaks = []
    for row_c, col_h in centroids:
        r = min(max(int(round(row_c)), 0), h - 1)
        c = min(max(int(round(col_h)), 0), w - 1)
        peaks.append((float(row_c), float(col_h), float(ch0[r, c]), float(ch1[r, c])))
    return peaks
```

- [ ] **Step 4: Confirmar que pasaría (revisión de código)**

Se revisa a mano contra los 3 casos del test: dos blobs separados por >8px no
se tocan (conectividad-8 con radio de gaussiana ~4px) -> 2 componentes; dos
centros a distancia `sqrt(2)` sí se tocan -> 1 componente; imagen vacía ->
`n_blobs=0` -> `[]`. Lucas confirma la ejecución real en el cluster (Task 6).

- [ ] **Step 5: Commit**

```bash
git add experiments/E_peaks_prep/blob_detect.py experiments/E_peaks_prep/tests/test_blob_detect.py
git commit -m "exp-e: blob_detect.py (deteccion de picos via componentes conexos)"
```

---

### Task 4: `extract_peaks.py` — orquestador principal (escribe `peaks_202465.h5`)

**Files:**
- Create: `experiments/E_peaks_prep/extract_peaks.py`
- Test: `experiments/E_peaks_prep/tests/test_padding.py`

**Interfaces:**
- Consumes: `calibration.bin_to_ppm` (Task 2), `blob_detect.detect_peaks` (Task 3).
- Produces: `extract_peaks_from_molecule(hsqc, calib) -> list[tuple[float,float,float,float]]`
  y `build_padded_arrays(peaks_per_molecule) -> (peaks_array, mask_array)` —
  `build_padded_arrays` es pura numpy, testeable localmente. Escribe
  `peaks_202465.h5` con datasets `peaks (N, max_peaks, 4)` float32 y
  `peaks_mask (N, max_peaks)` bool — usado por `validate_peaks.py` (Task 5).

- [ ] **Step 1: Escribir el test de `build_padded_arrays` (corre localmente, solo numpy)**

```python
# experiments/E_peaks_prep/tests/test_padding.py
# coding: ascii
"""Test de build_padded_arrays -- corre localmente, solo numpy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from extract_peaks import build_padded_arrays


def test_build_padded_arrays_shapes_and_mask():
    peaks_per_molecule = [
        [(55.0, 1.2, 1.0, 0.33)],
        [(55.0, 1.2, 1.0, 0.33), (110.0, 4.5, -2.0, 0.67)],
        [],  # molecula sin picos detectados (caso limite, ej. imagen vacia)
    ]
    peaks_array, mask_array = build_padded_arrays(peaks_per_molecule)

    assert peaks_array.shape == (3, 2, 4), peaks_array.shape
    assert mask_array.shape == (3, 2), mask_array.shape
    assert mask_array.dtype == bool

    assert mask_array[0].tolist() == [True, False]
    assert mask_array[1].tolist() == [True, True]
    assert mask_array[2].tolist() == [False, False]

    assert np.allclose(peaks_array[0, 0], [55.0, 1.2, 1.0, 0.33])
    assert np.allclose(peaks_array[1, 1], [110.0, 4.5, -2.0, 0.67])
    # Filas con mask=False deben quedar en cero (padding)
    assert np.allclose(peaks_array[0, 1], [0.0, 0.0, 0.0, 0.0])
    print("[OK] test_build_padded_arrays_shapes_and_mask")


def test_build_padded_arrays_empty_input():
    peaks_array, mask_array = build_padded_arrays([])
    assert peaks_array.shape == (0, 0, 4)
    assert mask_array.shape == (0, 0)
    print("[OK] test_build_padded_arrays_empty_input")


if __name__ == "__main__":
    test_build_padded_arrays_shapes_and_mask()
    test_build_padded_arrays_empty_input()
    print("\n>>> test_padding.py OK <<<")
```

- [ ] **Step 2: Correr el test y confirmar que falla**

Run: `python experiments/E_peaks_prep/tests/test_padding.py`
Expected: `ModuleNotFoundError: No module named 'extract_peaks'`

- [ ] **Step 3: Implementar `extract_peaks.py`**

```python
# experiments/E_peaks_prep/extract_peaks.py
# coding: ascii
"""
extract_peaks.py -- Exp E Fase 1: convierte el HSQC de imagen a un conjunto
de picos (delta_C, delta_H, amp_ch0, amp_ch1) por molecula, via
blob-detection (blob_detect.py) + calibracion pixel->ppm (calibration.py).

Recorre el h5 de imagenes en chunks (config: extraction.chunk_size) para no
cargar las 202465 imagenes completas en RAM (2x256x256 float32 x 202465 ~=
106 GB sin comprimir). Los picos extraidos son mucho mas chicos que las
imagenes, asi que se acumulan enteros en memoria y el padding se calcula
al final, en una sola pasada sobre el h5.

Uso (en el cluster, login node, sin GPU):
    python extract_peaks.py --config config.yaml
"""
import argparse
from pathlib import Path

import numpy as np

from blob_detect import detect_peaks
from calibration import bin_to_ppm


def extract_peaks_from_molecule(hsqc, calib):
    """hsqc: (2, H, W) float array de una sola molecula. calib: dict con
    c13_ppm_min/max, h1_ppm_min/max, resolution. Devuelve lista de
    (delta_c, delta_h, amp_ch0, amp_ch1)."""
    ch0, ch1 = hsqc[0], hsqc[1]
    raw_peaks = detect_peaks(ch0, ch1)
    resolution = calib["resolution"]
    out = []
    for row_c, col_h, amp0, amp1 in raw_peaks:
        delta_c = bin_to_ppm(row_c, calib["c13_ppm_min"], calib["c13_ppm_max"], resolution)
        delta_h = bin_to_ppm(col_h, calib["h1_ppm_min"], calib["h1_ppm_max"], resolution)
        out.append((delta_c, delta_h, amp0, amp1))
    return out


def build_padded_arrays(peaks_per_molecule):
    """peaks_per_molecule: lista de N listas de tuplas de 4 floats.
    Devuelve (peaks_array (N, max_peaks, 4) float32, mask_array (N, max_peaks) bool)."""
    n = len(peaks_per_molecule)
    max_peaks = max((len(p) for p in peaks_per_molecule), default=0)
    peaks_array = np.zeros((n, max_peaks, 4), dtype=np.float32)
    mask_array = np.zeros((n, max_peaks), dtype=bool)
    for i, peaks in enumerate(peaks_per_molecule):
        for j, peak in enumerate(peaks):
            peaks_array[i, j] = peak
            mask_array[i, j] = True
    return peaks_array, mask_array


def main(config_path):
    import h5py
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(cfg["paths"]["base_dir"])
    h5_path = base_dir / cfg["paths"]["h5_filename"]
    out_path = base_dir / cfg["paths"]["peaks_output_filename"]
    chunk_size = int(cfg["extraction"]["chunk_size"])
    calib = cfg["hsqc_calibration"]

    print("=" * 60)
    print("  EXP E FASE 1: extraccion de picos")
    print("=" * 60)
    print(f"-> Leyendo: {h5_path}")

    peaks_per_molecule = []
    with h5py.File(h5_path, "r") as f:
        n_total = f["hsqc"].shape[0]
        print(f"-> Moleculas totales: {n_total}")
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk = f["hsqc"][start:end]  # (chunk, 2, 256, 256)
            for i in range(chunk.shape[0]):
                peaks_per_molecule.append(extract_peaks_from_molecule(chunk[i], calib))
            print(f"   procesadas {end}/{n_total}", end="\r")
    print()

    peaks_array, mask_array = build_padded_arrays(peaks_per_molecule)
    n_counts = mask_array.sum(axis=1)
    print(f"-> max_peaks detectado: {peaks_array.shape[1]}")
    print(f"-> picos por molecula: min={n_counts.min()} max={n_counts.max()} "
          f"promedio={n_counts.mean():.2f}")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("peaks", data=peaks_array, compression="lzf")
        f.create_dataset("peaks_mask", data=mask_array, compression="lzf")
        f.attrs["peak_fields"] = "delta_c_ppm,delta_h_ppm,amp_ch0,amp_ch1"
        f.attrs["source_h5"] = str(h5_path)
        for k, v in calib.items():
            f.attrs[f"calib_{k}"] = v

    print(f"\n[SAVE] {out_path}")
    print(">>> EXP E FASE 1 extract_peaks.py OK <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp E Fase 1: extraccion de picos")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
```

- [ ] **Step 4: Correr el test y confirmar que pasa**

Run: `python experiments/E_peaks_prep/tests/test_padding.py`
Expected:
```
[OK] test_build_padded_arrays_shapes_and_mask
[OK] test_build_padded_arrays_empty_input

>>> test_padding.py OK <<<
```

Nota: este `python` de import también importa `blob_detect.py` (que
requiere scipy) por la línea `from blob_detect import detect_peaks` en
`extract_peaks.py`. Si el intérprete local no tiene scipy, este test
tampoco corre localmente pese a que `build_padded_arrays` en sí es pura
numpy — se documenta como limitación de entorno, no se finge ejecución.
Lucas lo confirma en el cluster junto con Task 6.

- [ ] **Step 5: Commit**

```bash
git add experiments/E_peaks_prep/extract_peaks.py experiments/E_peaks_prep/tests/test_padding.py
git commit -m "exp-e: extract_peaks.py (orquestador, escribe peaks_202465.h5)"
```

---

### Task 5: `validate_peaks.py` — reporte de validación (blobs vs label)

**Files:**
- Create: `experiments/E_peaks_prep/validate_peaks.py`
- Test: `experiments/E_peaks_prep/tests/test_validation_report.py`

**Interfaces:**
- Consumes: `peaks_202465.h5` (dataset `peaks_mask`) producido por Task 4,
  `vectors_13c_19v_202465.npy` (label existente, sin cambios), `classes_19v`
  del config.
- Produces: `visible_label_counts(labels, class_names) -> np.ndarray (N,)`,
  `blob_counts_from_mask(peaks_mask) -> np.ndarray (N,)`,
  `validation_report(blob_counts, visible_counts) -> dict` — las 3 son puras
  numpy, testeables localmente.

- [ ] **Step 1: Escribir el test (corre localmente, solo numpy)**

```python
# experiments/E_peaks_prep/tests/test_validation_report.py
# coding: ascii
"""Tests de validate_peaks.py -- corre localmente, solo numpy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from validate_peaks import (
    blob_counts_from_mask,
    validation_report,
    visible_label_counts,
)

CLASS_NAMES = [
    "CH3", "CH2", "CH", "Cq", "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N", "=CH2", "=CH/Ar", "Cqsp2",
    "Aldeh", "Imina", "C-2X", "C-3X",
]


def test_visible_label_counts_excludes_invisible_classes():
    labels = np.zeros((2, 19), dtype=int)
    labels[0, CLASS_NAMES.index("CH3")] = 3       # visible
    labels[0, CLASS_NAMES.index("Cq")] = 5         # invisible, no debe contar
    labels[1, CLASS_NAMES.index("Cqsp2")] = 4      # invisible, no debe contar
    labels[1, CLASS_NAMES.index("CH")] = 2         # visible

    counts = visible_label_counts(labels, CLASS_NAMES)
    assert counts.tolist() == [3, 2], counts.tolist()
    print("[OK] test_visible_label_counts_excludes_invisible_classes")


def test_blob_counts_from_mask():
    mask = np.array([
        [True, True, False],
        [True, False, False],
        [False, False, False],
    ])
    counts = blob_counts_from_mask(mask)
    assert counts.tolist() == [2, 1, 0], counts.tolist()
    print("[OK] test_blob_counts_from_mask")


def test_validation_report_exact_match_and_collision():
    blob_counts = np.array([3, 1, 5])
    visible_counts = np.array([3, 2, 5])  # molecula 1: deficit=1 (colision real)

    report = validation_report(blob_counts, visible_counts)
    assert report["n"] == 3
    assert abs(report["pct_exact_match"] - (2 / 3 * 100.0)) < 1e-6
    assert report["n_collision"] == 1
    assert abs(report["mean_deficit_positive"] - 1.0) < 1e-6
    print(f"[OK] test_validation_report_exact_match_and_collision -> {report}")


if __name__ == "__main__":
    test_visible_label_counts_excludes_invisible_classes()
    test_blob_counts_from_mask()
    test_validation_report_exact_match_and_collision()
    print("\n>>> test_validation_report.py OK <<<")
```

- [ ] **Step 2: Correr el test y confirmar que falla**

Run: `python experiments/E_peaks_prep/tests/test_validation_report.py`
Expected: `ModuleNotFoundError: No module named 'validate_peaks'`

- [ ] **Step 3: Implementar `validate_peaks.py`**

```python
# experiments/E_peaks_prep/validate_peaks.py
# coding: ascii
"""
validate_peaks.py -- Exp E Fase 1: valida que la extraccion de picos
(extract_peaks.py) no pierda informacion respecto al label de 19 clases.

Compara, por molecula, el numero de picos detectados (peaks_mask.sum) contra
el conteo VISIBLE del label (excluyendo Cq/Cq-O/Cq-N/Cqsp2 -- mismo criterio
que scripts/audit_data_pipeline.py). Un deficit positivo (visible > blobs)
es una colision real: dos o mas carbonos distintos cuyas gaussianas se
fusionaron en un solo componente conexo.

Uso (en el cluster, login node, sin GPU):
    python validate_peaks.py --config config.yaml
"""
import argparse
from pathlib import Path

import numpy as np

INVISIBLE_CLASSES = ["Cq", "Cq-O", "Cq-N", "Cqsp2"]


def visible_label_counts(labels, class_names):
    idx_invisible = [class_names.index(c) for c in INVISIBLE_CLASSES]
    idx_visible = [i for i in range(len(class_names)) if i not in idx_invisible]
    return labels[:, idx_visible].sum(axis=1).astype(int)


def blob_counts_from_mask(peaks_mask):
    return peaks_mask.sum(axis=1).astype(int)


def validation_report(blob_counts, visible_counts):
    deficit = visible_counts.astype(int) - blob_counts.astype(int)
    n = len(deficit)
    n_exact = int((deficit == 0).sum())
    n_collision = int((deficit > 0).sum())
    mean_deficit_positive = float(deficit[deficit > 0].mean()) if n_collision > 0 else 0.0
    return {
        "n": n,
        "pct_exact_match": n_exact / n * 100.0,
        "n_collision": n_collision,
        "pct_collision": n_collision / n * 100.0,
        "mean_deficit_positive": mean_deficit_positive,
        "deficit": deficit,
    }


def main(config_path):
    import h5py
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(cfg["paths"]["base_dir"])
    labels_path = base_dir / cfg["paths"]["labels_filename"]
    peaks_path = base_dir / cfg["paths"]["peaks_output_filename"]
    class_names = cfg["classes_19v"]

    print("=" * 60)
    print("  EXP E FASE 1: validacion de picos vs label")
    print("=" * 60)

    labels = np.load(labels_path).astype(int)
    with h5py.File(peaks_path, "r") as f:
        peaks_mask = f["peaks_mask"][:]

    visible_counts = visible_label_counts(labels, class_names)
    blob_counts = blob_counts_from_mask(peaks_mask)
    report = validation_report(blob_counts, visible_counts)

    print(f"\nMoleculas evaluadas: {report['n']}")
    print(f"Match exacto (blobs == visible): {report['pct_exact_match']:.2f}%")
    print(f"Con colision (visible > blobs): {report['n_collision']} "
          f"({report['pct_collision']:.2f}%)")
    print(f"Deficit promedio en las que tienen colision: "
          f"{report['mean_deficit_positive']:.2f}")

    deficit = report["deficit"]
    worst_idx = np.argsort(deficit)[::-1][:3]
    print("\nEjemplos con mayor colision (para inspeccion manual):")
    for idx in worst_idx:
        if deficit[idx] <= 0:
            break
        print(f"  molecula {idx}: blobs={blob_counts[idx]} "
              f"visible_label={visible_counts[idx]} deficit={deficit[idx]}")

    print("\n>>> EXP E FASE 1 validate_peaks.py OK <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp E Fase 1: validacion de picos")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
```

- [ ] **Step 4: Correr el test y confirmar que pasa**

Run: `python experiments/E_peaks_prep/tests/test_validation_report.py`
Expected:
```
[OK] test_visible_label_counts_excludes_invisible_classes
[OK] test_blob_counts_from_mask
[OK] test_validation_report_exact_match_and_collision -> {...}

>>> test_validation_report.py OK <<<
```

Este test SÍ corre completo localmente: `validate_peaks.py` no importa
`blob_detect.py` (no necesita scipy), solo numpy — a diferencia de Task 4.

- [ ] **Step 5: Commit**

```bash
git add experiments/E_peaks_prep/validate_peaks.py experiments/E_peaks_prep/tests/test_validation_report.py
git commit -m "exp-e: validate_peaks.py (reporte blobs vs label visible)"
```

---

### Task 6: Smoke test end-to-end + README.md

**Files:**
- Create: `experiments/E_peaks_prep/tests/test_smoke.py`
- Create: `experiments/E_peaks_prep/README.md`

**Interfaces:**
- Consumes: `extract_peaks.extract_peaks_from_molecule`, `extract_peaks.build_padded_arrays`
  (Task 4), `validate_peaks.visible_label_counts`, `validate_peaks.blob_counts_from_mask`,
  `validate_peaks.validation_report` (Task 5).

- [ ] **Step 1: Escribir el smoke test (requiere scipy — no corre localmente)**

```python
# experiments/E_peaks_prep/tests/test_smoke.py
# coding: ascii
"""
Smoke test OFFLINE de Exp E Fase 1 (rule 5 de CLAUDE.md) -- construye 3
moleculas sinteticas en memoria (sin h5py, sin cluster), corre el pipeline
completo extract_peaks_from_molecule -> build_padded_arrays -> reporte de
validacion, y confirma que las formas y los conteos son consistentes.

Requiere scipy (via blob_detect.py) -- correr en el cluster (login node)
antes de correr extract_peaks.py sobre el h5 completo:
    python tests/test_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from extract_peaks import build_padded_arrays, extract_peaks_from_molecule
from validate_peaks import blob_counts_from_mask, validation_report, visible_label_counts

CALIB = {"c13_ppm_min": 0, "c13_ppm_max": 220, "h1_ppm_min": -1, "h1_ppm_max": 15, "resolution": 256}

CLASS_NAMES = [
    "CH3", "CH2", "CH", "Cq", "CH3-O", "CH2-O", "CH-O", "Cq-O",
    "CH3-N", "CH2-N", "CH-N", "Cq-N", "=CH2", "=CH/Ar", "Cqsp2",
    "Aldeh", "Imina", "C-2X", "C-3X",
]


def _gaussian_blob(matrix, row, col, sigma=0.5, intensity=1.0):
    size = matrix.shape[0]
    radius = int(6 * sigma) + 1
    r0, r1 = max(0, row - radius), min(size, row + radius + 1)
    c0, c1 = max(0, col - radius), min(size, col + radius + 1)
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    g = intensity * np.exp(-0.5 * (((rr - row) / sigma) ** 2 + ((cc - col) / sigma) ** 2))
    matrix[r0:r1, c0:c1] += g
    return matrix


def _make_molecule(peak_specs):
    """peak_specs: lista de (row, col, ch0_intensity, ch1_intensity)."""
    hsqc = np.zeros((2, 256, 256), dtype=np.float32)
    for row, col, i0, i1 in peak_specs:
        _gaussian_blob(hsqc[0], row, col, intensity=i0)
        _gaussian_blob(hsqc[1], row, col, intensity=i1)
    return hsqc


def test_pipeline_end_to_end_three_molecules():
    # Molecula 0: 1 CH3 (2 picos separados, imitando 2 carbonos distintos)
    mol0 = _make_molecule([(30, 40, 1.0, 1.0), (100, 120, 1.0, 1.0)])
    # Molecula 1: 1 CH2 (fase negativa) + colision deliberada (2 carbonos
    # a 1px de distancia se funden en un blob)
    mol1 = _make_molecule([(80, 90, -2.0, 0.67), (150, 150, 1.0, 1.0), (151, 151, 1.0, 1.0)])
    # Molecula 2: sin picos (imagen vacia)
    mol2 = np.zeros((2, 256, 256), dtype=np.float32)

    peaks_per_molecule = [
        extract_peaks_from_molecule(mol0, CALIB),
        extract_peaks_from_molecule(mol1, CALIB),
        extract_peaks_from_molecule(mol2, CALIB),
    ]
    assert len(peaks_per_molecule[0]) == 2
    assert len(peaks_per_molecule[1]) == 2  # 1 CH2 + 1 blob fusionado
    assert len(peaks_per_molecule[2]) == 0

    peaks_array, mask_array = build_padded_arrays(peaks_per_molecule)
    assert peaks_array.shape == (3, 2, 4)
    assert mask_array.tolist() == [[True, True], [True, True], [False, False]]

    # Validacion: labels sinteticos donde molecula 1 tiene 3 carbonos
    # visibles reales (el pipeline solo pudo recuperar 2 -> deficit=1)
    labels = np.zeros((3, 19), dtype=int)
    labels[0, CLASS_NAMES.index("CH3")] = 2
    labels[1, CLASS_NAMES.index("CH2")] = 1
    labels[1, CLASS_NAMES.index("CH")] = 2  # los 2 carbonos fusionados
    labels[2, CLASS_NAMES.index("Cq")] = 1  # invisible -> no cuenta

    visible_counts = visible_label_counts(labels, CLASS_NAMES)
    blob_counts = blob_counts_from_mask(mask_array)
    report = validation_report(blob_counts, visible_counts)

    assert report["n"] == 3
    assert report["n_collision"] == 1  # solo la molecula 1
    assert report["deficit"][1] == 1
    print(f"[OK] test_pipeline_end_to_end_three_molecules -> {report}")


if __name__ == "__main__":
    test_pipeline_end_to_end_three_molecules()
    print("\n>>> SMOKE EXP E FASE 1 OK - listo para correr extract_peaks.py sobre el h5 completo <<<")
```

- [ ] **Step 2: Confirmar (revisión de código, no ejecución — falta scipy localmente)**

Se revisa a mano: molécula 0 dos picos separados por 70+ pixeles → 2
componentes; molécula 1 tiene 1 CH2 aislado + 2 centros a `sqrt(2)` px →
según Task 3 eso funde en 1 componente → total 2 picos, pero el label real
tiene 3 carbonos visibles (1 CH2 + 2 CH fusionados) → deficit=1, confirmado
por `validation_report`. Lucas corre esto de verdad en el cluster.

- [ ] **Step 3: Crear `experiments/E_peaks_prep/README.md`**

```markdown
# Exp E — Fase 1: Extracción y Validación de Picos

Checklist para correr esto en el cluster. Es solo procesamiento de datos —
no usa GPU, no hay `sbatch`, corre directo en el login node.

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/E_peaks_prep`
3. Confirmar que existen en `/home/lpassaglia.iquir/DB_200k/`:
   - `nmr_dataset_v3_202465_fast.h5`
   - `vectors_13c_19v_202465.npy`
4. Smoke test obligatorio (regla 5 de CLAUDE.md) — corre en segundos, no
   toca el h5 real:
   ```bash
   python tests/test_calibration.py
   python tests/test_blob_detect.py
   python tests/test_padding.py
   python tests/test_validation_report.py
   python tests/test_smoke.py
   ```
   Todos deben terminar en `>>> ... OK <<<`. Si `test_blob_detect.py` o
   `test_smoke.py` fallan por `ModuleNotFoundError: scipy`, instalar scipy
   en el env (`conda install scipy` o `pip install scipy` dentro de
   `NMR_env`) antes de seguir.
5. Extraer los picos del dataset completo (202465 moléculas, corre en el
   login node, esperá que imprima el progreso — no debería tardar más que
   varios minutos, es procesamiento de imágenes en CPU, no entrenamiento):
   ```bash
   python extract_peaks.py --config config.yaml
   ```
   Al final debería mostrar `max_peaks detectado` y las stats de picos por
   molécula, y guardar `peaks_202465.h5` en `DB_200k/`.
6. Correr la validación:
   ```bash
   python validate_peaks.py --config config.yaml
   ```
7. Copiar la salida completa (match exacto %, % con colisión, deficit
   promedio, ejemplos) a `docs/Runs/RESULTS.md`, sección nueva "Exp E Fase 1".
8. Avisá a Claude Code con los números — con eso decidimos si la extracción
   por blob-detection es lo suficientemente limpia para pasar a la Fase 2
   (armar y entrenar el modelo de conjuntos), o si hace falta el fallback de
   reprocesar desde el pkl original.

## Nota

Esta fase no entrena nada ni usa GPU — es la validación previa a comprometer
horas de entrenamiento en una representación que todavía no se probó.
```

- [ ] **Step 4: Commit**

```bash
git add experiments/E_peaks_prep/tests/test_smoke.py experiments/E_peaks_prep/README.md
git commit -m "exp-e: smoke test end-to-end + README de ejecucion en cluster"
```

---

## Al terminar

Housekeeping final (mismo patrón que Exp C/D): actualizar
`docs/Runs/RESULTS.md` con una entrada "Exp E Fase 1" en cuanto Lucas reporte
los números reales del cluster, y decidir con esos datos si se escribe el
spec de Fase 2 (arquitectura DeepSets) o si hace falta ajustar la extracción
primero.
