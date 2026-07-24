# Exp G — Multi-vector (cobertura@K) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un generador de candidatos post-hoc que, a partir del output crudo del modelo Fase 3, emite hasta K vectores FM-consistentes por molécula, y una métrica de cobertura@K sobre el val congelado — para alimentar un generador de estructuras sin perder el vector verdadero.

**Architecture:** Puro post-procesamiento sobre el checkpoint existente (no reentrena). Desde el ancla FM-consistente (oráculo v2), se enumeran movimientos unitarios *intra-grupo-de-nH* (que preservan total, totales por grupo y cupo CH2 por construcción), se rankean por distancia L1 al crudo y se emiten los top-K. La métrica corre 100% local una vez dumpeado el output crudo.

**Tech Stack:** Python 3, numpy (generador y métrica, sin torch), pandas + pyarrow (parquet), rdkit (N/O desde SMILES), torch (solo el cambio de dump, corre en cluster/XPU).

## Global Constraints

- **numpy puro en el generador y la métrica** — `candidates.py`, `coverage.py` y sus tests NO importan torch (deben correr en la PC local sin GPU). Solo `dump_predictions.py` usa torch.
- **Carpeta autocontenida:** `experiments/G_multivector/` copia `oraculo.py` de E3, no lo importa (convención del proyecto: copiar, no importar entre experimentos).
- **Orden de clases 19v fijo** (config/db.yaml): `[CH3, CH2, CH, Cq, CH3-O, CH2-O, CH-O, Cq-O, CH3-N, CH2-N, CH-N, Cq-N, =CH2, =CH/Ar, Cqsp2, Aldeh, Imina, C-2X, C-3X]`. No reordenar.
- **Grupos de nH (multiplicidad), derivados de `Gen_vector.py`:** 3H=[0,4,8], 2H=[1,5,9,12], 1H=[2,6,10,13,15,16], 0H=[3,7,11,14,17,18]. El grupo 2H == `IDX_CH2` del oráculo.
- **Tests estilo proyecto:** cada test es un `.py` con `if __name__ == "__main__": _run()` que imprime `>>> N TESTS OK <<<`, ejecutable con `python tests/test_x.py` desde la carpeta del experimento (no requiere pytest).
- **Checkpoint de referencia:** el mejor actual (Set Transformer Fase 3 en Intel XPU/Clementina, EMA v2 92.14%). El dump de `y_pred_raw` sale de ese checkpoint.
- **Alcance v1:** solo intra-nH (techo de cobertura 98.7%). Cross-nH y reentrenar-con-distribución = fase 2, fuera de este plan.

---

### Task 1: Generador de candidatos intra-nH (`candidates.py`)

**Files:**
- Create: `experiments/G_multivector/oraculo.py` (copia byte-idéntica de `experiments/E3_dos_conjuntos/oraculo.py`)
- Create: `experiments/G_multivector/candidates.py`
- Test: `experiments/G_multivector/tests/test_candidates.py`

**Interfaces:**
- Consumes: de `oraculo.py` → `N_CLASSES`, `IDX_CH2`, `IDX_N`, `IDX_O`, `IDX_2X`, `IDX_3X`, `ajustar_conteo_hetero(pred_cruda, total_real, ch2_real, n_atoms, o_atoms) -> np.ndarray[int19]`.
- Produces: `NH_GROUPS: dict[int, list[int]]`; `generate_candidates(raw, total, ch2, n_atoms, o_atoms, K, max_swaps=2) -> list[np.ndarray]` (lista de vectores int de largo 19, largo ≤ K; el elemento [0] es SIEMPRE el oráculo v2; el resto rankeado por `sum|c-raw|` ascendente; sin duplicados).

- [ ] **Step 1: Copiar `oraculo.py` a la carpeta del experimento**

Run:
```bash
mkdir -p experiments/G_multivector/tests
cp experiments/E3_dos_conjuntos/oraculo.py experiments/G_multivector/oraculo.py
```
Expected: `experiments/G_multivector/oraculo.py` existe (numpy puro, con `ajustar_conteo_hetero`, `IDX_N`, `IDX_O`, `IDX_2X`, `IDX_3X`).

- [ ] **Step 2: Escribir el test que falla** (`experiments/G_multivector/tests/test_candidates.py`)

```python
# coding: ascii
"""Test del generador de candidatos intra-nH -- numpy PURO, corre local sin torch.
Correr:  python tests/test_candidates.py   (desde experiments/G_multivector)"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oraculo import IDX_CH2, ajustar_conteo_hetero
from candidates import generate_candidates, NH_GROUPS

N = 19


def _raw(**kv):
    v = np.zeros(N, dtype=np.float64)
    for i, val in kv.items():
        v[int(i)] = val
    return v


def _fm_ok(c, total, ch2):
    return c.sum() == total and sum(c[i] for i in IDX_CH2) == ch2


def test_top1_is_oraculo_v2():
    raw = _raw(**{1: 1.4, 9: 0.6, 0: 1.9})
    cands = generate_candidates(raw, total=4, ch2=2, n_atoms=1, o_atoms=1, K=3)
    v2 = ajustar_conteo_hetero(raw, 4, 2, 1, 1)
    assert np.array_equal(cands[0], v2), "el top-1 debe ser el oraculo v2"


def test_2h_split_genera_ambos_candidatos():
    # CH2=1.4 vs CH2-N=0.6, cupo 2 -> deben aparecer (CH2=2,CH2-N=0) y (CH2=1,CH2-N=1)
    raw = _raw(**{1: 1.4, 9: 0.6})
    cands = generate_candidates(raw, total=2, ch2=2, n_atoms=1, o_atoms=1, K=4)
    got = {(c[1], c[9]) for c in cands}
    assert (2, 0) in got and (1, 1) in got, f"faltan candidatos 2H: {got}"


def test_todos_fm_consistentes():
    rng = np.random.RandomState(0)
    for _ in range(100):
        raw = rng.rand(N) * 2.5
        total = int(round(raw.sum()))
        ch2 = int(round(sum(raw[i] for i in IDX_CH2)))
        if total < ch2:
            total = ch2
        cands = generate_candidates(raw, total, ch2, n_atoms=2, o_atoms=2, K=5)
        for c in cands:
            assert _fm_ok(c, total, ch2), f"candidato no FM-consistente: {c}"
            assert (c >= 0).all()


def test_no_puebla_clases_prohibidas():
    # n_atoms=0 -> ninguna clase -N poblada en ningun candidato
    raw = _raw(**{1: 1.1, 9: 0.9})
    cands = generate_candidates(raw, total=2, ch2=2, n_atoms=0, o_atoms=0, K=5)
    for c in cands:
        assert all(c[i] == 0 for i in [8, 9, 10, 11, 16]), f"poblo clase -N: {c}"


def test_sin_duplicados_y_len_max_K():
    raw = _raw(**{1: 1.4, 9: 0.6, 2: 1.5, 6: 0.5})
    cands = generate_candidates(raw, total=4, ch2=2, n_atoms=1, o_atoms=1, K=3)
    assert len(cands) <= 3
    keys = {c.tobytes() for c in cands}
    assert len(keys) == len(cands), "hay candidatos duplicados"


def test_ranking_por_L1_al_crudo():
    # el 2do candidato debe estar mas cerca en L1 del crudo que el 3ro
    raw = _raw(**{1: 1.4, 9: 0.6, 2: 1.4, 6: 0.6})
    cands = generate_candidates(raw, total=4, ch2=2, n_atoms=1, o_atoms=1, K=5)
    d = [np.abs(c - raw).sum() for c in cands[1:]]
    assert d == sorted(d), f"el resto no esta rankeado por L1: {d}"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n>>> {len(fns)} TESTS OK <<<")


if __name__ == "__main__":
    _run()
```

- [ ] **Step 3: Correr el test y verificar que falla**

Run: `cd experiments/G_multivector && python tests/test_candidates.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'candidates'`.

- [ ] **Step 4: Escribir la implementación mínima** (`experiments/G_multivector/candidates.py`)

```python
# coding: ascii
"""Generador de candidatos multi-vector (Exp G, cobertura@K). numpy PURO.

Desde el ancla FM-consistente (oraculo v2), enumera movimientos unitarios
INTRA-grupo-de-nH. Como el grupo 2H es exactamente IDX_CH2, todo movimiento
intra-grupo preserva total, totales por grupo y cupo CH2 -> todo candidato es
FM-consistente por construccion. Se rankea el resto por distancia L1 al crudo.

Ver docs/superpowers/specs/2026-07-24-exp-g-multivector-coverage-design.md
"""
import numpy as np
from oraculo import IDX_N, IDX_O, IDX_2X, IDX_3X, ajustar_conteo_hetero

# Grupos por multiplicidad (nH). Derivados de Gen_vector.py.
NH_GROUPS = {
    3: [0, 4, 8],
    2: [1, 5, 9, 12],
    1: [2, 6, 10, 13, 15, 16],
    0: [3, 7, 11, 14, 17, 18],
}


def _forbidden_set(n_atoms, o_atoms):
    """Clases que la FM prohibe (mismas reglas que el oraculo v2)."""
    f = set()
    if n_atoms == 0:
        f.update(IDX_N)
    if o_atoms == 0:
        f.update(IDX_O)
    if n_atoms + o_atoms < 2:
        f.add(IDX_2X)
    if n_atoms + o_atoms < 3:
        f.add(IDX_3X)
    return f


def _intra_group_moves(vec, forbidden):
    """Vectores a UN movimiento unitario intra-grupo: resta 1 a un donante
    (conteo>0) y suma 1 a un receptor permitido del mismo grupo (a != b)."""
    out = []
    for group in NH_GROUPS.values():
        allowed = [c for c in group if c not in forbidden]
        for a in group:
            if vec[a] <= 0:
                continue
            for b in allowed:
                if b == a:
                    continue
                nv = vec.copy()
                nv[a] -= 1
                nv[b] += 1
                out.append(nv)
    return out


def generate_candidates(raw, total, ch2, n_atoms, o_atoms, K, max_swaps=2):
    """Hasta K vectores int (largo 19), FM-consistentes. [0] = oraculo v2;
    el resto rankeado por sum|c-raw| ascendente; sin duplicados."""
    raw = np.asarray(raw, dtype=np.float64)
    anchor = ajustar_conteo_hetero(raw, total, ch2, n_atoms, o_atoms).astype(int)
    forbidden = _forbidden_set(n_atoms, o_atoms)

    seen = {anchor.tobytes()}
    all_vecs = [anchor]
    frontier = [anchor]
    for _ in range(max_swaps):
        nxt = []
        for v in frontier:
            for nv in _intra_group_moves(v, forbidden):
                key = nv.tobytes()
                if key not in seen:
                    seen.add(key)
                    all_vecs.append(nv)
                    nxt.append(nv)
        frontier = nxt
        if not frontier:
            break

    rest = sorted(all_vecs[1:], key=lambda c: float(np.abs(c - raw).sum()))
    return [anchor] + rest[:max(0, K - 1)]
```

- [ ] **Step 5: Correr el test y verificar que pasa**

Run: `cd experiments/G_multivector && python tests/test_candidates.py`
Expected: `>>> 6 TESTS OK <<<`

- [ ] **Step 6: Commit**

```bash
git add experiments/G_multivector/oraculo.py experiments/G_multivector/candidates.py experiments/G_multivector/tests/test_candidates.py
git commit -m "exp G: generador de candidatos intra-nH (multi-vector) + tests"
```

---

### Task 2: Métrica cobertura@K (`coverage.py`)

**Files:**
- Create: `experiments/G_multivector/coverage.py`
- Test: `experiments/G_multivector/tests/test_coverage.py`

**Interfaces:**
- Consumes: `generate_candidates` (Task 1); `IDX_CH2` de `oraculo.py`.
- Produces: `coverage_curve(y_true, y_pred_raw, n_atoms, o_atoms, Ks, max_swaps=2) -> dict[int, dict]` donde cada valor es `{"coverage": float_pct, "k_mean": float, "k_max": int}`. `y_true`/`y_pred_raw` son arrays `(M, 19)`; `n_atoms`/`o_atoms` arrays `(M,)`. `total` y `ch2` se derivan de `y_true` (suma y suma de IDX_CH2), igual que el condicionante del oraculo.
- Produces: `main(parquet_path)` que carga el parquet, obtiene N/O desde SMILES (rdkit), imprime la curva.

- [ ] **Step 1: Escribir el test que falla** (`experiments/G_multivector/tests/test_coverage.py`)

```python
# coding: ascii
"""Test de la metrica cobertura@K -- numpy puro, corre local sin torch.
Correr:  python tests/test_coverage.py   (desde experiments/G_multivector)"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coverage import coverage_curve

N = 19


def _v(**kv):
    v = np.zeros(N, dtype=int)
    for i, val in kv.items():
        v[int(i)] = val
    return v


def test_cobertura_monotona_y_top1_reproduce_v2():
    # Molecula 1: el crudo redondea exacto al verdadero -> cubierta en K=1.
    yt1 = _v(**{0: 2, 1: 2}); raw1 = np.array(yt1, float) + 0.05
    # Molecula 2: confusion CH2 vs CH2-N; verdadero CH2=2, crudo reparte 1.4/0.6.
    yt2 = _v(**{1: 2}); raw2 = _v(**{1: 1}).astype(float); raw2[1] = 1.4; raw2[9] = 0.6
    yt = np.vstack([yt1, yt2]); raw = np.vstack([raw1, raw2])
    n = np.array([0, 1]); o = np.array([0, 0])
    res = coverage_curve(yt, raw, n, o, Ks=[1, 2, 3])
    assert res[1]["coverage"] <= res[2]["coverage"] <= res[3]["coverage"]
    assert res[3]["coverage"] == 100.0, "con K=3 deben estar las 2"
    assert res[1]["k_max"] == 1


def test_k_mean_y_k_max_reportados():
    yt = _v(**{1: 2})[None, :]
    raw = np.zeros((1, N)); raw[0, 1] = 1.4; raw[0, 9] = 0.6
    res = coverage_curve(yt, raw, np.array([1]), np.array([1]), Ks=[3])
    assert res[3]["k_mean"] >= 1.0 and res[3]["k_max"] >= 1


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n>>> {len(fns)} TESTS OK <<<")


if __name__ == "__main__":
    _run()
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `cd experiments/G_multivector && python tests/test_coverage.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'coverage'` (o `ImportError` de `coverage_curve`).

- [ ] **Step 3: Escribir la implementación** (`experiments/G_multivector/coverage.py`)

```python
# coding: ascii
"""Metrica cobertura@K para Exp G (multi-vector). numpy puro.

cobertura@K = fraccion de moleculas con y_true dentro de los K candidatos.
total y ch2 se derivan de y_true (= condicionante del oraculo); N y O de la FM.

Uso (tras dumpear y_pred_raw):
  python coverage.py --parquet predictions_<exp>.parquet
"""
import argparse
import numpy as np
from oraculo import IDX_CH2
from candidates import generate_candidates


def coverage_curve(y_true, y_pred_raw, n_atoms, o_atoms, Ks, max_swaps=2):
    y_true = np.asarray(y_true, dtype=int)
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)
    Kmax = max(Ks)
    M = len(y_true)
    hit_at = np.zeros(M, dtype=int)   # menor K que cubre (0 = no cubierta)
    n_emit = np.zeros(M, dtype=int)
    for m in range(M):
        total = int(y_true[m].sum())
        ch2 = int(sum(y_true[m, i] for i in IDX_CH2))
        cands = generate_candidates(y_pred_raw[m], total, ch2,
                                    int(n_atoms[m]), int(o_atoms[m]),
                                    K=Kmax, max_swaps=max_swaps)
        n_emit[m] = len(cands)
        for rank, c in enumerate(cands, start=1):
            if np.array_equal(c, y_true[m]):
                hit_at[m] = rank
                break
    res = {}
    for K in Ks:
        covered = (hit_at >= 1) & (hit_at <= K)
        emit_k = np.minimum(n_emit, K)
        res[K] = {
            "coverage": float(covered.mean() * 100),
            "k_mean": float(emit_k.mean()),
            "k_max": int(emit_k.max()),
        }
    return res


def _formulas(smiles):
    from rdkit import Chem
    n = np.zeros(len(smiles), int); o = np.zeros(len(smiles), int)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            n[i] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
            o[i] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    return n, o


def main(parquet_path, Ks=(1, 2, 3, 4, 5), max_swaps=2):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    yt = np.vstack(df["y_true"].apply(lambda v: np.array(v, dtype=int)))
    raw = np.vstack(df["y_pred_raw"].apply(lambda v: np.array(v, dtype=float)))
    n, o = _formulas(df["smiles"].tolist())
    res = coverage_curve(yt, raw, n, o, list(Ks), max_swaps=max_swaps)
    print("=" * 56)
    print("  COBERTURA@K (Exp G multi-vector) -- val congelado")
    print("=" * 56)
    print(f"  moleculas: {len(df)}   max_swaps: {max_swaps}")
    print(f"\n  {'K':>3}  {'cobertura':>10}  {'K prom emit':>12}  {'K max emit':>11}")
    print("  " + "-" * 44)
    for K in Ks:
        r = res[K]
        print(f"  {K:>3}  {r['coverage']:>9.2f}%  {r['k_mean']:>12.2f}  {r['k_max']:>11d}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cobertura@K Exp G")
    ap.add_argument("--parquet", required=True, help="parquet con y_true, y_pred_raw, smiles")
    ap.add_argument("--max-swaps", type=int, default=2)
    args = ap.parse_args()
    main(args.parquet, max_swaps=args.max_swaps)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `cd experiments/G_multivector && python tests/test_coverage.py`
Expected: `>>> 2 TESTS OK <<<`

- [ ] **Step 5: Commit**

```bash
git add experiments/G_multivector/coverage.py experiments/G_multivector/tests/test_coverage.py
git commit -m "exp G: metrica cobertura@K + test (numpy puro, local)"
```

---

### Task 3: Columna `y_pred_raw` en el dump (E3)

**Files:**
- Modify: `experiments/E3_dos_conjuntos/dump_predictions.py`

**Interfaces:**
- Produces: el parquet `predictions_<experiment_name>.parquet` gana la columna `y_pred_raw` (lista de 19 floats, el output crudo del modelo pre-redondeo, redondeado a 4 decimales). Es el insumo de `coverage.py` (Task 2).

- [ ] **Step 1: Agregar la columna al dict de filas**

En `experiments/E3_dos_conjuntos/dump_predictions.py`, en el `rows.append({...})` dentro del loop, agregar la clave `y_pred_raw` junto a las existentes. El array crudo del modelo ya está en la variable `out[k]` (es `model(...).cpu().numpy()`):

```python
                    "y_pred_crude": np.clip(np.floor(out[k]), 0, None).astype(int).tolist(),
                    "y_pred_assisted": ajustar_conteo_doble_exacto(out[k], total, ch2).tolist(),
                    "y_pred_assisted_v2": ajustar_conteo_hetero(
                        out[k], total, ch2, n_at, o_at).tolist(),
                    "y_pred_raw": [round(float(x), 4) for x in out[k]],
                    "crosspeaks": crosspeaks,
                    "c13_shifts": c13_shifts,
```

- [ ] **Step 2: Actualizar el docstring de columnas de salida**

En el docstring del módulo (arriba de todo), en la lista "Salida: ... con columnas:", agregar `y_pred_raw (19 floats, output crudo pre-redondeo)`:

```python
Salida: predictions_<experiment_name>.parquet con columnas:
  idx, smiles, y_true (19 ints), y_pred_crude (19 ints), y_pred_assisted (v1, 19 ints),
  y_pred_assisted_v2 (oraculo hetero, 19 ints), y_pred_raw (19 floats, output crudo pre-redondeo),
  crosspeaks (lista de [delta_c, delta_h] en ppm), c13_shifts (lista de delta_c en ppm).
```

- [ ] **Step 3: Verificar que compila**

Run: `cd experiments/E3_dos_conjuntos && python -m py_compile dump_predictions.py`
Expected: sin salida (compila OK). No se puede correr el dump local (requiere torch + checkpoint); corre en el cluster.

- [ ] **Step 4: Commit**

```bash
git add experiments/E3_dos_conjuntos/dump_predictions.py
git commit -m "exp G: dump y_pred_raw (output crudo) para el generador de candidatos"
```

---

### Task 4: Documentación (`README.md`, `RATIONALE.md`)

**Files:**
- Create: `experiments/G_multivector/README.md`
- Create: `experiments/G_multivector/RATIONALE.md`

**Interfaces:**
- Consumes: nada (docs). Cierra el experimento con el checklist de cómo correrlo.

- [ ] **Step 1: Escribir `RATIONALE.md`**

```markdown
# Exp G — Rationale

**Objetivo:** cobertura@K, no EMA. El vector alimenta un generador de estructuras;
su trabajo es achicar el espacio de generacion sin perder el vector verdadero.
Perder especificidad (emitir 2-3, generar de mas) es tolerable; perder al verdadero
es gravisimo.

**Hipotesis:** como el 85% de las fallas conserva la multiplicidad (nH) — el modelo
acierta cuantos H tiene el carbono y solo confunde el entorno dentro de ese nH —,
un set chico de candidatos generados moviendo conteos intra-nH cubre al verdadero.

**Evidencia (parquet Fase 3, val congelado 14428):** top-1 91.36% -> +1 swap intra-nH
98.18% -> cap 98.72%. 984 moleculas a exactamente 1 swap; 185 (1.28%) necesitan
cross-nH (fuera de alcance v1).

**Exito:** cobertura@K >= 98% con K promedio <= 3, top-1 == oraculo v2.
**Fracaso:** cobertura muy por debajo de la curva teorica a K chico -> revisar el
generador o el ranking.

Spec: docs/superpowers/specs/2026-07-24-exp-g-multivector-coverage-design.md
```

- [ ] **Step 2: Escribir `README.md`**

```markdown
# Exp G — Multi-vector (cobertura@K)

Generador de candidatos post-hoc sobre el checkpoint Fase 3 Set Transformer
(no reentrena). Emite hasta K vectores FM-consistentes por molecula; metrica =
cobertura@K sobre el val congelado.

## Piezas

- `candidates.py` — `generate_candidates(raw, total, ch2, n, o, K, max_swaps=2)`.
  Movimientos unitarios intra-grupo-de-nH desde el oraculo v2; todos FM-consistentes.
- `coverage.py` — curva cobertura@K sobre un parquet con `y_pred_raw`.
- `oraculo.py` — copia de E3 (reglas del ancla v2).
- `tests/` — numpy puro, corren local sin torch/GPU.

## Como correrlo

1. Tests locales (sin GPU):
   ```bash
   cd experiments/G_multivector
   python tests/test_candidates.py
   python tests/test_coverage.py
   ```
2. Generar el parquet con `y_pred_raw` en el cluster (checkpoint Fase 3 Set
   Transformer; XPU/Clementina o A10):
   ```bash
   cd experiments/E3_dos_conjuntos
   python dump_predictions.py --config config_settransformer.yaml
   ```
3. Traer el parquet a la PC y correr la metrica (100% local):
   ```bash
   cd experiments/G_multivector
   python coverage.py --parquet /ruta/al/predictions_nmr_202k_e3_settransformer_2sets_19v.parquet
   ```
4. Elegir el K operativo (~2-3) que da cobertura ~=98% con K promedio chico.
   Agregar la curva a `docs/Runs/RESULTS.md`.

## Alcance

v1 = solo intra-nH (techo 98.7%). Fase 2: candidatos cross-nH (el 1.28% de
multiplicidad mal) y/o reentrenar con distribucion calibrada por grupo de nH.
```

- [ ] **Step 3: Commit**

```bash
git add experiments/G_multivector/README.md experiments/G_multivector/RATIONALE.md
git commit -m "exp G: README + RATIONALE"
```

---

## Notas de ejecución

- Tras Task 3, correr el dump en el cluster produce el parquet con `y_pred_raw`; recién
  ahí `coverage.py --parquet ...` da los números reales. Todo el resto (Tasks 1, 2, 4) se
  valida 100% local.
- Sanity check al correr sobre datos reales: `coverage@1` debe reproducir la EMA asistida v2
  del checkpoint (~92.1%). Si no, hay un desalineo entre el ancla del generador y el oráculo.
