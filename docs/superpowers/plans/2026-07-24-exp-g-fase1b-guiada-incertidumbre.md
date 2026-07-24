# Exp G Fase 1b — Generación guiada por incertidumbre — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un generador de candidatos que emite alternativas SOLO donde el modelo tiene duda (K adaptativo), y su métrica de barrido de τ, para cubrir el vector verdadero con K promedio bajo.

**Architecture:** Extiende `experiments/G_multivector/` (Fase 1). Nueva función `generate_candidates_uncertainty` que reusa el ancla del oráculo v2 y `_intra_group_moves`, pero PODA los candidatos por `extra-L1 < τ`. La métrica barre τ. Todo numpy puro, corre local sobre el parquet con `y_pred_raw`.

**Tech Stack:** Python 3, numpy (generador y métrica, sin torch), pandas + pyarrow (parquet), rdkit (N/O desde SMILES), streamlit + altair (GUI).

## Global Constraints

- **numpy puro** en `candidates.py`, `coverage.py` y sus tests (sin torch; corren local sin GPU).
- **No romper Fase 1:** `generate_candidates` (Fase 1) y `coverage_curve` quedan INTACTAS. Fase 1b agrega funciones nuevas al lado, para poder comparar.
- **Reusar, no duplicar:** `generate_candidates_uncertainty` debe reusar `ajustar_conteo_hetero`, `_forbidden_set`, `_intra_group_moves` y `NH_GROUPS` ya existentes en `candidates.py`.
- **Orden de clases 19v fijo**; grupos de nH: 3H=[0,4,8], 2H=[1,5,9,12], 1H=[2,6,10,13,15,16], 0H=[3,7,11,14,17,18]; el grupo 2H es exactamente `IDX_CH2`.
- **Propiedad a preservar:** todo candidato debe ser FM-consistente (suma==total, cupo CH2==ch2, clases prohibidas en 0) y el top-1 (elemento [0]) debe ser SIEMPRE el oráculo v2.
- **Tests estilo proyecto:** cada test es un `.py` con `if __name__ == "__main__": _run()` que imprime `>>> N TESTS OK <<<`, ejecutable con `python tests/test_x.py` desde `experiments/G_multivector/`. **Ojo:** los helpers tipo `_raw`/`_v` deben recibir un dict (`_raw({1: 1.4})`), NO usar `**kwargs` con claves int (falla con `TypeError: keywords must be strings`).

---

### Task 1: `generate_candidates_uncertainty` (poda por τ) + tests

**Files:**
- Modify: `experiments/G_multivector/candidates.py` (agregar función al final; no tocar lo existente)
- Test: `experiments/G_multivector/tests/test_candidates_uncertainty.py`

**Interfaces:**
- Consumes (ya existen en `candidates.py`): `ajustar_conteo_hetero` (de `oraculo`), `_forbidden_set(n,o)`, `_intra_group_moves(vec, forbidden)`, `NH_GROUPS`.
- Produces: `generate_candidates_uncertainty(raw, total, ch2, n_atoms, o_atoms, tau, K_max, max_swaps=2) -> list[np.ndarray]` — lista de vectores int19, largo ≤ K_max; [0] SIEMPRE el oráculo v2; se conservan solo alternativas con `sum|c-raw| <= sum|anchor-raw| + tau`; el resto rankeado por `sum|c-raw|` ascendente; sin duplicados.

- [ ] **Step 1: Escribir el test que falla** (`experiments/G_multivector/tests/test_candidates_uncertainty.py`)

```python
# coding: ascii
"""Test del generador guiado por incertidumbre (Fase 1b) -- numpy puro, corre local.
Correr:  python tests/test_candidates_uncertainty.py   (desde experiments/G_multivector)"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oraculo import IDX_CH2, ajustar_conteo_hetero
from candidates import generate_candidates_uncertainty

N = 19


def _raw(kv):
    v = np.zeros(N, dtype=np.float64)
    for i, val in kv.items():
        v[int(i)] = val
    return v


def _fm_ok(c, total, ch2):
    return c.sum() == total and sum(c[i] for i in IDX_CH2) == ch2


def test_top1_es_oraculo_v2():
    raw = _raw({1: 1.55, 9: 0.55})
    cands = generate_candidates_uncertainty(raw, 2, 2, 1, 0, tau=0.5, K_max=6)
    v2 = ajustar_conteo_hetero(raw, 2, 2, 1, 0)
    assert np.array_equal(cands[0], v2), "el top-1 debe ser el oraculo v2"


def test_molecula_segura_emite_uno():
    # CH3=1.02, CH2=1.98 (hidrocarburo, sin duda). total=3, ch2=2, n=o=0.
    raw = _raw({0: 1.02, 1: 1.98})
    cands = generate_candidates_uncertainty(raw, 3, 2, 0, 0, tau=0.5, K_max=6)
    assert len(cands) == 1, f"molecula segura deberia emitir 1 solo candidato, emitio {len(cands)}"


def test_molecula_con_duda_incluye_alternativa():
    # CH2=1.55 vs CH2-N=0.55 (empate) -> debe incluir (CH2=1, CH2-N=1)
    raw = _raw({1: 1.55, 9: 0.55})
    cands = generate_candidates_uncertainty(raw, 2, 2, 1, 0, tau=0.5, K_max=6)
    got = {(c[1], c[9]) for c in cands}
    assert (1, 1) in got, f"falta la alternativa de la duda CH2/CH2-N: {got}"
    assert len(cands) >= 2


def test_tau0_solo_empates_exactos():
    # CH2=1.6 vs CH2-N=0.4: la alternativa cuesta extra-L1=0.4 > 0 -> con tau=0 NO se emite.
    raw = _raw({1: 1.6, 9: 0.4})
    cands = generate_candidates_uncertainty(raw, 2, 2, 1, 0, tau=0.0, K_max=6)
    assert len(cands) == 1, f"con tau=0 y sin empate exacto deberia emitir 1, emitio {len(cands)}"


def test_todos_fm_consistentes():
    rng = np.random.RandomState(0)
    for _ in range(100):
        raw = rng.rand(N) * 2.5
        total = int(round(raw.sum()))
        ch2 = int(round(sum(raw[i] for i in IDX_CH2)))
        if total < ch2:
            total = ch2
        cands = generate_candidates_uncertainty(raw, total, ch2, 2, 2, tau=1.0, K_max=6)
        for c in cands:
            assert _fm_ok(c, total, ch2), f"candidato no FM-consistente: {c}"
            assert (c >= 0).all()


def test_no_puebla_clases_prohibidas():
    raw = _raw({1: 1.1, 9: 0.9})
    cands = generate_candidates_uncertainty(raw, 2, 2, 0, 0, tau=2.0, K_max=6)
    for c in cands:
        assert all(c[i] == 0 for i in [8, 9, 10, 11, 16]), f"poblo clase -N con n=0: {c}"


def test_monotonia_en_tau():
    # el set de candidatos con tau grande contiene al de tau chico (antes del corte K_max)
    raw = _raw({1: 1.55, 9: 0.55, 2: 1.4, 6: 0.6})
    chico = generate_candidates_uncertainty(raw, 4, 2, 1, 1, tau=0.2, K_max=20)
    grande = generate_candidates_uncertainty(raw, 4, 2, 1, 1, tau=1.5, K_max=20)
    s_chico = {c.tobytes() for c in chico}
    s_grande = {c.tobytes() for c in grande}
    assert s_chico <= s_grande, "tau grande deberia contener a tau chico"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n>>> {len(fns)} TESTS OK <<<")


if __name__ == "__main__":
    _run()
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `cd experiments/G_multivector && python tests/test_candidates_uncertainty.py`
Expected: FAIL con `ImportError: cannot import name 'generate_candidates_uncertainty'`.

- [ ] **Step 3: Agregar la función al final de `experiments/G_multivector/candidates.py`**

```python
def generate_candidates_uncertainty(raw, total, ch2, n_atoms, o_atoms, tau, K_max,
                                    max_swaps=2):
    """Fase 1b: candidatos guiados por incertidumbre. Emite una alternativa solo
    si su distancia L1 al crudo supera a la del ancla en menos de tau
    (sum|c-raw| <= sum|anchor-raw| + tau). K adaptativo: molecula segura -> [anchor].

    [0] es siempre el oraculo v2; el resto rankeado por L1; sin duplicados; <= K_max.
    """
    raw = np.asarray(raw, dtype=np.float64)
    anchor = ajustar_conteo_hetero(raw, total, ch2, n_atoms, o_atoms).astype(int)
    forbidden = _forbidden_set(n_atoms, o_atoms)
    thresh = float(np.abs(anchor - raw).sum()) + tau

    seen = {anchor.tobytes()}
    kept = [anchor]
    frontier = [anchor]
    for _ in range(max_swaps):
        nxt = []
        for v in frontier:
            for nv in _intra_group_moves(v, forbidden):
                key = nv.tobytes()
                if key in seen:
                    continue
                if float(np.abs(nv - raw).sum()) <= thresh:
                    seen.add(key)
                    kept.append(nv)
                    nxt.append(nv)
        frontier = nxt
        if not frontier:
            break

    rest = sorted(kept[1:], key=lambda c: float(np.abs(c - raw).sum()))
    return [anchor] + rest[:max(0, K_max - 1)]
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `cd experiments/G_multivector && python tests/test_candidates_uncertainty.py`
Expected: `>>> 7 TESTS OK <<<`

- [ ] **Step 5: Commit**

```bash
git add experiments/G_multivector/candidates.py experiments/G_multivector/tests/test_candidates_uncertainty.py
git commit -m "exp G fase 1b: generate_candidates_uncertainty (poda por tau) + tests"
```

---

### Task 2: Métrica de barrido de τ en `coverage.py`

**Files:**
- Modify: `experiments/G_multivector/coverage.py` (agregar función + flag CLI; no tocar `coverage_curve`)
- Test: `experiments/G_multivector/tests/test_coverage_uncertainty.py`

**Interfaces:**
- Consumes: `generate_candidates_uncertainty` (Task 1); `IDX_CH2`; `_formulas` (ya en `coverage.py`).
- Produces: `coverage_uncertainty(y_true, y_pred_raw, n_atoms, o_atoms, tau, K_max, max_swaps=2) -> dict` con `{"coverage": float_pct, "k_mean": float, "k_max": int}`. `total`/`ch2` se derivan de `y_true`. Y `main_uncertainty(parquet_path, taus, K_max)` que barre τ e imprime la tabla.

- [ ] **Step 1: Escribir el test que falla** (`experiments/G_multivector/tests/test_coverage_uncertainty.py`)

```python
# coding: ascii
"""Test de la metrica de barrido de tau (Fase 1b) -- numpy puro, corre local.
Correr:  python tests/test_coverage_uncertainty.py   (desde experiments/G_multivector)"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coverage import coverage_uncertainty

N = 19


def _v(kv):
    v = np.zeros(N, dtype=int)
    for i, val in kv.items():
        v[int(i)] = val
    return v


def test_molecula_segura_kmean_1():
    # verdadero = crudo redondeado exacto -> 1 candidato, cubierto.
    yt = _v({0: 1, 1: 2})
    raw = np.zeros((1, N)); raw[0, 0] = 1.02; raw[0, 1] = 1.98
    res = coverage_uncertainty(yt[None, :], raw, np.array([0]), np.array([0]),
                               tau=0.5, K_max=6)
    assert res["coverage"] == 100.0
    assert res["k_mean"] == 1.0, f"molecula segura deberia dar k_mean=1, dio {res['k_mean']}"


def test_duda_cubierta_con_tau():
    # verdadero CH2=1, CH2-N=1; crudo 1.55/0.55; con tau razonable se cubre.
    yt = _v({1: 1, 9: 1})
    raw = np.zeros((1, N)); raw[0, 1] = 1.55; raw[0, 9] = 0.55
    res = coverage_uncertainty(yt[None, :], raw, np.array([1]), np.array([0]),
                               tau=0.5, K_max=6)
    assert res["coverage"] == 100.0
    assert res["k_mean"] >= 2.0


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n>>> {len(fns)} TESTS OK <<<")


if __name__ == "__main__":
    _run()
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `cd experiments/G_multivector && python tests/test_coverage_uncertainty.py`
Expected: FAIL con `ImportError: cannot import name 'coverage_uncertainty'`.

- [ ] **Step 3: Agregar a `experiments/G_multivector/coverage.py`**

Agregar el import de la función nueva (junto al import existente de `candidates`):

```python
from candidates import generate_candidates, generate_candidates_uncertainty
```

Agregar la función `coverage_uncertainty` (después de `coverage_curve`):

```python
def coverage_uncertainty(y_true, y_pred_raw, n_atoms, o_atoms, tau, K_max, max_swaps=2):
    """Cobertura, K promedio y K maximo emitido para el generador guiado (Fase 1b)
    a un tau dado. total/ch2 se derivan de y_true."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)
    M = len(y_true)
    covered = np.zeros(M, dtype=bool)
    nemit = np.zeros(M, dtype=int)
    for m in range(M):
        total = int(y_true[m].sum())
        ch2 = int(sum(y_true[m, i] for i in IDX_CH2))
        cands = generate_candidates_uncertainty(
            y_pred_raw[m], total, ch2, int(n_atoms[m]), int(o_atoms[m]),
            tau, K_max, max_swaps=max_swaps)
        nemit[m] = len(cands)
        covered[m] = any(np.array_equal(c, y_true[m]) for c in cands)
    return {"coverage": float(covered.mean() * 100),
            "k_mean": float(nemit.mean()),
            "k_max": int(nemit.max())}
```

Agregar `main_uncertainty` y el flag `--uncertainty` en el `__main__`:

```python
def main_uncertainty(parquet_path, taus=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0), K_max=6):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    yt = np.vstack(df["y_true"].apply(lambda v: np.array(v, dtype=int)))
    raw = np.vstack(df["y_pred_raw"].apply(lambda v: np.array(v, dtype=float)))
    n, o = _formulas(df["smiles"].tolist())
    print("=" * 60)
    print("  COBERTURA guiada por incertidumbre (Exp G Fase 1b) -- val congelado")
    print("=" * 60)
    print(f"  moleculas: {len(df)}   K_max: {K_max}")
    print(f"\n  {'tau':>5}  {'cobertura':>10}  {'K prom emit':>12}  {'K max emit':>11}")
    print("  " + "-" * 46)
    res_all = {}
    for tau in taus:
        r = coverage_uncertainty(yt, raw, n, o, tau, K_max)
        res_all[tau] = r
        print(f"  {tau:>5.2f}  {r['coverage']:>9.2f}%  {r['k_mean']:>12.2f}  {r['k_max']:>11d}")
    return res_all
```

En el bloque `if __name__ == "__main__":`, agregar el flag y el ruteo:

```python
    ap.add_argument("--uncertainty", action="store_true",
                    help="Corre el barrido de tau (Fase 1b) en vez de la curva de K (Fase 1).")
    args = ap.parse_args()
    if args.uncertainty:
        main_uncertainty(args.parquet, K_max=6)
    else:
        main(args.parquet, max_swaps=args.max_swaps)
```

(Nota: el `argparse` existente ya define `--parquet` y `--max-swaps`; solo agregá `--uncertainty` y el ruteo `if args.uncertainty`.)

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `cd experiments/G_multivector && python tests/test_coverage_uncertainty.py`
Expected: `>>> 2 TESTS OK <<<`

- [ ] **Step 5: Verificar que Fase 1 sigue andando (no-regresión)**

Run: `cd experiments/G_multivector && python tests/test_coverage.py && python tests/test_candidates.py`
Expected: `>>> 2 TESTS OK <<<` y `>>> 9 TESTS OK <<<` (sin cambios).

- [ ] **Step 6: Commit**

```bash
git add experiments/G_multivector/coverage.py experiments/G_multivector/tests/test_coverage_uncertainty.py
git commit -m "exp G fase 1b: coverage_uncertainty + barrido de tau (--uncertainty)"
```

---

### Task 3: Toggle Fase 1 / Fase 1b en la GUI

**Files:**
- Modify: `src/gui/gui_inspector.py`

**Interfaces:**
- Consumes: `generate_candidates` (ya importada) y `generate_candidates_uncertainty` (Task 1).
- Produces: en el panel multi-vector, un radio "Generador" (Fase 1 todos / Fase 1b guiada) y, si Fase 1b, un slider de τ; el panel usa el generador elegido.

- [ ] **Step 1: Importar la función nueva**

En `src/gui/gui_inspector.py`, en el bloque `try: from candidates import ...`, cambiar el import para traer ambas:

```python
try:
    from candidates import generate_candidates, generate_candidates_uncertainty
    HAS_GEN = True
except Exception:
    HAS_GEN = False
```

- [ ] **Step 2: Parametrizar `_candidates_for_row` por generador**

Reemplazar la función `_candidates_for_row` por una versión que acepta el modo y τ:

```python
def _candidates_for_row(row, K, gen="fase1", tau=0.5):
    """Lista de candidatos para una fila y el rank (1-based) del verdadero (0=no cubierto).
    gen='fase1' -> generate_candidates (todos, top-K); gen='fase1b' -> guiada por tau."""
    raw = np.array(row["y_pred_raw"], dtype=float)
    yt = np.array(row["y_true"], dtype=int)
    total = int(yt.sum())
    ch2 = int(sum(yt[i] for i in IDX_CH2))
    n_at, o_at = _formula_no(row["smiles"])
    if gen == "fase1b":
        cands = generate_candidates_uncertainty(raw, total, ch2, n_at, o_at, tau, K)
    else:
        cands = generate_candidates(raw, total, ch2, n_at, o_at, K=K)
    rank = 0
    for r, c in enumerate(cands, 1):
        if np.array_equal(c, yt):
            rank = r
            break
    return cands, rank
```

(Nota: `mv_ranks` llama `_candidates_for_row(d.iloc[m], K_max)` con los defaults `gen='fase1'` → sigue midiendo Fase 1, sin cambios. No hace falta tocar `mv_ranks`.)

- [ ] **Step 3: Agregar el control en el sidebar (junto al slider de K)**

Reemplazar el bloque del sidebar de multi-vector:

```python
K = 3
_gen = "fase1"
_tau = 0.5
if MV_AVAILABLE:
    st.sidebar.header("Multi-vector (Exp G)")
    _gen = st.sidebar.radio(
        "Generador",
        ["fase1", "fase1b"],
        format_func=lambda s: "Fase 1 (todos, top-K)" if s == "fase1"
        else "Fase 1b (guiada por duda)")
    K = st.sidebar.slider("K (tope de candidatos)", 1, 6, 3)
    if _gen == "fase1b":
        _tau = st.sidebar.slider("tau (umbral de duda)", 0.0, 2.0, 0.5, 0.05)
```

- [ ] **Step 4: Usar el generador elegido en el panel**

En el panel "Candidatos multi-vector", cambiar la llamada:

```python
    cands, cov_rank = _candidates_for_row(row, K, gen=_gen, tau=_tau)
```

y en el `st.subheader` del panel, reflejar el modo:

```python
    _gen_lbl = "Fase 1b (τ=%.2f)" % _tau if _gen == "fase1b" else "Fase 1 (todos)"
    st.subheader(f"Candidatos multi-vector — {_gen_lbl}, K_max={K}, emitidos={len(cands)}")
```

- [ ] **Step 5: Verificar que compila y la lógica anda**

Run:
```bash
cd "$(git rev-parse --show-toplevel)" && python -m py_compile src/gui/gui_inspector.py && python -c "
import sys, numpy as np
sys.path.insert(0, 'experiments/G_multivector')
from candidates import generate_candidates_uncertainty
raw=np.zeros(19); raw[1]=1.55; raw[9]=0.55
c=generate_candidates_uncertainty(raw,2,2,1,0,0.5,6)
print('fase1b emitidos:', len(c))
raw2=np.zeros(19); raw2[0]=1.02; raw2[1]=1.98
c2=generate_candidates_uncertainty(raw2,3,2,0,0,0.5,6)
print('segura emitidos:', len(c2))
assert len(c2)==1, 'la segura deberia emitir 1'
print('OK')
"
```
Expected: `GUI compila`, `fase1b emitidos: 2`, `segura emitidos: 1`, `OK`.

- [ ] **Step 6: Commit**

```bash
git add src/gui/gui_inspector.py
git commit -m "gui: toggle Fase 1 / Fase 1b (guiada por tau) en el panel multi-vector"
```

---

### Task 4: Documentación (README + RATIONALE de Fase 1b)

**Files:**
- Modify: `experiments/G_multivector/README.md` (agregar sección Fase 1b)
- Modify: `experiments/G_multivector/RATIONALE.md` (agregar sección Fase 1b)

**Interfaces:**
- Consumes: nada (docs).

- [ ] **Step 1: Agregar al final de `experiments/G_multivector/README.md`**

```markdown

## Fase 1b — guiada por incertidumbre

Emite alternativas SOLO donde el modelo duda (K adaptativo), podando por
`extra-L1 < tau` desde el ancla v2. Molecula segura -> 1 vector.

Tests locales:
```bash
python tests/test_candidates_uncertainty.py
python tests/test_coverage_uncertainty.py
```

Barrido de tau sobre el parquet (elegir el punto cobertura vs K promedio):
```bash
python coverage.py --parquet /ruta/al/predictions_...parquet --uncertainty
```

En la GUI: el panel multi-vector tiene un toggle "Fase 1 / Fase 1b" y un slider de tau.
```

- [ ] **Step 2: Agregar al final de `experiments/G_multivector/RATIONALE.md`**

```markdown

## Fase 1b

Fase 1 emitia K vectores SIEMPRE (K prom == K), sin especificidad. Fase 1b emite
alternativas solo donde hay duda real (costo extra de L1 < tau), dando K adaptativo:
molecula segura -> 1 vector; con dudas -> pocas mas. Objetivo: misma cobertura que
Fase 1 (~97-98%) con K promedio mucho mas bajo. tau y K_max se calibran con el barrido.
Spec: docs/superpowers/specs/2026-07-24-exp-g-fase1b-guiada-incertidumbre-design.md
```

- [ ] **Step 3: Commit**

```bash
git add experiments/G_multivector/README.md experiments/G_multivector/RATIONALE.md
git commit -m "exp G fase 1b: README + RATIONALE"
```

---

## Notas de ejecución

- Todo se valida 100% local (numpy puro + el parquet con `y_pred_raw` que ya se generó en Fase 1).
  No se necesita el cluster.
- El número de la verdad sale de `python coverage.py --parquet ... --uncertainty`: la tabla de τ.
  Comparar contra la fila de Fase 1 (`coverage.py --parquet ...`): mismo ~97% de cobertura, ¿con
  K promedio más bajo? Ese es el resultado del experimento; volcarlo a `docs/Runs/RESULTS.md`.
