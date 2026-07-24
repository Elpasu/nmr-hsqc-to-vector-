# Diseño — Exp G Fase 1b: generación de candidatos guiada por incertidumbre

**Fecha:** 2026-07-24
**Autor:** Lucas Passaglia (con Claude Code)
**Estado:** propuesto (pendiente de revisión)

---

## 1. Motivación (qué arregla respecto de Fase 1)

Fase 1 (`generate_candidates`) genera **todos** los swaps intra-nH posibles y corta en top-K por
L1. Medido: coverage@1 92.14% → @3 96.75% → @5 97.39%, con `K prom emit == K` para todo K. Dos
problemas: (a) emite K candidatos **siempre**, aún cuando el modelo está seguro (sin especificidad
→ infla la generación aguas abajo); (b) el presupuesto K se gasta en candidatos "de relleno" de
grupos sin duda, dejando afuera al verdadero en ~1.3% de casos donde sí estaba en el vecindario.

Fase 1b genera alternativas **solo donde hay duda real**, dando **K adaptativo** por molécula:
molécula segura → 1 vector; molécula con dudas → unos pocos. Mismo checkpoint, sin reentrenar.

## 2. Criterio de "duda": costo extra de L1 < τ (elegido por el usuario)

El ancla (oráculo v2) es la repartición entera más cercana al crudo → tiene L1 mínima
`L1a = sum|ancla - raw|`. Una alternativa (otro vector entero FM-consistente, alcanzable moviendo
conteos **dentro de grupos de nH**) tiene su propia L1 al crudo. Se **emite** una alternativa solo
si `sum|alt - raw| <= L1a + τ`, es decir, si empeora el ajuste al crudo en **menos de τ** (en
unidades de conteo).

- Grupo **partido** (CH2=1.55 / CH2-N=0.55): mover 1 de CH2→CH2-N cuesta extra-L1 ≈ 0 → **se
  emite** (τ cualquiera).
- Grupo **seguro** (CH=1.05, resto ≈0): mover CH→CH-N cuesta extra-L1 ≈ 2 → **NO se emite** (τ<2).

τ es la perilla principal. K_max es un tope duro de candidatos por molécula (por si una molécula
tiene muchas dudas y no querés que explote la generación).

**Por qué esto da K adaptativo:** en una molécula donde el modelo está seguro de todo, ningún
movimiento intra-nH queda dentro de `L1a + τ` → se emite **solo el ancla** (K=1). Donde hay una
duda → 2. Donde hay dos dudas independientes → hasta 4 (producto), acotado por K_max.

## 3. Algoritmo

```
generate_candidates_uncertainty(raw, total, ch2, n_atoms, o_atoms, tau, K_max, max_swaps=2):
    anchor  = ajustar_conteo_hetero(raw, total, ch2, n_atoms, o_atoms)   # ancla FM-consistente
    forbidden = _forbidden_set(n_atoms, o_atoms)                          # reglas del oraculo v2
    L1a = sum|anchor - raw|
    # BFS de movimientos unitarios intra-grupo-de-nH (reusa _intra_group_moves),
    # PODANDO por el umbral: se conserva un vector solo si sum|v - raw| <= L1a + tau.
    seen = {anchor}; kept = [anchor]; frontier = [anchor]
    repeat max_swaps veces:
        for v in frontier:
            for nv in _intra_group_moves(v, forbidden):
                if nv no visto y sum|nv - raw| <= L1a + tau:
                    marcar visto; kept.append(nv); frontier_next.append(nv)
    rest = sorted(kept - {anchor}, key = lambda c: sum|c - raw|)
    return [anchor] + rest[:K_max-1]
```

**Propiedades (heredadas de Fase 1, ya verificadas):** todo movimiento es intra-grupo → preserva
total, totales por grupo y cupo CH2 (el grupo 2H es exactamente `IDX_CH2`); las clases prohibidas
nunca se pueblan (el ancla las deja en 0 y `_intra_group_moves` no las usa de receptor). Por lo
tanto **todo candidato es FM-consistente**. El top-1 es siempre el oráculo v2.

Con `tau → ∞` y el mismo `max_swaps`, recupera el comportamiento de Fase 1 (sin poda). Con
`tau = 0`, solo emite alternativas con L1 exactamente igual al ancla (empates perfectos).

## 4. Métrica: barrido de τ (cobertura vs K promedio)

A diferencia de Fase 1 (curva sobre K fijo), acá el número de candidatos es **variable** por
molécula. La métrica es un **barrido de τ**: para cada τ (con un K_max fijo, ej. 6), sobre el val
congelado:

- `cobertura` = fracción de moléculas con `y_true` dentro del set emitido.
- `K promedio emitido` (la especificidad — cuánto más generás aguas abajo).
- `K máximo emitido`.

El deliverable es esa tabla (τ ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, ...}), para elegir el punto de
operación: **la τ más chica que da cobertura ≈ objetivo con K promedio bajo.** Se compara contra
la curva de Fase 1 (misma cobertura, ¿menor K promedio?). Éxito = alcanzar ~la misma cobertura que
Fase 1 (~97-98%) con **K promedio << K de Fase 1** (idealmente ~1.x).

## 5. Alcance de la implementación

**Dentro:**
- `experiments/G_multivector/candidates.py`: función nueva `generate_candidates_uncertainty(...)`
  (se mantiene `generate_candidates` de Fase 1 intacta para comparar).
- `experiments/G_multivector/coverage.py`: función `coverage_uncertainty(y_true, y_pred_raw,
  n_atoms, o_atoms, tau, K_max) -> {"coverage","k_mean","k_max"}` y un `main` que barre una lista
  de τ e imprime la tabla.
- Tests numpy puros (corren local): molécula segura → K=1; molécula con duda → incluye la
  alternativa; todos FM-consistentes; `tau=0` solo empates; `tau` grande ⊇ `tau` chico (monótono
  en cobertura); top-1 == oráculo v2.
- `gui_inspector.py`: toggle "Fase 1b (guiada, τ)" vs "Fase 1 (todos)" en el panel multi-vector,
  con un slider de τ, para inspeccionar caso por caso.

**Fuera (fase 2):** candidatos cross-nH (el 1.28% de multiplicidad mal) y reentrenar con cabeza
distribucional calibrada.

## 6. Criterios de éxito / fracaso

- **Éxito:** existe una τ donde cobertura ≈ 97-98% (comparable a Fase 1 con K=5) pero con **K
  promedio bajo** (ej. ≤ 2), demostrando la ganancia de especificidad.
- **Neutro/informativo:** si para igualar la cobertura de Fase 1 hace falta una τ tan grande que
  el K promedio no baja → la incertidumbre por L1 del modelo actual no discrimina bien, y el
  próximo paso es Fase 2 (reentrenar con distribución calibrada).
- **Fracaso (improbable):** cobertura muy por debajo de Fase 1 a cualquier τ con K promedio bajo →
  revisar la poda.

## 7. Verificación

- Tests de `candidates.py`/`coverage.py` ejecutables local (numpy puro, sin torch/GPU).
- `coverage.py --uncertainty` sobre el parquet con `y_pred_raw` (el mismo que Fase 1) → tabla de τ.
  Todo local, sin cluster.
- Sanity: con `tau=0` la cobertura debe ser ≥ coverage@1 de Fase 1 (92.14%) y el K promedio ≈ 1.0
  (casi nadie tiene empates exactos); al subir τ, cobertura sube y K promedio sube.
