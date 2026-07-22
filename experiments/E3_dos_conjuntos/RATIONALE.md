# RATIONALE — Exp E Fase 3: dos conjuntos de picos, dos arquitecturas

## Hipótesis

Fase 2 (`docs/Runs/RESULTS.md`, "Exp E — Fase 2") fracasó: el DeepSets sobre
picos dio EMA cruda 0.74% (< 0.89% de Exp C) y las confusiones de cuaternarios
(`Cqsp2`↔`=CH/Ar`, `CH2`↔`CH2-N`) **empeoraron**. La causa raíz no fue un bug
ni la capacidad del modelo: los picos de Fase 1b son **crosspeaks C-H puros**
(`extract_peaks_pkl.py` descarta todo carbono sin H), así que los cuaternarios
(`Cqsp2`, `Cq`, `Cq-O`, `Cq-N`) nunca entraban al modelo. Y Fase 2 además
eliminó las proyecciones 1D (`vec_c`/`vec_h`) — que son espectros ¹³C/¹H reales
calculados aparte desde el pkl y que **sí contienen los cuaternarios** —
creyéndolas redundantes.

Esta fase corrige el input y prueba si una arquitectura relacional aprovecha
mejor esa información.

## Qué cambia respecto a Fase 2

- **Segundo conjunto de picos ¹³C.** Se extrae del pkl un `(δC,)` por carbono
  con entorno químico distinto — **todos** los carbonos, incluidos los sin H.
  Los cuaternarios son justamente los δC que están en el ¹³C pero no en los
  crosspeaks. Feature: solo δC (nunca el nº de H — sería fuga del label).
- **Normalización min-max** de los desplazamientos desde el config (δC/[0,220],
  δH/[-1,15]), en ambos conjuntos. Facilita la convergencia; E2 no normalizaba.
- **Dos arquitecturas** sobre la misma pipeline:
  - **DeepSets de dos ramas:** MLP por pico + promedio enmascarado por conjunto,
    fusión con la FM. No modela interacciones entre picos.
  - **Set Transformer:** self-attention sobre la unión de ambos conjuntos (con
    embedding de tipo) + pooling por atención (PMA). Sí puede aprender el
    "match entre conjuntos" que identifica cuaternarios por ausencia.

## Por qué dos arquitecturas

La tarea de detectar un cuaternario es **relacional** ("¿qué δC del ¹³C no
tiene match en los crosspeaks?"), y el promedio enmascarado del DeepSets es la
operación menos capaz de expresarla. Como entrenar cuesta ~16 min y la GPU
sobra, se corren las dos sobre el mismo input para aislar dos efectos:

- **DeepSets(F3) vs DeepSets(F2):** efecto de completar el input (cuaternarios).
- **Set Transformer vs DeepSets (F3):** efecto de la capacidad relacional.

## Capacidad: deliberadamente chica

DeepSets ~35k parámetros, Set Transformer ~70k — muy por debajo de los 8.6M de
V10 (que sobreajustó y perdió contra Exp C). Misma decisión de Fase 2: no
agrandar el modelo para no mezclar variables.

## Criterio de éxito / fracaso

- **EMA cruda:** ≥ 0.89% (Exp C, el mejor limpio hasta ahora).
- **EMA asistida (oráculo):** superar a Exp C (70.02%) y a E2 (70.90%),
  idealmente a V10 (74.92%). Objetivo del proyecto: ~90% asistida.
- **Indicador real:** que las confusiones `Cqsp2`↔`=CH/Ar` y `CH2`↔`CH2-N`
  **bajen** respecto a V10/B/C/E2 — confirmaría que el problema era el input
  incompleto.
- **Si ambas arquitecturas siguen fallando en los cuaternarios** pese al input
  completo, el cuello no es ni el input ni la arquitectura de sets → apunta al
  **HMBC simulado** (`docs/WORKFLOW_V11_para_ClaudeCode.md` líneas 39-41 /
  344-346) como el verdadero fix de dominio.

Spec completo: `docs/superpowers/specs/2026-07-22-exp-e-fase3-dos-conjuntos-picos-design.md`.
Plan: `docs/superpowers/plans/2026-07-22-exp-e-fase3-dos-conjuntos-picos.md`.
