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
