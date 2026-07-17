# RATIONALE — Exp D: Val set congelado

## Hipótesis

Los resultados de V6..V10 no son estrictamente comparables entre sí: cada
versión particiona un dataset de tamaño distinto con `random_split(seed=42)`,
así que el 10% de val cambia de composición en cada corrida. Además, las
144k moléculas originales tienen del orden de miles de duplicados internos
por SMILES canónico, que pueden caer uno en train y su gemelo en val (fuga
de datos silenciosa). Congelar un val fijo y deduplicado hace que las EMAs
de Exp B, Exp C y cualquier versión futura sean comparables pie a pie.

## Qué causa del diagnóstico ataca

Rigor de comparación (crítica #5 del diagnóstico del Exp A). No ataca el
overfitting (eso lo ataca Exp B) ni el modality collapse (Exp C) — es una
precondición metodológica para que esos dos experimentos den números
confiables entre sí.

## Qué cambia exactamente respecto al V10

- Nada en el modelo ni en el dataset: `model_v10.py` y `dataset_v10.py` se
  copian sin modificar.
- El split: en vez de `random_split(seed=42, val_split=0.1)` sobre las
  202465 moléculas (lo que hace `train_v10.py`), el val queda fijo en las
  14428 moléculas "originales" de las 144k — el mismo `random_split`
  histórico que usó el training de V6-V9, aplicado al rango `[0, 144280)`
  del dataset de 202k (las 58185 nuevas se agregaron al final, sin
  reordenar las originales). Se guarda en `val_indices_frozen.npy`
  (vive en `DB_200k/`, no se versiona en git — igual que los `.h5` y los
  checkpoints).
- Cualquier fila de train cuyo SMILES canónico coincide con una fila de val
  se elimina de train (leak = 0). Las 58185 moléculas nuevas van siempre a
  train (ninguna está en `[0, 144280)`, donde vive el val histórico).
- Se re-evalúa el checkpoint V10 **ya entrenado**
  (`nmr_202k_v10_2ch_fm_19v_best.pth`) sobre este split nuevo — no se
  reentrena nada — para tener una referencia "V10-on-frozen-val" contra la
  que se compararán Exp B y Exp C.

## Qué métrica esperás mover y cuánto

**⚠️ Salvedad importante — "V10-on-frozen-val" no es un held-out limpio de V10.**
El val congelado son las 14428 moléculas "originales" de las 144k, elegidas
reproduciendo el `random_split(seed=42)` histórico de V6-V9 sobre el rango
`[0, 144280)`. Pero el checkpoint que se re-evalúa acá
(`nmr_202k_v10_2ch_fm_19v_best.pth`) fue entrenado con OTRO `random_split`,
el de `train_v10.py` (seed=42 también, pero aplicado sobre las 202465
moléculas completas — una permutación aleatoria distinta, sobre un rango
distinto). Los dos splits no tienen relación estructural entre sí: comparten
la seed, no la partición. En la práctica esto significa que ~90% de las
14428 moléculas del val congelado ya fueron vistas por V10 durante su propio
entrenamiento. El número "V10-on-frozen-val" es entonces una referencia
aproximada / ancla, no una evaluación held-out honesta de V10 específicamente.

Por eso, la heurística ingenua de "un cambio grande (>5pp) es sospechoso, hay
que parar e investigar" no aplica tal cual a este número: un salto real y
legítimo respecto al 0.61% / 74.92% original es **esperable** acá, justamente
por este solapamiento train/val, y no es por sí solo señal de bug. La señal
real de que algo está roto no es la magnitud del delta, sino que fallen las
verificaciones que ya corre `split.py`: leak=0 (intersección train∩val por
SMILES canónico) y los conteos de duplicados/SMILES inválidos reportados. Si
esas verificaciones pasan, un EMA alto en "V10-on-frozen-val" es coherente
con la contaminación descripta arriba, no un error.

Exp B y Exp C, a diferencia de V10, SÍ van a excluir el val congelado de su
propio entrenamiento — así que sus números sobre este mismo val van a ser
evaluaciones held-out limpias y honestas. La comparación "B/C vs
V10-on-frozen-val" hay que leerla con esta asimetría en mente: no es
apples-to-apples estricto, es "número limpio de B/C" contra "ancla
aproximada de V10".

## Criterio de éxito/fracaso

- **Éxito:** se reporta el número de duplicados canónicos encontrados, se
  genera `val_indices_frozen.npy` con ~14428 índices, la verificación
  train ∩ val (por SMILES canónico) da 0, y se obtiene el EMA
  cruda/asistida del V10 sobre ese split nuevo.
- **Fracaso:** el script no logra reproducir un split de tamaño razonable
  (p. ej. por desalineación de orden entre el dataset de 144k histórico y
  el de 202k), o la verificación de leak falla y no se puede corregir sin
  tocar el val (que debe quedar fijo).
