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

Ninguna EMA debería moverse por una razón de aprendizaje — es el mismo
checkpoint congelado. Se espera un cambio *pequeño* en el número (el val
ahora es más chico, sin los 58k scaffolds nuevos, sin duplicados con fuga)
respecto al 0.61% / 74.92% original. Un cambio grande (>5pp) sería señal de
que el val original y el nuevo tienen dificultad muy distinta, y vale la
pena investigarlo antes de seguir con B/C.

## Criterio de éxito/fracaso

- **Éxito:** se reporta el número de duplicados canónicos encontrados, se
  genera `val_indices_frozen.npy` con ~14428 índices, la verificación
  train ∩ val (por SMILES canónico) da 0, y se obtiene el EMA
  cruda/asistida del V10 sobre ese split nuevo.
- **Fracaso:** el script no logra reproducir un split de tamaño razonable
  (p. ej. por desalineación de orden entre el dataset de 144k histórico y
  el de 202k), o la verificación de leak falla y no se puede corregir sin
  tocar el val (que debe quedar fijo).
