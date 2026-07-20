# RATIONALE — Exp C: Rebalanceo de la fusión (GAP)

## Hipótesis

El modelo V10 tiene un desbalance severo en cómo fusiona sus tres ramas: la rama
convolucional (imagen HSQC) aporta 65536 features al vector de fusión, contra 128 de
la rama 1D (proyecciones ¹³C/¹H) y 8 de la Fórmula Molecular — una proporción de 512:1.
Ese único `Linear(65664 → 128)` concentra ~8.4M de los ~8.6M de parámetros del modelo
entero. La hipótesis (documentada desde V4 como "modality collapse") es que la red se
"ahoga" en la imagen e ignora las otras dos ramas, que llevan información que la imagen
NO tiene (ej. la Fórmula Molecular es la única señal directa sobre carbonos cuaternarios
sp2, invisibles en HSQC). Reemplazar el flatten por Global Average Pooling (GAP) fuerza
a la red a resumir la imagen en 64 números (uno por canal), balanceando la fusión a
64+128+8=200 y dejando que las otras ramas compitan en igualdad de condiciones.

## Qué causa del diagnóstico ataca

Causa #3 del diagnóstico del Exp A: desbalance de ramas / modality collapse. No ataca
el overfitting (eso lo intentó Exp B, sin éxito) — es un cambio arquitectónico
ortogonal.

## Qué cambia exactamente respecto al V10

- `model_c.py`: copia de `model_v10.py` donde el único cambio es reemplazar
  `x1.view(-1, self.flat_dim)` (flatten a 65536) por `nn.AdaptiveAvgPool2d(1)` seguido
  de `x1.view(-1, 64)`. `fusion_dim` pasa de 65672 a 200; `fc_fusion1 = nn.Linear(200, 128)`.
  Efecto colateral (no buscado pero real): el modelo pasa de ~8.6M a ~223k parámetros
  (~38x menos), muy por encima del "~10x" que estimaba el workflow original.
- **Sin regularización.** Exp B (dropout=0.25 + weight_decay=1e-5) dio underfitting
  real (val loss 6x más alto que V10, EMA cruda 0%) — se abandona esa línea por ahora
  para no mezclar dos variables. Si Exp C funciona, combinar con regularización más
  suave queda para una fase posterior.
- **Mismo split congelado que Exp D/B:** `val_indices_frozen.npy`, reconstruido vía
  `split_utils.py` (copiado, misma lógica que ya usó Exp B). Mismo loss
  (`ConstrainedMSELoss`), mismo scheduler (`patience=8, factor=0.7`), 100 épocas,
  `num_workers=0`.

## Riesgo conocido (por la experiencia de Exp B)

GAP es, en sí mismo, una compresión agresiva: descarta toda la información espacial y
se queda solo con el promedio por canal. Existe el mismo riesgo de underfitting que
tumbó a Exp B, aunque por una vía distinta (arquitectura, no regularización). Señal de
alerta temprana (ya instrumentada en `train.py`): si en la época 10, 20 o 30 el val
loss sigue por encima de 0.10 (V10 real está en 0.031), es la misma pinta que tuvo Exp
B — no hace falta esperar a que termine el entrenamiento completo para sospechar. Si
eso pasa, el siguiente paso es la variante alternativa que preveía el workflow original:
un bottleneck `Linear` explícito (ej. 65536→256) en vez de GAP puro, que preserva más
información que el promedio pero sigue rebalanceando la fusión. Esa variante NO está
construida todavía (se arma si hace falta, con el diagnóstico real en mano).

## Qué métrica esperás mover y cuánto

- Parámetros totales: de ~8.6M a ~223k (~38x menos) — verificable sin entrenar, vía el
  smoke test.
- Val loss: se espera en el mismo orden de magnitud que V10 (0.03-0.05). Si queda
  pegado en 0.15+ como Exp B, es la señal de que GAP también underfittea.
- EMA cruda: objetivo mínimo ≥ 0.61% (el baseline real de V10). El verdadero indicador
  de que la hipótesis es correcta es el **MAE de Cqsp2 y `=CH/Ar`** (las clases que
  dependen de la rama 1D/FM, no de la imagen): si el rebalanceo funciona, estas dos
  deberían mejorar más que el resto, porque antes la rama que las informa era ignorada.

## Criterio de éxito/fracaso

- **Éxito:** val loss en el rango de V10 (no explota a 0.15+), EMA cruda ≥ 0.61%, y el
  MAE de Cqsp2/`=CH/Ar` mejora respecto a V10.
- **Fracaso (underfitting por GAP):** val loss estancado por encima de 0.10 desde la
  época ~20-30 — pasar a la variante FC-bottleneck en un experimento de seguimiento.
- **Fracaso (sin mejora pero sin underfitting):** EMA similar a V10 pero MAE de
  Cqsp2/`=CH/Ar` sin cambios — indicaría que el desbalance de ramas no era la causa
  dominante, y hay que mirar otra cosa (ej. la representación de picos como conjunto,
  Exp E, o la loss function).
