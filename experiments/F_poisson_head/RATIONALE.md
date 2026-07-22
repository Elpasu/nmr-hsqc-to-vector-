# RATIONALE — Exp F: cabeza Poisson + entrenamiento extendido

## Hipótesis

La EMA cruda nunca superó ~2-3% en ningún experimento del proyecto
(V10 0.61%, Exp C 0.89%, E2 0.74%, DeepSets F3 2.28%, Set Transformer F3
2.26%), pese al salto de Fase 3 en la asistida (91.35%). `WORKFLOW_V11`
(sección "Notas de dominio") ya señalaba que la loss actual (MSE sobre
conteos + redondeo) es conceptualmente floja para conteos: "la
incertidumbre de un conteo escala con su magnitud; MSE la trata como
constante". Nunca se probó una cabeza Poisson. Además, en Fase 3 el LR
nunca bajó de 0.001 en 100 épocas (el scheduler no hizo plateau) — señal
de que el presupuesto de épocas pudo quedarse corto.

## Qué cambia respecto a Fase 3 (Set Transformer)

- **Cabeza:** activación `softplus` en la salida (`model_f_settransformer.py`)
  en vez de una salida lineal cruda. Garantiza `lambda >= 0` sin la
  inestabilidad de `exp` en logits grandes al inicio del entrenamiento.
- **Loss:** `ConstrainedPoissonLoss` (Poisson NLL por clase,
  `log_input=False` porque el input ya es `lambda`, más el mismo término
  de restricción de suma que `ConstrainedMSELoss`, `lambda_sum=0.5`) en
  vez de MSE.
- **Épocas:** 100 → 250. Scheduler sin cambios (`patience=8, factor=0.7`,
  regla dura del proyecto).
- El oráculo de post-proceso, el dataset, el split congelado y la
  arquitectura del Set Transformer (`d_model=64/n_heads=4/n_layers=2/n_seeds=1`)
  no cambian.

## Por qué Poisson y no clasificación ordinal

Ambas se mencionan en `WORKFLOW_V11` como alternativas a MSE. Poisson es
un cambio quirúrgico: no toca la forma de la salida (`(B, 19)`, no
`(B, 19, K)`), no toca el dataset, y el oráculo sigue funcionando sin
rediseño porque sigue recibiendo un valor continuo con parte fraccionaria
interpretable. Ordinal exigiría rediseñar la salida y el oráculo — se deja
como línea futura si Poisson no rinde.

## Caveat de comparabilidad

El val loss (Poisson NLL) no es comparable numéricamente contra el 0.0097
(MSE) de Fase 3 — son escalas distintas. La comparación real es por EMA
cruda/asistida y por el mapa de confusiones cruzadas (`evaluate.py`).

## Criterio de éxito / fracaso

- **EMA asistida:** ≥ 91.35% (Fase 3), apuntando a acercarse a 95%+.
- **EMA cruda:** mejora real y medible por sobre el techo histórico ~2-3%.
  Cualquier valor que se quede en ese rango es evidencia de que la cabeza
  Poisson no era el cuello de botella.
- **Confusiones:** `CH2`↔`CH2-N`, `C-2X` y `=CH/Ar`↔`Imina` deben bajar
  respecto a Fase 3.
- **Si falla:** ni la cabeza de salida ni el presupuesto de entrenamiento
  eran el cuello — apunta a un problema de información (HMBC simulado,
  fuera de alcance de este experimento) más que de modelado.

Spec completo: `docs/superpowers/specs/2026-07-22-exp-f-poisson-y-escalado-design.md`.
Plan: `docs/superpowers/plans/2026-07-22-exp-f-poisson-y-escalado.md`.
