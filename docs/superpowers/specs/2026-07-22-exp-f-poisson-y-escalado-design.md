# Exp F: cabeza Poisson + entrenamiento extendido, y estudio de escalado de datos (design)

> Punto de partida: Exp E Fase 3 — Set Transformer (`docs/Runs/RESULTS.md`, sección
> "Exp E — Fase 3"), **mejor resultado del proyecto**: EMA asistida 91.35%, EMA cruda
> 2.26%, primera vez que se cruza el objetivo ~90% asistida en evaluación limpia, y
> rompe la confusión estructural `Cqsp2`↔`=CH/Ar` que sobrevivió a cinco arquitecturas
> (V10/B/C/E2/DeepSets-F3). Quedan dos preguntas abiertas, independientes entre sí, que
> este documento cubre en dos partes.

## Motivación y alcance

**Parte 1 (Exp F).** La EMA cruda (2.26%) sigue muy por debajo de la asistida pese al
salto de Fase 3 — nunca superó ~2-3% en ningún experimento del proyecto. `WORKFLOW_V11`
(sección "Notas de dominio") ya señalaba, antes de que existiera Fase 3, que la loss
actual (MSE sobre conteos + redondeo) es conceptualmente floja: "la incertidumbre de un
conteo escala con su magnitud; MSE la trata como constante", y sugiere una cabeza
Poisson u ordinal. Nunca se probó. Además, en ninguna corrida de Fase 3 el LR bajó de
0.001 en 100 épocas (el scheduler nunca hizo plateau) — señal de que el presupuesto de
épocas puede haberse quedado corto. Se combinan ambos cambios en un solo experimento
(decisión explícita del usuario, sabiendo que mezcla dos variables): cabeza Poisson +
más épocas, sobre el Set Transformer de Fase 3 sin ningún otro cambio.

**Parte 2 (estudio de escalado).** Pregunta separada: si en el futuro se vuelve a
ampliar el dataset (como ya pasó una vez, 144k→202k), ¿serviría? Se mide la EMA del Set
Transformer de Fase 3 (arquitectura y loss **sin cambios**, para no mezclar esta
pregunta con la de la Parte 1) entrenado sobre fracciones crecientes del train set,
manteniendo siempre el mismo val congelado de Exp D. La pendiente de la curva entre el
75% y el 100% es el indicador real: si sigue subiendo fuerte, más datos ayudaría: si se
aplana, ya estamos en meseta y el esfuerzo rinde más en otro lado (representación,
arquitectura, dominio).

**Alcance:** como en toda fase anterior, dejar todo listo para `sbatch` (extracción de
datos si hiciera falta — acá no hace falta, se reusa `peaks_pkl_202465.npz` y
`peaks_13c_202465.npz` de Fase 3 tal cual). Entrenar y evaluar lo corre Lucas en el
cluster.

---

## Parte 1 — Exp F: cabeza Poisson + entrenamiento extendido

### Carpeta nueva

`experiments/F_poisson_head/` — copia (no import) de los archivos de la variante Set
Transformer de `E3_dos_conjuntos/`: `dataset_e3.py` → `dataset_f.py` (sin cambios de
lógica, solo nombre), `model_e3_settransformer.py` → `model_f_settransformer.py` (sin
cambios de arquitectura: mismo `d_model=64/n_heads=4/n_layers=2/n_seeds=1`), `train.py`,
`evaluate.py`, `dump_predictions.py`, `split_utils.py`, `config.yaml`, `run_train.sh`,
`run_eval.sh`, `tests/`, `README.md`, `RATIONALE.md`. Solo se entrena Set Transformer
(el ganador de Fase 3) — DeepSets no se toca en esta fase.

### Cambio 1 — cabeza y loss

- El modelo agrega una activación `softplus` sobre la salida de `fc_out` antes de
  devolverla: `λ = softplus(logits)`, garantiza `λ ≥ 0` sin la inestabilidad numérica de
  `exp` en logits grandes al inicio del entrenamiento.
- `PoissonNLLLoss` (`torch.nn.PoissonNLLLoss(log_input=False, full=True)`) reemplaza el
  término `mse(pred, target)` de `ConstrainedMSELoss`. Se mantiene el término de
  restricción de suma (`0.5 * MSE(sum(pred), sum(target))`) igual que hoy — mismo
  espíritu de "doble restricción" que ya usa el oráculo de post-proceso. Nueva clase
  `ConstrainedPoissonLoss(lambda_sum=0.5)`, mismo patrón que `ConstrainedMSELoss`.
- El post-proceso (`ajustar_conteo_doble_exacto` / oráculo, en `evaluate.py` y
  `dump_predictions.py`) **no cambia**: sigue recibiendo un `pred_raw` continuo (ahora
  es `λ` en vez del valor MSE) con parte fraccionaria interpretable como confianza
  relativa, exactamente como hoy.
- **Por qué Poisson y no clasificación ordinal** (la otra opción que menciona
  `WORKFLOW_V11`): es un cambio quirúrgico — no toca el dataset, no cambia la forma de
  la salida (`(B, 19)`, no `(B, 19, K)`), y el oráculo sigue funcionando sin rediseño.
  Ordinal se deja fuera de alcance (ver abajo).

### Cambio 2 — entrenamiento extendido

- `epochs: 100 → 250` en `config.yaml`. Scheduler **sin cambios**: `patience=8,
  factor=0.7` (regla dura del proyecto, no se toca). Mismo seed 42, mismo batch 64,
  mismo `num_workers=0`.
- Instrumentación: el log de `train.py` ya imprime el LR por época (no requiere cambio).
  Señal de alerta temprana: si a la época 100 el LR sigue en el valor inicial
  (0.001) igual que en Fase 3, es evidencia de que ni con Poisson se movió la loss lo
  suficiente para que el scheduler reaccione — se puede leer sin esperar las 250
  completas.

### Caveat de comparabilidad

El valor numérico del val loss (Poisson NLL) **no es comparable** contra el 0.0097 MSE
de Fase 3 — son escalas distintas. La comparación entre F3 y F se hace exclusivamente
por EMA cruda/asistida y por la matriz de confusiones cruzadas, igual que se hizo con
Exp C (que cambió el conteo de parámetros de forma no buscada y por eso también se leyó
por EMA, no por arquitectura).

### Qué se mantiene idéntico (comparabilidad con Fase 3)

Dataset (`NMRTwoSetsDataset`, dos conjuntos de picos, normalización min-max), split
congelado de Exp D, arquitectura del Set Transformer (`d_model/n_heads/n_layers/n_seeds`
sin cambios), `cond_tensor` (FM, 8 valores), scheduler `patience=8/factor=0.7`, seed 42,
`num_workers=0`, `num_classes=19` y orden de clases fijo.

### Testing

Smoke test offline obligatorio antes de `sbatch` (regla 5): forward con batch sintético
verificando shape `(B, 19)`, `λ ≥ 0` para cualquier input (incluso logits negativos
grandes), y conteo de parámetros idéntico a Fase 3 (~70k, la cabeza Poisson no agrega
parámetros, solo cambia la activación final). Test unitario de `ConstrainedPoissonLoss`
con valores sintéticos (verificar que no da NaN/Inf con `λ=0` y `target=0`, caso límite
real: moléculas con cero átomos de una clase).

### Criterio de éxito / fracaso

- **EMA asistida:** ≥ 91.35% (Fase 3), apuntando a acercarse a 95%+.
- **EMA cruda:** mejora real y medible por sobre el techo histórico ~2-3% — cualquier
  valor por debajo de eso (p.ej. quedarse en 2.x%) es evidencia de que la cabeza Poisson
  no fue el cuello de botella.
- **Confusiones:** `CH2`↔`CH2-N`, `C-2X` y el frente `=CH/Ar`↔`Imina` deben bajar
  respecto a Fase 3 (mapa de confusiones cruzadas, ya lo reporta `evaluate.py`).
- **Lectura si falla:** si ni la EMA cruda ni las confusiones se mueven, el cuello ya no
  es la cabeza de salida ni el presupuesto de entrenamiento — apunta a un problema de
  información (HMBC simulado, fuera de alcance) más que de modelado.

---

## Parte 2 — Estudio de escalado de datos

### Dónde vive

**No** es una arquitectura nueva — es una ablación sobre Fase 3 sin cambios de modelo
ni de loss. Vive dentro de `experiments/E3_dos_conjuntos/` (no se crea carpeta nueva):

- `train.py` gana un campo opcional de config `hyperparameters.train_fraction` (default
  `1.0`, no rompe los configs existentes de Fase 3). Si `train_fraction < 1.0`, después
  de calcular `train_idx` en `build_frozen_split`, se subsamplea de forma determinística
  con `np.random.RandomState(42).permutation(train_idx)[:int(len(train_idx) *
  train_fraction)]`. El val congelado (`val_idx`, 14428 moléculas) **nunca se toca**.
- 5 configs nuevos: `config_scaling_10.yaml`, `_25.yaml`, `_50.yaml`, `_75.yaml`,
  `_100.yaml` — copias de `config_settransformer.yaml` con distinto
  `train_fraction`, `experiment_name` y `checkpoint_dir` (para no pisar el checkpoint
  de Fase 3). Arquitectura y loss (`ConstrainedMSELoss`, MSE — no Poisson) sin cambios.
- `run_train_scaling.sh`: mismo patrón que `run_train_settransformer.sh`, parametrizado
  por `--config` para lanzar las 5 corridas con `sbatch`.

### Fracciones

10/25/50/75/100% de las ~187 314 moléculas de train (Exp D) → aprox. 18 731 / 46 829 /
93 657 / 140 486 / 187 314 moléculas. 100 épocas cada una (no 250 — esto es
diagnóstico, no el experimento principal; los subconjuntos chicos entrenan más rápido
por época). Costo estimado: ~15-20 min por corrida al 100% (igual que Fase 3), menos en
los subconjuntos chicos → **~1.5h de GPU en total para las 5**.

### Qué se mide

Cada corrida se evalúa con el `evaluate.py` de Fase 3 tal cual (mismo val congelado para
las 5). Salida: una tabla EMA cruda/asistida vs tamaño de train, agregada a mano en
`docs/Runs/RESULTS.md` (una fila por fracción). No se arma script de ploteo en el
cluster — Lucas copia los 5 números y yo puedo graficar la curva localmente si hace
falta (matplotlib, EMA vs N en escala log en X).

### Qué NO se aísla (fuera de alcance, ya charlado)

No se separa el efecto de las 58k moléculas nuevas (composicionalmente distintas: mucho
más contenido nitrogenado, más sp2 — ver "Auditoría de distribución de clases" en
`RESULTS.md`) del efecto de volumen puro. El muestreo de cada fracción es aleatorio
sobre el pool completo (144k+58k mezcladas), así que cada subconjunto tiene
aproximadamente la misma composición que el 100% — la curva mide únicamente "importa la
cantidad", no "importa qué tipo de moléculas". Se decidió así explícitamente por
alcance/tiempo; queda como posible estudio de seguimiento si la curva de volumen da señal
ambigua.

### Testing

Test unitario del subsampleo determinístico: mismo `train_fraction` + mismo seed →
mismos índices en dos corridas; fracciones distintas → subconjuntos anidados (el 25% es
superset del 10% en la misma corrida de `RandomState(42).permutation`, útil para leer la
curva como genuinamente incremental). No requiere torch, corre local. El resto reusa los
smoke tests ya existentes de Fase 3 (`test_forward_settransformer.py`), sin cambios.

---

## Estructura final de archivos

```
experiments/F_poisson_head/
  config.yaml, dataset_f.py, model_f_settransformer.py, train.py, evaluate.py,
  dump_predictions.py, split_utils.py, run_train.sh, run_eval.sh,
  tests/test_forward.py, tests/test_poisson_loss.py, README.md, RATIONALE.md

experiments/E3_dos_conjuntos/          (ya existe, se agrega:)
  config_scaling_10.yaml, config_scaling_25.yaml, config_scaling_50.yaml,
  config_scaling_75.yaml, config_scaling_100.yaml, run_train_scaling.sh,
  tests/test_train_fraction.py
```

## Fuera de alcance

- Clasificación ordinal como cabeza de conteo (alternativa a Poisson) — cambio más
  invasivo (forma de salida, rediseño del oráculo), se deja como línea futura si Poisson
  no rinde.
- Aislar el efecto composicional de las 58k moléculas nuevas (144k-solas vs 144k+58k) —
  decisión explícita del usuario, ver sección "Qué NO se aísla" arriba.
- Ejecutar la Parte 1 (Poisson) también sobre DeepSets — el foco es el modelo ganador
  (Set Transformer).
- HMBC simulado — cambio de datos mayor, pipeline nuevo en snmgt01, fuera del alcance
  diario de este repo.
- Ploteo automatizado de la curva de escalado en el cluster — se arma localmente con los
  5 números que traiga Lucas.
