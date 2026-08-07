# Exp I — Estudio de hiperparámetros del Set Transformer (E3)

**Fecha:** 2026-08-07
**Estado:** diseño aprobado, pendiente de plan de implementación
**Modelo bajo estudio:** `experiments/E3_dos_conjuntos/model_e3_settransformer.py` (Set Transformer,
70 163 parámetros), config congelada `config_settransformer.yaml`.

---

## 1. Motivación

El checkpoint E3 está **congelado** (decisión del 2026-07-27): EMA asistida v2 92.14 % en Clementina
XPU, ~97 % de cobertura@K con Fase 1b, y el trabajo se movió a inferencia experimental (Exp H) y al
acople con el generador. Este estudio **no busca reemplazarlo**.

Lo que busca es responder, con evidencia y no con una anécdota, a la pregunta que un revisor va a
hacer: *"¿por qué esta arquitectura y no otra? ¿probaron más capas, más neuronas?"*. Hoy la respuesta
sería "es lo que salió del Exp E Fase 3", que no es una justificación.

Existe además una **predicción previa fuerte** que este estudio pone a prueba: el Exp F (cabeza
Poisson + 250 épocas, con el scheduler llegando a saturación real, LR 0.001 → 0.00002) **no mejoró**
la EMA, y el estudio de escalado de datos mostró **meseta** entre 75 % y 100 % del train set. Ambos
apuntan a que el cuello de botella ya no es de optimización ni de datos, sino de información/dominio.
Si el sweep de hiperparámetros también da meseta, las tres evidencias convergen y la conclusión pasa
de ser una hipótesis a estar triangulada.

**Una meseta es un resultado válido y esperado, no un fracaso del estudio.** Permite la afirmación
más fuerte que se puede hacer sobre una elección de arquitectura: *"el modelo elegido es el más chico
dentro de la meseta de rendimiento"*.

---

## 2. Ubicación en el repo (y por qué NO una carpeta nueva)

El estudio vive **dentro de `experiments/E3_dos_conjuntos/`**, no en `experiments/I_.../`.

Esto rompe a propósito la convención del proyecto de "carpeta autocontenida, copiar no importar". La
razón: el estudio solo tiene valor si el código de entrenamiento y evaluación es **bit a bit el
mismo** que produjo el checkpoint congelado. Copiar `train.py` / `dataset_e3.py` / `evaluate.py` a
otra carpeta abriría exactamente el riesgo de divergencia silenciosa que invalidaría toda la
comparación — el mismo tipo de error que la regla dura 7 previene para el orden de clases.

Existe precedente exacto: el **estudio de escalado de datos** (Exp E, 2026-07-22) vivió en esta misma
carpeta como `config_scaling_{10,25,50,75,100}.yaml` + `run_train_scaling.sh` parametrizado, y su
tabla está en `docs/Runs/RESULTS.md`.

```
experiments/E3_dos_conjuntos/
  hp_sweep/
    sweep_grid.yaml            # el DISEÑO del estudio: baseline + ejes + valores. Fuente única.
    make_configs.py            # genera los config_hp_*.yaml desde sweep_grid.yaml (yaml puro)
    configs/                   # los 22 configs generados, COMMITEADOS (auditables)
    collect_results.py         # parsea los .out de SLURM -> tabla markdown + CSV
    make_plot.py               # figura OFAT con banda de ruido
    tests/test_make_configs.py
    tests/test_all_archs_forward.py
  run_sweep.sh                 # sbatch run_sweep.sh hp_sweep/configs/X.yaml (train + eval en UN job)
```

`configs/` se commitea aunque sea generado: un revisor tiene que poder leer exactamente qué se
entrenó, sin ejecutar nada.

---

## 3. Diseño experimental

Baseline (la config congelada):
`d_model=64, n_heads=4, n_layers=2, n_seeds=1, learning_rate=1e-3, batch_size=64, fusion=128→64`.

### 3.1 OFAT — una dimensión a la vez (16 corridas)

| # | Eje | Valores probados (baseline entre paréntesis) | Corridas |
|---|---|---|---|
| 1 | `model.d_model` | 32, 128, 256 (64) | 3 |
| 2 | `model.n_layers` | 1, 3, 4 (2) | 3 |
| 3 | `model.n_heads` | 2, 8 (4) | 2 |
| 4 | `model.n_seeds` (semillas del PMA) | 2, 4 (1) | 2 |
| 5 | `hyperparameters.learning_rate` | 3e-4, 3e-3 (1e-3) | 2 |
| 6 | `hyperparameters.batch_size` | 32, 128 (64) | 2 |
| 7 | `model.fusion_hidden` | [64, 32], [256, 128] ([128, 64]) | 2 |

**Total OFAT: 16.**

Los ejes 1–4 y 7 son "capas y número de neuronas" (la pregunta literal del revisor). Los ejes 5–6 son
de optimización y se incluyen porque son la primera objeción alternativa: *"quizá la arquitectura
grande sí ayuda pero con otro learning rate"*.

**Restricción dura de la arquitectura:** `MAB.forward` calcula `d = dim_V // num_heads`, así que
`d_model` debe ser divisible por `n_heads`. Todas las combinaciones listadas la cumplen
(64/2, 64/8, 32/4, 128/4, 256/4). El generador de configs valida esto y el test lo verifica.

### 3.2 Mini-grid 2D `d_model × n_layers` (4 corridas nuevas)

El grid completo {32, 64, 128} × {1, 2, 4} son 9 celdas, de las cuales **5 ya están cubiertas** por el
baseline y el OFAT (64×2 = baseline; 32×2, 128×2 del eje 1; 64×1, 64×4 del eje 2). Faltan **4**:

| d_model | n_layers |
|---|---|
| 32 | 1 |
| 32 | 4 |
| 128 | 1 |
| 128 | 4 |

Responde a la objeción específica: *"¿probaron que `d_model=128` no ayuda **con más profundidad**?"* —
que el OFAT puro, por construcción, no puede contestar.

### 3.3 Piso de ruido (2 corridas)

Baseline reentrenado con `seed = 43` y `seed = 44`. La `seed = 42` ya existe: es la corrida A10
original del Exp E Fase 3 (EMA asistida 91.35 %), que se reutiliza como tercera réplica.

Sin esto no hay forma de decirle a un revisor si un +0.6 pp es señal o azar. Evidencia previa que
motiva la medición: la **misma config exacta** dio 91.35 % en A10 y 92.14 % en XPU — ~0.8 pp de
variación atribuible solo a la trayectoria estocástica en FP32 (documentado en `RESULTS.md`,
sección de migración XPU). Ese dato es n=2 y confunde hardware con seed; las 3 réplicas lo miden
limpio.

### 3.4 Presupuesto

**22 corridas nuevas** × ~39 min (A10, 100 épocas) ≈ **14 h de GPU**, más `evaluate.py` por corrida
(~2–3 min cada una, incluida en el mismo job SLURM).

---

## 4. Invariantes — lo que NO cambia en ninguna de las 22

Si algo de esta lista varía entre corridas, las EMAs dejan de ser comparables (regla dura 8) y el
estudio no vale nada:

- **Val set congelado:** `val_indices_frozen.npy` (Exp D), 14 428 moléculas, en las 22.
- **Épocas:** 100 en todas. Sin screening corto — a 39 min/corrida el ahorro es marginal y meter
  presupuestos distintos introduce un confound: un modelo más grande podría necesitar más épocas y
  perdería injustamente.
- **Loss:** `ConstrainedMSELoss(lambda_sum=0.5)`.
- **Scheduler:** `ReduceLROnPlateau(patience=8, factor=0.7)` — regla dura 6.
- **`num_workers: 0`** — regla dura 1.
- **19 clases en el orden de `config/db.yaml`** — regla dura 7.
- **Dataset:** los mismos `.npz` / `.npy` de 202 465 moléculas, `train_fraction = 1.0`.
- **Un solo cluster para las 22.** Recomendado: **login-1 / A10**, porque es donde vive toda la serie
  histórica (incluida la corrida seed=42 que se reutiliza) y es 1.8× más rápido que XPU. El
  checkpoint desplegado sigue siendo el de Clementina; este estudio es sobre la *elección de
  arquitectura*, no sobre qué checkpoint se sirve.

**Excepción explícita:** en las corridas del piso de ruido (§3.3) la seed **sí** varía — es la
variable bajo estudio de esa sub-medición.

### 4.1 `checkpoint_dir` único por corrida

Cada config generado escribe en su propio `checkpoint_dir`. Si dos configs lo compartieran, la
segunda corrida pisaría el checkpoint de la primera **sin tirar ningún error**, y la evaluación
posterior reportaría la EMA equivocada atribuida al modelo equivocado. El generador lo garantiza por
construcción (deriva el nombre del `experiment_name`) y `test_make_configs.py` lo verifica.

---

## 5. Métrica de decisión

**Primaria: EMA asistida v2** (oráculo con zeroing por heteroátomos) — es la métrica del checkpoint
congelado y la que define el ranking.

**Desempate: best val loss** (`ConstrainedMSELoss`), útil porque es continua y menos ruidosa.

**Se reporta pero NO decide: EMA cruda.** `RESULTS.md` ya documenta que es ruidosa e inestable
(0.9 %–2.3 % a lo largo del scaling study, sin tendencia monótona pese a que la asistida subía
monótonamente). Usarla para elegir sería elegir por ruido.

**Regla de interpretación:** una diferencia menor a la banda de ruido medida en §3.3 se declara
**no significativa**. Esto se fija *antes* de ver los resultados, para que no haya sospecha de haber
elegido el criterio después de mirar los números.

`evaluate.py --oraculo all` ya emite las tres métricas en una tabla de 3 vías; no hace falta tocarlo.

---

## 6. Cambios de código necesarios

Los tres son **aditivos y con default idéntico al comportamiento actual**. El `state_dict` del
checkpoint congelado sigue cargando sin cambios.

1. **`train.py` — seed configurable.** Hoy la línea 105 hace `set_seed(42)` hardcodeado. Pasa a
   `set_seed(cfg['hyperparameters'].get('seed', 42))`. Necesario para §3.3.
2. **`model_e3_settransformer.py` — `fusion_hidden` parametrizable.** Hoy `NMR_SetTransformer.__init__`
   crea `fc_fusion1 = Linear(fusion_dim, 128)` y `fc_fusion2 = Linear(128, 64)` con dimensiones
   fijas. Pasa a recibir `fusion_hidden=(128, 64)` como parámetro con ese default. Necesario para el
   eje 7.
3. **`train.py::build_model` — propagar `fusion_hidden`** desde `cfg['model']` al constructor, con el
   mismo default.

`train_fraction`, `d_model`, `n_heads`, `n_layers`, `n_seeds`, `learning_rate`, `batch_size` y el
scheduler **ya** son configurables — no requieren cambios.

---

## 7. Componentes

### 7.1 `sweep_grid.yaml` — el diseño, declarativo

Fuente única de verdad del estudio: el baseline, los ejes con sus valores, las celdas del grid 2D y
las seeds del piso de ruido. Un revisor lee este archivo y entiende el diseño completo sin leer
código Python.

### 7.2 `make_configs.py` — generador

Lee `sweep_grid.yaml` + `config_settransformer.yaml` (el baseline real, no una copia) y escribe los
22 YAML en `configs/`. Deriva `experiment_name` y `checkpoint_dir` de forma determinística y única.
Valida `d_model % n_heads == 0` y falla fuerte si no se cumple. Sin torch: corre y se testea en la PC
local.

### 7.3 `run_sweep.sh` — un job SLURM por combinación

Toma el config como argumento y corre **`train.py` y después `evaluate.py` en el mismo job**:

```bash
sbatch run_sweep.sh hp_sweep/configs/hp_dmodel_128.yaml
```

Que sean un solo job (y no dos) es deliberado: 22 `sbatch` en vez de 44, y elimina la clase de error
"evalué un checkpoint que todavía no terminó de entrenarse". `#SBATCH --time=01:30:00` cubre
39 min de train + eval con margen. Usa `--gres=gpu:1` (regla dura 2).

### 7.4 `collect_results.py` — recolector

Parsea los `.out` de SLURM (best val loss de `train.py`, las tres EMAs de `evaluate.py`, tiempo, nº de
parámetros) y emite un CSV + la tabla markdown lista para pegar en `RESULTS.md`. Corre local sobre
los `.out` que Lucas baje del cluster. Tolerante a corridas faltantes (marca la fila como pendiente
en vez de romper).

### 7.5 `make_plot.py` — figura

Un panel por eje OFAT: EMA asistida v2 vs. valor del eje, con la **banda de ruido ±σ sombreada** y el
baseline marcado. Es la figura que contesta visualmente "¿por qué esta combinación?". Sigue el patrón
de `experiments/E3_dos_conjuntos/plots/make_plots.py` (matplotlib, sin torch).

---

## 8. Testing

Todo local, en CPU, **antes** de gastar cola de GPU (regla dura 5).

### 8.1 `tests/test_make_configs.py`

- Se generan exactamente 22 configs.
- Cada config OFAT difiere del baseline en **exactamente una** clave (comparación recursiva de dicts).
  Las claves de identidad `experiment_name` y `paths.checkpoint_dir` se excluyen de esta comparación:
  son únicas por construcción en las 22 y no forman parte del diseño experimental.
- `d_model % n_heads == 0` en las 22.
- `experiment_name` único en las 22; `checkpoint_dir` único en las 22.
- Los invariantes de §4 (épocas, scheduler, `num_workers`, `val_indices_filename`, `train_fraction`,
  paths del dataset) son idénticos en las 22.
- Las corridas del grid 2D tienen los `(d_model, n_layers)` esperados y no duplican celdas del OFAT.
- Las corridas de ruido difieren del baseline **solo** en `seed` (con la misma exclusión de las claves
  de identidad).

### 8.2 `tests/test_all_archs_forward.py`

Instancia el modelo de **cada uno de los 22 configs** y hace un forward en CPU con shapes reales
(batch pequeño, máscaras mixtas, incluida una molécula totalmente enmascarada). Verifica que la
salida sea `(B, 19)` y no contenga NaN.

Cumple la regla dura 5 para las 22 de una sola vez: un mismatch de dimensiones se descubre en
segundos en la PC local, no después de 40 minutos de cola.

---

## 9. Entregables

1. Sección nueva en `docs/Runs/RESULTS.md`: **"Exp I — estudio de hiperparámetros del Set
   Transformer"**, con la tabla de 22 filas (eje, valor, nº de parámetros, best val loss, EMA cruda,
   EMA asistida v2, minutos) y la banda de ruido medida explícita.
2. `experiments/E3_dos_conjuntos/plots/hp_sweep_ofat.png`.
3. Un párrafo de conclusión honesta, incluyendo el resultado tal como salga.

---

## 10. Alcance

**Dentro:**
- Generación de los 22 configs, el `.sh`, el recolector, la figura y los tests.
- Los tres cambios aditivos de código de §6.
- La sección de `RESULTS.md` (con las celdas de resultados vacías hasta que Lucas corra los jobs).

**Fuera:**
- Lanzar los jobs. Claude Code no puede hacer `sbatch` ni leer logs del cluster — eso lo hace Lucas
  por SSH. Este trabajo deja todo listo para `sbatch`.
- Reemplazar el checkpoint congelado. Si el estudio encontrara un ganador claro (mejora **mayor** a la
  banda de ruido), esa sería una decisión aparte, con su propio experimento de confirmación.
- Regularización (dropout, weight decay). Ya se estudió y falló en el Exp B (underfitting real,
  EMA 0.00 %/27.09 %); reabrirlo no aporta.
- Buscar hiperparámetros del oráculo o de Fase 1b (τ, K_max). Son de *post-procesamiento*, no de
  entrenamiento, y ya tienen su propio barrido medido en `RESULTS.md` (Exp G).
- HMBC, features nuevas o cualquier cambio de representación. Es la línea que el Exp F señaló como
  prometedora, pero es otro proyecto.

---

## 11. Criterios de éxito

El estudio es exitoso si, al terminar, se puede responder a un revisor con una tabla y una figura:

1. Se probaron **7 dimensiones de hiperparámetros** con 22 corridas controladas, mismo val congelado,
   mismo presupuesto de épocas, mismo cluster.
2. Se **midió** cuánta variación produce el azar (3 réplicas del baseline), y se fijó *antes* de mirar
   los resultados que las diferencias por debajo de esa banda no se interpretan.
3. La conclusión se sostiene sea cual sea el resultado:
   - **Si hay meseta** (lo esperado): el modelo elegido es el más chico dentro de la meseta, y la
     conclusión converge con las de Exp F (optimización) y el scaling study (datos) — el cuello es de
     información/dominio.
   - **Si hay un ganador real** (mejora > banda de ruido): se documenta, y queda como candidato para
     un experimento de confirmación con réplicas propias.

El éxito **no** se define como "encontrar un modelo mejor".
