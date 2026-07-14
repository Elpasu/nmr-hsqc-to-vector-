# Workflow de Experimentos — Proyecto NMR Deep Learning (Fase V11+)

> **Documento de especificación para Claude Code.**
> Escrito para que un agente genere los scripts sin ambigüedad. Cada tarea es
> autocontenida: objetivo, archivos a tocar, cambios exactos, criterio de
> aceptación y verificación.
>
> **Cluster objetivo:** `login-1` (usuario `lpassaglia.iquir`, partición
> `gpua10_hi`, env conda `NMR_env`). Todo corre acá.
>
> **Autor:** Lucas Passaglia · **Estado al generar este doc:** V10 (2 canales +
> FM + 19 clases + 202k) encolado y validado. Este doc define los experimentos
> **posteriores** al V10.

---

## 0. Contexto mínimo del proyecto (leer antes de tocar nada)

**Qué hace el modelo.** Recibe un espectro HSQC simulado (imagen 2D) + proyecciones
1D (¹³C y ¹H) + un tensor condicional, y predice un **vector de conteos enteros de
19 clases** de entornos de carbono (CH3, CH2, …, C-2X, C-3X). Métrica principal:
**Exact Match Accuracy (EMA)** — fracción de moléculas con el vector *entero* correcto.

**Serie histórica (EMA en val hermético, 144k):** V3 72.42 → V6-12v 83.79 (máximo
histórico) → V7 83.36 → V8 (2ch, sin FM) 78.51 → V9 (FM+19v) 79.82. El V10 (objeto
recién lanzado) combina 2 canales + FM + 19 clases + 202k. Objetivo del proyecto: **>85%**.

**Dato clave.** El dataset se amplió de 144 280 a **202 465** moléculas
(144 280 + 58 185 nuevas de scaffolds diversos). Anti-leak entre ambos sets ya
verificado = 0 solapamientos.

**Los dos formatos de HSQC** (no intercambiables):
- **V1 (1 canal)** `nmr_dataset_{N}.h5` → `hsqc (N, 256, 256)`. Modelos V7/V9.
- **V3 (2 canales)** `nmr_dataset_v3_{N}.h5` → `hsqc (N, 2, 256, 256)`. Modelos V8/V10.
  Canal 0 = DEPT escalado por N_H; Canal 1 = tipo CH (0.33/0.67/1.0).

El V10 y todos los experimentos de este doc usan **el formato de 2 canales (V3)**.

**Cuello de botella de dominio (no se resuelve con más datos):** los carbonos
cuaternarios sp2 (Cqsp2) son invisibles en HSQC. La FM ayuda; el fix real sería
HMBC simulado (ver Exp E / nota final).

---

## 1. Configuración canónica — `config/db.yaml`

> **Fuente única de verdad.** Todos los scripts leen rutas, N y constantes de acá.
> Prohibido hardcodear rutas o nombres de archivo en los `.py` (bug real que ya
> ocurrió: `smiles_path` hardcodeado en `train_v9.py` ignorando el config).

```yaml
# config/db.yaml
project:
  name: "nmr-hsqc-to-vector"
  cluster: "login-1"
  user: "lpassaglia.iquir"
  conda_env: "/home/lpassaglia.iquir/anaconda3/envs/NMR_env"
  partition: "gpua10_hi"

data:
  base_dir: "/home/lpassaglia.iquir/DB_200k"
  N: 202465                       # 144280 + 58185
  # --- dataset de 2 canales (V3) — el que usan V8/V10/V11 ---
  h5_v3:      "nmr_dataset_v3_202465.h5"      # hsqc (N,2,256,256)
  labels_19v: "vectors_13c_19v_202465.npy"    # (N,19)
  smiles:     "smiles_202465.npy"             # (N,)
  # --- referencia histórica (144k), para comparaciones ---
  base_dir_144k: "/home/lpassaglia.iquir/DB_144K"
  N_144k: 144280

model:
  num_classes: 19
  resolution: 256
  hsqc_channels: 2
  proj_dim: 512          # vec_c(256) + vec_h(256)
  cond_dim: 8            # [total_signals, total_CH2, C,H,N,O,S,Hal]

classes_19v:             # orden EXACTO — no reordenar
  - CH3
  - CH2
  - CH
  - Cq
  - CH3-O
  - CH2-O
  - CH-O
  - Cq-O
  - CH3-N
  - CH2-N
  - CH-N
  - Cq-N
  - "=CH2"
  - "=CH/Ar"
  - Cqsp2
  - Aldeh
  - Imina
  - C-2X
  - C-3X
  # índices CH2 (para el condicionante total_CH2 y el proyector de conteo): 1,5,9,12

hyperparameters:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  val_split: 0.1
  seed: 42
  scheduler:
    patience: 8          # scheduler CORREGIDO — estándar irrenunciable
    factor: 0.7

# Valores que se PONEN A PRUEBA en los experimentos de este doc:
regularization:
  dropout: 0.25          # Exp B
  weight_decay: 0.00001  # Exp B
```

---

## 2. Estructura de repo objetivo

```
nmr-hsqc-to-vector/
├── config/
│   └── db.yaml                 # fuente única de verdad (sección 1)
├── src/
│   ├── data/
│   │   └── dataset.py          # NMRDataset (2 canales + FM), lee db.yaml
│   ├── models/
│   │   ├── model_v10.py        # baseline actual (referencia, NO tocar)
│   │   └── model_v11.py        # variantes de arquitectura (Exp C, E)
│   ├── train.py                # entrenamiento genérico param. por config
│   ├── evaluate.py             # eval con flag --oraculo on/off (Exp A)
│   └── utils/
│       ├── split.py            # split fijo/congelado (Exp D)
│       └── config.py           # loader de db.yaml
├── configs/
│   ├── v10_baseline.yaml       # el que ya corre
│   ├── v11a_eval_crudo.yaml
│   ├── v11b_reg.yaml
│   ├── v11c_gap.yaml
│   └── v11d_valfijo.yaml
├── slurm/
│   └── run_train.sh            # plantilla, --gres=gpu:1
├── tests/
│   └── test_forward.py         # smoke test 1 batch (obligatorio pre-sbatch)
└── README.md
```

---

## 3. Convenciones para el agente (reglas duras, sacadas de errores reales)

Estas reglas NO son opcionales. Cada una corresponde a un bug que ya ocurrió:

1. **Encoding UTF-8 sí o sí.** Al crear archivos en el cluster usar heredoc con
   comillas (`cat > f.py << 'PYEOF'`), nunca pegar en editor que guarde Latin-1.
   Si un `.py` tira `SyntaxError: invalid continuation byte`, es encoding roto →
   `iconv -f latin1 -t utf-8`.
2. **SLURM usa `#SBATCH --gres=gpu:1`**, NO `--gpus=1`. Con `--gpus=1` el job queda
   pending con "Nodes DOWN/DRAINED" para siempre.
3. **Nunca hardcodear rutas ni nombres** en los `.py`. Todo sale de `db.yaml`.
   (Bug real: `train_v9.py` tenía `smiles_path = ".../smiles_144280.npy"` fijo.)
4. **Smoke test obligatorio antes de cada `sbatch`.** Correr `tests/test_forward.py`
   (1 batch, CPU, en el nodo de login) y confirmar los shapes esperados. No gastar
   cola de GPU para descubrir un mismatch de dimensiones.
5. **Jobs largos → siempre por SLURM (`sbatch`), nunca en el login node.** Para
   scripts de datos (no-GPU) que tarden, usar `nohup ... &` + `python -u` (unbuffered,
   para ver el log en vivo).
6. **El scheduler es `patience=8, factor=0.7`.** Es el estándar del proyecto. No
   volver al agresivo (`patience=3, factor=0.5`) que colapsó el LR en V7/V9.
7. **Cualquier cambio de arquitectura conserva `num_classes=19` y el orden de clases
   de `db.yaml`.** Reordenar clases desalinea los labels sin tirar error.
8. **Comparabilidad histórica:** el val set y la seed (42) deben ser idénticos entre
   experimentos para que las EMAs sean comparables (ver Exp D).

---

## 4. Los experimentos (orden de ejecución A → E)

> Cada experimento se puede lanzar de forma independiente. A, B, D son rápidos/baratos;
> C es el de mayor impacto arquitectónico; E es el salto conceptual a futuro.
> **El V10 baseline queda intacto como referencia.**

---

### EXP A — EMA cruda vs. asistida (diagnóstico de honestidad de la métrica)

**Objetivo.** Saber cuánto del rendimiento viene del modelo y cuánto del
post-procesamiento que fuerza los conteos. Hoy la EMA reportada usa
`ajustar_conteo_doble_exacto`, que **obliga** a la predicción a sumar exactamente
`total_signals` y `total_CH2` — dos valores calculados *desde el target*. Eso puede
estar inflando la EMA sin que la red realmente aprenda.

**Por qué importa.** Es la crítica #1 de la review. Sin este número, no sabés si tus
+pp entre versiones son aprendizaje o son el constraint. En un paper te lo exigen.

**Qué generar.** Un `evaluate.py` con un flag:
- `--oraculo on`  → comportamiento actual (proyector de doble restricción). = **EMA asistida**.
- `--oraculo off` → predicción = `np.floor(pred_raw)` con clip a ≥0, sin forzar sumas. = **EMA cruda**.

**Cambios exactos.**
- Refactorizar `evaluate_v9/v10.py`: envolver el bloque de `ajustar_conteo_doble_exacto`
  en un `if args.oraculo == "on"`. En el `else`, `pred_int = np.clip(np.floor(pred_raw), 0, None).astype(int)`.
- Reportar EMA, MAE por grupo y EMA por entorno para **ambos modos** en la misma corrida
  (correr el loop dos veces sobre el mismo val, o computar las dos predicciones por batch).
- Imprimir una tabla comparativa final: `EMA_cruda | EMA_asistida | Δ`.

**Criterio de aceptación.**
- Corre sobre el checkpoint del V10 (`checkpoints_V10_202k/..._best.pth`) sin error.
- Devuelve las dos EMAs. La cruda será menor; el Δ es el dato de interés.
- Si el Δ es grande (>10 pp), documentar: el modelo depende fuertemente del oráculo.

**Verificación.** No necesita GPU nueva: corre sobre el val set con el modelo ya
entrenado. Se puede lanzar apenas el V10 termine.

**Config:** `configs/v11a_eval_crudo.yaml` (hereda de v10, agrega `oraculo_modes: [on, off]`).

---

### EXP B — Reponer regularización (dropout + weight_decay)

**Objetivo.** Reintroducir la regularización que existía en V4/V5/V6 y que se perdió
entre V6 y V9. El proyecto tiene overfitting documentado (train ~0.010 vs val ~0.045)
y los modelos actuales (`model_v9.py`, `model_v10.py`) **no tienen dropout**, y el
optimizer **no tiene weight_decay**.

**Por qué importa.** Crítica #3. Es gratis, ataca un problema ya medido, y con 202k
+ overfitting conocido es casi seguro que ayuda.

**Cambios exactos.**
- En `model_v11.py` (copia de `model_v10.py`): agregar `nn.Dropout(p=cfg.dropout)`
  después de la ReLU de `fc_fusion1` y de `fc_fusion2`.
- En `train.py`: `optim.Adam(params, lr=..., weight_decay=cfg.weight_decay)`.
- Leer `dropout` y `weight_decay` de `db.yaml` (sección `regularization`).

**Criterio de aceptación.**
- El gap train/val al final del entrenamiento **baja** respecto al V10 baseline.
- La EMA (asistida y cruda) no empeora; idealmente mejora en val.
- Smoke test pasa (shapes idénticos al V10; dropout no cambia dimensiones).

**Verificación.** Entrenamiento completo por SLURM. Comparar curva train/val vs V10.

**Config:** `configs/v11b_reg.yaml` (v10 + `dropout: 0.25`, `weight_decay: 1e-5`).

---

### EXP C — Balancear las ramas (GAP / bottleneck convolucional)

**Objetivo.** Resolver el desbalance de la fusión. Hoy:
`fusion_dim = 65536 (conv) + 128 (1D) + 8 (FM)`. La rama conv domina 512:1 sobre la
1D — la red se ahoga en la imagen 2D e ignora las otras ramas. **Ese es el mecanismo
del "modality collapse"** observado desde el V4.

**Por qué importa.** Crítica #2, la de mayor techo entre las rápidas. Bonus: el
`Linear(65664 → 128)` actual tiene ~8.4M params en una sola capa (el ~90% del modelo);
reemplazarlo adelgaza el modelo ~10x y probablemente mejora la generalización.

**Cambios exactos (en `model_v11.py`).**
- Tras el último bloque conv (salida `64×32×32`), en vez de `view(-1, 65536)`:
  aplicar **Global Average Pooling** → `nn.AdaptiveAvgPool2d(1)` → salida `(batch, 64)`.
  (Alternativa: una FC `65536→128` como bottleneck explícito; GAP es más limpio y con
  menos params. Preferir GAP; dejar la FC como variante si GAP baja demasiado la EMA.)
- Nueva fusión: `fusion_dim = 64 (conv GAP) + 128 (1D) + 8 (FM) = 200`. Ramas
  balanceadas (mismo orden de magnitud).
- Ajustar `fc_fusion1 = nn.Linear(200, 128)`.

**Criterio de aceptación.**
- Nº de parámetros del modelo **baja drásticamente** (imprimir `sum(p.numel())` antes/después).
- La rama 1D deja de ser ignorada: como proxy, la EMA en grupos de carbonos
  **invisibles en HSQC** (Cqsp2 — depende del 1D/FM) debería mejorar o al menos no caer.
- EMA global ≥ V10 baseline. Si cae mucho con GAP puro, probar la variante FC-bottleneck.

**Verificación.** Entrenamiento completo. Comparar EMA global y MAE de Cqsp2 vs V10.
Este es el experimento con más chance de mover la aguja hacia el 85%.

**Config:** `configs/v11c_gap.yaml`.

---

### EXP D — Congelar el val set (rigor de comparación)

**Objetivo.** Hacer que todas las versiones se evalúen sobre **el mismo** conjunto de
validación, para que las EMAs sean estrictamente comparables y publicables.

**Por qué importa.** Crítica #5. Hoy `random_split(seed=42)` se aplica sobre el dataset
concatenado; al pasar de 144k a 202k, la composición del val cambia → V9-en-144k y
V10-en-202k **no comparten val** y no son comparables al pie de la letra. Además, los
~3400 duplicados internos de las 144k pueden hacer que un compuesto caiga en train y su
gemelo en val (fuga residual).

**Cambios exactos (en `src/utils/split.py`).**
1. **Deduplicación interna:** cargar `smiles_202465.npy`, canonicalizar con RDKit
   (`Chem.MolToSmiles(Chem.MolFromSmiles(s))`), detectar duplicados. Generar un array
   de índices únicos. Reportar cuántos duplicados hay.
2. **Val set congelado:** definir el val como las 14 428 moléculas originales de las
   144k (Opción B del workflow histórico), guardando sus índices en
   `val_indices_frozen.npy`. El resto (incluidas TODAS las 58k nuevas) va a train.
   Alternativa más simple (Opción A): mismo seed + 10%, aceptando que el val crece; solo
   si Opción B resulta muy invasiva. **Preferir Opción B para publicación.**
3. `train.py` y `evaluate.py` deben cargar el split desde el archivo de índices, no
   regenerarlo con `random_split`.

**Criterio de aceptación.**
- Reporta el nº de duplicados internos canónicos encontrados.
- El val set es idéntico entre cualquier par de experimentos (mismo `val_indices_frozen.npy`).
- Ninguna molécula de train comparte SMILES canónico con una de val (leak = 0).

**Verificación.** Script de chequeo: intersección de SMILES canónicos train∩val = ∅.

**Config:** `configs/v11d_valfijo.yaml` (apunta a `val_indices_frozen.npy`).

---

### EXP E — Representación de picos como conjunto (salto arquitectónico, futuro)

**Objetivo.** Reemplazar la imagen HSQC de 256×256 (casi toda ceros) por el objeto real:
un **conjunto de picos** `{(δC, δH, multiplicidad)}`, de tamaño variable y
permutation-invariant.

**Por qué importa.** Crítica #4, la de mayor techo a largo plazo. El CNN desperdicia
capacidad convolucionando espacio vacío. Un DeepSets o un Transformer chico sobre tokens
de pico es mucho más eficiente en parámetros y natural para el dato.

**Esto es exploratorio — no bloquea nada.** Se hace después de A-D. Dos sub-variantes:
- **E1 — DeepSets:** cada pico → MLP compartido → agregación (sum/mean) permutation-invariant
  → fusión con proj 1D + FM. Simple, fuerte baseline.
- **E2 — Set Transformer:** self-attention sobre los tokens de pico. Más capacidad,
  más caro. Solo si E1 muestra que el enfoque de sets supera al CNN.

**Requiere un cambio de datos:** generar, junto al h5, una representación de picos
(lista `(δC, δH, mult)` por molécula, paddeada a longitud máxima con máscara). Reutilizar
el pkl de shifts + la conectividad C-H de RDKit (ya está en `Genera_mapas_de_pkl_v2.py`).

**Criterio de aceptación (exploratorio).**
- Un modelo de sets con **órdenes de magnitud menos parámetros** que el CNN alcanza EMA
  comparable o superior al V10.
- Si lo logra, es el candidato a V12 y el argumento central para el paper.

**Config:** `configs/v12_sets.yaml` (fase separada).

---

## 5. Notas de dominio (no arquitectónicas, para el paper)

- **Cqsp2 es un límite de información, no de modelo.** Más datos no lo resuelve. El fix
  real es **HMBC simulado** (ve conectividad de cuaternarios) — mencionado en los reportes
  históricos. Vale más que volver a duplicar el dataset.
- **La loss actual (MSE sobre conteos + redondeo) es subóptima.** Para conteos, una cabeza
  **Poisson** o clasificación **ordinal** por posición es más natural (la incertidumbre de
  un conteo escala con su magnitud; MSE la trata como constante). Mejora conceptual, no urgente.

---

## 6. Orden de ejecución sugerido y qué esperar

| Exp | Costo | Riesgo | Impacto esperado | Depende de |
|-----|-------|--------|------------------|------------|
| A   | minutos (sin GPU nueva) | nulo | diagnóstico (no sube EMA, revela verdad) | V10 terminado |
| B   | 1 entrenamiento | bajo | +pp por menos overfitting | — |
| C   | 1 entrenamiento | medio | **el mayor salto hacia 85%** | — |
| D   | minutos + reentrenar | bajo | rigor (no sube EMA, la hace válida) | — |
| E   | fase de exploración | alto | techo a largo plazo | nueva repr. de datos |

**Recomendación:** A primero (barato, informa todo lo demás), después B y D en
paralelo si hay 2 GPUs (baratos, ortogonales), después C (el importante), y E como
línea de investigación separada.

---

## 7. Instrucción para Claude Code

> Generá el repo con la estructura de la sección 2. Empezá creando `config/db.yaml`
> (sección 1) y `src/utils/config.py` (loader). Después, para cada experimento A→E,
> generá los archivos indicados en "Qué generar / Cambios exactos", respetando TODAS
> las convenciones de la sección 3. Antes de proponer cualquier `sbatch`, generá y
> mencioná el smoke test (`tests/test_forward.py`). No modifiques los scripts del V10
> baseline: son la referencia. Trabajá un experimento a la vez, empezando por A.
