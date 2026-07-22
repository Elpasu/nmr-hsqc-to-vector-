# Exp E — Fase 3: dos conjuntos de picos (crosspeaks C-H + ¹³C), dos arquitecturas (design)

> Fase 3 de Exp E. Fase 2 (`docs/runs/RESULTS.md`, sección "Exp E — Fase 2")
> **fracasó**: el DeepSets sobre picos dio EMA cruda 0.74% (< 0.89% de Exp C) y las
> confusiones de cuaternarios (`Cqsp2`↔`=CH/Ar`, `CH2`↔`CH2-N`) **empeoraron**. La
> causa raíz no fue un bug ni la capacidad del modelo: los picos de Fase 1b son
> **crosspeaks C-H puros** (`extract_peaks_pkl.py` descarta todo carbono sin H,
> `if not h_shifts: continue`), así que los carbonos cuaternarios (`Cqsp2`, `Cq`,
> `Cq-O`, `Cq-N`) **nunca entran al modelo**. Encima, Fase 2 eliminó las
> proyecciones 1D (`vec_c`/`vec_h`) — que son espectros ¹³C/¹H reales calculados
> aparte desde el pkl y que **sí contienen los cuaternarios** — creyéndolas
> redundantes. Esta fase corrige el input y, ya que estamos, prueba si una
> arquitectura relacional (Set Transformer) aprovecha mejor esa información que un
> DeepSets puro.

## Motivación y alcance

Un HSQC (imagen o picos) es físicamente incapaz de mostrar carbonos sin H directo:
son invisibles. La única ventana a los cuaternarios en el pipeline es el espectro
¹³C 1D, que se calcula por separado desde el pkl DFT. En vez de reinyectar ese ¹³C
como un vector denso binneado (el binning de 256 fue justo lo que descartó Fase 1),
se lo trata como **conjunto de picos** — coherente con la evidencia de Fase 1b
(picos sin binning: 97% de match vs 11% de la imagen).

La tarea de detectar un cuaternario es, en el fondo, **relacional**: "¿qué δC del
¹³C NO tiene correspondencia en los crosspeaks?". Un DeepSets con promedio
enmascarado procesa cada pico aislado y no compara elementos entre sí, así que es la
arquitectura menos capaz de expresar esa operación. Un Set Transformer
(self-attention) sí puede aprenderla. Como entrenar cuesta ~16 min y el presupuesto
de GPU sobra largamente, se corren **las dos arquitecturas sobre exactamente el
mismo input y pipeline** para aislar dos efectos por separado:

- DeepSets (Fase 3) vs DeepSets (Fase 2) → efecto de **completar el input** (agregar
  cuaternarios).
- Set Transformer vs DeepSets (ambos Fase 3) → efecto de la **capacidad relacional**.

**Alcance:** dejar la carpeta lista para `sbatch` (extracción de datos incluida).
Entrenar y evaluar lo corre Lucas en el cluster, como en Exp B/C/E2.

## Datos

### Extracción de picos ¹³C (paso nuevo, corre local)

`experiments/E_peaks_prep/extract_peaks_13c_pkl.py` — variante de
`extract_peaks_pkl.py`. Diferencia única de fondo: en vez de agrupar por par C-H y
descartar los carbonos sin H, **itera todos los carbonos** del pkl (átomos con
`GetAtomicNum() == 6`, índices post-`AddHs`) y toma su `δC = nmr_shifts[c_idx]`.
Cada pico ¹³C es un vector de **una sola feature: `(δC,)`**. Se mantiene el dedup por
simetría (`_dedupe_symmetric_peaks` sobre `δC` a 6 decimales: carbonos equivalentes
por simetría tienen el mismo shift DFT y el label los cuenta una vez).

- **Nunca se incluye el nº de H del carbono** como feature: eso es esencialmente el
  label (CH3/CH2/CH/Cq) y sería fuga. Solo posición (δC).
- **Salida:** `peaks_13c_202465.npz` (`peaks_13c (N, M, 1)`, `mask_13c (N, M)`), donde
  `M` = máximo de picos ¹³C por molécula, determinado al extraer (`build_padded_arrays`
  de Fase 1). Copiar al cluster (`/home/lpassaglia.iquir/DB_200k/`) junto al
  `peaks_pkl_202465.npz` que ya está.
- **Validación de sanidad (obligatoria antes de entrenar):** contar #picos_¹³C por
  molécula vs `total_signals` (suma del label de 19 clases). Como ahora se incluyen
  los cuaternarios, el match esperado es **~100%** (los picos ¹³C = todos los carbonos
  con entorno distinto = exactamente lo que cuenta el label), muy por encima del 97%
  de los crosspeaks. Un match bajo indica que el pkl no guardó shifts de algún tipo de
  carbono → hay que revisarlo antes de gastar GPU. Reportar % igual que Fase 1b.

### Dataset (`dataset_e3.py`)

`NMRTwoSetsDataset`: carga en memoria los **dos** npz (`peaks_pkl_202465.npz` para
crosspeaks C-H, `peaks_13c_202465.npz` para ¹³C). El `cond_tensor` (8:
`[total_señales, total_CH2, C,H,N,O,S,Hal]`) se calcula EXACTO igual que
`dataset_v10.py`/`dataset_e2.py` (RDKit sobre `smiles_202465.npy` + label). Reutiliza
el split congelado de Exp D (`val_indices_frozen.npy` + `split_utils.py` copiado,
mismo patrón que B/C/E2). `__getitem__` devuelve
`((peaks_ch, mask_ch, peaks_13c, mask_13c, cond), target)`.

**Normalización (nueva vs E2), min-max desde `config/db.yaml` (`hsqc_calibration`):**

- `δC → (δC − 0) / (220 − 0)` (usa `c13_ppm_min`/`c13_ppm_max`), en ambos conjuntos.
- `δH → (δH − (−1)) / (15 − (−1))` (usa `h1_ppm_min`/`h1_ppm_max`), solo crosspeaks.
- `amp_ch0 → amp_ch0 / 3` (queda en ~[−1, 1]; `amp_ch0 = phase·mult`, `mult ∈ {1,2,3}`).
- `amp_ch1` se deja como está (`mult/3 ∈ {0.33, 0.67, 1.0}`, ya en [0,1]).

Min-max con la calibración física (no z-score): determinista, sale del config (no se
hardcodea, regla 3), y sin riesgo de filtrar estadísticas de train al val. Rompe la
identidad exacta de la rama crosspeaks con E2, lo cual es aceptable: el baseline a
batir es **Exp C (0.89% cruda)**, no E2 (que fracasó).

`num_workers: 0` se mantiene (estándar del proyecto; además todo está en memoria, sin
h5py ni I/O por ítem).

## Arquitecturas

`train.py` es único y lee `model.arch` del config (`"deepsets"` | `"settransformer"`)
para instanciar el modelo. Todo lo demás (loss, scheduler, épocas, split, dataset) es
idéntico entre las dos corridas — es lo que hace la comparación limpia. Dos configs
(`config_deepsets.yaml`, `config_settransformer.yaml`) que solo difieren en ese campo
(y en `experiment_name`/`checkpoint_dir`).

### A — DeepSets (`model_e3_deepsets.py`)

Dos ramas permutation-invariant:

- **Rama crosspeaks:** MLP compartido `4 → 64 → 64` por pico, promedio enmascarado
  sobre `mask_ch` → `aggA` (64). (Misma estructura que E2, con input ya normalizado.)
- **Rama ¹³C:** MLP compartido `1 → 64 → 64` por pico, promedio enmascarado sobre
  `mask_13c` → `aggB` (64).
- **Fusión:** `[aggA(64), aggB(64), cond(8)] = 136 → 128 → 64 → 19`.

Si una molécula no tiene picos válidos en un conjunto, ese agregado queda en cero
(clamp del divisor a `min=1.0`, igual que E2).

### B — Set Transformer (`model_e3_settransformer.py`)

Los dos conjuntos se **unen en un solo set de tokens** para que la atención pueda
cruzar información entre crosspeaks y ¹³C (que es donde vive la señal de "cuaternario =
δC del ¹³C sin match en crosspeaks"):

- **Tokenización:** cada pico se proyecta a `d_model = 64` con un lineal **según su
  tipo** — `Linear(4 → 64)` para crosspeaks, `Linear(1 → 64)` para ¹³C — más un
  **embedding de tipo** (2 entradas: crosspeak/¹³C) sumado al token. Se concatenan en
  la dimensión de tokens: `T = max_peaks_ch + M`. La máscara combinada
  `[mask_ch, mask_13c]` marca los tokens válidos.
- **Encoder:** 2 bloques de self-attention tipo SAB (Set Transformer, Lee et al. 2019):
  multihead self-attention (`n_heads = 4`) + feed-forward, con LayerNorm y residual. La
  atención respeta la máscara: los tokens de padding no son atendidos ni atienden.
- **Pooling:** PMA (Pooling by Multihead Attention) con **1 seed vector** → un vector
  de `d_model` (agregación por atención, permutation-invariant).
- **Fusión:** `[pooled(64), cond(8)] → 128 → 64 → 19`.

Capacidad: del orden de decenas de miles de parámetros (2 capas, `d_model=64`,
`n_heads=4`, conjuntos ≤ ~40 tokens) — trivial frente a los 8.6M de V10, en línea con
la decisión de mantener el modelo chico (evidencia: V10 grande sobreajustó y perdió
contra Exp C).

**Hiperparámetros del Set Transformer** (`d_model=64`, `n_heads=4`, `n_layers=2`,
`n_seeds=1`) viven en el config, no hardcodeados.

## Qué se mantiene idéntico (comparabilidad)

Loss `ConstrainedMSELoss`, scheduler `ReduceLROnPlateau` patience=8/factor=0.7 (regla
6), 100 épocas, seed 42, batch 64, `num_workers=0`, `cond_tensor` (FM, misma
convención), split congelado (mismo val 14428 que B/C/E2), sin regularización
explícita. `num_classes=19` y el orden de clases de `db.yaml` fijo (regla 7). Único
cambio real de datos vs E2: se agrega el conjunto ¹³C y la normalización.

## Testing

Smoke test offline obligatorio antes de cualquier `sbatch` (regla 5), **uno por
arquitectura**: forward con un batch sintético de los dos conjuntos + máscaras,
verificando shape de salida `(B, 19)` y conteo de parámetros razonable (chico, del
orden de E2/Exp C, no de V10). Más un test de la extracción ¹³C sobre unas pocas
moléculas (validar #picos_¹³C vs `total_signals` y el dedup por simetría). Mismo patrón
que `tests/test_forward.py` de E2.

## Estructura de carpeta

`experiments/E3_dos_conjuntos/`:
`config_deepsets.yaml`, `config_settransformer.yaml`, `dataset_e3.py`,
`model_e3_deepsets.py`, `model_e3_settransformer.py`, `train.py` (parametrizado por
`arch`), `evaluate.py`, `dump_predictions.py`, `split_utils.py`,
`run_train_deepsets.sh`, `run_train_settransformer.sh`, `run_eval.sh`, `tests/`,
`README.md`, `RATIONALE.md`. El extractor `extract_peaks_13c_pkl.py` va en
`experiments/E_peaks_prep/` (junto a `extract_peaks_pkl.py`).

## Criterio de éxito / fracaso

`evaluate.py` reutiliza el patrón de E2 y reporta **las dos EMAs** — modo crudo
(`--oraculo off`) y modo asistido (`--oraculo on`, oráculo de doble restricción con
`idx_ch2 = [1, 5, 9, 12]`) — más el mapa de confusiones cruzadas y el desglose por
entorno. Ambas métricas se comparan sobre el val congelado de 14428 (idéntico a
B/C/E2).

- **EMA cruda:** objetivo mínimo ≥ 0.89% (Exp C, el mejor limpio hasta ahora).
- **EMA asistida (oráculo):** debe superar al menos a Exp C (70.02%) y a E2 (70.90%),
  e idealmente a V10 (74.92%, la mejor asistida con evaluación limpia). El objetivo que
  se marcó Lucas para el proyecto es **~90% asistida** — ninguna corrida limpia se
  acercó todavía. Se interpreta con la cautela ya documentada (Exp A): la asistida
  depende de dónde caen los errores vía el oráculo, así que se lee junto a la cruda y a
  las confusiones, no en aislamiento.
- **Indicador real de éxito:** que las confusiones `Cqsp2`↔`=CH/Ar` y `CH2`↔`CH2-N`
  bajen respecto a V10/B/C/E2. Es lo que confirmaría que el problema era el input
  incompleto (cuaternarios ausentes), no la arquitectura.
- **Lectura comparativa:** si DeepSets(F3) > DeepSets(F2) pero se estanca, y Set
  Transformer lo supera, la ganancia extra es de la capacidad relacional. Si ambos
  siguen fallando en los cuaternarios pese al input completo, el cuello no es ni el
  input ni la arquitectura de sets → apunta al HMBC simulado (`WORKFLOW_V11` líneas
  39-41 / 344-346) como el verdadero fix de dominio.

## Fuera de alcance

- Perceiver u otras variantes de atención más pesadas (los conjuntos son chicos,
  ≤ ~40 tokens; no justifican un latent bottleneck).
- Enriquecer el pico ¹³C con codificación de Fourier/RBF de δC (queda como palanca si
  la rama de 1 feature entrena flojo).
- Cabeza de conteo Poisson / loss ponderada por clase (mejora conceptual separada, ver
  `docs/runs/RESULTS.md` "Auditoría de distribución de clases").
- HMBC simulado (cambio de datos mayor, experimento aparte).
