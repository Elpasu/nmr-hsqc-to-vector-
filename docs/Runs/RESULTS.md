# Results Log — NMR HSQC→Vector

One entry per run. Raw logs live in `docs/runs/<name>_train.out`.
**Target metric is EMA (Exact Match Accuracy), not Val Loss.** EMA comes from
`evaluate_v10.py`; Val Loss (MSE) only tracks convergence.

| Model | Ch | Cls | Data | Reg. | Best Val Loss (ep) | EMA crude | EMA assist | Notes |
|-------|----|----|------|------|--------------------|-----------|------------|-------|
| V10 baseline | 2 | 19 | 202k | none | 0.0303 (76) | 0.61% | 74.92% | overfits from ~ep48; assisted EMA inflated by oracle (Exp A) |
| V10-on-frozen-val (Exp D) | 2 | 19 | 202k | none | — (no retrain) | 0.93% | 90.66% | same ckpt as V10; val is ~90% train-contaminated, NOT a clean number — see below |
| Exp B — regularizacion | 2 | 19 | 202k | drop=0.25, wd=1e-5 | 0.1764 (97) | 0.00% | 27.09% | **regression, not fix** — underfits, see below |
| Exp C — GAP (fusion) | 2 | 19 | 202k | none | 0.0370 (100) | 0.89% | 70.02% | crude EMA improved vs V10 true baseline; assisted below target, see below |
| Exp E Fase 1 — extraccion de picos (blob-detection) | n/a (sin imagen) | n/a | 202k | n/a | n/a (sin entrenar) | n/a | n/a | 88.75% de moleculas con colision de blobs — imagen 256x256 no alcanza, ver seccion propia |
| Exp E Fase 1b — extraccion de picos (pkl original) | n/a (sin imagen) | n/a | 202k | n/a | n/a (sin entrenar) | n/a | n/a | 97.19% match exacto, 2.19% colision real — valida pasar a Fase 2, ver seccion propia |
| Exp E Fase 2 — DeepSets (picos) | n/a (sin imagen) | 19 | 202k | none | 0.0323 (97) | 0.74% | 70.90% | **FRACASO segun criterio propio** — crude < Exp C (0.89%), confusiones de cuaternarios EMPEORARON; picos son crosspeaks puros y se quito la proj 1D. Ver seccion |
| Exp E Fase 3 — DeepSets (2 conjuntos: crosspeaks + 13C) | n/a (sin imagen) | 19 | 202k | none | 0.0201 (97) | 2.28% | 82.96% | **exito** — agrega conjunto 13C (con cuaternarios); Cqsp2 con error 13.5%->5.2% vs E2. Ver seccion |
| **Exp E Fase 3 — Set Transformer (2 conjuntos: crosspeaks + 13C)** | n/a (sin imagen) | 19 | 202k | none | **0.0097 (97)** | 2.26% | **91.35%** | **mejor resultado del proyecto** — primera vez que cruza el objetivo ~90% asistida limpio; rompe la confusion estructural Cqsp2<->=CH/Ar (persistia en V10/B/C/E2). Ver seccion |
| Exp E Fase 3 — estudio de escalado (Set Transformer, 10-100% train) | n/a (sin imagen) | 19 | 18.7k-202k | none | 0.0097 (100%) | 0.9-2.3% | 83.67% -> 91.35% | **meseta de datos** — 75%->100% no mueve EMA ni val loss; ampliar el dataset no rendiria. Ver seccion |
| Exp F — cabeza Poisson + 250 epocas (Set Transformer) | n/a (sin imagen) | 19 | 202k | none | 0.3051 Poisson-NLL (no comp.) | 0.60% | 90.63% | **NO mejoro** vs Fase 3 ST (91.35%): asistida -0.72pp, cruda cae (2.26->0.60); confusiones CH2<->CH2-N e Imina->=CH/Ar intactas. Ver seccion |
| Migración XPU — Exp E Fase 3 Set Transformer en Intel XPU (Clementina) | n/a (sin imagen) | 19 | 202k | none | **0.0086 (99)** | 1.71% | **92.12% (v1) / 92.14% (v2)** | **mismo codigo/config que la fila de arriba, corrido en Intel XPU en vez de NVIDIA A10** — paridad confirmada (val loss y EMA dentro de la varianza esperada, incluso mejores). Ver seccion y `docs/MIGRACION_XPU_Clementina_XXI.md` |

---

## V10 — baseline (2ch + FM + 19v + 202k)

- **Date:** 2026-07-15 · **SLURM:** 2374453 · A10
- **Config:** `configs/config_V10.yaml` · **Data:** `nmr_dataset_v3_202465_fast.h5`
- **Arch:** `model_v10.py` (Conv2d 2→16→32→64 + 1D 512→256→128 + cond 8) — **no dropout / no weight_decay**
- **Sched:** ReduceLROnPlateau patience=8 factor=0.7 · **Split:** random_split(seed=42), val 10%
- **Run:** 100 epochs · 620 min (~6.1 min/ep) · `.err` clean · ckpt `checkpoints_V10_202k/nmr_202k_v10_2ch_fm_19v_best.pth`
- **Best Val Loss:** 0.0303 @ ep76
- **EMA:** cruda 0.61% · asistida 74.92% (Exp A, oráculo de doble restricción). Δ = +74.3pp — la EMA asistida satura la métrica, no sirve para comparar versiones. Ver `docs/PROMPT_superpowers_mejoras.md`.

**Takeaways:**
- Scheduler behaved correctly (LR 0.001 held to ep62, then smooth decay). No premature LR collapse (unlike V9).
- **Clear overfitting:** Train 0.013 vs Val 0.031; Val plateaus at ~0.031 from ep~48 while Train keeps dropping → validates **Exp B** (dropout + weight_decay). Epochs ~50–100 added no generalization.

---

## Exp D — val set congelado (V10 checkpoint, no retrain)

- **Date:** 2026-07-17 · **SLURM:** 2375430 · **Config:** `experiments/D_val_congelado/config.yaml`
- **Change vs baseline:** same V10 checkpoint (`nmr_202k_v10_2ch_fm_19v_best.pth`), re-evaluated on a
  frozen val (14428 "original 144k" molecules, historical `random_split(seed=42)` reproduced over
  `[0,144280)`), instead of the random 10%-of-202k split V10 was actually trained/evaluated on.
  `split.py` dedup report: 2928 canonical-duplicate groups (3433 excess molecules), 723 train rows
  dropped for leak against val, verified leak=0 (train∩val canonical SMILES).
- **EMA crude / assisted:** 0.93% / 90.66% (Δ +89.7pp). By entorno (assisted): Alifáticos 96.5%,
  Heteroatómicos O/N 95.0%, sp2 94.8%, X-Multiples 98.5%.
- **⚠️ Not a clean number:** ~90% of this frozen val was part of V10's actual training set (V10
  trained on a random 10%-of-202k split, a different permutation than this historical 144k-based
  split). The jump vs the original 0.61%/74.92% (Exp A, V10's true held-out split) is expected
  contamination, not a real improvement — treat this row as a rough anchor only. Exp B and Exp C
  will exclude this frozen val from their own training, so **their numbers on this same val ARE
  clean** — compare B vs C directly; compare either vs this row only with the caveat above in mind.
  See `experiments/D_val_congelado/RATIONALE.md`.
- **Takeaway:** split/dedup machinery verified working end-to-end (val landed exactly on the
  expected 14428, leak=0 confirmed). `val_indices_frozen.npy` now fixed for all future experiments.

---

## Exp B — regularización (dropout + weight_decay)

- **Date:** 2026-07-17→18 · **SLURM train:** 2375431 (100 ep, ~10.2h) · **SLURM eval:** 2376413
- **Config:** `experiments/B_regularizacion/config.yaml` · **Change vs baseline:** `model_v11b.py`
  (dropout=0.25 after ReLU of fc_fusion1/fc_fusion2) + weight_decay=1e-5 (Adam). Trained from
  scratch (not a re-eval like Exp D) on the Exp D frozen split (train=187314, val=14428, leak
  removed=723 — split machinery reproduced Exp D's numbers exactly, ruling out a split bug).
- **Loss:** Train 0.3745 / Val 0.1764 (best, ep97) — **~6-12x higher than V10's 0.013/0.031**, not
  in the same regime at all. LR decayed correctly (0.001→0.000058, no premature collapse). This
  is genuine underfitting, not measurement noise: 100 full epochs, LR annealed 17x, and train loss
  never got remotely close to V10's.
- **EMA crude / assisted:** 0.00% / 27.09% — both **worse than V10's true baseline** (0.61%/74.92%),
  not just worse than the inflated Exp-D reference. Δ=+27.09pp (much smaller than V10's +74.3pp,
  consistent with a model that's further from correct overall, so the oracle has less to work with).
- **Diagnosis (not yet confirmed, next step's job):** `fc_fusion1 = Linear(65664, 128)` is already
  a severe bottleneck — the entire HSQC image gets compressed into 128 numbers before dropout even
  applies. Stacking 25% dropout on that already-thin, overloaded channel (dropout again on the
  64-unit layer after it) likely starves the network of the image signal it needs, compounding the
  "modality collapse" Exp C already targets rather than fixing the overfitting gap. weight_decay=1e-5
  alone is too mild to explain a 6-12x loss inflation; dropout placement/magnitude is the prime
  suspect.
- **Takeaway:** regularization as specified (0.25 / 1e-5) is **not a safe default for this
  architecture** — do not carry these values into Exp C without re-testing. This result is
  evidence FOR prioritizing Exp C (rebalance the fusion bottleneck) over a milder regularization
  retry, since the failure mode implicates the same bottleneck Exp C targets.

---

## Exp C — GAP (rebalanceo de fusión) — resultado final

- **Date:** 2026-07-20→21 · **SLURM train:** 2376427 (100 ep, 603.7 min ≈ 10.1h) ·
  **SLURM eval:** 2376888 · **Params:** 222,883 (V10: 8,603,299, ~38.6x menos).
- **Loss:** Train 0.0203 / Val 0.0370 (mejor, ep100, todavía descendiendo — el scheduler
  nunca terminó de estabilizar). Mismo orden de magnitud que V10 (0.031); nada que ver con
  el 0.176 de Exp B. Ningún `[WARN]` de underfitting se disparó en las 100 épocas.
- **EMA crude / assisted:** 0.89% / 70.02% (Δ +69.1pp). Por entorno (asistida): Alifáticos
  82.71%, Heteroatómicos O/N 80.03%, sp2 83.54%, X-Multiples 96.32%.
- **Comparación honesta:** EMA cruda 0.89% > 0.61% de V10 (baseline real, split limpio) —
  mejora real, aunque chica en términos absolutos. EMA asistida 70.02% < 74.92% de V10 —
  por debajo, pero la asistida depende de dónde caen los errores del modelo (vía el
  oráculo), no es la métrica de comparación primaria del proyecto. Lejos del objetivo de
  ~90% asistida que se marcó el usuario — ningún experimento corrido hasta ahora se acercó
  a eso con una evaluación limpia.
- **Confusiones que sobreviven al oráculo — MISMAS que en V10 y Exp B:** `Cqsp2`↔`=CH/Ar`
  (~40% de los errores cruzados de esas dos clases), `CH2`↔`CH2-N` (44-52%), `CH`↔`=CH/Ar`.
  Tres arquitecturas distintas (V10 sin cambios, Exp B con regularización, Exp C con GAP),
  mismo patrón de confusión — evidencia fuerte de que el problema **no es arquitectónico**,
  es de representación/información. Motiva pasar a Exp E (representación de picos).
- **Takeaway:** rebalancear la fusión ayudó (crude EMA sube, 38.6x menos parámetros, sin
  underfitting) pero no resuelve las confusiones estructurales. Siguiente paso: Exp E
  (conjunto de picos en vez de imagen), no más iteración sobre la arquitectura CNN.

---

## Exp E — Fase 1: extracción de picos vía blob-detection

- **Fecha:** 2026-07-21 · **Scripts:** `experiments/E_peaks_prep/extract_peaks.py`,
  `validate_peaks.py` (login node, sin GPU) · **Dataset:** las 202465 moléculas completas.
- **Qué se hizo:** convertir el HSQC de imagen (2×256×256) a una lista de picos
  `(δC, δH, amp_ch0, amp_ch1)` por molécula, detectando componentes conexos
  (conectividad-8) sobre el canal 0. Calibración exacta (δC `[0,220]` ppm, δH `[-1,15]`
  ppm, uniforme, 256 bins) copiada de `Genera_mapas_de_pkl_v2.py`.
- **Resultado de extracción:** `max_peaks=14`, picos por molécula min=0 max=14
  promedio=4.42.
- **Resultado de validación (blobs detectados vs conteo visible del label):**
  - Match exacto: **11.24%** de las moléculas.
  - Con colisión (visible > blobs, pico perdido por fusión): **179 695 / 202 465
    (88.75%)**.
  - Déficit promedio en las que colisionan: **3.81** picos perdidos.
  - Peores casos: moléculas con 32 carbonos visibles en el label, de las cuales el
    blob-detector solo separó 3-4 (deficit=28-29) — moléculas grandes con zona
    alifática muy poblada.
- **Diagnóstico:** no es un bug de la extracción — es un límite físico de la imagen
  fuente. Cada pico ocupa un radio de ~4px (`sigma=0.5`), que en ppm reales es
  **~3.45 ppm en δC y ~0.25 ppm en δH**. Dos carbonos dentro de esa ventana se funden
  en un solo blob de forma indistinguible — la CNN de V10/Exp B/Exp C ve exactamente
  la misma fusión (no puede separarlos tampoco), así que esto no es una desventaja de
  blob-detection frente al enfoque de imagen: es un techo compartido por ambos, que
  solo se hizo visible al contar blobs en vez de píxeles (el chequeo viejo de
  `audit_data_pipeline.py`, basado en conteo de píxeles, no lo detectaba — comparaba
  contra una magnitud que no medía colisión real).
- **Decisión (con Lucas):** pasar al plan de contingencia ya pactado — reprocesar los
  picos directamente desde los datos originales del pkl/DFT (sin pasar por el binning
  de 256×256), donde δC/δH son valores reales sin cuantizar y la colisión debería caer
  a niveles marginales. Spec de esa fase, pendiente.

---

## Exp E — Fase 1b: extracción de picos desde el pkl original (sin binning)

- **Fecha:** 2026-07-21 · **Script:** `experiments/E_peaks_prep/extract_peaks_pkl.py`
  (local, máquina Windows de Lucas — sin GPU, sin cluster) · **Dataset:** las
  202465 moléculas completas, matching por posición
  (`mol_ids_144280.npy`/`mol_ids_58185.npy` ↔ `smiles_202465.npy`, verificado
  con `verify_smiles_alignment` antes de generar cualquier pico).
- **Qué se hizo:** en vez de detectar picos en la imagen 256×256 (Fase 1,
  blob-detection), extraerlos directamente de los shifts DFT del pkl original
  (`nmr_calculated_data_scaled_144K.pkl` + `nmr_calculated_data_scaled_58k.pkl`),
  agrupando por carbono (no por par C-H) — sin ningún binning de por medio.
- **Bug encontrado y corregido durante la corrida:** la primera pasada dio
  38.51% match exacto con ~61% de moléculas en *exceso* de picos (no déficit).
  Causa: carbonos químicamente equivalentes por simetría (ej. las 2 posiciones
  orto de un anillo para-sustituido) reciben el mismo shift DFT — en un HSQC
  real son indistinguibles (una sola señal), y el label los cuenta una vez,
  pero la extracción contaba un pico por átomo sin deduplicar. Fix: colapsar
  picos con `(δC, δH)` idénticos (hasta 6 decimales) antes de contar. Verificado
  en 3 moléculas reales antes de aplicar el fix al dataset completo.
- **Resultado final (post-fix):** `max_peaks=32`, picos por molécula
  promedio=7.79 (min=0, max=32).
  - Match exacto: **97.19%**.
  - Con colisión real (visible > picos): **4425 / 202465 (2.19%)**.
  - Déficit promedio en las que colisionan: **1.06**.
- **Comparación directa con Fase 1 (blob-detection sobre la imagen):**

  | | Fase 1 (imagen 256×256) | Fase 1b (pkl, sin binning) |
  |---|---|---|
  | Match exacto | 11.24% | **97.19%** |
  | Colisión | 88.75% | **2.19%** |
  | Déficit promedio | 3.81 | 1.06 |
  | Picos/molécula (prom.) | 4.42 | 7.79 |

- **Diagnóstico:** confirma que la pérdida de información de Fase 1 era del
  binning de la imagen (sigma=0.5 ⇒ ~3.45 ppm δC / ~0.25 ppm δH por blob), no
  un límite del dato en sí. Trabajando con los shifts reales, la colisión cae
  a niveles marginales (2.19%, genuina — carbonos distintos con shifts
  realmente muy cercanos, no artefacto de resolución).
- **Salida:** `peaks_pkl_202465.npz` (`peaks (N, max_peaks, 4)`,
  `peaks_mask (N, max_peaks)`), en `DB_nmr_to_vector/202K_suma/` local.
- **Decisión:** este resultado valida pasar a Exp E Fase 2 (armar y entrenar
  el modelo de conjuntos, DeepSets como primer candidato) sobre esta
  representación. Pendiente: escribir el spec de Fase 2.

---

## Exp E — Fase 2: modelo DeepSets sobre picos — resultado final

- **Fecha:** 2026-07-21 · **SLURM train:** 2376953 (100 ep, ~9.2s/ep, ~16 min total) ·
  **SLURM eval:** 2376954 · **Params:** 23,315 (V10: 8.6M, Exp C: 223k).
- **Config:** `experiments/E2_deepsets/config.yaml` · **Arch:** `model_e2.py` (DeepSets:
  MLP por pico 4→64→64, promedio enmascarado, fusión 72→128→64→19).
- **Loss:** Train 0.0388 / Val **0.0323** (mejor, ep97). Mismo orden que V10 (0.031) y Exp C
  (0.037). Convergencia limpia, scheduler OK (LR 0.001→0.00049, sin colapso), `.err` vacío.
  **No hay ningún bug de código** — el experimento corrió como se diseñó.
- **EMA crude / assisted:** 0.74% / 70.90% (Δ +70.16pp). Por entorno (asistida): Alifáticos
  86.69%, Heteroatómicos O/N 83.20%, sp2 81.71%, X-Multiples 95.27%.
- **Veredicto vs criterio propio (`RATIONALE.md`): FRACASO.** EMA cruda 0.74% < 0.89%
  (objetivo mínimo = Exp C). EMA asistida 70.90% ≈ Exp C (70.02%), sin mejora real. Y el
  indicador de éxito que fijó la propia RATIONALE — que las confusiones `Cqsp2`↔`=CH/Ar` y
  `CH2`↔`CH2-N` mejoraran — **empeoró**: `Cqsp2`→`=CH/Ar` 53.6% (Exp C ~40%),
  `=CH/Ar`→`Cqsp2` 50.2%, `CH2`→`CH2-N` 69.6% (Exp C 44-52%), `CH2-N`→`CH2` 56.7%.
- **Causa probable (hallazgo del análisis post-mortem, no un bug):** los picos de Exp E son
  **crosspeaks C-H puros** — `experiments/E_peaks_prep/extract_peaks_pkl.py:65-70` descarta
  todo carbono sin H (`if not h_shifts: continue`). Los cuaternarios (`Cqsp2`, `Cq`, `Cq-O`,
  `Cq-N`) **nunca entran al set de picos**: el modelo sólo puede inferirlos vía la FM +
  restricciones de conteo. Además, a diferencia de V10/B/C, Exp E **eliminó las proyecciones
  1D (`vec_c`/`vec_h`)** con el argumento de que eran "redundantes con los picos" (spec Fase 2,
  líneas 40-43) — pero el diseño ORIGINAL del workflow (E1, `WORKFLOW_V11` líneas 324-325)
  mandaba fusionar "picos + **proj 1D** + FM". La desviación quitó una entrada que el plan
  original mantenía.
- **⚠️ PREGUNTA ABIERTA (decide el próximo experimento):** ¿`vec_c`/`vec_h` son **(a)**
  proyecciones de la imagen HSQC 2D — que por física NO contienen cuaternarios, exactamente el
  mismo conjunto de carbonos que los picos — o **(b)** espectros ¹³C/¹H 1D reales desde el pkl,
  que SÍ ven cuaternarios? Lo decide `Genera_mapas_de_pkl_v2.py` (snmgt01), no verificable
  desde este repo. **Si (b):** su eliminación fue una pérdida de información real y el próximo
  experimento obvio y barato es reincorporar la proj 1D al DeepSets. **Si (a):** el cuello es
  información pura (Cqsp2 invisible en HSQC) y el fix es HMBC simulado, como ya anticipa
  `WORKFLOW_V11` líneas 39-41 / 344-346.
- **Takeaway:** cambiar imagen→picos NO resolvió las confusiones estructurales; las agravó al
  quitar la proj 1D. Refuerza que `Cqsp2`/`Cq` son un límite de INFORMACIÓN (invisibles en
  HSQC), no de arquitectura — cuatro representaciones distintas (V10 CNN, Exp B, Exp C GAP,
  Exp E picos) fallan igual en ellos. Antes de saltar a Set Transformer, el experimento
  correcto es reañadir la proj 1D al DeepSets (si aporta cuaternarios) o el HMBC simulado.

---

## Exp E — Fase 3: dos conjuntos de picos (crosspeaks C-H + ¹³C), dos arquitecturas

- **Fecha:** 2026-07-22 · **SLURM train:** 2376980 (DeepSets, 20.7 min) / 2376981
  (Set Transformer, 39.0 min) · **SLURM eval:** 2377009 (DeepSets) / 2377018
  (Set Transformer) · **Params:** DeepSets 35,795 / Set Transformer 70,163
  (ambos chicos por diseño; V10: 8,603,299).
- **Qué cambia vs Fase 2:** se agrega un segundo conjunto de picos ¹³C
  (`peaks_13c_202465.npz`, extraído con `extract_peaks_13c_pkl.py` — todos los
  carbonos del pkl, sin filtrar por H, a diferencia de los crosspeaks de Fase
  1b). Validación de la extracción sobre las 202465 moléculas reales: match
  exacto 99.35%, déficit real 0.01% (21 moléculas), exceso 0.64% (1295
  moléculas, causado por carbonos equivalentes por simetría con δC
  ligeramente distinto — benigno, no filtrado). Los desplazamientos se
  normalizan min-max desde `config/db.yaml` (no se normalizaba en Fase 2). Se
  prueban dos arquitecturas sobre el mismo dataset/loss/split/scheduler: un
  DeepSets de dos ramas (una por conjunto) y un Set Transformer (self-attention
  sobre la unión de ambos conjuntos con embedding de tipo + pooling PMA).
- **Loss (100 épocas, ambos):** DeepSets Train 0.0266 / Val **0.0201** (mejor,
  ep97). Set Transformer Train 0.0130 / Val **0.0097** (mejor, ep97) — el
  valor más bajo del proyecto, menos de un tercio del de Fase 2 (0.0323). En
  ninguno de los dos el LR bajó de 0.001 en 100 épocas (scheduler no encontró
  meseta) — margen de mejora, no estancamiento.
- **EMA crude / assisted:**
  - **DeepSets:** 2.28% / 82.96% (Δ +80.68pp). Por entorno (asistida):
    Alifáticos 92.75%, Heteroatómicos O/N 89.83%, sp2 91.02%, X-Multiples
    97.08%.
  - **Set Transformer:** 2.26% / **91.35%** (Δ +89.09pp). Por entorno
    (asistida): Alifáticos 96.06%, Heteroatómicos O/N 93.80%, sp2 96.56%,
    X-Multiples 98.15%.
- **Éxito según criterio propio (`RATIONALE.md` de la carpeta):** ambas
  arquitecturas superan el objetivo mínimo de EMA cruda (0.89%, Exp C) y la
  asistida de referencia (Exp C 70.02%, E2 70.90%). El Set Transformer además
  supera a V10 (74.92%, hasta ahora el techo) y **cruza por primera vez, con
  evaluación limpia, el objetivo de ~90% asistida** que se marcó el usuario
  para el proyecto.
- **Las confusiones persistentes de cuaternarios bajaron y cambiaron de
  naturaleza** (el indicador de éxito real definido en la RATIONALE):

  | Clase — % moléculas con error | E2 (Fase 2, fracaso) | DeepSets F3 | Set Transformer F3 |
  |---|---|---|---|
  | `Cqsp2` | 13.5% | 5.2% | **1.2%** |
  | `=CH/Ar` | 8.9% | 6.0% | **2.1%** |
  | `CH2` | 3.6% | 3.0% | 1.9% |
  | `CH2-N` | 3.7% | 3.4% | 2.2% |

  Más importante que la caída absoluta: en V10/Exp B/Exp C/E2/DeepSets-F3, la
  confusión dominante de `Cqsp2` era siempre `=CH/Ar` (40-63% de sus errores).
  En el Set Transformer, `Cqsp2` ya no confunde principalmente con `=CH/Ar`
  (ahora `Cq-O` 34.1%, `=CH/Ar` 21.6%, `C-2X` 12.5%), y `=CH/Ar` tampoco
  confunde ya con `Cqsp2` como principal (pasa a `Imina`/`CH-O`). La pareja
  `Cqsp2`↔`=CH/Ar`, que sobrevivió a cinco arquitecturas distintas sobre la
  imagen/crosspeaks, se rompió al combinar el conjunto ¹³C (que sí ve los
  cuaternarios) con una arquitectura capaz de comparar ambos conjuntos entre sí.
- **Diagnóstico:** confirma las dos hipótesis del diseño — (1) el input
  incompleto de Fase 2 (crosspeaks puros, sin cuaternarios) era la causa raíz
  de su fracaso, no la capacidad del modelo; (2) la capacidad relacional
  (Set Transformer vs DeepSets, mismo input) aporta una mejora adicional
  sustancial (82.96% → 91.35% asistida, 0.0201 → 0.0097 val loss), consistente
  con que detectar un cuaternario es una operación de comparación entre
  conjuntos que el promedio enmascarado del DeepSets no puede expresar bien.
- **Resto del error (para el próximo experimento):** en el Set Transformer,
  lo que más pesa ahora es la pareja `CH2`↔`CH2-N` (1.9%/2.2%, la otra
  confusión histórica, que bajó pero no se rompió como `Cqsp2`↔`=CH/Ar`),
  seguida de `C-2X` (1.7%, clase rara, sistemas polihalogenados) y un frente
  nuevo y menor `=CH/Ar`↔`Imina` (25.8% de los errores de `=CH/Ar`).
- **Takeaway:** mejor resultado limpio del proyecto hasta ahora. Spec:
  `docs/superpowers/specs/2026-07-22-exp-e-fase3-dos-conjuntos-picos-design.md`.
  Plan: `docs/superpowers/plans/2026-07-22-exp-e-fase3-dos-conjuntos-picos.md`.
  Código: `experiments/E3_dos_conjuntos/`.

---

## Exp E — Fase 3: estudio de escalado de datos (Set Transformer, sin cambios de modelo/loss)

- **Fecha:** 2026-07-22 · **SLURM train:** 2377114-2377118 · **SLURM eval:** 2377168-2377172 ·
  **Config:** `experiments/E3_dos_conjuntos/config_scaling_{10,25,50,75,100}.yaml`.
- **Qué es:** ablación sobre el Set Transformer de Fase 3 (arquitectura y loss `ConstrainedMSELoss`
  **sin cambios** — no es Poisson) entrenado sobre fracciones crecientes del train set, con el
  **mismo val congelado** (14428) para las 5. Subsampleo determinístico anidado
  (`RandomState(42)`), 100 épocas cada una. Pregunta: si se volviera a ampliar el dataset (como
  144k→202k), ¿serviría?

  | Fracción | Train N | Best Val Loss | EMA cruda | EMA asistida |
  |---|---|---|---|---|
  | 10%  | 18,731  | 0.0203 | 1.29% | 83.67% |
  | 25%  | 46,828  | 0.0138 | 1.48% | 88.38% |
  | 50%  | 93,657  | 0.0117 | 0.91% | 90.04% |
  | 75%  | 140,485 | 0.0097 | 1.28% | 91.55% |
  | 100% | 187,314 | 0.0097 | 2.26% | 91.35% |

- **Resultado — meseta de datos.** Saltos de EMA asistida por tramo: 10→25% **+4.71pp**, 25→50%
  +1.66pp, 50→75% +1.51pp, **75→100% −0.20pp** (plano, dentro del ruido). El val loss lo confirma
  independientemente: 75% y 100% dan **exactamente 0.0097**. Las últimas ~47k moléculas (140k→187k)
  no aportaron nada medible.
- **Sanity checks:** el 100% (91.35%) reproduce exacto el Set Transformer de Fase 3 (mismo
  modelo/data/seed ✓). La EMA cruda es ruidosa (0.9-2.3%, sin tendencia monótona), como se espera de
  una métrica que exige los 19 conteos exactos sin oráculo.
- **Takeaway:** volver a ampliar el dataset **no rendiría** — el esfuerzo va mejor a la cabeza de
  salida (Exp F), a la representación, o al dominio (HMBC). Figuras en
  `experiments/E3_dos_conjuntos/plots/` (`scaling_curve_ema.png` + las demás; se regeneran con
  `python plots/make_plots.py`, sin torch).

---

## Exp F — cabeza Poisson + entrenamiento extendido (no mejoró)

- **Fecha:** 2026-07-22 · **SLURM train:** 2377113 (250 ep, 98.3 min) · **SLURM eval:** 2377247 ·
  **Config:** `experiments/F_poisson_head/config.yaml` · **Params:** 70,163 (idéntico a Fase 3 Set
  Transformer — `softplus` no agrega parámetros).
- **Qué cambia vs Fase 3 Set Transformer:** (1) cabeza `softplus` (`λ ≥ 0`) +
  `ConstrainedPoissonLoss` (Poisson NLL + restricción de suma) en vez de `ConstrainedMSELoss`;
  (2) épocas 100 → 250. Dataset, split, arquitectura y scheduler (`patience=8/factor=0.7`) sin
  cambios. Decisión explícita del usuario de combinar las dos variables en un solo experimento.
- **Loss (Poisson NLL):** Train 0.3273 / Val **0.3051** (mejor, ep~245). **⚠️ Este valor NO es
  comparable con el 0.0097 (MSE) de Fase 3** — son escalas distintas (Poisson NLL vs MSE). La
  comparación F vs F3 se hace SOLO por EMA (pendiente).
- **Comportamiento del entrenamiento — lo informativo hasta ahora:** a diferencia de Fase 3 (donde
  el LR nunca bajó de 0.001 en 100 épocas), acá **el scheduler sí actuó**: LR 0.001 → 0.00002 (17×),
  y las últimas ~10 épocas están planas (Val 0.3052-0.3054). Entrenó **hasta saturación real** — lo
  que valida que en Fase 3 el presupuesto de épocas se había quedado corto. Si la EMA no mejora pese
  a esto, el cuello no era ni la cabeza de salida ni las épocas.
- **EMA crude / assist:** 0.60% / 90.63%. **No mejoró** vs Fase 3 Set Transformer (2.26% / 91.35%):
  la asistida quedó marginalmente por debajo (−0.72pp) y la cruda **cayó** (−1.66pp, 2.26→0.60). Por
  entorno (asistida): Alifáticos 95.79%, Heteroatómicos O/N 93.57%, sp2 95.74%, X-Múltiples 98.13% —
  casi calcado a Fase 3.
- **Confusiones intactas:** el patrón residual es el MISMO que Fase 3, la cabeza Poisson y las 250
  épocas no lo tocaron: `CH2`↔`CH2-N` sigue dominante (CH2→CH2-N 79.5%, CH2-N→CH2 60.5%),
  `Imina`→`=CH/Ar` 82.6%, `C-2X`→`=CH/Ar`/`Cqsp2`.
- **Interpretación (informativo, no solo negativo):** con el scheduler ya saturado (LR→0.00002) y una
  cabeza de conteo más natural (Poisson), el modelo **no superó** a Fase 3 → el cuello ya **no es de
  optimización** (loss / épocas / cabeza de salida). Fase 3 Set Transformer ya estaba en el techo de
  lo que esta representación permite. El error residual (`CH2`↔`CH2-N`, `Imina`↔`=CH/Ar`) es robusto a
  estos cambios ⇒ apunta a **límite de información/dominio** (separar un CH2/CH alifático de su versión
  unida a N, o una imina de un aromático, sin señales adicionales), no a un problema de modelo. Nota:
  la cabeza Poisson además **empeora el modo crudo** (softplus + NLL redondea peor sin el oráculo).
- **Takeaway:** **Fase 3 Set Transformer (MSE, 100 ép) queda como el mejor modelo del proyecto** —
  Exp F no lo desplaza. El próximo paso no es afinar la optimización sino inyectar **información
  nueva**: HMBC simulado (ve conectividad de cuaternarios) o features que separen X-alifático de X-N.
  Spec: `docs/superpowers/specs/2026-07-22-exp-f-poisson-y-escalado-design.md`.

---

## Migración XPU — Exp E Fase 3 Set Transformer en Intel XPU (Clementina XXI)

- **Fecha:** 2026-07-23/24 · **SLURM train:** 1489559 (`cn073`, 70.3 min) · **SLURM eval:** 1489606 ·
  **Config:** `experiments/E3_dos_conjuntos/config_settransformer.yaml` (idéntico al de la fila
  "Exp E Fase 3 — Set Transformer" de arriba, mismo checkpoint name, mismo split congelado,
  mismo `patience=8/factor=0.7`). **Único cambio:** hardware — NVIDIA A10 (`login-1`, CUDA) →
  Intel Data Center GPU Max 1550 (Clementina XXI, backend `xpu`). Detalle completo del proceso
  de migración, decisiones de diseño y gotchas de infraestructura en
  `docs/MIGRACION_XPU_Clementina_XXI.md`.
- **Qué NO cambia:** arquitectura (`model_e3_settransformer.py`), dataset (`dataset_e3.py`),
  loss (`ConstrainedMSELoss`), split (`val_indices_frozen.npy`, Exp D), hiperparámetros. El código
  de entrenamiento/evaluación es aditivo (`device_utils.pick_device()`): sigue corriendo en CUDA
  sin cambios de comportamiento si no se exporta nada.
- **Paridad numérica CPU↔XPU (pre-requisito, `tests/test_paridad_cpu_xpu.py`):** mismos pesos,
  mismas entradas, forward/gradientes/molécula-100%-enmascarada — diferencias de `~1e-7`–`1e-9`
  (ruido de FP32 entre hardware, `atol=2e-5`). Validado antes de correr el entrenamiento real.
- **Best Val Loss:** **0.0086 @ ep99** (vs 0.0097 @ ep97 en A10 — mejor, no peor).
- **EMA crude / assist:** 1.71% / **92.12%** (oráculo v1) / **92.14%** (oráculo v2, zeroing por
  heteroatómos) — vs 2.26% / 91.35% en A10. Por entorno (asistida v1): Alifáticos 96.47%,
  Heteroatómicos O/N 94.63%, sp2 96.71%, X-Múltiples 98.17% — prácticamente calcado a A10.
- **Por qué difiere un poco (no es un bug):** el `ReduceLROnPlateau` disparó en un punto distinto
  entre las dos corridas (A10: LR se mantuvo en 0.001 las 100 épocas; XPU: bajó a 0.00049 desde
  ep95) — consecuencia acumulada del ruido de FP32 entre hardware sobre 100 épocas, no un error de
  implementación. Confirmado por la paridad numérica pre-entrenamiento: los operadores dan lo
  mismo, la trayectoria estocástica diverge un poco, como es esperable.
- **Tiempo:** 70.3 min (XPU) vs 39.0 min (A10) — **~1.8× más lento**, esperado y fuera de alcance
  de esta migración: FP32 puro (sin XMX/BF16, eso es Fase 5 opcional), modelo chico (70k
  parámetros, el overhead de lanzamiento de kernels pesa proporcionalmente más), y
  `num_workers=0` (regla dura 1) sin optimizar. El objetivo de esta fase era paridad de
  resultados, no velocidad.
- **Takeaway:** **migración de E3 a Intel XPU funcionalmente validada** — mismo código, mismos
  hiperparámetros, resultado equivalente (de hecho ligeramente mejor) al baseline A10. Logs crudos
  en `docs/runs/XPU_Clementina_E3_settransformer/`.

---

## Auditoría de distribución de clases (dataset completo, 202465 moléculas)

- **Fecha:** 2026-07-20 · **Script:** `scripts/audit_class_frequency.py` (login node, sin GPU).
- **Hipótesis original (descartada por los datos):** que `Cqsp2` y `=CH/Ar` fallan tanto en el
  modelo por ser clases *raras* en el dataset. Los datos dicen lo contrario.
- **Hallazgo real:** `Cqsp2` (92.62% de las moléculas, promedio 3.378/molécula) y `=CH/Ar` (80.34%,
  promedio 3.271/molécula) son las **dos clases más comunes de las 19**, no las más raras — están
  al fondo de la tabla ordenada por rareza, no arriba. Entre las dos suman ~6.6 señales promedio
  por molécula, una porción enorme del total de carbonos de una molécula típica.
- **Por qué importa:** como la EMA exige acertar las 19 clases simultáneamente, un error en la
  clase más frecuente y de mayor conteo arruina muchísimas más moléculas que un error en una clase
  rara. `Cqsp2` además es literalmente invisible en HSQC (carbono cuaternario, sin H propio) — el
  modelo solo puede inferirlo indirectamente (probablemente vía la Fórmula Molecular). La
  combinación "clase dominante + invisible en la imagen" la vuelve la sospechosa número uno de la
  EMA cruda baja, más que un problema aislado entre 19.
- **Bonus — composición de las 58k moléculas nuevas vs las 144k originales:** el dataset ampliado
  no es "más de lo mismo". Deltas de presencia más grandes: `CH2-N` 18.71%→57.34% (+38.6pp),
  `CH-N` 7.70%→33.78% (+26.1pp), `=CH/Ar` 76.57%→89.70% (+13.1pp), `Cqsp2` 90.79%→97.17% (+6.4pp).
  Las moléculas nuevas tienen mucho más contenido nitrogenado y más carbonos sp2/cuaternarios.
- **Implicancia para la estrategia:** el cuello de botella puede no ser solo arquitectónico
  (Exp C) — el desbalance de clases (`Cqsp2`/`=CH/Ar` dominan el conteo) sugiere que una loss que
  pondere por clase, o una cabeza de conteo específica para las clases dominantes, podría mover
  más la aguja que solo rebalancear la fusión. Ver discusión de próximos pasos en curso.

---

<!-- Template for next entries — keep it short:

## <Exp> — <one-line description>
- **Date:** · **SLURM:** · **Config:** · **Data:**
- **Change vs baseline:** <what this experiment tests>
- **Best Val Loss:** X @ epY · **EMA crude / assisted:** X / Y
- **Takeaway:** <1-2 lines>
- Also add a row to the top table.
-->
