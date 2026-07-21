# Exp E — Fase 1: extracción y validación de picos (design)

> Fase 1 de Exp E. Alcance: convertir el HSQC de imagen (256×256×2) a un conjunto de
> picos `(δC, δH, amp_ch0, amp_ch1)` por molécula, y validar que esa conversión no
> pierde información respecto al label. **No incluye entrenar ningún modelo todavía**
> — eso es Fase 2 (spec separado, después de ver los resultados de esta fase).

## Motivación

Exp C (rebalanceo de fusión, GAP) mejoró la EMA cruda levemente (0.61%→0.89%) pero
las confusiones de clase persisten idénticas en V10, Exp B y Exp C — tres
arquitecturas distintas, mismo patrón (`Cqsp2`↔`=CH/Ar`, `CH2`↔`CH2-N`). Evidencia de
que el cuello de botella es de representación, no de arquitectura. La auditoría de
pipeline (`scripts/audit_data_pipeline.py`) mostró además que la imagen HSQC es
99.2% espacio vacío — la CNN gasta la mayor parte de su cómputo en convolucionar
ceros.

La hipótesis de Exp E: reemplazar la imagen dispersa por una lista compacta de picos
reales (10-20 por molécula típica en vez de 131 072 valores) le da al modelo una
entrada mucho más limpia, sin el problema de localización espacial que hoy tiene que
resolver la CNN.

## Fuente de los picos y calibración

Se encontró el generador original del dataset:
`E:\Proyectos\SciTrix\ScitrixDB\DB-Batch0\Genera mapas de pkl v2.py` (fuera del repo,
solo como referencia — no se ejecuta, no se modifica). De ahí sale la calibración
exacta, sin inferir nada:

- δC (canal filas): rango `[0, 220]` ppm, binning **uniforme** en 256 bins.
- δH (canal columnas): rango `[-1, 15]` ppm, binning **uniforme** en 256 bins.
- Cada pico se dibuja como una gaussiana 2D, `sigma=0.5`, radio `6·sigma+1 ≈ 4px`
  (blob de ~9×9 px) — coincide con lo observado en la auditoría (~65 px no-cero por
  señal antes de deduplicar por blob).
- Canal 0 = DEPT escalado por N_H (signo: CH2 negativo, CH/CH3 positivo).
- Canal 1 = tipo de carbono normalizado (CH=0.33, CH2=0.67, CH3=1.0) — dato que ya
  existe en la imagen pero que hoy la CNN tiene que redescubrir por convolución.
- Los H de un mismo carbono (ej. los 2 H de un CH2) se dibujan en el **mismo pixel**
  — no es colisión real, es una sola señal de carbono. La colisión real (pérdida de
  información) es cuando **dos carbonos distintos** caen tan cerca en (δC, δH) que
  sus gaussianas se funden en un blob único.

Se elige **extraer los picos desde el h5 existente** (blob-detection), no
reprocesar desde el pkl original — evita depender de `snmgt01` y del pipeline DFT,
que están fuera del scope de este repo. Si la validación (más abajo) muestra pérdida
de información significativa, se reevalúa el fallback de reprocesar desde el pkl.

## Método de extracción

Script nuevo, self-contained: `experiments/E_peaks_prep/extract_peaks.py`.

1. **Calibración a `config/db.yaml`** (no hardcodeada en el script):
   ```yaml
   hsqc_calibration:
     c13_ppm_min: 0
     c13_ppm_max: 220
     h1_ppm_min: -1
     h1_ppm_max: 15
   ```
2. **Detección de blobs**: `scipy.ndimage.label` sobre la máscara de píxeles no-cero
   del canal 0, conectividad 8 (estructura `np.ones((3,3))`). Un componente conexo =
   un pico candidato.
3. **Por cada blob**: centroide ponderado por `|canal0|` en esa región (más robusto
   que el centroide sin ponderar para blobs fusionados/asimétricos) → convertido a
   `(δC, δH)` reales con la inversa exacta del binning uniforme:
   `ppm = bin/(resolution-1) * (ppm_max-ppm_min) + ppm_min`.
   Se guardan 4 valores crudos por pico: `(δC, δH, amp_ch0, amp_ch1)`, donde
   `amp_ch0`/`amp_ch1` son los valores de cada canal en el pixel de centroide
   (redondeado al entero más cercano). **No se decodifica CH/CH2/CH3 a mano** — el
   modelo de Fase 2 aprende a usar estos 4 números, igual que hoy aprende de la
   imagen cruda.
4. **Padding**: cada molécula tiene un número distinto de picos. Se guarda un tensor
   `(N, max_peaks, 4)` + máscara `(N, max_peaks)` de picos válidos. `max_peaks` se
   calcula dinámicamente como el máximo real de blobs encontrado en el dataset
   procesado (no un valor supuesto de antemano).

## Validación

Objetivo: medir si la extracción por blobs pierde información respecto al label,
reemplazando el chequeo por-píxel de `audit_data_pipeline.py` (que no podía
distinguir un blob de una señal real) por un chequeo por-blob correcto.

- Para cada molécula: `n_blobs` (de la extracción) vs `visible_label_count` (suma
  del label excluyendo `Cq`, `Cq-O`, `Cq-N`, `Cqsp2` — mismo criterio que
  `audit_data_pipeline.py`).
- Reporta: % de moléculas con match exacto, distribución del déficit
  (`visible_label_count - n_blobs`; positivo = colisión real, blobs fusionados),
  y algún caso ejemplo de colisión para inspección manual.
- Corre sobre una muestra aleatoria (mismo patrón que la auditoría anterior:
  `n_sample` configurable, seed=42) para no cargar el h5 completo en memoria en el
  login node.

## Salida

Nuevo archivo `peaks_202465.h5` en `base_dir` (mismo directorio que el resto de los
datasets, ruta desde `config/db.yaml`), con datasets `peaks (N, max_peaks, 4)` y
`peaks_mask (N, max_peaks)`. No se modifica el h5 de imágenes existente — queda
disponible para comparar ambas representaciones si hiciera falta.

## Testing (dadas las limitaciones del entorno local)

La máquina de desarrollo local no tiene `h5py`/`scipy` instalados — igual que en
experimentos anteriores, la verificación de la lógica de extracción se hace con:
- Tests unitarios de las funciones puras (`ppm_to_bin`/`bin_to_ppm` inversas,
  cálculo de centroide ponderado, cálculo de `max_peaks`) usando arrays sintéticos
  de numpy, sin depender de h5py.
- Un test de blob-detection con una imagen sintética pequeña construida a mano
  (2-3 blobs conocidos, uno de ellos deliberadamente fusionado) para verificar que
  el conteo y las posiciones salen correctos.
- El smoke test real contra el h5 de 202k lo corre Lucas en el cluster
  (`tests/test_forward.py`-equivalente para este script, 1 batch chico).

## Fuera de esta fase (spec futuro)

- Arquitectura del modelo de conjuntos (DeepSets primero, Set Transformer solo si
  DeepSets muestra que el approach de conjuntos supera a la CNN — según el criterio
  ya establecido en `docs/WORKFLOW_V11_para_ClaudeCode.md`).
- Cómo fusiona la rama de picos con las proyecciones 1D (`vec_c`/`vec_h`) y la
  Fórmula Molecular existentes — mantener ambas ramas por continuidad con el resto
  de los experimentos es el default a revisar en ese momento.
- Entrenamiento y evaluación end-to-end.

Ese diseño se hace después de ver los resultados de esta fase — si la extracción de
picos pierde información significativa, cambia el punto de partida de la Fase 2.
