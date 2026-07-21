# Exp E — Fase 2: Modelo DeepSets sobre Picos

Checklist para correr esto en el cluster. A diferencia de Exp B/C, este
modelo entrena mucho más rápido (sin CNN sobre imágenes de 256×256) — no
esperes que tarde 10-14h como los anteriores, probablemente termine las 100
épocas en bastante menos de una hora.

## Antes de empezar: copiar el archivo de picos al cluster

`peaks_pkl_202465.npz` (generado en Exp E Fase 1b) está solo en tu máquina
local Windows (`E:\Proyectos\SciTrix\ScitrixDB\DB_nmr_to_vector\202K_suma\`).
Copialo a `/home/lpassaglia.iquir/DB_200k/` en el cluster (vía `scp` o lo
que uses normalmente) antes de seguir.

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/E2_deepsets`
3. Confirmar que existen en `/home/lpassaglia.iquir/DB_200k/`:
   - `peaks_pkl_202465.npz` (recién copiado)
   - `vectors_13c_19v_202465.npy`, `smiles_202465.npy` (ya deberían estar)
   - `val_indices_frozen.npy` (lo generó Exp D)
4. Smoke test obligatorio antes de cualquier `sbatch`:
   ```bash
   python tests/test_forward.py
   python tests/test_split_utils.py
   ```
   El primero debería mostrar ~23,315 parámetros (mucho menos que los
   ~223k de Exp C o los ~8.6M de V10) — si el número no coincide, algo
   está mal conectado en `model_e2.py`, avisá antes de entrenar.
5. Lanzar el entrenamiento:
   ```bash
   sbatch run_train.sh
   ```
6. **A diferencia de Exp B/C, revisá el log temprano** — como este modelo
   entrena mucho más rápido, probablemente veas las 100 épocas completas
   en minutos, no en horas. Mirá `expE2_train_<jobid>.out`.
7. Cuando termine, revisar el val loss final y compararlo contra V10
   (0.031) y Exp C (0.037) — no hay una referencia previa de picos, así
   que cualquier valor en ese orden de magnitud es razonable.
8. Evaluar el checkpoint sobre el mismo val congelado:
   ```bash
   sbatch run_eval.sh
   ```
9. (Opcional, para la GUI) Volcar las predicciones:
   ```bash
   python dump_predictions.py --config config.yaml
   ```
10. Revisar `expE2_eval_<jobid>.out`: copiar la tabla "EMA CRUDA vs
    ASISTIDA" y, sobre todo, si las confusiones `Cqsp2`↔`=CH/Ar` y
    `CH2`↔`CH2-N` (idénticas en V10/Exp B/Exp C) mejoraron o
    desaparecieron — es el indicador real de si la representación de
    picos resolvió el problema. Agregar los resultados a
    `docs/Runs/RESULTS.md`, fila "Exp E Fase 2".
11. Avisá a Claude Code con los números.

## Nota

Modelo deliberadamente chico (~23k parámetros) — no es un descuido, ver
`RATIONALE.md`. Si el resultado es bueno, el próximo paso (Set Transformer
o una variante más grande) es un experimento aparte, no una modificación
de este.
