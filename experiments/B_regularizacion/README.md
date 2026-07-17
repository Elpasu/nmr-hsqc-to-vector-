# Exp B — Regularización (dropout + weight_decay)

Checklist para correr esto en el cluster. A diferencia de Exp D, esto SÍ
entrena un modelo nuevo desde cero — consume horas de GPU.

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/B_regularizacion`
3. Confirmar que existe `/home/lpassaglia.iquir/DB_200k/val_indices_frozen.npy`
   (lo generó Exp D). Si no está, avisá antes de seguir — no se puede entrenar sin eso.
4. Smoke test obligatorio antes de cualquier `sbatch`:
   ```bash
   python tests/test_forward.py
   python tests/test_split_utils.py
   ```
5. Lanzar el entrenamiento (dura horas — V10 tardó ~10.3h para 100 épocas
   con el mismo dataset):
   ```bash
   sbatch run_train.sh
   ```
6. Cuando termine, revisar `expB_train_<jobid>.out`: comparar el gap
   train/val final contra el de V10 (train 0.013 vs val 0.031 —
   `docs/Runs/RESULTS.md`). Confirmar que `.err` está limpio.
7. Evaluar el checkpoint nuevo sobre el mismo val congelado:
   ```bash
   sbatch run_eval.sh
   ```
8. (Opcional, para inspeccionar en la GUI) Volcar las predicciones:
   ```bash
   python dump_predictions.py --config config.yaml
   ```
   Bajate el `.parquet` a tu PC y abrilo con `src/gui/gui_inspector.py`.
9. Revisar `expB_eval_<jobid>.out`: copiar la tabla "EMA CRUDA vs
   ASISTIDA" a `docs/Runs/RESULTS.md`, fila "Exp B — regularización".
   Compará contra "V10-on-frozen-val" (0.93% / 90.66%) sabiendo que esa
   referencia está inflada por contaminación train/val — la comparación
   más honesta es también el gap train/val del paso 6.
10. Avisá a Claude Code con los números — con eso decidimos si seguimos
    con Exp C o si hace falta ajustar dropout/weight_decay.
