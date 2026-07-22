# Exp F: cabeza Poisson + entrenamiento extendido

Checklist para correr esto en el cluster. Un solo modelo (Set Transformer,
igual al ganador de Fase 3) con dos cambios: cabeza Poisson en vez de MSE,
y 250 épocas en vez de 100 (el LR nunca bajó de 0.001 en Fase 3— 100
épocas puede haberse quedado corto).

## Antes de empezar

No hace falta generar datos nuevos — se reusan `peaks_pkl_202465.npz`,
`peaks_13c_202465.npz`, `vectors_13c_19v_202465.npy`, `smiles_202465.npy`
y `val_indices_frozen.npy`, ya en `/home/lpassaglia.iquir/DB_200k/` desde
Fase 3.

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/F_poisson_head`
3. **Smoke tests obligatorios antes de cualquier `sbatch` (regla 5):**
   ```bash
   python tests/test_dataset_f.py
   python tests/test_forward_settransformer.py
   python tests/test_poisson_loss.py
   python tests/test_oraculo.py
   ```
   El de forward imprime el conteo de parámetros: debería dar ~70k, igual
   que Fase 3 (softplus no agrega parámetros). Si no es chico (< 200k),
   avisá antes de entrenar.
4. Lanzar el entrenamiento:
   ```bash
   sbatch run_train.sh
   ```
5. **Revisá el log temprano**: `expF_train_<jobid>.out`. A diferencia de
   Fase 3, el val loss es Poisson NLL, no MSE — **no lo compares
   numéricamente contra el 0.0097 de Fase 3**. Mirá en cambio si el LR
   baja de 0.001 antes de la época 250 (en Fase 3 nunca bajó en 100).
6. Evaluar el checkpoint sobre el val congelado:
   ```bash
   sbatch run_eval.sh
   ```
7. (Opcional, para la GUI) Volcar predicciones:
   ```bash
   python dump_predictions.py --config config.yaml
   ```
8. Revisar `expF_eval_<jobid>.out`: comparar EMA cruda y asistida contra
   Fase 3 (2.26% / 91.35%), y el mapa de confusiones cruzadas — sobre todo
   si `CH2`↔`CH2-N`, `C-2X` y `=CH/Ar`↔`Imina` bajaron.
9. Agregar una fila "Exp F" a `docs/Runs/RESULTS.md`.
10. Avisá a Claude Code con los números.
