# Exp D — Val set congelado

Checklist para correr esto en el cluster (`login-1`, env `NMR_env`). No
reentrena nada: reutiliza el checkpoint V10 ya entrenado.

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/D_val_congelado`
3. Generar el split congelado (no necesita GPU, corre en el login node,
   dura unos minutos):
   ```bash
   python split.py --config config.yaml
   ```
   Revisar el reporte impreso: cuántos duplicados canónicos aparecen,
   cuántas filas de train se eliminan por leak, y que el val final sea
   ~14428. Si `val final` se aleja mucho de 14428, o `SMILES invalidos` es
   grande, **parar y avisar** antes de seguir — puede indicar que el orden
   de las 144280 moléculas originales no se preservó al construir el
   dataset de 202k (ver `RATIONALE.md`, "Criterio de éxito/fracaso").
4. Confirmar que `val_indices_frozen.npy` quedó en
   `/home/lpassaglia.iquir/DB_200k/`.
5. Smoke test obligatorio antes de cualquier `sbatch` (CPU, sin checkpoint
   real):
   ```bash
   python tests/test_forward.py
   ```
6. Lanzar la re-evaluación del checkpoint V10 sobre el split nuevo:
   ```bash
   sbatch run_eval.sh
   ```
7. Cuando termine, revisar `expD_eval_<jobid>.out`: copiar la tabla
   "EMA CRUDA vs ASISTIDA" a `docs/Runs/RESULTS.md`, como fila nueva
   "V10-on-frozen-val (Exp D)".
8. Avisar a Claude Code con los números — con eso arrancamos Exp B.

## Nota

`configs/config_V10.yaml` (en la raíz del repo, el que se usó para
entrenar V10) tiene `h5_filename` sin `_fast` y `num_workers=4` — ambos
inconsistentes con las reglas duras del proyecto y con lo que documenta
`docs/Runs/RESULTS.md` sobre la corrida real. El `config.yaml` de esta
carpeta ya usa los valores correctos; no copiar `config_V10.yaml` tal cual
para futuros experimentos.
