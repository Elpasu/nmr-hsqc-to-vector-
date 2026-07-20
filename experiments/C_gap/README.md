# Exp C — Rebalanceo de la fusión (GAP)

Checklist para correr esto en el cluster. Entrena un modelo nuevo desde cero (no
reutiliza ningún checkpoint anterior) — consume horas de GPU.

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/C_gap`
3. Confirmar que existe `/home/lpassaglia.iquir/DB_200k/val_indices_frozen.npy`
   (lo generó Exp D). Si no está, avisá antes de seguir.
4. Smoke test obligatorio antes de cualquier `sbatch`:
   ```bash
   python tests/test_forward.py
   python tests/test_split_utils.py
   ```
   El primero debería mostrar ~223k parámetros (contra los ~8.6M de V10) — si el
   número no baja así de fuerte, algo está mal conectado, avisá antes de entrenar.
5. Lanzar el entrenamiento (~10-14h, igual que Exp B):
   ```bash
   sbatch run_train.sh
   ```
6. **Mientras corre**, podés ir mirando `expC_train_<jobid>.out` en las épocas 10, 20 y
   30. El script ya imprime una alerta (`[WARN]`) si el val loss sigue por encima de
   0.10 en esos puntos — es la misma señal que dio Exp B antes de terminar mal. Si ves
   esa alerta repetida, no hace falta esperar a que termine para avisar.
7. Cuando termine, revisar `expC_train_<jobid>.out`: comparar el val loss final contra
   el de V10 (0.031) y el de Exp B (0.176, que fue un fracaso).
8. Evaluar el checkpoint nuevo sobre el mismo val congelado:
   ```bash
   sbatch run_eval.sh
   ```
9. (Opcional, para la GUI) Volcar las predicciones:
   ```bash
   python dump_predictions.py --config config.yaml
   ```
10. Revisar `expC_eval_<jobid>.out`: copiar la tabla "EMA CRUDA vs ASISTIDA" y, sobre
    todo, el MAE de `Cqsp2` y `=CH/Ar` (las clases que más deberían mejorar si el
    rebalanceo funciona) a `docs/Runs/RESULTS.md`, fila "Exp C — GAP".
11. Avisá a Claude Code con los números — con eso decidimos si seguimos con la
    variante FC-bottleneck (si GAP underfitteó), con combinar C + regularización más
    suave (si C funcionó), o con otra cosa.

## Nota

`configs/config_V10.yaml` (en la raíz del repo) tiene `h5_filename` sin `_fast` y
`num_workers=4` — ambos inconsistentes con las reglas duras del proyecto. El
`config.yaml` de esta carpeta ya usa los valores correctos; no copiar
`config_V10.yaml` tal cual para futuros experimentos.
